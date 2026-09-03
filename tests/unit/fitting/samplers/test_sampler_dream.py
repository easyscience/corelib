# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause
"""Unit tests for ``DreamSampler`` — mirrors
``src/easyscience/fitting/samplers/sampler_dream.py``.

Ported from the former ``TestBumpsSample`` suite in
``tests/unit/fitting/minimizers/test_minimizer_bumps.py`` when the former
``Bumps.mcmc_sample`` moved here as ``DreamSampler.run`` (easyscience/core#280).
That entry point, and ``Fitter.mcmc_sample``, have since been removed —
``Sampler`` and ``DreamSampler.run`` are the supported APIs.
"""

import logging
from unittest.mock import MagicMock

import numpy as np
import pytest

import easyscience.fitting.samplers.sampler_dream
from easyscience.fitting.engine_base import EngineBase
from easyscience.fitting.minimizers.bumps_utils import BumpsProgressMonitor
from easyscience.fitting.minimizers.utils import FitError
from easyscience.fitting.samplers import DreamSampler


class TestDreamSamplerRun:
    """Tests for ``DreamSampler.run()`` and its helpers."""

    # Sentinel value to signal "set fitter.state = None" in _setup_driver_mock
    ABORT = object()

    @pytest.fixture
    def engine(self) -> DreamSampler:
        return DreamSampler(obj='obj', fit_function='fit_function')

    @pytest.fixture(autouse=True)
    def _mock_bumps_internals(self, monkeypatch):
        """Prevent run() from constructing real BUMPS objects.

        ``run()`` imports ``DreamFit`` from the real ``bumps`` package
        internally and builds its problem via ``build_curve_problem``,
        which would try to build real model objects.  We redirect those
        to mocks and also mock ``FitDriver`` (a module-level import) so
        the whole flow stays under test control.
        """
        import bumps.fitters

        monkeypatch.setattr(bumps.fitters, 'DreamFit', MagicMock())
        self._set_problem(monkeypatch, MagicMock())

    @staticmethod
    def _set_problem(monkeypatch, problem):
        """Point ``build_curve_problem`` at a canned (problem, counter, curve) triple."""
        monkeypatch.setattr(
            easyscience.fitting.samplers.sampler_dream,
            'build_curve_problem',
            MagicMock(return_value=(problem, MagicMock(), MagicMock())),
        )

    def _setup_driver_mock(
        self, monkeypatch, fitter_state_value=None, fit_result=None, fit_side_effect=None
    ):
        """Helper to create a mocked FitDriver with configurable behavior.

        :param fitter_state_value: If ``None``, ``driver.fitter.state`` will be
            a regular MagicMock (non-None).  Pass ``ABORT`` to set it to ``None``
            and simulate user abort.
        """
        from easyscience import global_object

        global_object.stack.enabled = False

        mock_driver = MagicMock()
        mock_driver.clip = MagicMock()

        if fit_side_effect is not None:
            mock_driver.fit.side_effect = fit_side_effect
        else:
            mock_driver.fit.return_value = fit_result or (np.array([1.0]), 0.0)

        mock_driver.stderr = MagicMock(return_value=np.array([0.1]))

        if fitter_state_value is TestDreamSamplerRun.ABORT:
            mock_driver.fitter.state = None
        else:
            mock_state = MagicMock()
            mock_state.Nvar = 1
            mock_state.Npop = 5
            mock_state.labels = ['p_param_0']
            mock_draw = MagicMock()
            mock_draw.points = np.array([[1.0]])
            mock_draw.logp = np.array([0.5])
            mock_state.draw.return_value = mock_draw
            mock_driver.fitter.state = mock_state

        mock_FitDriver = MagicMock(return_value=mock_driver)
        monkeypatch.setattr(
            easyscience.fitting.samplers.sampler_dream, 'FitDriver', mock_FitDriver
        )
        return mock_FitDriver, mock_driver

    def test_is_an_engine(self, engine: DreamSampler) -> None:
        assert isinstance(engine, EngineBase)
        assert engine.package == 'bumps'

    @pytest.mark.parametrize(
        'kwargs, match',
        [
            ({'samples': 0}, 'samples must be a positive integer'),
            ({'samples': -1}, 'samples must be a positive integer'),
            ({'burn': -1}, 'burn must be a non-negative integer'),
            ({'thin': 0}, 'thin must be a positive integer'),
        ],
    )
    def test_run_invalid_args(self, engine: DreamSampler, kwargs, match) -> None:
        """Invalid samples/burn/thin values raise ValueError before any sampling.

        This is the single source of truth for these checks — the higher-level
        ``Sampler`` relies on it.
        """
        with pytest.raises(ValueError, match=match):
            engine.run(
                x=np.array([1.0]),
                y=np.array([0.1]),
                weights=np.array([1.0]),
                samples=kwargs.get('samples', 10),
                burn=kwargs.get('burn', 0),
                thin=kwargs.get('thin', 1),
            )

    @pytest.mark.parametrize(
        'overrides, match',
        [
            ({'y': np.array([0.1])}, 'x and y must have the same shape'),
            ({'x': np.array([1.0, np.nan])}, 'x cannot contain NaN'),
            ({'y': np.array([0.1, np.inf])}, 'y cannot contain NaN'),
            ({'weights': np.array([1.0])}, 'Weights must have the same shape'),
            ({'weights': np.array([1.0, np.nan])}, 'Weights cannot be NaN'),
            ({'weights': np.array([1.0, 0.0])}, 'Weights must be strictly positive'),
        ],
    )
    def test_run_invalid_data(self, engine: DreamSampler, overrides, match) -> None:
        """Shape mismatches and non-finite/non-positive data raise ValueError
        before any sampling."""
        data = {
            'x': np.array([1.0, 2.0]),
            'y': np.array([0.1, 0.2]),
            'weights': np.array([1.0, 1.0]),
        }
        data.update(overrides)
        with pytest.raises(ValueError, match=match):
            engine.run(**data, samples=10, burn=0, thin=1)

    def test_run_rejects_none_weights(self, engine: DreamSampler) -> None:
        """weights=None gets a clear ValueError instead of a shape error
        from ``np.asarray(None)`` (CR-5)."""
        with pytest.raises(ValueError, match='weights must not be None'):
            engine.run(
                x=np.array([1.0]),
                y=np.array([0.1]),
                weights=None,
                samples=10,
                burn=0,
                thin=1,
            )

    def test_run_basic(self, engine: DreamSampler, monkeypatch) -> None:
        """Verify that run() returns a dict with expected keys."""
        mock_FitDriver, _ = self._setup_driver_mock(monkeypatch)

        result = engine.run(
            x=np.array([1.0, 2.0]),
            y=np.array([0.1, 0.2]),
            weights=np.array([1.0, 1.0]),
            samples=100,
            burn=20,
            thin=2,
            population=5,
        )

        assert isinstance(result, dict)
        assert 'draws' in result
        assert 'param_names' in result
        assert 'internal_bumps_object' in result
        assert 'logp' in result
        mock_FitDriver.assert_called_once()

    def test_run_with_progress_callback(self, engine: DreamSampler, monkeypatch) -> None:
        """Verify progress callback is wired up as a monitor."""
        mock_FitDriver, _ = self._setup_driver_mock(monkeypatch)
        progress_callback = MagicMock()

        result = engine.run(
            x=np.array([1.0]),
            y=np.array([0.1]),
            weights=np.array([1.0]),
            samples=10,
            burn=5,
            thin=1,
            progress_callback=progress_callback,
        )

        assert result is not None
        call_kwargs = mock_FitDriver.call_args.kwargs
        assert 'monitors' in call_kwargs
        assert len(call_kwargs['monitors']) == 1
        assert isinstance(call_kwargs['monitors'][0], BumpsProgressMonitor)

    def test_run_aborted_by_user_raises_fit_error(self, engine: DreamSampler, monkeypatch) -> None:
        """Verify that sampling abortion raises FitError."""
        self._setup_driver_mock(monkeypatch, fitter_state_value=TestDreamSamplerRun.ABORT)

        with pytest.raises(FitError, match='Sampling aborted by user'):
            engine.run(x=np.array([1.0]), y=np.array([0.1]), weights=np.array([1.0]))

    def test_run_driver_exception_restores_parameters(
        self, engine: DreamSampler, monkeypatch
    ) -> None:
        """Verify that a driver exception during sampling restores parameter values."""
        self._setup_driver_mock(monkeypatch, fit_side_effect=RuntimeError('driver failed'))
        engine._restore_parameter_values = MagicMock()

        with pytest.raises(RuntimeError, match='driver failed'):
            engine.run(x=np.array([1.0]), y=np.array([0.1]), weights=np.array([1.0]))

        engine._restore_parameter_values.assert_called_once()

    def test_run_population_param(self, engine: DreamSampler, monkeypatch) -> None:
        """population kwarg is forwarded to DREAM as pop."""
        mock_FitDriver, _ = self._setup_driver_mock(monkeypatch)

        engine.run(
            x=np.array([1.0]),
            y=np.array([0.1]),
            weights=np.array([1.0]),
            samples=10,
            burn=0,
            thin=1,
            population=7,
        )

        call_kwargs = mock_FitDriver.call_args.kwargs
        assert call_kwargs['pop'] == 7

    def test_run_sampler_kwargs_forwarded(self, engine: DreamSampler, monkeypatch) -> None:
        """sampler_kwargs entries are merged into the DREAM kwargs."""
        mock_FitDriver, _ = self._setup_driver_mock(monkeypatch)

        engine.run(
            x=np.array([1.0]),
            y=np.array([0.1]),
            weights=np.array([1.0]),
            samples=10,
            burn=0,
            thin=1,
            sampler_kwargs={'trim': False},
        )

        assert mock_FitDriver.call_args.kwargs['trim'] is False

    def test_run_rejects_non_callable_callback(self, engine: DreamSampler) -> None:
        with pytest.raises(ValueError, match='progress_callback must be callable'):
            engine.run(
                x=np.array([1.0]),
                y=np.array([0.1]),
                weights=np.array([1.0]),
                samples=10,
                burn=5,
                thin=1,
                progress_callback='not-callable',
            )

    # --- Resume-state tests -------------------------------------------------

    def _make_resume_state_mock(self, *, nvar=2, npop=10, labels=None):
        """Build a mock MCMCDraw for resume tests.

        BUMPS labels follow the pattern ``'p<param_name>'`` (the
        ``PARAMETER_PREFIX`` concatenated with the unique name),
        e.g. ``'pFilm_thickness'``.

        :param nvar: Number of parameters.
        :param npop: Population size.
        :param labels: Parameter labels (defaults to ``['pa', 'pb']``
            which strip to ``['a', 'b']``).
        """
        if labels is None:
            labels = ['pa', 'pb']
        mock_state = MagicMock()
        mock_state.Nvar = nvar
        mock_state.Npop = npop
        mock_state.labels = labels
        mock_draw = MagicMock()
        mock_draw.points = np.ones((20, nvar))
        mock_draw.logp = np.ones(20)
        mock_state.draw.return_value = mock_draw
        return mock_state

    def _make_problem_with_parameters(self, param_names):
        """Build a mock FitProblem whose ``_parameters`` yields the given names."""
        params = []
        for name in param_names:
            p = MagicMock()
            p.name = 'p' + name
            params.append(p)
        mock_problem = MagicMock()
        mock_problem._parameters = params
        return mock_problem

    def test_run_resume_state(self, engine: DreamSampler, monkeypatch) -> None:
        """Verify resume_state is forwarded to driver.fit()."""
        mock_FitDriver, mock_driver = self._setup_driver_mock(monkeypatch)
        resume_state = self._make_resume_state_mock()
        self._set_problem(monkeypatch, self._make_problem_with_parameters(['a', 'b']))

        result = engine.run(
            x=np.array([1.0, 2.0]),
            y=np.array([0.1, 0.2]),
            weights=np.array([1.0, 1.0]),
            samples=10,
            burn=0,
            thin=1,
            resume_state=resume_state,
        )

        assert result is not None
        # Verify a fit_state (defensive copy of resume_state) was passed to driver.fit()
        call_kwargs = mock_driver.fit.call_args.kwargs
        assert call_kwargs.get('fit_state') is not None
        assert call_kwargs['fit_state'] is not resume_state

    def test_run_resume_param_mismatch_raises(self, engine: DreamSampler, monkeypatch) -> None:
        """Parameter count mismatch raises ValueError before driver.fit()."""
        self._set_problem(monkeypatch, self._make_problem_with_parameters(['a']))
        # resume_state has 2 params, model has 1
        resume_state = self._make_resume_state_mock(nvar=2)

        with pytest.raises(ValueError, match='resume_state has 2 parameters'):
            engine.run(
                x=np.array([1.0]),
                y=np.array([0.1]),
                weights=np.array([1.0]),
                samples=10,
                burn=0,
                thin=1,
                resume_state=resume_state,
            )

    def test_run_resume_param_name_mismatch_raises(
        self, engine: DreamSampler, monkeypatch
    ) -> None:
        """Parameter name/order mismatch raises ValueError before driver.fit()."""
        self._set_problem(monkeypatch, self._make_problem_with_parameters(['a', 'b']))
        # resume_state has labels ['px', 'py'] → stripped to ['x', 'y']
        # Current model has params ['pa', 'pb'] → stripped to ['a', 'b']
        # → mismatch
        resume_state = self._make_resume_state_mock(nvar=2, labels=['px', 'py'])

        with pytest.raises(ValueError, match='Parameter names/order mismatch'):
            engine.run(
                x=np.array([1.0]),
                y=np.array([0.1]),
                weights=np.array([1.0]),
                samples=10,
                burn=0,
                thin=1,
                resume_state=resume_state,
            )

    def test_run_resume_population_mismatch_raises(
        self, engine: DreamSampler, monkeypatch
    ) -> None:
        """Explicit population differing from state.Npop raises ValueError."""
        self._set_problem(monkeypatch, self._make_problem_with_parameters(['a', 'b']))
        resume_state = self._make_resume_state_mock(nvar=2, npop=10)

        with pytest.raises(ValueError, match='would produce'):
            engine.run(
                x=np.array([1.0]),
                y=np.array([0.1]),
                weights=np.array([1.0]),
                samples=10,
                burn=0,
                thin=1,
                population=3,  # ceil(3*2)=6 ≠ 10
                resume_state=resume_state,
            )

    def test_run_resume_forces_burn_to_zero(
        self, engine: DreamSampler, monkeypatch, caplog: 'pytest.LogCaptureFixture'
    ) -> None:
        """burn>0 with resume_state warns and is forced to 0."""
        mock_FitDriver, _ = self._setup_driver_mock(monkeypatch)
        self._set_problem(monkeypatch, self._make_problem_with_parameters(['a', 'b']))
        resume_state = self._make_resume_state_mock()

        with caplog.at_level(logging.WARNING, logger='easyscience.fitting.bumps'):
            engine.run(
                x=np.array([1.0]),
                y=np.array([0.1]),
                weights=np.array([1.0]),
                samples=10,
                burn=5,
                thin=1,
                resume_state=resume_state,
            )

        assert 'ignored on resume' in caplog.text
        # burn must be forced to 0 in the kwargs passed to BUMPS
        assert mock_FitDriver.call_args.kwargs['burn'] == 0

    def test_run_resume_unlabeled_state_warns_and_uses_absolute_pop(
        self, engine: DreamSampler, monkeypatch, caplog: 'pytest.LogCaptureFixture'
    ) -> None:
        """A state reloaded from disk carries default labels ('P0', ...), so
        name validation is skipped with a warning, and the saved population is
        reproduced as a negative pop (BUMPS' absolute-chain-count convention)."""
        mock_FitDriver, _ = self._setup_driver_mock(monkeypatch)
        self._set_problem(monkeypatch, self._make_problem_with_parameters(['a', 'b']))
        resume_state = self._make_resume_state_mock(nvar=2, npop=10, labels=['P0', 'P1'])

        with caplog.at_level(logging.WARNING, logger='easyscience.fitting.bumps'):
            engine.run(
                x=np.array([1.0]),
                y=np.array([0.1]),
                weights=np.array([1.0]),
                samples=10,
                burn=0,
                thin=1,
                resume_state=resume_state,
            )

        assert 'does not carry parameter names' in caplog.text
        assert mock_FitDriver.call_args.kwargs['pop'] == -10


class TestDreamSamplerProgressPayload:
    """``DreamSampler._build_sample_progress_payload``."""

    @pytest.fixture
    def engine(self) -> DreamSampler:
        return DreamSampler(obj='obj', fit_function='fit_function')

    def test_payload_structure_and_sampling_flag(self, engine: DreamSampler) -> None:
        mock_problem = MagicMock()
        mock_problem.chisq.side_effect = [25.0, 12.5]
        mock_problem.labels.return_value = ['palpha']
        mock_problem.getp.return_value = np.array([1.0])

        payload = engine._build_sample_progress_payload(
            mock_problem, 7, np.array([1.0]), 12.5, 100
        )

        assert payload['iteration'] == 7
        assert payload['total_steps'] == 100
        assert payload['chi2'] == 25.0
        assert payload['reduced_chi2'] == 12.5
        assert payload['parameter_values'] == {'alpha': 1.0}
        assert payload['sampling'] is True
        assert payload['finished'] is False
        assert payload['refresh_plots'] is False
        # The nllf already computed by the sampler is reused — no model
        # re-evaluation via setp.
        mock_problem.chisq.assert_any_call(nllf=12.5, norm=False)
        mock_problem.chisq.assert_any_call(nllf=12.5, norm=True)
        mock_problem.setp.assert_not_called()

    def test_payload_keys(self, engine: DreamSampler) -> None:
        """Same keys as the classical-fit payload, plus ``sampling``."""
        mock_problem = MagicMock()
        mock_problem.chisq.side_effect = [10.0, 5.0]
        mock_problem.labels.return_value = ['pa']
        mock_problem.getp.return_value = np.array([5.0])

        payload = engine._build_sample_progress_payload(
            mock_problem, 1, np.array([5.0]), nllf=5.0, total_steps=50
        )

        expected_keys = {
            'iteration',
            'chi2',
            'reduced_chi2',
            'parameter_values',
            'refresh_plots',
            'finished',
            'sampling',
            'total_steps',
        }
        assert set(payload.keys()) == expected_keys
