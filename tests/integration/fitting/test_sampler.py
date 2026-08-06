# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause
"""Integration tests for the ``Sampler`` class (Bayesian MCMC via BUMPS DREAM).

Only full sampling runs and save/load/resume roundtrips live here; error
paths, argument validation and sidecar parsing are unit-tested in
``tests/unit/fitting/test_sampler.py``.
"""

import json

import numpy as np
import pytest

from easyscience import ObjBase
from easyscience import Parameter
from easyscience.fitting import Sampler
from easyscience.fitting import SamplingResults
from easyscience.fitting.multi_fitter import MultiFitter


class Line(ObjBase):
    m: Parameter
    c: Parameter

    def __init__(self, m_val: float, c_val: float):
        m = Parameter('m', m_val)
        c = Parameter('c', c_val)
        super(Line, self).__init__('line', m=m, c=c)

    def __call__(self, x):
        return self.m.value * x + self.c.value


class AbsSin(ObjBase):
    phase: Parameter
    offset: Parameter

    def __init__(self, offset_val: float, phase_val: float):
        offset = Parameter('offset', offset_val)
        phase = Parameter('phase', phase_val)
        super().__init__('sin', offset=offset, phase=phase)

    def __call__(self, x):
        return np.abs(np.sin(self.phase.value * x + self.offset.value))


class AbsSin2D(ObjBase):
    phase: Parameter
    offset: Parameter

    def __init__(self, offset_val: float, phase_val: float):
        offset = Parameter('offset', offset_val)
        phase = Parameter('phase', phase_val)
        super().__init__('sin2D', offset=offset, phase=phase)

    def __call__(self, x):
        X = x[:, :, 0]  # x is a 2D array
        Y = x[:, :, 1]
        return np.abs(np.sin(self.phase.value * X + self.offset.value)) * np.abs(
            np.sin(self.phase.value * Y + self.offset.value)
        )


def _fitter_and_data():
    """Build a 2-parameter MultiFitter over a small sine model.

    The fitter keeps its default (LMFit) minimizer: sampling no longer
    requires switching to BUMPS, only an installed ``bumps`` package.
    """
    pytest.importorskip('bumps')
    ref_sin = AbsSin(0.2, np.pi)
    sp = AbsSin(0.354, 3.05)
    sp.offset.fixed = False
    sp.phase.fixed = False
    x = np.linspace(0, 5, 50)
    y = ref_sin(x)
    weights = np.ones_like(x)
    f = MultiFitter([sp], [sp])
    return f, sp, x, y, weights


class TestSampler:
    """Integration tests for ``Sampler(f, ...)`` / ``Sampler``."""

    @pytest.mark.filterwarnings('ignore::UserWarning')
    def test_sample_returns_results_object(self):
        """sample() returns a populated SamplingResults; to_legacy_dict() has the legacy shape."""
        f, sp, x, y, weights = _fitter_and_data()
        sampler = Sampler(f, [x], [y], [weights])

        results = sampler.sample(samples=100, burn=20, thin=2)

        assert isinstance(results, SamplingResults)
        assert results.draws.ndim == 2
        assert results.draws.shape[1] == len(results.param_names)
        assert results.logp.shape[0] == results.draws.shape[0]
        assert results.state is not None

        # param_names should match the model's fit parameters
        expected_pars = {p.unique_name for p in sp.get_fit_parameters()}
        assert set(results.param_names) == expected_pars

        # results cached on the sampler
        assert sampler.results is results
        assert sampler.state is results.state
        assert sampler.draws is results.draws
        assert sampler.param_names == results.param_names

        # legacy dict shape
        legacy = results.to_legacy_dict()
        assert set(legacy.keys()) == {'draws', 'param_names', 'internal_bumps_object', 'logp'}
        assert legacy['internal_bumps_object'] is results.state

    @pytest.mark.filterwarnings('ignore::UserWarning')
    def test_sample_multi_dataset(self):
        """Multi-dataset sampling via Sampler(f, ...) has correct param_names."""
        ref_sin_1 = AbsSin(0.2, np.pi)
        sp_sin_1 = AbsSin(0.354, 3.05)
        sp_line = Line(0.43, 6.1)

        # Link a parameter across models
        sp_line.m.make_dependent_on(
            dependency_expression='sp_sin1', dependency_map={'sp_sin1': sp_sin_1.offset}
        )

        x1 = np.linspace(0, 5, 50)
        y1 = ref_sin_1(x1)
        x2 = np.copy(x1)
        y2 = Line(1, 4.6)(x2)
        weights = np.ones_like(x1)

        sp_sin_1.offset.fixed = False
        sp_sin_1.phase.fixed = False
        sp_line.c.fixed = False

        pytest.importorskip('bumps')
        f = MultiFitter([sp_sin_1, sp_line], [sp_sin_1, sp_line])

        sampler = Sampler(f, [x1, x2], [y1, y2], [weights, weights])
        results = sampler.sample(samples=100, burn=20, thin=2)

        # All parameters across both models should appear
        all_params = {p.unique_name for p in sp_sin_1.get_fit_parameters()}
        all_params |= {p.unique_name for p in sp_line.get_fit_parameters()}
        assert set(results.param_names) == all_params

    def test_sample_population(self):
        """Passing population should succeed and produce valid draws."""
        f, _, x, y, weights = _fitter_and_data()
        sampler = Sampler(f, [x], [y], [weights])

        results = sampler.sample(samples=100, burn=20, thin=2, population=5)
        assert results.draws.shape[0] > 0

    @pytest.mark.filterwarnings('ignore::UserWarning')
    def test_sample_vectorized_2d(self):
        """sample() with vectorized=True and 2D input should produce valid draws."""
        sp = AbsSin2D(0.1, 1.75)

        x = np.linspace(0, 5, 50)
        X, Y = np.meshgrid(x, x)
        x2D = np.stack((X, Y), axis=2)
        y2D = np.abs(np.sin(X)) * np.abs(np.sin(Y))
        weights = np.ones_like(y2D)

        sp.offset.fixed = False
        sp.phase.fixed = False

        pytest.importorskip('bumps')
        f = MultiFitter([sp], [sp])

        sampler = Sampler(f, [x2D], [y2D], [weights], vectorized=True)
        results = sampler.sample(samples=100, burn=20, thin=2)

        assert results.draws.ndim == 2
        assert results.draws.shape[0] > 0
        assert results.draws.shape[1] == len(results.param_names)

    @pytest.mark.filterwarnings('ignore::UserWarning')
    def test_fit_function_restored_on_success(self):
        """fit_function must be restored after a successful sample()."""
        f, _, x, y, weights = _fitter_and_data()
        sampler = Sampler(f, [x], [y], [weights])
        original_func = f.fit_function

        sampler.sample(samples=100, burn=20, thin=2)
        assert f.fit_function is original_func

    @pytest.mark.filterwarnings('ignore::UserWarning')
    def test_sampler_kwargs_forwarded(self):
        """Per-call sampler_kwargs dict is forwarded to the BUMPS DREAM sampler."""
        f, _, x, y, weights = _fitter_and_data()
        sampler = Sampler(f, [x], [y], [weights])

        results = sampler.sample(samples=100, burn=20, thin=2, sampler_kwargs={'init': 'random'})

        assert results.draws.ndim == 2
        assert results.draws.shape[0] > 0

    @pytest.mark.filterwarnings('ignore::UserWarning')
    def test_default_sampler_kwargs_merged(self, monkeypatch):
        """Constructor-level sampler_kwargs defaults are used; per-call kwargs win."""
        from easyscience.fitting.samplers.sampler_dream import DreamSampler

        f, _, x, y, weights = _fitter_and_data()
        sampler = Sampler(f, [x], [y], [weights], sampler_kwargs={'init': 'random'})

        captured = {}
        original_run = DreamSampler.run

        def spy(self, **kwargs):
            captured.update(kwargs.get('sampler_kwargs') or {})
            return original_run(self, **kwargs)

        monkeypatch.setattr(DreamSampler, 'run', spy)

        sampler.sample(samples=100, burn=20, thin=2)
        assert captured == {'init': 'random'}

        captured.clear()
        sampler.sample(samples=100, burn=20, thin=2, sampler_kwargs={'init': 'lhs'})
        assert captured == {'init': 'lhs'}  # per-call overrides default

    @pytest.mark.filterwarnings('ignore::UserWarning')
    def test_sample_with_lmfit_minimizer_active(self):
        """Sampling works without switching the fitter's minimizer to BUMPS —
        the new capability enabled by the ``DreamSampler`` engine (#280)."""
        f, _, x, y, weights = _fitter_and_data()
        assert f.minimizer.package == 'lmfit'  # the default LMFit minimizer

        sampler = Sampler(f, [x], [y], [weights])
        results = sampler.sample(samples=100, burn=20, thin=2)

        assert results.draws.shape[0] > 0
        # The active minimizer is untouched by sampling.
        assert f.minimizer.package == 'lmfit'

    @pytest.mark.filterwarnings('ignore::UserWarning')
    def test_extend_chain(self):
        """extend(additional_samples=) continues the chain; ring-buffer math is done for the user."""
        f, _, x, y, weights = _fitter_and_data()
        sampler = Sampler(f, [x], [y], [weights])

        first = sampler.sample(samples=100, burn=20, thin=1)
        n_first = first.draws.shape[0]

        extended = sampler.extend(additional_samples=100, thin=1)

        assert extended.draws.shape[1] == first.draws.shape[1]
        assert extended.draws.shape[0] >= n_first
        assert sampler.results is extended

    @pytest.mark.filterwarnings('ignore::UserWarning')
    def test_extend_with_thinning_keeps_existing_draws(self):
        """extend() must not drop stored draws when thin > 1.

        Regression test: the ring buffer is sized from the state's stored
        generations (``Ngen * Npop``), not from the retained-draw count,
        which BUMPS divides by the thinning interval.
        """
        f, _, x, y, weights = _fitter_and_data()
        sampler = Sampler(f, [x], [y], [weights])

        first = sampler.sample(samples=1000, burn=20, thin=10)
        n_first = first.draws.shape[0]

        extended = sampler.extend(additional_samples=1000, thin=10)

        # All previously retained draws are kept, plus ~additional/thin new ones.
        assert extended.draws.shape[0] > n_first

    @pytest.mark.filterwarnings('ignore::UserWarning')
    def test_extend_total_samples_override(self):
        """extend(total_samples=) bypasses the additional_samples arithmetic."""
        f, _, x, y, weights = _fitter_and_data()
        sampler = Sampler(f, [x], [y], [weights])

        sampler.sample(samples=100, burn=20, thin=1)
        extended = sampler.extend(total_samples=150, thin=1)

        assert extended.draws.shape[0] > 0

    @pytest.mark.filterwarnings('ignore::UserWarning')
    def test_extend_after_save_load_roundtrip(self, tmp_path, caplog):
        """save() -> new sampler -> load_state() -> extend() continues the chain.

        Regression coverage: BUMPS ``load_state`` does not preserve parameter
        labels, so resume validation must proceed by parameter order (with a
        logged warning) instead of raising.
        """
        import logging

        f, _, x, y, weights = _fitter_and_data()
        sampler = Sampler(f, [x], [y], [weights])
        first = sampler.sample(samples=100, burn=20, thin=1)

        prefix = str(tmp_path / 'chain')
        sampler.save(prefix)

        sampler2 = Sampler(f, [x], [y], [weights])
        loaded = sampler2.load_state(prefix)
        assert loaded.draws.shape[1] == first.draws.shape[1]

        with caplog.at_level(logging.WARNING, logger='easyscience.fitting.bumps'):
            extended = sampler2.extend(additional_samples=100, thin=1)

        assert 'does not carry parameter names' in caplog.text
        assert extended.draws.shape[1] == first.draws.shape[1]
        assert extended.draws.shape[0] > 0

    @pytest.mark.filterwarnings('ignore::UserWarning')
    def test_extend_preserves_nondefault_population(self):
        """Extending a chain that used a non-default population must not fail.

        Regression test: the population scale factor is recovered from the
        saved state on resume, otherwise BUMPS regenerates the default
        population and raises ``Cannot change Nvar, Npop or Ncr on resize``.
        """
        f, _, x, y, weights = _fitter_and_data()
        sampler = Sampler(f, [x], [y], [weights])

        first = sampler.sample(samples=100, burn=20, thin=1, population=5)
        first_npop = first.state.Npop

        extended = sampler.extend(additional_samples=100, thin=1)

        assert extended.draws.shape[1] == first.draws.shape[1]
        assert extended.draws.shape[0] > 0
        # Population must be preserved across the extend.
        assert extended.state.Npop == first_npop

    @pytest.mark.filterwarnings('ignore::UserWarning')
    def test_save_warns_when_fingerprint_unavailable(self, tmp_path, caplog, monkeypatch):
        """save() still succeeds when the data cannot be fingerprinted, but
        logs a warning and records ``null`` in the sidecar."""
        import logging

        f, _, x, y, weights = _fitter_and_data()
        sampler = Sampler(f, [x], [y], [weights])
        sampler.sample(samples=100, burn=20, thin=2)

        monkeypatch.setattr('easyscience.fitting.sampler._data_fingerprint', lambda *args: None)

        prefix = str(tmp_path / 'chain')
        with caplog.at_level(logging.WARNING, logger='easyscience.fitting'):
            sampler.save(prefix)

        assert 'Could not compute a fingerprint' in caplog.text
        with open(f'{prefix}.params.json') as fh:
            assert json.load(fh)['data_fingerprint'] is None

    @pytest.mark.filterwarnings('ignore::UserWarning')
    def test_load_state_populates_results(self, tmp_path):
        """A freshly loaded sampler reports draws/logp/param_names without resampling."""
        f, sp, x, y, weights = _fitter_and_data()
        sampler = Sampler(f, [x], [y], [weights])
        first = sampler.sample(samples=100, burn=20, thin=2)

        prefix = str(tmp_path / 'chain')
        sampler.save(prefix)

        sampler2 = Sampler(f, [x], [y], [weights])
        assert sampler2.draws is None
        loaded = sampler2.load_state(prefix)

        assert sampler2.state is loaded.state
        assert loaded.draws.ndim == 2
        assert loaded.draws.shape[1] == first.draws.shape[1]
        assert loaded.logp.shape[0] == loaded.draws.shape[0]
        # Sidecar restores the original (prefix-stripped) parameter names.
        assert loaded.param_names == first.param_names

        # save() wrote a v2 sidecar carrying names and a data fingerprint.
        with open(f'{prefix}.params.json') as fh:
            sidecar = json.load(fh)
        assert sidecar['schema_version'] == 2
        assert sidecar['param_names'] == first.param_names
        assert sidecar['data_fingerprint'] is not None

    @pytest.mark.filterwarnings('ignore::UserWarning')
    def test_load_short_chain_regression(self, tmp_path):
        """Short chain save/load — pins the BUMPS >= 1.0.4 loadtxt workaround.

        A short chain stores a single CR-weight update row; BUMPS' numpy-based
        reader collapses it to a 1-D array and ``load_state`` raises
        ``IndexError`` without the 2-D coercion workaround.
        """
        f, _, x, y, weights = _fitter_and_data()
        sampler = Sampler(f, [x], [y], [weights])
        sampler.sample(samples=20, burn=5, thin=1)

        prefix = str(tmp_path / 'short_chain')
        sampler.save(prefix)

        sampler2 = Sampler(f, [x], [y], [weights])
        loaded = sampler2.load_state(prefix)
        assert loaded.draws.shape[0] > 0

    @pytest.mark.filterwarnings('ignore::UserWarning')
    def test_load_fingerprint_mismatch_warns(self, tmp_path, caplog):
        """Loading a chain into a sampler bound to different data warns."""
        f, _, x, y, weights = _fitter_and_data()
        sampler = Sampler(f, [x], [y], [weights])
        sampler.sample(samples=100, burn=20, thin=2)

        prefix = str(tmp_path / 'chain')
        sampler.save(prefix)

        import logging

        other_y = y + 0.5
        sampler2 = Sampler(f, [x], [other_y], [weights])
        with caplog.at_level(logging.WARNING, logger='easyscience.fitting'):
            sampler2.load_state(prefix)
        assert 'does not match the data fingerprint' in caplog.text


class TestDeprecatedMcmcSampleShim:
    """The released ``Fitter.mcmc_sample`` entry point keeps working as a
    deprecation shim delegating to ``Sampler`` (the ``Bumps.mcmc_sample``
    delegate is unit-tested in ``test_minimizer_bumps.py``)."""

    @pytest.mark.filterwarnings('ignore::UserWarning')
    def test_fitter_mcmc_sample_warns_and_returns_legacy_dict(self):
        f, sp, x, y, weights = _fitter_and_data()

        with pytest.warns(DeprecationWarning, match='Fitter.mcmc_sample'):
            legacy = f.mcmc_sample([x], [y], [weights], samples=100, burn=20, thin=2)

        assert set(legacy.keys()) == {'draws', 'param_names', 'internal_bumps_object', 'logp'}
        assert legacy['draws'].ndim == 2
        assert legacy['draws'].shape[0] > 0
        assert legacy['draws'].shape[1] == len(legacy['param_names'])
        expected_pars = {p.unique_name for p in sp.get_fit_parameters()}
        assert set(legacy['param_names']) == expected_pars
        # The shim no longer requires the BUMPS minimizer — the default
        # LMFit minimizer stays active throughout.
        assert f.minimizer.package == 'lmfit'
