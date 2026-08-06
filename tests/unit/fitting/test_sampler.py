# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause
"""Unit tests for ``sampler.py`` — mirrors ``src/easyscience/fitting/sampler.py``.

Full sampling runs and save/load/resume roundtrips live in
``tests/integration/fitting/test_sampler.py``.
"""

import json
import logging
from types import SimpleNamespace

import numpy as np
import pytest

from easyscience import ObjBase
from easyscience import Parameter
from easyscience.fitting import Sampler
from easyscience.fitting import SamplingResults
from easyscience.fitting.minimizers.minimizer_base import MINIMIZER_PARAMETER_PREFIX
from easyscience.fitting.multi_fitter import MultiFitter
from easyscience.fitting.sampler import _data_fingerprint
from easyscience.fitting.sampler import load_chain


class AbsSin(ObjBase):
    phase: Parameter
    offset: Parameter

    def __init__(self, offset_val: float, phase_val: float):
        offset = Parameter('offset', offset_val)
        phase = Parameter('phase', phase_val)
        super().__init__('sin', offset=offset, phase=phase)

    def __call__(self, x):
        return np.abs(np.sin(self.phase.value * x + self.offset.value))


class _StubFitter:
    """Duck-types the Fitter attributes checked by the Sampler constructor."""

    minimizer = None
    fit_function = None


class _StubState:
    """Minimal stand-in for a BUMPS ``MCMCDraw`` as seen by ``load_chain``."""

    def __init__(self, labels):
        self.labels = list(labels)


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


def _xyw():
    x = np.linspace(0, 5, 50)
    return x, np.sin(x), np.ones_like(x)


def _make_state(ngen=6, npop=5, nvar=2, seed=7):
    """Synthesize a small, fully populated BUMPS ``MCMCDraw``.

    ``Nupdate=1`` is deliberate: on ``save_state`` it writes a single
    CR-weight row, which is exactly the single-row stats file that trips
    the BUMPS ``loadtxt`` bug worked around by ``_load_bumps_state``.
    """
    bumps_state = pytest.importorskip('bumps.dream.state')
    rng = np.random.default_rng(seed)
    state = bumps_state.MCMCDraw(
        Ngen=ngen, Nthin=ngen, Nupdate=1, Nvar=nvar, Npop=npop, Ncr=3, thinning=1
    )
    for _ in range(ngen):
        points = rng.normal(size=(npop, nvar))
        logp = rng.normal(size=npop)
        state._generation(new_draws=npop, x=points, logp=logp, accept=np.ones(npop))
    state._update(CR_weight=np.array([0.2, 0.3, 0.5]))
    return state


class TestSamplerConstructorValidation:
    def test_rejects_fitter_without_minimizer(self):
        x, y, w = _xyw()
        with pytest.raises(TypeError, match='fitter must be a configured Fitter'):
            Sampler(object(), [x], [y], [w])

    def test_rejects_mixed_array_and_list(self):
        x, y, w = _xyw()
        with pytest.raises(ValueError, match='both be arrays or both be lists'):
            Sampler(_StubFitter(), [x], y)

    def test_rejects_dataset_count_mismatch(self):
        x, y, w = _xyw()
        with pytest.raises(ValueError, match='same number of datasets'):
            Sampler(_StubFitter(), [x, x], [y])

    def test_rejects_weights_structure_mismatch(self):
        x, y, w = _xyw()
        with pytest.raises(ValueError, match='weights must match the structure'):
            Sampler(_StubFitter(), [x], [y], w)

    def test_rejects_weights_count_mismatch(self):
        x, y, w = _xyw()
        with pytest.raises(ValueError, match='weights must hold the same number'):
            Sampler(_StubFitter(), [x], [y], [w, w])

    def test_rejects_non_bool_vectorized(self):
        x, y, w = _xyw()
        with pytest.raises(TypeError, match='vectorized must be a bool'):
            Sampler(_StubFitter(), [x], [y], [w], vectorized=1)

    def test_rejects_non_dict_sampler_kwargs(self):
        x, y, w = _xyw()
        with pytest.raises(TypeError, match='sampler_kwargs must be a dict'):
            Sampler(_StubFitter(), [x], [y], [w], sampler_kwargs=[('init', 'random')])

    def test_accepts_single_arrays(self):
        x, y, w = _xyw()
        sampler = Sampler(_StubFitter(), x, y, w)
        assert sampler.results is None


class TestSamplerDataBinding:
    def test_properties_expose_bound_data(self):
        x, y, w = _xyw()
        f = _StubFitter()
        sampler = Sampler(f, [x], [y], [w])
        assert sampler.fitter is f
        np.testing.assert_array_equal(sampler.x[0], x)
        np.testing.assert_array_equal(sampler.y[0], y)
        np.testing.assert_array_equal(sampler.weights[0], w)

    def test_weights_property_none_when_unset(self):
        x, y, _ = _xyw()
        sampler = Sampler(_StubFitter(), [x], [y])
        assert sampler.weights is None

    def test_inputs_are_copied(self):
        """Mutating the caller's arrays after construction must not change the
        bound data (nor the save() fingerprint derived from it)."""
        x, y, w = _xyw()
        y_original = y.copy()
        sampler = Sampler(_StubFitter(), [x], [y], [w])
        fingerprint_before = sampler._fingerprint()

        y[:] = 0.0

        np.testing.assert_array_equal(sampler.y[0], y_original)
        assert sampler._fingerprint() == fingerprint_before

    def test_bound_arrays_are_read_only(self):
        x, y, w = _xyw()
        sampler = Sampler(_StubFitter(), x, y, w)
        with pytest.raises(ValueError, match='read-only'):
            sampler.x[0] = 99.0

    def test_data_properties_have_no_setters(self):
        """Bound data is deliberately immutable — sample new data with a new
        Sampler, so a chain can never be extended against different data."""
        x, y, w = _xyw()
        sampler = Sampler(_StubFitter(), [x], [y], [w])
        for name in ('fitter', 'x', 'y', 'weights'):
            with pytest.raises(AttributeError):
                setattr(sampler, name, None)


class TestSamplerPathValidation:
    def test_save_rejects_non_pathlike(self):
        x, y, w = _xyw()
        sampler = Sampler(_StubFitter(), [x], [y], [w])
        with pytest.raises(TypeError, match='path must be a str or os.PathLike'):
            sampler.save(123)

    def test_save_accepts_pathlike(self, tmp_path):
        """A Path object passes validation; the empty sampler then raises RuntimeError."""
        x, y, w = _xyw()
        sampler = Sampler(_StubFitter(), [x], [y], [w])
        with pytest.raises(RuntimeError, match='No chain state to save'):
            sampler.save(tmp_path / 'chain')

    def test_load_state_rejects_non_pathlike(self):
        x, y, w = _xyw()
        sampler = Sampler(_StubFitter(), [x], [y], [w])
        with pytest.raises(TypeError, match='path must be a str or os.PathLike'):
            sampler.load_state(123)

    def test_load_chain_rejects_non_pathlike(self):
        with pytest.raises(TypeError, match='path must be a str or os.PathLike'):
            load_chain(None)

    @pytest.mark.parametrize('skip', [-1, 1.5, True, 'no'])
    def test_load_chain_rejects_bad_skip(self, tmp_path, skip):
        with pytest.raises(ValueError, match='skip must be a non-negative integer'):
            load_chain(str(tmp_path / 'chain'), skip=skip)


class TestSamplerErrorPaths:
    def test_sample_requires_bumps_package(self, monkeypatch):
        """sample() must raise RuntimeError when the bumps package is not
        installed — regardless of the active minimizer — and must not touch
        the fitter."""
        sp = AbsSin(0.354, 3.05)
        f = MultiFitter([sp], [sp])

        x, y, w = _xyw()
        sampler = Sampler(f, [x], [y], [w])
        minimizer_before = f.minimizer
        monkeypatch.setattr(
            'easyscience.fitting.available_minimizers.bumps_engine_available', False
        )
        with pytest.raises(RuntimeError, match='requires the bumps package'):
            sampler.sample(samples=10, burn=5, thin=1)
        assert f.minimizer is minimizer_before

    def test_fitter_untouched_on_error(self):
        """The fitter is never mutated by sampling, even when the engine
        raises."""
        f, _, x, y, weights = _fitter_and_data()
        sampler = Sampler(f, [x], [y], [weights])
        original_func = f.fit_function
        minimizer_before = f.minimizer

        # Invalid `samples` is rejected by the engine (single source of
        # validation).
        with pytest.raises(ValueError, match='samples must be a positive integer'):
            sampler.sample(samples=-1, burn=5, thin=1)

        assert f.fit_function is original_func
        assert f.minimizer is minimizer_before

    def test_extend_requires_existing_state(self):
        """extend() before sample()/load_state() raises RuntimeError."""
        x, y, w = _xyw()
        sampler = Sampler(_StubFitter(), [x], [y], [w])

        with pytest.raises(RuntimeError, match='No chain to extend'):
            sampler.extend(additional_samples=10)

    def test_save_raises_without_state(self, tmp_path):
        """save() before sample() raises RuntimeError."""
        x, y, w = _xyw()
        sampler = Sampler(_StubFitter(), [x], [y], [w])

        with pytest.raises(RuntimeError, match='No chain state to save'):
            sampler.save(str(tmp_path / 'chain'))

    def test_sample_warns_when_replacing_existing_chain(self, monkeypatch, caplog):
        """sample() over an existing chain logs a replace warning; a fresh
        sampler does not."""
        x, y, w = _xyw()
        sampler = Sampler(_StubFitter(), [x], [y], [w])

        dummy = SamplingResults(
            draws=np.zeros((1, 1)), param_names=['p'], logp=np.zeros(1), state=object()
        )
        monkeypatch.setattr(Sampler, '_run', lambda self, **kwargs: dummy)

        with caplog.at_level(logging.WARNING, logger='easyscience.fitting'):
            sampler.sample(samples=10)
        assert 'Replacing the existing chain' not in caplog.text

        sampler._state = object()
        with caplog.at_level(logging.WARNING, logger='easyscience.fitting'):
            sampler.sample(samples=10)
        assert 'Replacing the existing chain' in caplog.text


class TestLoadChainSidecar:
    """Sidecar parsing in ``load_chain``, with the BUMPS reader stubbed out."""

    @pytest.fixture(autouse=True)
    def _stub_bumps_reader(self, monkeypatch):
        self.state = _StubState(['P0', 'P1'])
        monkeypatch.setattr(
            'easyscience.fitting.sampler._load_bumps_state',
            lambda path, skip=0: self.state,
        )

    @staticmethod
    def _write_sidecar(prefix, payload):
        with open(f'{prefix}.params.json', 'w') as fh:
            json.dump(payload, fh)

    def test_v2_sidecar_restores_names(self, tmp_path):
        prefix = str(tmp_path / 'chain')
        self._write_sidecar(
            prefix,
            {
                'schema_version': 2,
                'param_names': ['a', 'b'],
                'easyscience_version': '0.0.0',
                'data_fingerprint': 'abc',
            },
        )
        state, names, sidecar = load_chain(prefix)
        assert state is self.state
        assert names == ['a', 'b']
        assert sidecar['data_fingerprint'] == 'abc'

    def test_v1_sidecar_accepted(self, tmp_path):
        """v1 sidecars (easyreflectometry's save_posterior) still restore names."""
        prefix = str(tmp_path / 'chain')
        self._write_sidecar(
            prefix,
            {'schema_version': 1, 'param_names': ['a', 'b'], 'easyreflectometry_version': '0.0.0'},
        )
        _, names, _ = load_chain(prefix)
        assert names == ['a', 'b']

    def test_missing_sidecar_falls_back_to_state_labels(self, tmp_path):
        _, names, sidecar = load_chain(str(tmp_path / 'chain'))
        assert sidecar == {}
        assert names == ['P0', 'P1']

    def test_fallback_strips_minimizer_prefix_from_labels(self, tmp_path):
        self.state.labels = [f'{MINIMIZER_PARAMETER_PREFIX}a', f'{MINIMIZER_PARAMETER_PREFIX}b']
        _, names, _ = load_chain(str(tmp_path / 'chain'))
        assert names == ['a', 'b']

    def test_corrupt_sidecar_tolerated(self, tmp_path):
        """A corrupt sidecar falls back to labels rather than raising."""
        prefix = str(tmp_path / 'chain')
        (tmp_path / 'chain.params.json').write_text('{ not valid json')
        _, names, sidecar = load_chain(prefix)
        assert sidecar == {}
        assert names == ['P0', 'P1']

    def test_unknown_schema_version_names_not_trusted(self, tmp_path):
        """A sidecar from a future schema is read but its names are not trusted."""
        prefix = str(tmp_path / 'chain')
        self._write_sidecar(prefix, {'schema_version': 99, 'param_names': ['bogus']})
        _, names, sidecar = load_chain(prefix)
        assert sidecar['schema_version'] == 99
        assert names == ['P0', 'P1']


class TestSamplerConstructorDataValidation:
    """Scalars, strings and empty/ragged data must be rejected at construction."""

    def test_rejects_scalar_x(self):
        _, y, _ = _xyw()
        with pytest.raises(ValueError, match='x must be an array of values, got a scalar'):
            Sampler(_StubFitter(), 5.0, y)

    def test_rejects_scalar_dataset_in_list(self):
        x, y, _ = _xyw()
        with pytest.raises(ValueError, match=r'y\[1\] must be an array of values'):
            Sampler(_StubFitter(), [x, x], [y, 3.0])

    def test_rejects_string_data(self):
        x, _, _ = _xyw()
        with pytest.raises(TypeError, match='y must hold numeric values'):
            Sampler(_StubFitter(), x, 'abc')

    def test_rejects_non_numeric_object_array(self):
        _, y, _ = _xyw()
        with pytest.raises(TypeError, match='x must hold numeric values'):
            Sampler(_StubFitter(), np.array([{}, {}], dtype=object), y)

    def test_rejects_empty_array(self):
        with pytest.raises(ValueError, match='x must not be empty'):
            Sampler(_StubFitter(), np.array([]), np.array([]))

    def test_rejects_ragged_dataset(self):
        with pytest.raises(TypeError, match=r'x\[0\] could not be converted'):
            Sampler(_StubFitter(), [[1.0, [2.0, 3.0]]], [np.zeros(3)])

    def test_rejects_scalar_weights(self):
        x, y, _ = _xyw()
        with pytest.raises(ValueError, match='weights must be an array of values'):
            Sampler(_StubFitter(), x, y, 2.0)

    def test_weights_list_may_hold_none_entries(self):
        x, y, w = _xyw()
        sampler = Sampler(_StubFitter(), [x, x], [y, y], [w, None])
        assert sampler.weights[1] is None


class TestDataFingerprint:
    def test_returns_none_on_unhashable_data(self):
        assert _data_fingerprint([object()], [], []) is None

    def test_fingerprint_without_weights(self):
        x, y, _ = _xyw()
        sampler = Sampler(_StubFitter(), x, y)
        assert isinstance(sampler._fingerprint(), str)


class TestSamplingResultsLegacyDict:
    def test_to_legacy_dict_maps_fields(self):
        state = object()
        results = SamplingResults(
            draws=np.ones((2, 1)), param_names=['p'], logp=np.zeros(2), state=state
        )
        legacy = results.to_legacy_dict()
        assert legacy['internal_bumps_object'] is state
        assert legacy['param_names'] == ['p']
        np.testing.assert_array_equal(legacy['draws'], results.draws)
        np.testing.assert_array_equal(legacy['logp'], results.logp)


class TestSamplerRunEngine:
    """The ``_run`` tail: results construction, storage, and kwarg merging,
    with the ``DreamSampler`` engine stubbed out."""

    def test_run_stores_results_and_exposes_properties(self, monkeypatch):
        f, _, x, y, weights = _fitter_and_data()
        from easyscience.fitting.samplers.sampler_dream import DreamSampler

        canned = {
            'draws': np.arange(8.0).reshape(4, 2),
            'param_names': ['offset', 'phase'],
            'logp': np.zeros(4),
            'internal_bumps_object': object(),
        }
        captured = {}

        def fake_run(self, **kwargs):
            captured.update(kwargs)
            return dict(canned)

        monkeypatch.setattr(DreamSampler, 'run', fake_run)

        sampler = Sampler(f, [x], [y], [weights], sampler_kwargs={'trim': False})
        original_func = f.fit_function
        minimizer_before = f.minimizer
        results = sampler.sample(samples=100, burn=10, thin=2, sampler_kwargs={'init': 'lhs'})

        assert isinstance(results, SamplingResults)
        assert sampler.results is results
        assert sampler.state is canned['internal_bumps_object']
        np.testing.assert_array_equal(sampler.draws, canned['draws'])
        assert sampler.param_names == ['offset', 'phase']
        np.testing.assert_array_equal(sampler.logp, canned['logp'])
        # Per-call kwargs are merged over the instance defaults.
        assert captured['sampler_kwargs'] == {'trim': False, 'init': 'lhs'}
        assert captured['samples'] == 100
        assert captured['burn'] == 10
        assert captured['resume_state'] is None
        # The fitter is never mutated: a fresh engine gets the wrapped
        # function directly, and the active (LMFit) minimizer stays put.
        assert f.fit_function is original_func
        assert f.minimizer is minimizer_before

    def test_run_works_with_non_bumps_minimizer(self, monkeypatch):
        """Sampling works with the default LMFit minimizer active — the
        engine is constructed independently of the fitter's minimizer."""
        f, _, x, y, weights = _fitter_and_data()
        from easyscience.fitting.samplers.sampler_dream import DreamSampler

        assert f.minimizer.package != 'bumps'  # default is LMFit

        constructed = {}
        original_init = DreamSampler.__init__

        def spy_init(self, obj, fit_function):
            constructed['obj'] = obj
            constructed['fit_function'] = fit_function
            original_init(self, obj, fit_function)

        canned = {
            'draws': np.zeros((2, 2)),
            'param_names': ['offset', 'phase'],
            'logp': np.zeros(2),
            'internal_bumps_object': object(),
        }
        monkeypatch.setattr(DreamSampler, '__init__', spy_init)
        monkeypatch.setattr(DreamSampler, 'run', lambda self, **kwargs: dict(canned))

        sampler = Sampler(f, [x], [y], [weights])
        results = sampler.sample(samples=10, burn=0, thin=1)

        assert results.param_names == ['offset', 'phase']
        # The engine is bound to the fitter's model object and a wrapped
        # fit function, not to the minimizer.
        assert constructed['obj'] is f.fit_object
        assert callable(constructed['fit_function'])


class TestSamplerExtendArithmetic:
    """extend() ring-buffer sizing, with the sampling engine stubbed out."""

    @staticmethod
    def _sampler_with_stub_state(monkeypatch, ngen=7, npop=5):
        x, y, w = _xyw()
        sampler = Sampler(_StubFitter(), x, y, w)
        sampler._state = SimpleNamespace(Ngen=ngen, Npop=npop)
        captured = {}
        dummy = SamplingResults(
            draws=np.zeros((1, 1)), param_names=['p'], logp=np.zeros(1), state=object()
        )

        def fake_run(self, **kwargs):
            captured.update(kwargs)
            return dummy

        monkeypatch.setattr(Sampler, '_run', fake_run)
        return sampler, captured

    def test_buffer_sized_from_stored_generations(self, monkeypatch):
        sampler, captured = self._sampler_with_stub_state(monkeypatch, ngen=7, npop=5)

        sampler.extend(additional_samples=100, thin=3)

        assert captured['samples'] == 7 * 5 + 100
        assert captured['burn'] == 0
        assert captured['thin'] == 3
        assert captured['population'] is None
        assert captured['resume_state'] is sampler._state

    def test_total_samples_overrides_arithmetic(self, monkeypatch):
        sampler, captured = self._sampler_with_stub_state(monkeypatch)

        sampler.extend(additional_samples=100, total_samples=1234)

        assert captured['samples'] == 1234


class TestSamplerPersistenceRoundTrip:
    """Real ``save()``/``load_chain()``/``load_state()`` round-trips over a
    synthesized chain — no BUMPS reader mocking, unlike ``TestLoadChainSidecar``."""

    @staticmethod
    def _sampler_with_state():
        x, y, w = _xyw()
        sampler = Sampler(_StubFitter(), x, y, w)
        state = _make_state()
        _draw = state.draw()
        sampler._state = state
        sampler._results = SamplingResults(
            draws=_draw.points, param_names=['offset', 'phase'], logp=_draw.logp, state=state
        )
        return sampler, state

    def test_save_writes_chain_files_and_sidecar(self, tmp_path):
        sampler, _ = self._sampler_with_state()
        prefix = str(tmp_path / 'chain')

        sampler.save(prefix)

        for suffix in ('-chain.mc.gz', '-point.mc.gz', '-stats.mc.gz'):
            assert (tmp_path / f'chain{suffix}').exists()
        with open(f'{prefix}.params.json') as fh:
            sidecar = json.load(fh)
        assert sidecar['schema_version'] == 2
        assert sidecar['param_names'] == ['offset', 'phase']
        assert sidecar['data_fingerprint'] == sampler._fingerprint()
        assert 'easyscience_version' in sidecar

    def test_save_warns_when_fingerprint_unavailable(self, tmp_path, monkeypatch, caplog):
        sampler, _ = self._sampler_with_state()
        monkeypatch.setattr(Sampler, '_fingerprint', lambda self: None)
        prefix = str(tmp_path / 'chain')

        with caplog.at_level(logging.WARNING, logger='easyscience.fitting'):
            sampler.save(prefix)

        assert 'Could not compute a fingerprint' in caplog.text
        with open(f'{prefix}.params.json') as fh:
            assert json.load(fh)['data_fingerprint'] is None

    def test_load_chain_reads_back_single_update_row_state(self, tmp_path):
        """A short chain saves a single CR-weight row; reading it back
        exercises the ``_load_bumps_state`` loadtxt workaround for real."""
        sampler, state = self._sampler_with_state()
        prefix = str(tmp_path / 'chain')
        sampler.save(prefix)

        loaded_state, names, sidecar = load_chain(prefix)

        assert names == ['offset', 'phase']
        assert loaded_state.Npop == state.Npop
        assert loaded_state.Nvar == state.Nvar
        assert sidecar['schema_version'] == 2

    def test_load_bumps_state_restores_loader_after_failure(self, tmp_path):
        bumps_state = pytest.importorskip('bumps.dream.state')
        from easyscience.fitting.sampler import _load_bumps_state

        original = bumps_state.loadtxt_with_fallback
        with pytest.raises(RuntimeError, match='does not exist'):
            _load_bumps_state(str(tmp_path / 'missing'))
        assert bumps_state.loadtxt_with_fallback is original

    def test_load_state_populates_results(self, tmp_path, caplog):
        sampler, state = self._sampler_with_state()
        prefix = str(tmp_path / 'chain')
        sampler.save(prefix)

        x, y, w = _xyw()
        fresh = Sampler(_StubFitter(), x, y, w)
        with caplog.at_level(logging.WARNING, logger='easyscience.fitting'):
            results = fresh.load_state(prefix)

        # Same bound data: the fingerprint matches, so no warning.
        assert 'does not match the data' not in caplog.text
        assert fresh.results is results
        assert fresh.state is results.state
        assert results.param_names == ['offset', 'phase']
        assert results.draws.shape[1] == state.Nvar
        assert results.logp.shape[0] == results.draws.shape[0]

    def test_load_state_warns_on_different_data(self, tmp_path, caplog):
        sampler, _ = self._sampler_with_state()
        prefix = str(tmp_path / 'chain')
        sampler.save(prefix)

        x, y, w = _xyw()
        other = Sampler(_StubFitter(), x, 2.0 * y, w)
        with caplog.at_level(logging.WARNING, logger='easyscience.fitting'):
            other.load_state(prefix)

        assert 'does not match the data' in caplog.text
