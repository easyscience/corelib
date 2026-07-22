# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause
"""Unit tests for ``sampler.py`` — mirrors ``src/easyscience/fitting/sampler.py``.

Full sampling runs and save/load/resume roundtrips live in
``tests/integration/fitting/test_sampler.py``.
"""

import json
import logging

import numpy as np
import pytest

from easyscience import ObjBase
from easyscience import Parameter
from easyscience.fitting import Sampler
from easyscience.fitting import SamplingResults
from easyscience.fitting.minimizers.minimizer_base import MINIMIZER_PARAMETER_PREFIX
from easyscience.fitting.multi_fitter import MultiFitter
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


def _bumps_fitter_and_data():
    """Build a 2-parameter BUMPS MultiFitter over a small sine model."""
    ref_sin = AbsSin(0.2, np.pi)
    sp = AbsSin(0.354, 3.05)
    sp.offset.fixed = False
    sp.phase.fixed = False
    x = np.linspace(0, 5, 50)
    y = ref_sin(x)
    weights = np.ones_like(x)
    f = MultiFitter([sp], [sp])
    try:
        f.switch_minimizer('Bumps')
    except AttributeError:
        pytest.skip('BUMPS is not installed')
    return f, sp, x, y, weights


def _xyw():
    x = np.linspace(0, 5, 50)
    return x, np.sin(x), np.ones_like(x)


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
    def test_sample_requires_bumps(self):
        """sample() must raise RuntimeError if the minimizer is not BUMPS —
        and must not mutate the fitter (no needless minimizer rebuild)."""
        sp = AbsSin(0.354, 3.05)
        f = MultiFitter([sp], [sp])

        x, y, w = _xyw()
        sampler = Sampler(f, [x], [y], [w])
        minimizer_before = f.minimizer
        with pytest.raises(RuntimeError, match='Bayesian sampling requires a BUMPS minimizer'):
            sampler.sample(samples=10, burn=5, thin=1)
        assert f.minimizer is minimizer_before

    def test_fit_function_restored_on_error(self):
        """fit_function must be restored even when the minimizer raises."""
        f, _, x, y, weights = _bumps_fitter_and_data()
        sampler = Sampler(f, [x], [y], [weights])
        original_func = f.fit_function

        # Invalid `samples` is rejected by the minimizer (single source of
        # validation) *after* the fitter has been mutated for sampling.
        with pytest.raises(ValueError, match='samples must be a positive integer'):
            sampler.sample(samples=-1, burn=5, thin=1)

        assert f.fit_function is original_func

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
