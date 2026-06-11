# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause
"""Integration tests for the ``Sampler`` class (Bayesian MCMC via BUMPS DREAM).

The behavioural MCMC tests formerly in
``test_multi_fitter.py::TestMultiFitterMcmcSample`` were moved here and
adapted to the ``fitter.create_sampler(...)`` API. The deprecated
``Fitter.mcmc_sample()`` shim is covered by the ``TestDeprecatedMcmcSample``
class at the bottom.
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


class TestSampler:
    """Integration tests for ``fitter.create_sampler(...)`` / ``Sampler``."""

    def test_sample_requires_bumps(self):
        """sample() must raise RuntimeError if the minimizer is not BUMPS —
        and must not mutate the fitter (no needless minimizer rebuild)."""
        sp = AbsSin(0.354, 3.05)
        f = MultiFitter([sp], [sp])

        x = np.linspace(0, 5, 50)
        y = np.sin(x)
        weights = np.ones_like(x)

        sampler = f.create_sampler([x], [y], [weights])
        minimizer_before = f.minimizer
        with pytest.raises(RuntimeError, match='Bayesian sampling requires a BUMPS minimizer'):
            sampler.sample(samples=10, burn=5, thin=1)
        assert f.minimizer is minimizer_before

    @pytest.mark.filterwarnings('ignore::UserWarning')
    def test_sample_returns_results_object(self):
        """sample() returns a populated SamplingResults; to_legacy_dict() has the legacy shape."""
        f, sp, x, y, weights = _bumps_fitter_and_data()
        sampler = f.create_sampler([x], [y], [weights])

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
        """Multi-dataset sampling via MultiFitter.create_sampler has correct param_names."""
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

        f = MultiFitter([sp_sin_1, sp_line], [sp_sin_1, sp_line])
        try:
            f.switch_minimizer('Bumps')
        except AttributeError:
            pytest.skip('BUMPS is not installed')

        sampler = f.create_sampler([x1, x2], [y1, y2], [weights, weights])
        results = sampler.sample(samples=100, burn=20, thin=2)

        # All parameters across both models should appear
        all_params = {p.unique_name for p in sp_sin_1.get_fit_parameters()}
        all_params |= {p.unique_name for p in sp_line.get_fit_parameters()}
        assert set(results.param_names) == all_params

    def test_sample_population(self):
        """Passing population should succeed and produce valid draws."""
        f, _, x, y, weights = _bumps_fitter_and_data()
        sampler = f.create_sampler([x], [y], [weights])

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

        f = MultiFitter([sp], [sp])
        try:
            f.switch_minimizer('Bumps')
        except AttributeError:
            pytest.skip('BUMPS is not installed')

        sampler = f.create_sampler([x2D], [y2D], [weights], vectorized=True)
        results = sampler.sample(samples=100, burn=20, thin=2)

        assert results.draws.ndim == 2
        assert results.draws.shape[0] > 0
        assert results.draws.shape[1] == len(results.param_names)

    def test_fit_function_restored_on_error(self):
        """fit_function must be restored even when the minimizer raises."""
        f, _, x, y, weights = _bumps_fitter_and_data()
        sampler = f.create_sampler([x], [y], [weights])
        original_func = f.fit_function

        # Invalid `samples` is rejected by the minimizer (single source of
        # validation) *after* the fitter has been mutated for sampling.
        with pytest.raises(ValueError, match='samples must be a positive integer'):
            sampler.sample(samples=-1, burn=5, thin=1)

        assert f.fit_function is original_func

    @pytest.mark.filterwarnings('ignore::UserWarning')
    def test_fit_function_restored_on_success(self):
        """fit_function must be restored after a successful sample()."""
        f, _, x, y, weights = _bumps_fitter_and_data()
        sampler = f.create_sampler([x], [y], [weights])
        original_func = f.fit_function

        sampler.sample(samples=100, burn=20, thin=2)
        assert f.fit_function is original_func

    @pytest.mark.filterwarnings('ignore::UserWarning')
    def test_sampler_kwargs_forwarded(self):
        """Per-call sampler_kwargs dict is forwarded to the BUMPS DREAM sampler."""
        f, _, x, y, weights = _bumps_fitter_and_data()
        sampler = f.create_sampler([x], [y], [weights])

        results = sampler.sample(samples=100, burn=20, thin=2, sampler_kwargs={'init': 'random'})

        assert results.draws.ndim == 2
        assert results.draws.shape[0] > 0

    @pytest.mark.filterwarnings('ignore::UserWarning')
    def test_default_sampler_kwargs_merged(self):
        """Constructor-level sampler_kwargs defaults are used; per-call kwargs win."""
        f, _, x, y, weights = _bumps_fitter_and_data()
        sampler = f.create_sampler([x], [y], [weights], sampler_kwargs={'init': 'random'})

        captured = {}
        original_mcmc_sample = type(f.minimizer).mcmc_sample

        def spy(self, **kwargs):
            captured.update(kwargs.get('sampler_kwargs') or {})
            return original_mcmc_sample(self, **kwargs)

        try:
            type(f.minimizer).mcmc_sample = spy
            sampler.sample(samples=100, burn=20, thin=2)
            assert captured == {'init': 'random'}

            captured.clear()
            sampler.sample(samples=100, burn=20, thin=2, sampler_kwargs={'init': 'lhs'})
            assert captured == {'init': 'lhs'}  # per-call overrides default
        finally:
            type(f.minimizer).mcmc_sample = original_mcmc_sample

    @pytest.mark.filterwarnings('ignore::UserWarning')
    def test_extend_chain(self):
        """extend(additional_samples=) continues the chain; ring-buffer math is done for the user."""
        f, _, x, y, weights = _bumps_fitter_and_data()
        sampler = f.create_sampler([x], [y], [weights])

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
        f, _, x, y, weights = _bumps_fitter_and_data()
        sampler = f.create_sampler([x], [y], [weights])

        first = sampler.sample(samples=1000, burn=20, thin=10)
        n_first = first.draws.shape[0]

        extended = sampler.extend(additional_samples=1000, thin=10)

        # All previously retained draws are kept, plus ~additional/thin new ones.
        assert extended.draws.shape[0] > n_first

    @pytest.mark.filterwarnings('ignore::UserWarning')
    def test_extend_total_samples_override(self):
        """extend(total_samples=) bypasses the additional_samples arithmetic."""
        f, _, x, y, weights = _bumps_fitter_and_data()
        sampler = f.create_sampler([x], [y], [weights])

        sampler.sample(samples=100, burn=20, thin=1)
        extended = sampler.extend(total_samples=150, thin=1)

        assert extended.draws.shape[0] > 0

    def test_extend_requires_existing_state(self):
        """extend() before sample()/load_state() raises RuntimeError."""
        f, _, x, y, weights = _bumps_fitter_and_data()
        sampler = f.create_sampler([x], [y], [weights])

        with pytest.raises(RuntimeError, match='No chain to extend'):
            sampler.extend(additional_samples=10)

    @pytest.mark.filterwarnings('ignore::UserWarning')
    def test_extend_after_save_load_roundtrip(self, tmp_path):
        """save() -> new sampler -> load_state() -> extend() continues the chain.

        Regression coverage: BUMPS ``load_state`` does not preserve parameter
        labels, so resume validation must proceed by parameter order (with a
        warning) instead of raising.
        """
        f, _, x, y, weights = _bumps_fitter_and_data()
        sampler = f.create_sampler([x], [y], [weights])
        first = sampler.sample(samples=100, burn=20, thin=1)

        prefix = str(tmp_path / 'chain')
        sampler.save(prefix)

        sampler2 = f.create_sampler([x], [y], [weights])
        loaded = sampler2.load_state(prefix)
        assert loaded.draws.shape[1] == first.draws.shape[1]

        with pytest.warns(UserWarning, match='does not carry parameter names'):
            extended = sampler2.extend(additional_samples=100, thin=1)

        assert extended.draws.shape[1] == first.draws.shape[1]
        assert extended.draws.shape[0] > 0

    @pytest.mark.filterwarnings('ignore::UserWarning')
    def test_extend_preserves_nondefault_population(self):
        """Extending a chain that used a non-default population must not fail.

        Regression test: the population scale factor is recovered from the
        saved state on resume, otherwise BUMPS regenerates the default
        population and raises ``Cannot change Nvar, Npop or Ncr on resize``.
        """
        f, _, x, y, weights = _bumps_fitter_and_data()
        sampler = f.create_sampler([x], [y], [weights])

        first = sampler.sample(samples=100, burn=20, thin=1, population=5)
        first_npop = first.state.Npop

        extended = sampler.extend(additional_samples=100, thin=1)

        assert extended.draws.shape[1] == first.draws.shape[1]
        assert extended.draws.shape[0] > 0
        # Population must be preserved across the extend.
        assert extended.state.Npop == first_npop

    def test_save_raises_without_state(self, tmp_path):
        """save() before sample() raises RuntimeError."""
        f, _, x, y, weights = _bumps_fitter_and_data()
        sampler = f.create_sampler([x], [y], [weights])

        with pytest.raises(RuntimeError, match='No chain state to save'):
            sampler.save(str(tmp_path / 'chain'))

    @pytest.mark.filterwarnings('ignore::UserWarning')
    def test_load_state_populates_results(self, tmp_path):
        """A freshly loaded sampler reports draws/logp/param_names without resampling."""
        f, sp, x, y, weights = _bumps_fitter_and_data()
        sampler = f.create_sampler([x], [y], [weights])
        first = sampler.sample(samples=100, burn=20, thin=2)

        prefix = str(tmp_path / 'chain')
        sampler.save(prefix)

        sampler2 = f.create_sampler([x], [y], [weights])
        assert sampler2.draws is None
        loaded = sampler2.load_state(prefix)

        assert sampler2.state is loaded.state
        assert loaded.draws.ndim == 2
        assert loaded.draws.shape[1] == first.draws.shape[1]
        assert loaded.logp.shape[0] == loaded.draws.shape[0]
        # Sidecar restores the original (prefix-stripped) parameter names.
        assert loaded.param_names == first.param_names

    @pytest.mark.filterwarnings('ignore::UserWarning')
    def test_load_restores_param_names_from_sidecar(self, tmp_path):
        """v2 sidecar restores names; a legacy v1 sidecar is still accepted."""
        f, _, x, y, weights = _bumps_fitter_and_data()
        sampler = f.create_sampler([x], [y], [weights])
        first = sampler.sample(samples=100, burn=20, thin=2)

        prefix = str(tmp_path / 'chain')
        sampler.save(prefix)

        # v2 sidecar written by Sampler.save()
        with open(f'{prefix}.params.json') as fh:
            sidecar = json.load(fh)
        assert sidecar['schema_version'] == 2
        assert sidecar['param_names'] == first.param_names
        assert sidecar['data_fingerprint'] is not None

        # Rewrite as a legacy v1 sidecar (as written by old save_posterior)
        with open(f'{prefix}.params.json', 'w') as fh:
            json.dump(
                {
                    'schema_version': 1,
                    'param_names': first.param_names,
                    'easyreflectometry_version': '0.0.0',
                },
                fh,
            )

        sampler2 = f.create_sampler([x], [y], [weights])
        loaded = sampler2.load_state(prefix)
        assert loaded.param_names == first.param_names

    @pytest.mark.filterwarnings('ignore::UserWarning')
    def test_load_short_chain_regression(self, tmp_path):
        """Short chain save/load — pins the BUMPS >= 1.0.4 loadtxt workaround.

        A short chain stores a single CR-weight update row; BUMPS' numpy-based
        reader collapses it to a 1-D array and ``load_state`` raises
        ``IndexError`` without the 2-D coercion workaround.
        """
        f, _, x, y, weights = _bumps_fitter_and_data()
        sampler = f.create_sampler([x], [y], [weights])
        sampler.sample(samples=20, burn=5, thin=1)

        prefix = str(tmp_path / 'short_chain')
        sampler.save(prefix)

        sampler2 = f.create_sampler([x], [y], [weights])
        loaded = sampler2.load_state(prefix)
        assert loaded.draws.shape[0] > 0

    @pytest.mark.filterwarnings('ignore::UserWarning')
    def test_load_fingerprint_mismatch_warns(self, tmp_path):
        """Loading a chain into a sampler bound to different data warns."""
        f, _, x, y, weights = _bumps_fitter_and_data()
        sampler = f.create_sampler([x], [y], [weights])
        sampler.sample(samples=100, burn=20, thin=2)

        prefix = str(tmp_path / 'chain')
        sampler.save(prefix)

        other_y = y + 0.5
        sampler2 = f.create_sampler([x], [other_y], [weights])
        with pytest.warns(UserWarning, match='does not match the data fingerprint'):
            sampler2.load_state(prefix)


class TestDeprecatedMcmcSample:
    """The deprecated ``Fitter.mcmc_sample()`` shim keeps the full legacy
    signature working and returns the legacy dict shape."""

    @pytest.mark.filterwarnings('ignore::UserWarning')
    def test_warns_and_returns_legacy_dict(self):
        f, sp, x, y, weights = _bumps_fitter_and_data()

        with pytest.warns(DeprecationWarning, match='mcmc_sample.*deprecated'):
            result = f.mcmc_sample(x=[x], y=[y], weights=[weights], samples=100, burn=20, thin=2)

        assert isinstance(result, dict)
        assert set(result.keys()) == {'draws', 'param_names', 'internal_bumps_object', 'logp'}
        assert result['draws'].ndim == 2
        assert result['draws'].shape[1] == len(result['param_names'])
        expected_pars = {p.unique_name for p in sp.get_fit_parameters()}
        assert set(result['param_names']) == expected_pars

    def test_raises_runtime_error_when_not_bumps(self):
        """The shim still raises RuntimeError when the minimizer is not BUMPS."""
        sp = AbsSin(0.354, 3.05)
        f = MultiFitter([sp], [sp])

        x = np.linspace(0, 5, 50)
        y = np.sin(x)
        weights = np.ones_like(x)

        with pytest.warns(DeprecationWarning):
            with pytest.raises(RuntimeError, match='Bayesian sampling requires a BUMPS minimizer'):
                f.mcmc_sample(x=[x], y=[y], weights=[weights], samples=10, burn=5, thin=1)

    @pytest.mark.filterwarnings('ignore::UserWarning')
    @pytest.mark.filterwarnings('ignore::DeprecationWarning')
    def test_resume_state_still_works(self):
        """Legacy ``resume_state=`` continues a chain through the shim."""
        f, _, x, y, weights = _bumps_fitter_and_data()

        first = f.mcmc_sample(x=[x], y=[y], weights=[weights], samples=100, burn=20, thin=1)
        first_state = first['internal_bumps_object']

        extended = f.mcmc_sample(
            x=[x],
            y=[y],
            weights=[weights],
            samples=200,
            burn=0,
            thin=1,
            resume_state=first_state,
        )

        assert extended['draws'].shape[1] == first['draws'].shape[1]
        assert extended['draws'].shape[0] > 0
        assert extended['internal_bumps_object'].Npop == first_state.Npop
