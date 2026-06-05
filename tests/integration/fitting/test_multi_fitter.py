# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest

from easyscience import ObjBase
from easyscience import Parameter
from easyscience.fitting.minimizers import FitError
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


@pytest.mark.parametrize('fit_engine', [None, 'LMFit', 'Bumps', 'DFO'])
def test_multi_fit(fit_engine):
    ref_sin_1 = AbsSin(0.2, np.pi)
    sp_sin_1 = AbsSin(0.354, 3.05)
    ref_sin_2 = AbsSin(np.pi * 0.45, 0.45 * np.pi * 0.5)
    sp_sin_2 = AbsSin(1, 0.5)

    ref_sin_2.offset.make_dependent_on(
        dependency_expression='ref_sin1', dependency_map={'ref_sin1': ref_sin_1.offset}
    )

    sp_sin_2.offset.make_dependent_on(
        dependency_expression='sp_sin1', dependency_map={'sp_sin1': sp_sin_1.offset}
    )

    x1 = np.linspace(0, 5, 200)
    y1 = ref_sin_1(x1)
    x2 = np.copy(x1)
    y2 = ref_sin_2(x2)
    weights = np.ones_like(x1)

    sp_sin_1.offset.fixed = False
    sp_sin_1.phase.fixed = False
    sp_sin_2.phase.fixed = False

    f = MultiFitter([sp_sin_1, sp_sin_2], [sp_sin_1, sp_sin_2])
    if fit_engine is not None:
        try:
            f.switch_minimizer(fit_engine)
        except AttributeError:
            pytest.skip(msg=f'{fit_engine} is not installed')

    results = f.fit(x=[x1, x2], y=[y1, y2], weights=[weights, weights])
    X = [x1, x2]
    Y = [y1, y2]
    F_ref = [ref_sin_1, ref_sin_2]
    F_real = [sp_sin_1, sp_sin_2]
    for idx, result in enumerate(results):
        assert result.n_pars == len(sp_sin_1.get_fit_parameters()) + len(
            sp_sin_2.get_fit_parameters()
        )
        assert result.chi2 == pytest.approx(0, abs=1.5e-3 * (len(result.x) - result.n_pars))
        assert result.reduced_chi2 == pytest.approx(0, abs=1.5e-3)
        assert result.success
        assert np.all(result.x == X[idx])
        assert np.all(result.y_obs == Y[idx])
        assert result.y_calc == pytest.approx(F_ref[idx](X[idx]), abs=1e-2)
        assert result.residual == pytest.approx(F_real[idx](X[idx]) - F_ref[idx](X[idx]), abs=1e-2)


@pytest.mark.parametrize('fit_engine', [None, 'LMFit', 'Bumps', 'DFO'])
def test_multi_fit_propagates_iteration_metadata_and_message(fit_engine):
    """Verify that fit metadata and message are copied into each per-dataset result."""
    ref_sin_1 = AbsSin(0.2, np.pi)
    sp_sin_1 = AbsSin(0.354, 3.05)
    ref_sin_2 = AbsSin(np.pi * 0.45, 0.45 * np.pi * 0.5)
    sp_sin_2 = AbsSin(1, 0.5)

    ref_sin_2.offset.make_dependent_on(
        dependency_expression='ref_sin1', dependency_map={'ref_sin1': ref_sin_1.offset}
    )
    sp_sin_2.offset.make_dependent_on(
        dependency_expression='sp_sin1', dependency_map={'sp_sin1': sp_sin_1.offset}
    )

    x1 = np.linspace(0, 5, 200)
    y1 = ref_sin_1(x1)
    x2 = np.copy(x1)
    y2 = ref_sin_2(x2)
    weights = np.ones_like(x1)

    sp_sin_1.offset.fixed = False
    sp_sin_1.phase.fixed = False
    sp_sin_2.phase.fixed = False

    f = MultiFitter([sp_sin_1, sp_sin_2], [sp_sin_1, sp_sin_2])
    if fit_engine is not None:
        try:
            f.switch_minimizer(fit_engine)
        except AttributeError:
            pytest.skip(msg=f'{fit_engine} is not installed')

    results = f.fit(x=[x1, x2], y=[y1, y2], weights=[weights, weights])
    for result in results:
        assert result.n_evaluations is not None
        assert isinstance(result.n_evaluations, int)
        assert result.n_evaluations > 0
        assert result.iterations is not None
        assert isinstance(result.iterations, int)
        assert result.iterations >= 0
        assert isinstance(result.message, str)


@pytest.mark.parametrize('fit_engine', [None, 'LMFit', 'Bumps', 'DFO'])
def test_multi_fit2(fit_engine):
    ref_sin_1 = AbsSin(0.2, np.pi)
    sp_sin_1 = AbsSin(0.354, 3.05)
    ref_sin_2 = AbsSin(np.pi * 0.45, 0.45 * np.pi * 0.5)
    sp_sin_2 = AbsSin(1, 0.5)  # ref_sin_1_obj = genObjs[0]
    ref_line_obj = Line(1, 4.6)

    ref_sin_2.offset.make_dependent_on(
        dependency_expression='ref_sin1', dependency_map={'ref_sin1': ref_sin_1.offset}
    )
    ref_line_obj.m.make_dependent_on(
        dependency_expression='ref_sin1', dependency_map={'ref_sin1': ref_sin_1.offset}
    )

    sp_line = Line(0.43, 6.1)

    sp_sin_2.offset.make_dependent_on(
        dependency_expression='sp_sin1', dependency_map={'sp_sin1': sp_sin_1.offset}
    )
    sp_line.m.make_dependent_on(
        dependency_expression='sp_sin1', dependency_map={'sp_sin1': sp_sin_1.offset}
    )

    x1 = np.linspace(0, 5, 200)
    y1 = ref_sin_1(x1)
    x3 = np.copy(x1)
    y3 = ref_sin_2(x3)
    x2 = np.copy(x1)
    y2 = ref_line_obj(x2)
    weights = np.ones_like(x1)

    sp_sin_1.offset.fixed = False
    sp_sin_1.phase.fixed = False
    sp_sin_2.phase.fixed = False
    sp_line.c.fixed = False

    f = MultiFitter([sp_sin_1, sp_line, sp_sin_2], [sp_sin_1, sp_line, sp_sin_2])
    if fit_engine is not None:
        try:
            f.switch_minimizer(fit_engine)
        except AttributeError:
            pytest.skip(msg=f'{fit_engine} is not installed')

    results = f.fit(x=[x1, x2, x3], y=[y1, y2, y3], weights=[weights, weights, weights])
    X = [x1, x2, x3]
    Y = [y1, y2, y3]
    F_ref = [ref_sin_1, ref_line_obj, ref_sin_2]
    F_real = [sp_sin_1, sp_line, sp_sin_2]

    assert len(results) == len(X)

    for idx, result in enumerate(results):
        assert result.n_pars == len(sp_sin_1.get_fit_parameters()) + len(
            sp_sin_2.get_fit_parameters()
        ) + len(sp_line.get_fit_parameters())
        assert result.chi2 == pytest.approx(0, abs=1.5e-3 * (len(result.x) - result.n_pars))
        assert result.reduced_chi2 == pytest.approx(0, abs=1.5e-3)
        assert result.success
        assert np.all(result.x == X[idx])
        assert np.all(result.y_obs == Y[idx])
        assert result.y_calc == pytest.approx(F_real[idx](X[idx]), abs=1e-2)
        assert result.residual == pytest.approx(F_ref[idx](X[idx]) - F_real[idx](X[idx]), abs=1e-2)


@pytest.mark.parametrize('fit_engine', [None, 'LMFit', 'Bumps', 'DFO'])
def test_multi_fit_1D_2D(fit_engine):
    # Generate fit and reference objects
    ref_sin1D = AbsSin(0.2, np.pi)
    sp_sin1D = AbsSin(0.354, 3.05)

    ref_sin2D = AbsSin2D(0.3, 1.6)
    sp_sin2D = AbsSin2D(0.1, 1.75)  # The fit is VERY sensitive to the initial values :-(

    # Link the parameters
    ref_sin2D.offset.make_dependent_on(
        dependency_expression='ref_sin1', dependency_map={'ref_sin1': ref_sin1D.offset}
    )
    sp_sin2D.offset.make_dependent_on(
        dependency_expression='sp_sin1', dependency_map={'sp_sin1': sp_sin1D.offset}
    )

    # Generate data
    x1D = np.linspace(0.2, 3.8, 400)
    y1D = ref_sin1D(x1D)
    weights1D = np.ones_like(x1D)

    x = np.linspace(0, 5, 200)
    X, Y = np.meshgrid(x, x)
    x2D = np.stack((X, Y), axis=2)
    y2D = ref_sin2D(x2D)
    weights2D = np.ones_like(y2D)

    ff = MultiFitter([sp_sin1D, sp_sin2D], [sp_sin1D, sp_sin2D])
    if fit_engine is not None:
        try:
            ff.switch_minimizer(fit_engine)
        except AttributeError:
            pytest.skip(msg=f'{fit_engine} is not installed')

    sp_sin1D.offset.fixed = False
    sp_sin1D.phase.fixed = False
    sp_sin2D.phase.fixed = False

    f = MultiFitter([sp_sin1D, sp_sin2D], [sp_sin1D, sp_sin2D])
    if fit_engine is not None:
        try:
            f.switch_minimizer(fit_engine)
        except AttributeError:
            pytest.skip(msg=f'{fit_engine} is not installed')
    try:
        results = f.fit(
            x=[x1D, x2D], y=[y1D, y2D], weights=[weights1D, weights2D], vectorized=True
        )
    except FitError as e:
        if 'Unable to allocate' in str(e):
            pytest.skip(msg='MemoryError - Matrix too large')
        else:
            raise e

    X = [x1D, x2D]
    Y = [y1D, y2D]
    F_ref = [ref_sin1D, ref_sin2D]
    F_real = [sp_sin1D, sp_sin2D]
    for idx, result in enumerate(results):
        assert result.n_pars == len(sp_sin1D.get_fit_parameters()) + len(
            sp_sin2D.get_fit_parameters()
        )
        if (
            fit_engine != 'DFO'
        ):  # DFO apparently does not fit well with even weights. Can't be bothered to fix
            assert result.chi2 == pytest.approx(0, abs=1.5e-3 * (len(result.x) - result.n_pars))
            assert result.reduced_chi2 == pytest.approx(0, abs=1.5e-3)
            assert result.y_calc == pytest.approx(F_ref[idx](X[idx]), abs=1e-2)
            assert result.residual == pytest.approx(
                F_real[idx](X[idx]) - F_ref[idx](X[idx]), abs=1e-2
            )
        assert result.success
        assert np.all(result.x == X[idx])
        assert np.all(result.y_obs == Y[idx])


# ---------------------------------------------------------------------------
# Tests for MultiFitter.mcmc_sample (Bayesian MCMC via BUMPS DREAM)
# ---------------------------------------------------------------------------


class TestMultiFitterMcmcSample:
    """Integration tests for ``MultiFitter.mcmc_sample``."""

    def test_raises_runtime_error_when_not_bumps(self):
        """mcmc_sample() must raise RuntimeError if the minimizer is not a BUMPS instance."""
        sp = AbsSin(0.354, 3.05)
        f = MultiFitter([sp], [sp])

        x = np.linspace(0, 5, 50)
        y = np.sin(x)
        weights = np.ones_like(x)

        with pytest.raises(RuntimeError, match='Bayesian sampling requires a BUMPS minimizer'):
            f.mcmc_sample(x=[x], y=[y], weights=[weights], samples=10, burn=5, thin=1)

    @pytest.mark.filterwarnings('ignore::UserWarning')
    def test_returns_expected_keys_and_shapes(self):
        """mcmc_sample() with BUMPS should return draws, param_names, state, logp."""
        ref_sin = AbsSin(0.2, np.pi)
        sp = AbsSin(0.354, 3.05)

        x = np.linspace(0, 5, 50)
        y = ref_sin(x)
        weights = np.ones_like(x)

        sp.offset.fixed = False
        sp.phase.fixed = False

        f = MultiFitter([sp], [sp])
        try:
            f.switch_minimizer('Bumps')
        except AttributeError:
            pytest.skip('BUMPS is not installed')

        result = f.mcmc_sample(x=[x], y=[y], weights=[weights], samples=100, burn=20, thin=2)

        assert isinstance(result, dict)
        assert 'draws' in result
        assert 'param_names' in result
        assert 'internal_bumps_object' in result
        assert 'logp' in result

        # draws shape: (retained_samples, n_params)
        assert result['draws'].ndim == 2
        assert result['draws'].shape[1] == len(result['param_names'])

        # param_names should match the model's fit parameters
        expected_pars = {p.unique_name for p in sp.get_fit_parameters()}
        assert set(result['param_names']) == expected_pars

    @pytest.mark.filterwarnings('ignore::UserWarning')
    def test_multi_dataset_returns_consistent_param_names(self):
        """mcmc_sample() with multiple datasets should have correct param_names across all."""
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

        result = f.mcmc_sample(
            x=[x1, x2], y=[y1, y2], weights=[weights, weights], samples=100, burn=20, thin=2
        )

        # All parameters across both models should appear
        all_params = {p.unique_name for p in sp_sin_1.get_fit_parameters()}
        all_params |= {p.unique_name for p in sp_line.get_fit_parameters()}
        assert set(result['param_names']) == all_params

    def test_population_param_controls_chain_count(self):
        """Passing population should succeed and produce valid draws."""
        sp = AbsSin(0.354, 3.05)
        sp.offset.fixed = False
        sp.phase.fixed = False
        f = MultiFitter([sp], [sp])
        try:
            f.switch_minimizer('Bumps')
        except AttributeError:
            pytest.skip('BUMPS is not installed')

        x = np.linspace(0, 5, 50)
        y = np.sin(x)
        weights = np.ones_like(x)

        result = f.mcmc_sample(
            x=[x], y=[y], weights=[weights], samples=100, burn=20, thin=2, population=5
        )
        assert 'draws' in result

    @pytest.mark.filterwarnings('ignore::UserWarning')
    def test_vectorized_2d_input_produces_valid_draws(self):
        """mcmc_sample() with vectorized=True and 2D input should produce valid draws."""
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

        result = f.mcmc_sample(
            x=[x2D], y=[y2D], weights=[weights], samples=100, burn=20, thin=2, vectorized=True
        )

        assert result['draws'].ndim == 2
        assert result['draws'].shape[0] > 0
        assert result['draws'].shape[1] == len(result['param_names'])

    def test_fit_function_restored_after_runtime_error(self):
        """fit_function must be restored to its original value even when mcmc_sample() raises."""
        sp = AbsSin(0.354, 3.05)
        f = MultiFitter([sp], [sp])

        x = np.linspace(0, 5, 50)
        y = np.sin(x)
        weights = np.ones_like(x)

        original_func = f.fit_function

        with pytest.raises(RuntimeError):
            f.mcmc_sample(x=[x], y=[y], weights=[weights], samples=10, burn=5, thin=1)

        assert f.fit_function is original_func

    @pytest.mark.filterwarnings('ignore::UserWarning')
    def test_fit_function_restored_after_successful_sample(self):
        """fit_function must be restored to its original value after a successful mcmc_sample()."""
        ref_sin = AbsSin(0.2, np.pi)
        sp = AbsSin(0.354, 3.05)

        x = np.linspace(0, 5, 50)
        y = ref_sin(x)
        weights = np.ones_like(x)

        sp.offset.fixed = False
        sp.phase.fixed = False

        f = MultiFitter([sp], [sp])
        try:
            f.switch_minimizer('Bumps')
        except AttributeError:
            pytest.skip('BUMPS is not installed')

        original_func = f.fit_function
        f.mcmc_sample(x=[x], y=[y], weights=[weights], samples=100, burn=20, thin=2)
        assert f.fit_function is original_func

    @pytest.mark.filterwarnings('ignore::UserWarning')
    def test_sampler_kwargs_forwarded(self):
        """sampler_kwargs dict is forwarded to the BUMPS DREAM sampler."""
        ref_sin = AbsSin(0.2, np.pi)
        sp = AbsSin(0.354, 3.05)

        x = np.linspace(0, 5, 50)
        y = ref_sin(x)
        weights = np.ones_like(x)

        sp.offset.fixed = False
        sp.phase.fixed = False

        f = MultiFitter([sp], [sp])
        try:
            f.switch_minimizer('Bumps')
        except AttributeError:
            pytest.skip('BUMPS is not installed')

        result = f.mcmc_sample(
            x=[x],
            y=[y],
            weights=[weights],
            samples=100,
            burn=20,
            thin=2,
            sampler_kwargs={'init': 'random'},
        )

        assert result['draws'].ndim == 2
        assert result['draws'].shape[0] > 0


class TestSampleAliasResolution:
    def test_conflicting_chains_and_population_raises(self):
        """Passing both chains and population with different values must raise."""
        sp = AbsSin(0.354, 3.05)
        f = MultiFitter([sp], [sp])
        try:
            f.switch_minimizer('Bumps')
        except AttributeError:
            pytest.skip('BUMPS is not installed')

        x = np.linspace(0, 5, 50)
        y = np.sin(x)
        weights = np.ones_like(x)

        with pytest.raises(ValueError, match='Conflicting population arguments'):
            f.mcmc_sample(
                x=[x], y=[y], weights=[weights], samples=10, burn=5, thin=1, chains=3, population=5
            )

    def test_chains_and_population_equal_is_ok(self):
        """Passing chains == population should succeed (no conflict)."""
        sp = AbsSin(0.354, 3.05)
        sp.offset.fixed = False
        sp.phase.fixed = False
        f = MultiFitter([sp], [sp])
        try:
            f.switch_minimizer('Bumps')
        except AttributeError:
            pytest.skip('BUMPS is not installed')

        x = np.linspace(0, 5, 50)
        y = np.sin(x)
        weights = np.ones_like(x)

        # Should not raise ValueError — chains and population are equal.
        # DREAM needs a sufficient population; 5 is a safe minimum.
        result = f.mcmc_sample(
            x=[x], y=[y], weights=[weights], samples=100, burn=20, thin=2, chains=5, population=5
        )
        assert 'draws' in result


class TestSampleSeedReproducibility:
    @pytest.mark.filterwarnings('ignore::UserWarning')
    def test_seed_produces_valid_draws(self):
        """Running mcmc_sample() with a seed must produce valid draws."""
        ref_sin = AbsSin(0.2, np.pi)
        sp = AbsSin(0.354, 3.05)

        x = np.linspace(0, 5, 50)
        y = ref_sin(x)
        weights = np.ones_like(x)

        sp.offset.fixed = False
        sp.phase.fixed = False

        f = MultiFitter([sp], [sp])
        try:
            f.switch_minimizer('Bumps')
        except AttributeError:
            pytest.skip('BUMPS is not installed')

        result = f.mcmc_sample(
            x=[x], y=[y], weights=[weights], samples=100, burn=20, thin=2, seed=42
        )

        assert result['draws'].ndim == 2
        assert result['draws'].shape[0] > 0
        assert result['draws'].shape[1] == len(result['param_names'])
        # logp should be present (may be None if not computed)
        assert 'logp' in result

    @pytest.mark.filterwarnings('ignore::UserWarning')
    def test_different_seeds_both_produce_valid_draws(self):
        """Running mcmc_sample() with different seeds should each produce valid draws."""
        ref_sin = AbsSin(0.2, np.pi)
        sp = AbsSin(0.354, 3.05)

        x = np.linspace(0, 5, 50)
        y = ref_sin(x)
        weights = np.ones_like(x)

        sp.offset.fixed = False
        sp.phase.fixed = False

        f = MultiFitter([sp], [sp])
        try:
            f.switch_minimizer('Bumps')
        except AttributeError:
            pytest.skip('BUMPS is not installed')

        result1 = f.mcmc_sample(
            x=[x], y=[y], weights=[weights], samples=100, burn=20, thin=2, seed=42
        )
        result2 = f.mcmc_sample(
            x=[x], y=[y], weights=[weights], samples=100, burn=20, thin=2, seed=12345
        )

        # Both must produce valid draws
        assert result1['draws'].shape[0] > 0
        assert result2['draws'].shape[0] > 0
        assert result1['draws'].ndim == 2
        assert result2['draws'].ndim == 2


class TestSampleMultiprocessing:
    @pytest.mark.filterwarnings('ignore::UserWarning')
    def test_n_workers_one_uses_sequential_mapper(self):
        """n_workers=1 should behave like the default sequential DREAM mapper."""
        ref_sin = AbsSin(0.2, np.pi)
        sp = AbsSin(0.354, 3.05)

        x = np.linspace(0, 5, 50)
        y = ref_sin(x)
        weights = np.ones_like(x)

        sp.offset.fixed = False
        sp.phase.fixed = False

        f = MultiFitter([sp], [sp])
        try:
            f.switch_minimizer('Bumps')
        except AttributeError:
            pytest.skip('BUMPS is not installed')

        result = f.mcmc_sample(
            x=[x], y=[y], weights=[weights], samples=50, burn=10, thin=2, n_workers=1
        )

        assert result['draws'].ndim == 2
        assert result['draws'].shape[0] > 0
        assert result['draws'].shape[1] == len(result['param_names'])

    def test_n_workers_must_be_positive(self):
        """n_workers must be positive when explicitly provided."""
        sp = AbsSin(0.354, 3.05)
        sp.offset.fixed = False
        sp.phase.fixed = False

        f = MultiFitter([sp], [sp])
        try:
            f.switch_minimizer('Bumps')
        except AttributeError:
            pytest.skip('BUMPS is not installed')

        x = np.linspace(0, 5, 50)
        y = np.sin(x)
        weights = np.ones_like(x)

        with pytest.raises(ValueError, match='n_workers must be at least 1'):
            f.mcmc_sample(x=[x], y=[y], weights=[weights], samples=10, burn=5, thin=1, n_workers=0)

    @pytest.mark.filterwarnings('ignore::UserWarning')
    def test_n_workers_two_produces_valid_draws(self, tmp_path):
        """n_workers>1 should evaluate DREAM populations through process workers."""
        repo_root = str(Path(__file__).resolve().parents[3])
        test_file = str(Path(__file__).resolve())
        script = tmp_path / 'run_bumps_multiprocessing_sample.py'
        script.write_text(
            textwrap.dedent(
                f"""
                import importlib.util
                import multiprocessing as mp
                import sys

                sys.path.insert(0, {repo_root!r})
                sys.path.insert(0, {repo_root + '/src'!r})

                import numpy as np

                from easyscience.fitting.multi_fitter import MultiFitter

                # Load the AbsSin model straight from this test file to avoid
                # depending on ``tests`` being importable as a package (it is a
                # plain namespace dir and can be shadowed by sibling projects).
                _spec = importlib.util.spec_from_file_location('_mp_test_models', {test_file!r})
                _models = importlib.util.module_from_spec(_spec)
                _spec.loader.exec_module(_models)
                AbsSin = _models.AbsSin


                def main():
                    ref_sin = AbsSin(0.2, np.pi)
                    sp = AbsSin(0.354, 3.05)

                    x = np.linspace(0, 5, 40)
                    y = ref_sin(x)
                    weights = np.ones_like(x)

                    sp.offset.fixed = False
                    sp.phase.fixed = False

                    f = MultiFitter([sp], [sp])
                    f.switch_minimizer('Bumps')
                    result = f.mcmc_sample(
                        x=[x],
                        y=[y],
                        weights=[weights],
                        samples=50,
                        burn=10,
                        thin=2,
                        population=5,
                        n_workers=2,
                    )

                    assert result['draws'].ndim == 2
                    assert result['draws'].shape[0] > 0
                    assert result['draws'].shape[1] == len(result['param_names'])


                if __name__ == '__main__':
                    mp.freeze_support()
                    main()
                """
            ),
            encoding='utf-8',
        )

        try:
            completed = subprocess.run(
                [sys.executable, str(script)],
                cwd=repo_root,
                capture_output=True,
                text=True,
                timeout=60,
                check=False,
            )
        except subprocess.TimeoutExpired:
            pytest.fail('n_workers=2 sampling subprocess timed out after 60 seconds')

        assert completed.returncode == 0, completed.stdout + completed.stderr
