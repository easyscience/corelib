# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import pytest

from easyscience import AvailableMinimizers
from easyscience import Fitter
from easyscience import ObjBase
from easyscience import Parameter
from easyscience.base_classes import ModelBase
from easyscience.fitting.minimizers import FitError


# Model and container of parameters for tests
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


class AbsSin2DL(AbsSin2D):
    def __call__(self, x):
        X = x[:, 0]  # x is a 1D array
        Y = x[:, 1]
        return np.abs(np.sin(self.phase.value * X + self.offset.value)) * np.abs(
            np.sin(self.phase.value * Y + self.offset.value)
        )


class StraightLine(ModelBase):
    def __init__(self, slope: float, intercept: float):
        super().__init__()
        self._slope = Parameter('slope', slope)
        self._intercept = Parameter('intercept', intercept)

    @property
    def slope(self) -> Parameter:
        return self._slope

    @slope.setter
    def slope(self, value: float) -> None:
        self._slope.value = value

    @property
    def intercept(self) -> Parameter:
        return self._intercept

    @intercept.setter
    def intercept(self, value: float) -> None:
        self._intercept.value = value

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.slope.value * x + self.intercept.value


def check_fit_results(result, sp_sin, ref_sin, x, **kwargs):
    assert result.n_pars == len(sp_sin.get_fit_parameters())
    assert result.chi2 == pytest.approx(0, abs=1.5e-3 * (len(result.x) - result.n_pars))
    assert result.reduced_chi2 == pytest.approx(0, abs=1.5e-3)
    assert result.success
    if 'sp_ref1' in kwargs.keys():
        sp_ref1 = kwargs['sp_ref1']
        for key, value in sp_ref1.items():
            assert key in result.p.keys()
            assert key in result.p0.keys()
            assert result.p0[key] == pytest.approx(value)  # Bumps does something strange here
    assert np.all(result.x == x)
    for item1, item2 in zip(sp_sin._kwargs.values(), ref_sin._kwargs.values()):
        # assert item.error > 0 % This does not work as some methods don't calculate error
        assert item1.error == pytest.approx(0, abs=2.1e-1)
        assert item1.value == pytest.approx(item2.value, abs=5e-3)
    y_calc_ref = ref_sin(x)
    assert result.y_calc == pytest.approx(y_calc_ref, abs=1e-2)
    assert result.residual == pytest.approx(sp_sin(x) - y_calc_ref, abs=1e-2)


@pytest.mark.fast
@pytest.mark.parametrize(
    'fit_engine',
    [
        None,
        AvailableMinimizers.LMFit,
        AvailableMinimizers.Bumps,
        AvailableMinimizers.DFO,
    ],
)
def test_basic_fit(fit_engine: AvailableMinimizers):
    ref_sin = AbsSin(0.2, np.pi)
    sp_sin = AbsSin(0.354, 3.05)

    x = np.linspace(0, 5, 200)
    weights = np.ones_like(x)
    y = ref_sin(x)

    sp_sin.offset.fixed = False
    sp_sin.phase.fixed = False

    f = Fitter(sp_sin, sp_sin)
    if fit_engine is not None:
        try:
            f.switch_minimizer(fit_engine)
        except AttributeError:
            pytest.skip(reason=f'{fit_engine} is not installed')

    result = f.fit(x=x, y=y, weights=weights)

    if fit_engine is not None:
        assert (
            result.minimizer_engine.package == fit_engine.name.lower()
        )  # Special case where minimizer matches package
    assert sp_sin.phase.value == pytest.approx(ref_sin.phase.value, rel=1e-3)
    assert sp_sin.offset.value == pytest.approx(ref_sin.offset.value, rel=1e-3)


@pytest.mark.fast
@pytest.mark.parametrize(
    'fit_engine',
    [
        None,
        AvailableMinimizers.LMFit,
        AvailableMinimizers.Bumps,
        AvailableMinimizers.DFO,
    ],
)
def test_fit_result(fit_engine):
    ref_sin = AbsSin(0.2, np.pi)
    sp_sin = AbsSin(0.354, 3.05)

    x = np.linspace(0, 5, 200)
    weights = np.ones_like(x)
    y = ref_sin(x)

    sp_sin.offset.fixed = False
    sp_sin.phase.fixed = False

    sp_ref1 = {
        f'p{item1.unique_name}': item1.value
        for item1, item2 in zip(sp_sin._kwargs.values(), ref_sin._kwargs.values())
    }
    sp_ref2 = {
        f'p{item1.unique_name}': item2.value
        for item1, item2 in zip(sp_sin._kwargs.values(), ref_sin._kwargs.values())
    }

    f = Fitter(sp_sin, sp_sin)

    if fit_engine is not None:
        try:
            f.switch_minimizer(fit_engine)
        except AttributeError:
            pytest.skip(reason=f'{fit_engine} is not installed')

    result = f.fit(x, y, weights=weights)
    check_fit_results(result, sp_sin, ref_sin, x, sp_ref1=sp_ref1, sp_ref2=sp_ref2)


@pytest.mark.fast
@pytest.mark.parametrize(
    'fit_engine',
    [
        None,
        AvailableMinimizers.LMFit,
        AvailableMinimizers.Bumps,
        AvailableMinimizers.DFO,
    ],
)
def test_basic_max_evaluations(fit_engine):
    ref_sin = AbsSin(0.2, np.pi)
    sp_sin = AbsSin(0.354, 3.05)

    x = np.linspace(0, 5, 200)
    weights = np.ones_like(x)
    y = ref_sin(x)

    sp_sin.offset.fixed = False
    sp_sin.phase.fixed = False

    f = Fitter(sp_sin, sp_sin)
    if fit_engine is not None:
        try:
            f.switch_minimizer(fit_engine)
        except AttributeError:
            pytest.skip(reason=f'{fit_engine} is not installed')
    f.max_evaluations = 3
    result = f.fit(x=x, y=y, weights=weights)
    # Result should not be the same as the reference
    assert sp_sin.phase.value != pytest.approx(ref_sin.phase.value, rel=1e-3)
    assert sp_sin.offset.value != pytest.approx(ref_sin.offset.value, rel=1e-3)


@pytest.mark.fast
@pytest.mark.parametrize(
    'fit_engine',
    [
        None,
        AvailableMinimizers.LMFit,
        AvailableMinimizers.Bumps,
        AvailableMinimizers.DFO,
    ],
)
def test_max_evaluations_populates_fit_result_fields(fit_engine):
    """With a tight budget every engine must return success=False, n_evaluations>0, non-empty message."""
    ref_sin = AbsSin(0.2, np.pi)
    sp_sin = AbsSin(0.354, 3.05)

    x = np.linspace(0, 5, 200)
    weights = np.ones_like(x)
    y = ref_sin(x)

    sp_sin.offset.fixed = False
    sp_sin.phase.fixed = False

    f = Fitter(sp_sin, sp_sin)
    if fit_engine is not None:
        try:
            f.switch_minimizer(fit_engine)
        except AttributeError:
            pytest.skip(reason=f'{fit_engine} is not installed')
    f.max_evaluations = 3
    result = f.fit(x=x, y=y, weights=weights)

    assert result.success is False
    assert result.n_evaluations is not None
    assert result.n_evaluations > 0
    assert isinstance(result.message, str)
    assert len(result.message) > 0


@pytest.mark.fast
def test_bumps_max_evaluations_counts_objective_calls() -> None:
    """Bumps reports actual objective calls, which can exceed the step budget."""
    ref_sin = AbsSin(0.2, np.pi)
    sp_sin = AbsSin(0.354, 3.05)

    x = np.linspace(0, 5, 200)
    weights = np.ones_like(x)
    y = ref_sin(x)

    sp_sin.offset.fixed = False
    sp_sin.phase.fixed = False

    f = Fitter(sp_sin, sp_sin)
    try:
        f.switch_minimizer(AvailableMinimizers.Bumps)
    except AttributeError:
        pytest.skip(reason=f'{AvailableMinimizers.Bumps} is not installed')

    f.max_evaluations = 3
    result = f.fit(x=x, y=y, weights=weights)

    assert result.success is False
    assert result.n_evaluations is not None
    assert result.n_evaluations > f.max_evaluations
    assert f'maximum optimizer steps ({f.max_evaluations})' in result.message
    assert f'objective evaluated {result.n_evaluations} times' in result.message


@pytest.mark.fast
@pytest.mark.parametrize(
    'fit_engine,tolerance',
    [
        (None, 10),
        (AvailableMinimizers.LMFit, 10),
        (AvailableMinimizers.Bumps, 10),
        (AvailableMinimizers.DFO, 0.1),
    ],
)
def test_basic_tolerance(fit_engine, tolerance):
    ref_sin = AbsSin(0.2, np.pi)
    sp_sin = AbsSin(0.354, 3.05)

    x = np.linspace(0, 5, 200)
    weights = np.ones_like(x)
    y = ref_sin(x)

    sp_sin.offset.fixed = False
    sp_sin.phase.fixed = False

    f = Fitter(sp_sin, sp_sin)
    if fit_engine is not None:
        try:
            f.switch_minimizer(fit_engine)
        except AttributeError:
            pytest.skip(reason=f'{fit_engine} is not installed')
    f.tolerance = tolerance
    result = f.fit(x=x, y=y, weights=weights)
    # Result should not be the same as the reference
    assert sp_sin.phase.value != pytest.approx(ref_sin.phase.value, rel=1e-3)
    assert sp_sin.offset.value != pytest.approx(ref_sin.offset.value, rel=1e-3)


@pytest.mark.fast
@pytest.mark.parametrize('fit_method', ['leastsq', 'powell', 'cobyla'])
def test_lmfit_methods(fit_method):
    ref_sin = AbsSin(0.2, np.pi)
    sp_sin = AbsSin(0.354, 3.05)

    x = np.linspace(0, 5, 200)
    weights = np.ones_like(x)
    y = ref_sin(x)

    sp_sin.offset.fixed = False
    sp_sin.phase.fixed = False

    f = Fitter(sp_sin, sp_sin)
    assert fit_method in f._minimizer.supported_methods()
    result = f.fit(x, y, weights=weights, method=fit_method)
    check_fit_results(result, sp_sin, ref_sin, x)


# @pytest.mark.xfail(reason="known bumps issue")
@pytest.mark.fast
@pytest.mark.parametrize('fit_method', ['newton', 'lm'])
def test_bumps_methods(fit_method):
    ref_sin = AbsSin(0.2, np.pi)
    sp_sin = AbsSin(0.354, 3.05)

    x = np.linspace(0, 5, 200)
    weights = np.ones_like(x)
    y = ref_sin(x)

    sp_sin.offset.fixed = False
    sp_sin.phase.fixed = False

    f = Fitter(sp_sin, sp_sin)
    f.switch_minimizer('Bumps')
    assert fit_method in f._minimizer.supported_methods()
    result = f.fit(x, y, weights=weights, method=fit_method)
    check_fit_results(result, sp_sin, ref_sin, x)


@pytest.mark.fast
def test_bumps_fit_emits_no_fitness_deprecation_warning():
    """Regression (CR-1): the classical BUMPS fit path must not read the
    deprecated ``FitProblem.fitness`` property, which emits a ``UserWarning``
    on bumps >= 1.0.4 — the ``Curve`` comes back from ``build_curve_problem``
    directly."""
    import warnings

    ref_sin = AbsSin(0.2, np.pi)
    sp_sin = AbsSin(0.354, 3.05)

    x = np.linspace(0, 5, 200)
    weights = np.ones_like(x)
    y = ref_sin(x)

    sp_sin.offset.fixed = False
    sp_sin.phase.fixed = False

    f = Fitter(sp_sin, sp_sin)
    f.switch_minimizer('Bumps')

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        f.fit(x, y, weights=weights)

    fitness_warnings = [w for w in caught if 'fitness' in str(w.message)]
    assert fitness_warnings == []


@pytest.mark.fast
@pytest.mark.parametrize(
    'fit_engine',
    [AvailableMinimizers.LMFit, AvailableMinimizers.Bumps, AvailableMinimizers.DFO],
)
def test_dependent_parameter(fit_engine):
    ref_sin = AbsSin(np.pi * 0.45, 0.45 * np.pi * 0.5)
    sp_sin = AbsSin(1, 0.5)

    x = np.linspace(0, 5, 200)
    weights = np.ones_like(x)
    y = ref_sin(x)

    f = Fitter(sp_sin, sp_sin)

    sp_sin.offset.make_dependent_on(
        dependency_expression='2*phase', dependency_map={'phase': sp_sin.phase}
    )

    if fit_engine is not None:
        try:
            f.switch_minimizer(fit_engine)
        except AttributeError:
            pytest.skip(reason=f'{fit_engine} is not installed')

    result = f.fit(x, y, weights=weights)
    check_fit_results(result, sp_sin, ref_sin, x)


@pytest.mark.fast
@pytest.mark.parametrize(
    'fit_engine',
    [
        None,
        AvailableMinimizers.LMFit,
        AvailableMinimizers.Bumps,
        AvailableMinimizers.DFO,
    ],
)
def test_2D_vectorized(fit_engine):
    x = np.linspace(0, 5, 200)
    mm = AbsSin2D(0.3, 1.6)
    m2 = AbsSin2D(0.1, 1.8)  # The fit is quite sensitive to the initial values :-(
    X, Y = np.meshgrid(x, x)
    XY = np.stack((X, Y), axis=2)
    weights = np.ones_like(mm(XY))
    ff = Fitter(m2, m2)
    if fit_engine is not None:
        try:
            ff.switch_minimizer(fit_engine)
        except AttributeError:
            pytest.skip(reason=f'{fit_engine} is not installed')
    try:
        result = ff.fit(x=XY, y=mm(XY), weights=weights, vectorized=True)
    except FitError as e:
        if 'Unable to allocate' in str(e):
            pytest.skip(reason='MemoryError - Matrix too large')
        else:
            raise e
    assert result.n_pars == len(m2.get_fit_parameters())
    assert result.reduced_chi2 == pytest.approx(0, abs=1.5e-3)
    assert result.success
    assert np.all(result.x == XY)
    y_calc_ref = m2(XY)
    assert result.y_calc == pytest.approx(y_calc_ref, abs=1e-2)
    assert result.residual == pytest.approx(mm(XY) - y_calc_ref, abs=1e-2)


@pytest.mark.fast
@pytest.mark.parametrize(
    'fit_engine',
    [
        None,
        AvailableMinimizers.LMFit,
        AvailableMinimizers.Bumps,
        AvailableMinimizers.DFO,
    ],
)
def test_2D_non_vectorized(fit_engine):
    x = np.linspace(0, 5, 200)
    mm = AbsSin2DL(0.3, 1.6)
    m2 = AbsSin2DL(0.1, 1.8)  # The fit is quite sensitive to the initial values :-(
    X, Y = np.meshgrid(x, x)
    XY = np.stack((X, Y), axis=2)
    weights = np.ones_like(mm(XY.reshape(-1, 2)))
    ff = Fitter(m2, m2)
    if fit_engine is not None:
        try:
            ff.switch_minimizer(fit_engine)
        except AttributeError:
            pytest.skip(reason=f'{fit_engine} is not installed')
    try:
        result = ff.fit(x=XY, y=mm(XY.reshape(-1, 2)), weights=weights, vectorized=False)
    except FitError as e:
        if 'Unable to allocate' in str(e):
            pytest.skip(reason='MemoryError - Matrix too large')
        else:
            raise e
    assert result.n_pars == len(m2.get_fit_parameters())
    assert result.reduced_chi2 == pytest.approx(0, abs=1.5e-3)
    assert result.success
    assert np.all(result.x == XY)
    y_calc_ref = m2(XY.reshape(-1, 2))
    assert result.y_calc == pytest.approx(y_calc_ref, abs=1e-2)
    assert result.residual == pytest.approx(mm(XY.reshape(-1, 2)) - y_calc_ref, abs=1e-2)


@pytest.mark.fast
@pytest.mark.parametrize(
    'fit_engine',
    [
        None,
        AvailableMinimizers.LMFit,
        AvailableMinimizers.Bumps,
        AvailableMinimizers.DFO,
    ],
)
def test_fixed_parameter_does_not_change(fit_engine):
    # WHEN
    ref_sin = AbsSin(0.2, np.pi)
    sp_sin = AbsSin(0.354, 3.05)

    x = np.linspace(0, 5, 200)
    weights = np.ones_like(x)
    y = ref_sin(x)

    # Fix the offset, only phase should be optimized
    sp_sin.offset.fixed = True
    sp_sin.phase.fixed = False

    fixed_offset_before = sp_sin.offset.value

    # THEN
    f = Fitter(sp_sin, sp_sin)
    if fit_engine is not None:
        try:
            f.switch_minimizer(fit_engine)
        except AttributeError:
            pytest.skip(reason=f'{fit_engine} is not installed')

    result = f.fit(x=x, y=y, weights=weights)

    # EXPECT
    # Offset should remain unchanged
    assert sp_sin.offset.value == pytest.approx(fixed_offset_before, abs=1e-12)
    # Phase should be optimized
    assert sp_sin.phase.value != pytest.approx(ref_sin.phase.value, rel=1e-3)


@pytest.mark.fast
def test_fitter_new_model_base_integration():
    # WHEN
    ground_truth = StraightLine(slope=2.0, intercept=1.0)
    model = StraightLine(slope=0.5, intercept=0.0)

    x = np.linspace(0, 10, 100)
    weights = np.ones_like(x)
    y = ground_truth(x)

    # THEN
    model.slope.fixed = False
    model.intercept.fixed = False
    fitter = Fitter(model, model)
    result = fitter.fit(x=x, y=y, weights=weights)

    # EXPECT
    assert model.slope.value == pytest.approx(ground_truth.slope.value, rel=1e-3)
    assert model.intercept.value == pytest.approx(ground_truth.intercept.value, rel=1e-3)


@pytest.mark.fast
@pytest.mark.parametrize(
    'fit_engine',
    [
        None,
        AvailableMinimizers.LMFit,
        AvailableMinimizers.Bumps,
        AvailableMinimizers.DFO,
    ],
)
def test_fitter_variable_weights(fit_engine):
    # WHEN
    ref_sin = AbsSin(0.2, np.pi)
    sp_sin = AbsSin(0.354, 3.05)

    x = np.linspace(0, 5, 200)
    y_true = ref_sin(x)

    # Introduce bias in second half of data
    y = y_true.copy()
    y[100:] += 0.5  # Artificial distortion

    # Case 1: High weight on distorted region
    weights_high = np.ones_like(x)
    weights_high[100:] = 10.0

    # Case 2: Low weight on distorted region
    weights_low = np.ones_like(x)
    weights_low[100:] = 0.1

    sp_sin.offset.fixed = False
    sp_sin.phase.fixed = False

    def run_fit(weights):
        model = AbsSin(0.354, 3.05)
        model.offset.fixed = False
        model.phase.fixed = False

        f = Fitter(model, model)
        if fit_engine is not None:
            try:
                f.switch_minimizer(fit_engine)
            except AttributeError:
                pytest.skip(reason=f'{fit_engine} is not installed')

        f.fit(x=x, y=y, weights=weights)
        return model.offset.value, model.phase.value

    offset_high, phase_high = run_fit(weights_high)
    offset_low, phase_low = run_fit(weights_low)

    # The fit should shift more toward the distorted region
    # when it has higher weight
    assert abs(offset_high - ref_sin.offset.value) > abs(offset_low - ref_sin.offset.value)


def _analytic_wls(design: np.ndarray, y: np.ndarray, point_weights: np.ndarray):
    """Solve min_beta sum_i (point_weights_i * (y_i - design_i . beta))^2.

    Returns the solution and the unscaled covariance (X^T W X)^-1 of the
    corresponding weighted least-squares problem.
    """
    a = design * point_weights[:, None]
    b = y * point_weights
    beta, *_ = np.linalg.lstsq(a, b, rcond=None)
    covariance = np.linalg.inv(a.T @ a)
    return beta, covariance


@pytest.mark.fast
@pytest.mark.parametrize(
    'fit_engine',
    [
        None,
        AvailableMinimizers.LMFit,
        AvailableMinimizers.Bumps,
        AvailableMinimizers.DFO,
    ],
)
def test_weight_convention_matches_analytic_wls(fit_engine):
    """Pin the weights = 1/sigma convention against the analytic WLS solution.

    A straight line is linear in (slope, intercept), so weighted least squares
    has the closed-form solution beta = (X^T W X)^-1 X^T W y with
    W = diag(1/sigma^2). On heteroscedastic data the candidate conventions
    (weights = 1/sigma, sigma, 1/sigma^2) give measurably different solutions,
    so this test fails if any minimizer's interpretation of ``weights`` drifts.
    """
    x = np.linspace(0.0, 10.0, 21)
    sigma = 0.1 + 0.18 * x  # heteroscedastic, 0.1 .. 1.9
    slope_true, intercept_true = 2.0, 1.0
    # Deterministic 'noise' so the data does not lie exactly on any line
    perturbation = np.cos(2.0 * x + 0.5)
    y = slope_true * x + intercept_true + perturbation * sigma

    design = np.column_stack([x, np.ones_like(x)])
    beta, covariance = _analytic_wls(design, y, 1.0 / sigma)
    beta_if_sigma, _ = _analytic_wls(design, y, sigma)
    beta_if_inverse_variance, _ = _analytic_wls(design, y, 1.0 / sigma**2)

    # Sanity check: the conventions are distinguishable well beyond the fit tolerance
    tolerance = 2e-3
    margin = min(
        np.max(np.abs(beta_if_sigma - beta)),
        np.max(np.abs(beta_if_inverse_variance - beta)),
    )
    assert margin > 20 * tolerance, 'test data cannot discriminate weight conventions'

    model = StraightLine(slope=1.5, intercept=0.5)
    model.slope.fixed = False
    model.intercept.fixed = False

    f = Fitter(model, model)
    if fit_engine is not None:
        try:
            f.switch_minimizer(fit_engine)
        except AttributeError:
            pytest.skip(reason=f'{fit_engine} is not installed')

    result = f.fit(x=x, y=y, weights=1.0 / sigma)

    assert result.success
    assert model.slope.value == pytest.approx(beta[0], abs=tolerance)
    assert model.intercept.value == pytest.approx(beta[1], abs=tolerance)

    if fit_engine in (None, AvailableMinimizers.LMFit):
        # lmfit scales the covariance by reduced chi-square (scale_covar=True)
        residual = y - design @ beta
        chi2 = float(np.sum((residual / sigma) ** 2))
        reduced_chi2 = chi2 / (x.size - 2)
        expected_errors = np.sqrt(np.diag(covariance) * reduced_chi2)
        assert model.slope.error == pytest.approx(expected_errors[0], rel=0.05)
        assert model.intercept.error == pytest.approx(expected_errors[1], rel=0.05)
