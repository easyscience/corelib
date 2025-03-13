#  SPDX-FileCopyrightText: 2023 EasyScience contributors  <core@easyscience.software>
#  SPDX-License-Identifier: BSD-3-Clause
#  © 2021-2023 Contributors to the EasyScience project <https://github.com/easyScience/EasyScience

__author__ = "github.com/wardsimon"
__version__ = "0.0.1"

import pytest

import numpy as np
from easyscience.Constraints import ObjConstraint
from easyscience.fitting.fitter import Fitter
from easyscience.fitting.minimizers import FitError
from easyscience.fitting.available_minimizers import AvailableMinimizers
from easyscience.Objects.ObjectClasses import BaseObj
from easyscience.Objects.variable import Parameter


class AbsSin(BaseObj):
    phase: Parameter
    offset: Parameter

    def __init__(self, offset_val: float, phase_val: float):
        offset = Parameter("offset", offset_val)
        phase = Parameter("phase", phase_val)
        super().__init__("sin", offset=offset, phase=phase)

    def __call__(self, x):
        return np.abs(np.sin(self.phase.value * x + self.offset.value))


class AbsSin2D(BaseObj):
    phase: Parameter
    offset: Parameter

    def __init__(self, offset_val: float, phase_val: float):
        offset = Parameter("offset", offset_val)
        phase = Parameter("phase", phase_val)
        super().__init__("sin2D", offset=offset, phase=phase)

    def __call__(self, x):
        X = x[:, :, 0]   # x is a 2D array
        Y = x[:, :, 1]
        return np.abs(
            np.sin(self.phase.value * X + self.offset.value)
        ) * np.abs(np.sin(self.phase.value * Y + self.offset.value))


class AbsSin2DL(AbsSin2D):
    def __call__(self, x):
        X = x[:, 0]   # x is a 1D array
        Y = x[:, 1]
        return np.abs(
            np.sin(self.phase.value * X + self.offset.value)
        ) * np.abs(np.sin(self.phase.value * Y + self.offset.value))


def check_fit_results(result, sp_sin, ref_sin, x, **kwargs):
    assert result.n_pars == len(sp_sin.get_fit_parameters())
    assert result.chi2 == pytest.approx(0, abs=1.5e-3 * (len(result.x) - result.n_pars))
    assert result.reduced_chi == pytest.approx(0, abs=1.5e-3)
    assert result.success
    if "sp_ref1" in kwargs.keys():
        sp_ref1 = kwargs["sp_ref1"]
        for key, value in sp_ref1.items():
            assert key in result.p.keys()
            assert key in result.p0.keys()
            assert result.p0[key] == pytest.approx(
                value
            )  # Bumps does something strange here
    assert np.all(result.x == x)
    for item1, item2 in zip(sp_sin._kwargs.values(), ref_sin._kwargs.values()):
        # assert item.error > 0 % This does not work as some methods don't calculate error
        assert item1.error == pytest.approx(0, abs=1e-1)
        assert item1.value == pytest.approx(item2.value, abs=5e-3)
    y_calc_ref = ref_sin(x)
    assert result.y_calc == pytest.approx(y_calc_ref, abs=1e-2)
    assert result.residual == pytest.approx(sp_sin(x) - y_calc_ref, abs=1e-2)


@pytest.mark.parametrize("with_errors", [False, True])
@pytest.mark.parametrize("fit_engine", [None, AvailableMinimizers.LMFit, AvailableMinimizers.Bumps, AvailableMinimizers.DFO])
def test_basic_fit(fit_engine: AvailableMinimizers, with_errors):
    ref_sin = AbsSin(0.2, np.pi)
    sp_sin = AbsSin(0.354, 3.05)

    x = np.linspace(0, 5, 200)
    y = ref_sin(x)

    sp_sin.offset.fixed = False
    sp_sin.phase.fixed = False

    f = Fitter(sp_sin, sp_sin)
    if fit_engine is not None:
        try:
            f.switch_minimizer(fit_engine)
        except AttributeError:
            pytest.skip(msg=f"{fit_engine} is not installed")
    args = [x, y]
    kwargs = {}
    if with_errors:
        kwargs["weights"] = 1 / np.sqrt(y)
    result = f.fit(*args, **kwargs)

    if fit_engine is not None:
        assert result.minimizer_engine.package == fit_engine.name.lower() # Special case where minimizer matches package
    assert sp_sin.phase.value == pytest.approx(ref_sin.phase.value, rel=1e-3)
    assert sp_sin.offset.value == pytest.approx(ref_sin.offset.value, rel=1e-3)


@pytest.mark.parametrize("fit_engine", [None, AvailableMinimizers.LMFit, AvailableMinimizers.Bumps, AvailableMinimizers.DFO])
def test_fit_result(fit_engine):
    ref_sin = AbsSin(0.2, np.pi)
    sp_sin = AbsSin(0.354, 3.05)

    x = np.linspace(0, 5, 200)
    y = ref_sin(x)

    sp_sin.offset.fixed = False
    sp_sin.phase.fixed = False

    sp_ref1 = {
        f"p{item1.unique_name}": item1.value
        for item1, item2 in zip(sp_sin._kwargs.values(), ref_sin._kwargs.values())
    }
    sp_ref2 = {
        f"p{item1.unique_name}": item2.value
        for item1, item2 in zip(sp_sin._kwargs.values(), ref_sin._kwargs.values())
    }

    f = Fitter(sp_sin, sp_sin)

    if fit_engine is not None:
        try:
            f.switch_minimizer(fit_engine)
        except AttributeError:
            pytest.skip(msg=f"{fit_engine} is not installed")

    result = f.fit(x, y)
    check_fit_results(result, sp_sin, ref_sin, x, sp_ref1=sp_ref1, sp_ref2=sp_ref2)


@pytest.mark.parametrize("fit_engine", [None, AvailableMinimizers.LMFit, AvailableMinimizers.Bumps, AvailableMinimizers.DFO])
def test_basic_max_evaluations(fit_engine):
    ref_sin = AbsSin(0.2, np.pi)
    sp_sin = AbsSin(0.354, 3.05)

    x = np.linspace(0, 5, 200)
    y = ref_sin(x)

    sp_sin.offset.fixed = False
    sp_sin.phase.fixed = False

    f = Fitter(sp_sin, sp_sin)
    if fit_engine is not None:
        try:
            f.switch_minimizer(fit_engine)
        except AttributeError:
            pytest.skip(msg=f"{fit_engine} is not installed")
    args = [x, y]
    kwargs = {}
    f.max_evaluations = 3
    try:
        result = f.fit(*args, **kwargs)
        # Result should not be the same as the reference
        assert sp_sin.phase.value != pytest.approx(ref_sin.phase.value, rel=1e-3)
        assert sp_sin.offset.value != pytest.approx(ref_sin.offset.value, rel=1e-3)
    except FitError as e:
        # DFO throws a different error
        assert "Objective has been called MAXFUN times" in str(e)


@pytest.mark.parametrize("fit_engine,tolerance", [(None, 10), (AvailableMinimizers.LMFit, 10), (AvailableMinimizers.Bumps, 10), (AvailableMinimizers.DFO, 0.1)])
def test_basic_tolerance(fit_engine, tolerance):
    ref_sin = AbsSin(0.2, np.pi)
    sp_sin = AbsSin(0.354, 3.05)

    x = np.linspace(0, 5, 200)
    y = ref_sin(x)

    sp_sin.offset.fixed = False
    sp_sin.phase.fixed = False

    f = Fitter(sp_sin, sp_sin)
    if fit_engine is not None:
        try:
            f.switch_minimizer(fit_engine)
        except AttributeError:
            pytest.skip(msg=f"{fit_engine} is not installed")
    args = [x, y]
    kwargs = {}
    f.tolerance = tolerance
    result = f.fit(*args, **kwargs)
    # Result should not be the same as the reference
    assert sp_sin.phase.value != pytest.approx(ref_sin.phase.value, rel=1e-3)
    assert sp_sin.offset.value != pytest.approx(ref_sin.offset.value, rel=1e-3)


@pytest.mark.parametrize("fit_method", ["leastsq", "powell", "cobyla"])
def test_lmfit_methods(fit_method):
    ref_sin = AbsSin(0.2, np.pi)
    sp_sin = AbsSin(0.354, 3.05)

    x = np.linspace(0, 5, 200)
    y = ref_sin(x)

    sp_sin.offset.fixed = False
    sp_sin.phase.fixed = False

    f = Fitter(sp_sin, sp_sin)
    assert fit_method in f._minimizer.supported_methods()
    result = f.fit(x, y, method=fit_method)
    check_fit_results(result, sp_sin, ref_sin, x)


#@pytest.mark.xfail(reason="known bumps issue")
@pytest.mark.parametrize("fit_method", ["newton", "lm"])
def test_bumps_methods(fit_method):
    ref_sin = AbsSin(0.2, np.pi)
    sp_sin = AbsSin(0.354, 3.05)

    x = np.linspace(0, 5, 200)
    y = ref_sin(x)

    sp_sin.offset.fixed = False
    sp_sin.phase.fixed = False

    f = Fitter(sp_sin, sp_sin)
    f.switch_minimizer("Bumps")
    assert fit_method in f._minimizer.supported_methods()
    result = f.fit(x, y, method=fit_method)
    check_fit_results(result, sp_sin, ref_sin, x)


@pytest.mark.parametrize("fit_engine", [AvailableMinimizers.LMFit, AvailableMinimizers.Bumps, AvailableMinimizers.DFO])
def test_fit_constraints(fit_engine):
    ref_sin = AbsSin(np.pi * 0.45, 0.45 * np.pi * 0.5)
    sp_sin = AbsSin(1, 0.5)

    x = np.linspace(0, 5, 200)
    y = ref_sin(x)

    sp_sin.phase.fixed = False

    f = Fitter(sp_sin, sp_sin)

    assert len(f.fit_constraints()) == 0
    c = ObjConstraint(sp_sin.offset, "2*", sp_sin.phase)
    f.add_fit_constraint(c)

    if fit_engine is not None:
        try:
            f.switch_minimizer(fit_engine)
        except AttributeError:
            pytest.skip(msg=f"{fit_engine} is not installed")

    result = f.fit(x, y)
    check_fit_results(result, sp_sin, ref_sin, x)
    assert len(f.fit_constraints()) == 1
    f.remove_fit_constraint(0)
    assert len(f.fit_constraints()) == 0


@pytest.mark.parametrize("with_errors", [False, True])
@pytest.mark.parametrize("fit_engine", [None, AvailableMinimizers.LMFit, AvailableMinimizers.Bumps, AvailableMinimizers.DFO])
def test_2D_vectorized(fit_engine, with_errors):
    x = np.linspace(0, 5, 200)
    mm = AbsSin2D(0.3, 1.6)
    m2 = AbsSin2D(
        0.1, 1.8
    )  # The fit is quite sensitive to the initial values :-(
    X, Y = np.meshgrid(x, x)
    XY = np.stack((X, Y), axis=2)
    ff = Fitter(m2, m2)
    if fit_engine is not None:
        try:
            ff.switch_minimizer(fit_engine)
        except AttributeError:
            pytest.skip(msg=f"{fit_engine} is not installed")
    try:
        args = [XY, mm(XY)]
        kwargs = {"vectorized": True}
        if with_errors:
            kwargs["weights"] = 1 / np.sqrt(args[1])
        result = ff.fit(*args, **kwargs)
    except FitError as e:
        if "Unable to allocate" in str(e):
            pytest.skip(msg="MemoryError - Matrix too large")
        else:
            raise e
    assert result.n_pars == len(m2.get_fit_parameters())
    assert result.reduced_chi == pytest.approx(0, abs=1.5e-3)
    assert result.success
    assert np.all(result.x == XY)
    y_calc_ref = m2(XY)
    assert result.y_calc == pytest.approx(y_calc_ref, abs=1e-2)
    assert result.residual == pytest.approx(mm(XY) - y_calc_ref, abs=1e-2)


@pytest.mark.parametrize("with_errors", [False, True])
@pytest.mark.parametrize("fit_engine", [None, AvailableMinimizers.LMFit, AvailableMinimizers.Bumps, AvailableMinimizers.DFO])
def test_2D_non_vectorized(fit_engine, with_errors):
    x = np.linspace(0, 5, 200)
    mm = AbsSin2DL(0.3, 1.6)
    m2 = AbsSin2DL(
        0.1, 1.8
    )  # The fit is quite sensitive to the initial values :-(
    X, Y = np.meshgrid(x, x)
    XY = np.stack((X, Y), axis=2)
    ff = Fitter(m2, m2)
    if fit_engine is not None:
        try:
            ff.switch_minimizer(fit_engine)
        except AttributeError:
            pytest.skip(msg=f"{fit_engine} is not installed")
    try:
        args = [XY, mm(XY.reshape(-1, 2))]
        kwargs = {"vectorized": False}
        if with_errors:
            kwargs["weights"] = 1 / np.sqrt(args[1])
        result = ff.fit(*args, **kwargs)
    except FitError as e:
        if "Unable to allocate" in str(e):
            pytest.skip(msg="MemoryError - Matrix too large")
        else:
            raise e
    assert result.n_pars == len(m2.get_fit_parameters())
    assert result.reduced_chi == pytest.approx(0, abs=1.5e-3)
    assert result.success
    assert np.all(result.x == XY)
    y_calc_ref = m2(XY.reshape(-1, 2))
    assert result.y_calc == pytest.approx(y_calc_ref, abs=1e-2)
    assert result.residual == pytest.approx(
        mm(XY.reshape(-1, 2)) - y_calc_ref, abs=1e-2
    )
