#  SPDX-FileCopyrightText: 2023 EasyScience contributors  <core@easyscience.software>
#  SPDX-License-Identifier: BSD-3-Clause
#  © 2021-2023 Contributors to the EasyScience project <https://github.com/easyScience/EasyScience

__author__ = "github.com/wardsimon"
__version__ = "0.0.1"

import pytest

import numpy as np
from easyscience.Constraints import ObjConstraint
from easyscience.fitting.multi_fitter import MultiFitter
from easyscience.fitting.minimizers import FitError
from easyscience.Objects.ObjectClasses import BaseObj
from easyscience.Objects.variable import Parameter


class Line(BaseObj):
    m: Parameter
    c: Parameter

    def __init__(self, m_val: float, c_val: float):
        m = Parameter("m", m_val)
        c = Parameter("c", c_val)
        super(Line, self).__init__("line", m=m, c=c)

    def __call__(self, x):
        return self.m.value * x + self.c.value
    

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


@pytest.mark.parametrize("with_errors", [False, True])
@pytest.mark.parametrize("fit_engine", [None, "LMFit", "Bumps", "DFO"])
def test_multi_fit(fit_engine, with_errors):
    ref_sin_1 = AbsSin(0.2, np.pi)
    sp_sin_1 = AbsSin(0.354, 3.05)
    ref_sin_2 = AbsSin(np.pi * 0.45, 0.45 * np.pi * 0.5)
    sp_sin_2 = AbsSin(1, 0.5)

    ref_sin_1.offset.user_constraints["ref_sin2"] = ObjConstraint(
        ref_sin_2.offset, "", ref_sin_1.offset
    )
    ref_sin_1.offset.user_constraints["ref_sin2"]()

    sp_sin_1.offset.user_constraints["sp_sin2"] = ObjConstraint(
        sp_sin_2.offset, "", sp_sin_1.offset
    )
    sp_sin_1.offset.user_constraints["sp_sin2"]()

    x1 = np.linspace(0, 5, 200)
    y1 = ref_sin_1(x1)
    x2 = np.copy(x1)
    y2 = ref_sin_2(x2)

    sp_sin_1.offset.fixed = False
    sp_sin_1.phase.fixed = False
    sp_sin_2.phase.fixed = False

    f = MultiFitter([sp_sin_1, sp_sin_2], [sp_sin_1, sp_sin_2])
    if fit_engine is not None:
        try:
            f.switch_minimizer(fit_engine)
        except AttributeError:
            pytest.skip(msg=f"{fit_engine} is not installed")

    args = [[x1, x2], [y1, y2]]
    kwargs = {}
    if with_errors:
        kwargs["weights"] = [1 / np.sqrt(y1), 1 / np.sqrt(y2)]

    results = f.fit(*args, **kwargs)
    X = [x1, x2]
    Y = [y1, y2]
    F_ref = [ref_sin_1, ref_sin_2]
    F_real = [sp_sin_1, sp_sin_2]
    for idx, result in enumerate(results):
        assert result.n_pars == len(sp_sin_1.get_fit_parameters()) + len(
            sp_sin_2.get_fit_parameters()
        )
        assert result.chi2 == pytest.approx(
            0, abs=1.5e-3 * (len(result.x) - result.n_pars)
        )
        assert result.reduced_chi == pytest.approx(0, abs=1.5e-3)
        assert result.success
        assert np.all(result.x == X[idx])
        assert np.all(result.y_obs == Y[idx])
        assert result.y_calc == pytest.approx(F_ref[idx](X[idx]), abs=1e-2)
        assert result.residual == pytest.approx(
            F_real[idx](X[idx]) - F_ref[idx](X[idx]), abs=1e-2
        )


@pytest.mark.parametrize("with_errors", [False, True])
@pytest.mark.parametrize("fit_engine", [None, "LMFit", "Bumps", "DFO"])
def test_multi_fit2(fit_engine, with_errors):
    ref_sin_1 = AbsSin(0.2, np.pi)
    sp_sin_1 = AbsSin(0.354, 3.05)
    ref_sin_2 = AbsSin(np.pi * 0.45, 0.45 * np.pi * 0.5)
    sp_sin_2 = AbsSin(1, 0.5)#    ref_sin_1_obj = genObjs[0]
    ref_line_obj = Line(1, 4.6)

    ref_sin_1.offset.user_constraints["ref_sin2"] = ObjConstraint(
        ref_sin_2.offset, "", ref_sin_1.offset
    )
    ref_sin_1.offset.user_constraints["ref_line"] = ObjConstraint(
        ref_line_obj.m, "", ref_sin_1.offset
    )
    ref_sin_1.offset.user_constraints["ref_sin2"]()
    ref_sin_1.offset.user_constraints["ref_line"]()

    sp_line = Line(0.43, 6.1)

    sp_sin_1.offset.user_constraints["sp_sin2"] = ObjConstraint(
        sp_sin_2.offset, "", sp_sin_1.offset
    )
    sp_sin_1.offset.user_constraints["sp_line"] = ObjConstraint(
        sp_line.m, "", sp_sin_1.offset
    )
    sp_sin_1.offset.user_constraints["sp_sin2"]()
    sp_sin_1.offset.user_constraints["sp_line"]()

    x1 = np.linspace(0, 5, 200)
    y1 = ref_sin_1(x1)
    x3 = np.copy(x1)
    y3 = ref_sin_2(x3)
    x2 = np.copy(x1)
    y2 = ref_line_obj(x2)

    sp_sin_1.offset.fixed = False
    sp_sin_1.phase.fixed = False
    sp_sin_2.phase.fixed = False
    sp_line.c.fixed = False

    f = MultiFitter([sp_sin_1, sp_line, sp_sin_2], [sp_sin_1, sp_line, sp_sin_2])
    if fit_engine is not None:
        try:
            f.switch_minimizer(fit_engine)
        except AttributeError:
            pytest.skip(msg=f"{fit_engine} is not installed")

    args = [[x1, x2, x3], [y1, y2, y3]]
    kwargs = {}
    if with_errors:
        kwargs["weights"] = [1 / np.sqrt(y1), 1 / np.sqrt(y2), 1 / np.sqrt(y3)]

    results = f.fit(*args, **kwargs)
    X = [x1, x2, x3]
    Y = [y1, y2, y3]
    F_ref = [ref_sin_1, ref_line_obj, ref_sin_2]
    F_real = [sp_sin_1, sp_line, sp_sin_2]

    assert len(results) == len(X)

    for idx, result in enumerate(results):
        assert result.n_pars == len(sp_sin_1.get_fit_parameters()) + len(
            sp_sin_2.get_fit_parameters()
        ) + len(sp_line.get_fit_parameters())
        assert result.chi2 == pytest.approx(
            0, abs=1.5e-3 * (len(result.x) - result.n_pars)
        )
        assert result.reduced_chi == pytest.approx(0, abs=1.5e-3)
        assert result.success
        assert np.all(result.x == X[idx])
        assert np.all(result.y_obs == Y[idx])
        assert result.y_calc == pytest.approx(F_real[idx](X[idx]), abs=1e-2)
        assert result.residual == pytest.approx(
            F_ref[idx](X[idx]) - F_real[idx](X[idx]), abs=1e-2
        )


@pytest.mark.parametrize("with_errors", [False, True])
@pytest.mark.parametrize("fit_engine", [None, "LMFit", "Bumps", "DFO"])
def test_multi_fit_1D_2D(fit_engine, with_errors):
    # Generate fit and reference objects
    ref_sin1D = AbsSin(0.2, np.pi)
    sp_sin1D = AbsSin(0.354, 3.05)

    ref_sin2D = AbsSin2D(0.3, 1.6)
    sp_sin2D = AbsSin2D(
        0.1, 1.75
    )  # The fit is VERY sensitive to the initial values :-(

    # Link the parameters
    ref_sin1D.offset.user_constraints["ref_sin2"] = ObjConstraint(
        ref_sin2D.offset, "", ref_sin1D.offset
    )
    ref_sin1D.offset.user_constraints["ref_sin2"]()

    sp_sin1D.offset.user_constraints["sp_sin2"] = ObjConstraint(
        sp_sin2D.offset, "", sp_sin1D.offset
    )
    sp_sin1D.offset.user_constraints["sp_sin2"]()

    # Generate data
    x1D = np.linspace(0.2, 3.8, 400)
    y1D = ref_sin1D(x1D)

    x = np.linspace(0, 5, 200)
    X, Y = np.meshgrid(x, x)
    x2D = np.stack((X, Y), axis=2)
    y2D = ref_sin2D(x2D)

    ff = MultiFitter([sp_sin1D, sp_sin2D], [sp_sin1D, sp_sin2D])
    if fit_engine is not None:
        try:
            ff.switch_minimizer(fit_engine)
        except AttributeError:
            pytest.skip(msg=f"{fit_engine} is not installed")

    sp_sin1D.offset.fixed = False
    sp_sin1D.phase.fixed = False
    sp_sin2D.phase.fixed = False

    f = MultiFitter([sp_sin1D, sp_sin2D], [sp_sin1D, sp_sin2D])
    if fit_engine is not None:
        try:
            f.switch_minimizer(fit_engine)
        except AttributeError:
            pytest.skip(msg=f"{fit_engine} is not installed")
    try:
        args = [[x1D, x2D], [y1D, y2D]]
        kwargs = {"vectorized": True}
        if with_errors:
            kwargs["weights"] = [1 / np.sqrt(y1D), 1 / np.sqrt(y2D)]
        results = f.fit(*args, **kwargs)
    except FitError as e:
        if "Unable to allocate" in str(e):
            pytest.skip(msg="MemoryError - Matrix too large")
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
        assert result.chi2 == pytest.approx(
            0, abs=1.5e-3 * (len(result.x) - result.n_pars)
        )
        assert result.reduced_chi == pytest.approx(0, abs=1.5e-3)
        assert result.success
        assert np.all(result.x == X[idx])
        assert np.all(result.y_obs == Y[idx])
        assert result.y_calc == pytest.approx(F_ref[idx](X[idx]), abs=1e-2)
        assert result.residual == pytest.approx(
            F_real[idx](X[idx]) - F_ref[idx](X[idx]), abs=1e-2
        )
