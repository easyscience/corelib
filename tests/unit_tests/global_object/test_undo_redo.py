#  SPDX-FileCopyrightText: 2023 EasyScience contributors  <core@easyscience.software>
#  SPDX-License-Identifier: BSD-3-Clause
#  © 2021-2023 Contributors to the EasyScience project <https://github.com/easyScience/EasyScience

__author__ = "github.com/wardsimon"
__version__ = "0.0.1"

import math

import numpy as np
import pytest

from easyscience.Objects.Groups import BaseCollection
from easyscience.Objects.ObjectClasses import BaseObj
from easyscience.Objects.variable.parameter import Parameter
from easyscience.Objects.variable.descriptor_str import DescriptorStr
from easyscience.Objects.variable.descriptor_number import DescriptorNumber
from easyscience.Objects.variable.descriptor_bool import DescriptorBool

from easyscience.fitting import Fitter


def createSingleObjs(idx):
    alphabet = "abcdefghijklmnopqrstuvwxyz"
    reps = math.floor(idx / len(alphabet)) + 1
    name = alphabet[idx % len(alphabet)] * reps
    if idx % 2:
        return Parameter(name, idx,unit="m/s")
    else:
        return DescriptorNumber(name, idx,unit="m/s")


def createParam(option):
    return pytest.param(option, id=option[0])


def doUndoRedo(obj, attr, future, additional=""):
    from easyscience import global_object

    global_object.stack.enabled = True
    e = False

    def getter(_obj, _attr):
        value = getattr(_obj, _attr)
        if additional:
            value = getattr(value, additional)
        return value

    try:
        previous = getter(obj, attr)
        if attr == "unit" and hasattr(obj, "convert_unit"):
            obj.convert_unit(future)
        else:
            setattr(obj, attr, future)
        assert getter(obj, attr) == future
        assert global_object.stack.canUndo()
        global_object.stack.undo()
        assert getter(obj, attr) == previous
        assert global_object.stack.canRedo()
        global_object.stack.redo()
        assert getter(obj, attr) == future
    except Exception as err:
        e = err
    finally:
        global_object.stack.enabled = False
    return e


# @pytest.mark.parametrize(
#     "test",
#     [
#         createParam(option)
#         for option in [
#             ("value", 500),
#             ("error", 5),
#             ("enabled", False),
#             ("unit", "m/s"),
#             ("display_name", "boom"),
#             ("fixed", False),
#             ("max", 505),
#             ("min", -1),
#         ]
#     ],
# )
# @pytest.mark.parametrize(
#     "idx", [pytest.param(0, id="DescriptorNumber"), pytest.param(1, id="Parameter")]
# )

# def test_SinglesUndoRedo(idx, test):
#     obj = createSingleObjs(idx)
#     attr = test[0]
#     value = test[1]

#     if not hasattr(obj, attr):
#         pytest.skip(f"Not applicable: {obj} does not have field {attr}")
#     e = doUndoRedo(obj, attr, value)
#     if e:
#         raise e
#             ("enabled", False),
#             ("fixed", False),
#             ("max", 505),
#             ("min", -1),

@pytest.mark.parametrize(
    "test",
    [
        createParam(option)
        for option in [
            ("value", 500),
            ("error", 5),
            ("unit", "km/s"),
            ("display_name", "boom"),
        ]
    ],
)

def test_DescriptorNumberUndoRedo(test):
    obj = DescriptorNumber('DescriptorNumber',1,unit='m/s')
    attr = test[0]
    value = test[1]

    e = doUndoRedo(obj, attr, value)
    if e:
        raise e


@pytest.mark.parametrize(
    "test",
    [
        createParam(option)
        for option in [
            ("value", 500),
            ("error", 5),
            ("unit", "km/s"),
            ("display_name", "boom"),
            ("enabled", False),
            ("fixed", False),
            ("max", 505),
            ("min", -1),
        ]
    ],
)

def test_ParameterUndoRedo(test):
    obj = Parameter('Parameter',1,unit='m/s')
    attr = test[0]
    value = test[1]

    e = doUndoRedo(obj, attr, value)
    if e:
        raise e

@pytest.mark.parametrize("value", (True, False))
def test_Parameter_Bounds_UndoRedo(value):
    from easyscience import global_object

    global_object.stack.enabled = True
    p = Parameter("test", 1, enabled=value)
    assert p.min == -np.inf
    assert p.max == np.inf
    assert p.bounds == (-np.inf, np.inf)

    p.bounds = (0, 2)
    assert p.min == 0
    assert p.max == 2
    assert p.bounds == (0, 2)
    assert p.enabled is True

    global_object.stack.undo()
    assert p.min == -np.inf
    assert p.max == np.inf
    assert p.bounds == (-np.inf, np.inf)
    assert p.enabled is value


def test_BaseObjUndoRedo():
    objs = {obj.name: obj for obj in [createSingleObjs(idx) for idx in range(5)]}
    name = "test"
    obj = BaseObj(name, **objs)
    name2 = "best"

    # Test name
    # assert not doUndoRedo(obj, 'name', name2)

    # Test setting value
    for b_obj in objs.values():
        e = doUndoRedo(obj, b_obj.name, b_obj.value + 1, "value")
        if e:
            raise e


def test_BaseCollectionUndoRedo():
    objs = [createSingleObjs(idx) for idx in range(5)]
    name = "test"
    obj = BaseCollection(name, *objs)
    name2 = "best"

    # assert not doUndoRedo(obj, 'name', name2)

    from easyscience import global_object

    global_object.stack.enabled = True

    original_length = len(obj)
    p = Parameter("slip_in", 50)
    idx = 2
    obj.insert(idx, p)
    assert len(obj) == original_length + 1
    objs.insert(idx, p)
    for item, obj_r in zip(obj, objs):
        assert item == obj_r

    # Test inserting items
    global_object.stack.undo()
    assert len(obj) == original_length
    _ = objs.pop(idx)
    for item, obj_r in zip(obj, objs):
        assert item == obj_r
    global_object.stack.redo()
    assert len(obj) == original_length + 1
    objs.insert(idx, p)
    for item, obj_r in zip(obj, objs):
        assert item == obj_r

    # Test Del Items
    del obj[idx]
    del objs[idx]
    assert len(obj) == original_length
    for item, obj_r in zip(obj, objs):
        assert item == obj_r
    global_object.stack.undo()
    assert len(obj) == original_length + 1
    objs.insert(idx, p)
    for item, obj_r in zip(obj, objs):
        assert item == obj_r
    del objs[idx]
    global_object.stack.redo()
    assert len(obj) == original_length
    for item, obj_r in zip(obj, objs):
        assert item == obj_r

    # Test Place Item
    old_item = objs[idx]
    objs[idx] = p
    obj[idx] = p
    assert len(obj) == original_length
    for item, obj_r in zip(obj, objs):
        assert item == obj_r
    global_object.stack.undo()
    for i in range(len(obj)):
        if i == idx:
            item = old_item
        else:
            item = objs[i]
        assert obj[i] == item
    global_object.stack.redo()
    for item, obj_r in zip(obj, objs):
        assert item == obj_r

    global_object.stack.enabled = False


def test_UndoRedoMacros():
    items = [createSingleObjs(idx) for idx in range(5)]
    offset = 5
    undo_text = "test_macro"
    from easyscience import global_object

    global_object.stack.enabled = True
    global_object.stack.beginMacro(undo_text)
    values = [item.value for item in items]

    for item, value in zip(items, values):
        item.value = value + offset
    global_object.stack.endMacro()

    for item, old_value in zip(items, values):
        assert item.value == old_value + offset
    assert global_object.stack.undoText() == undo_text

    global_object.stack.undo()

    for item, old_value in zip(items, values):
        assert item.value == old_value
    assert global_object.stack.redoText() == undo_text

    global_object.stack.redo()
    for item, old_value in zip(items, values):
        assert item.value == old_value + offset


@pytest.mark.parametrize("fit_engine", ["LMFit", "Bumps", "DFO"])
def test_fittingUndoRedo(fit_engine):
    m_value = 6
    c_value = 2
    x = np.linspace(-5, 5, 100)
    dy = np.random.rand(*x.shape)

    class Line(BaseObj):
        def __init__(self, m: Parameter, c: Parameter):
            super(Line, self).__init__("basic_line", m=m, c=c)

        @classmethod
        def default(cls):
            m = Parameter("m", m_value)
            c = Parameter("c", c_value)
            return cls(m=m, c=c)

        @classmethod
        def from_pars(cls, m_value: float, c_value: float):
            m = Parameter("m", m_value)
            c = Parameter("c", c_value)
            return cls(m=m, c=c)

        def __call__(self, x: np.ndarray) -> np.ndarray:
            return self.m.value * x + self.c.value

    l1 = Line.default()
    m_sp = 4
    c_sp = -3

    l2 = Line.from_pars(m_sp, c_sp)
    l2.m.fixed = False
    l2.c.fixed = False

    y = l1(x) + 0.125 * (dy - 0.5)

    f = Fitter(l2, l2)
    try:
        f.switch_minimizer(fit_engine)
    except AttributeError:
        pytest.skip(msg=f"{fit_engine} is not installed")

    from easyscience import global_object

    global_object.stack.enabled = True
    res = f.fit(x, y)

    # assert l1.c.value == pytest.approx(l2.c.value, rel=l2.c.error * 3)
    # assert l1.m.value == pytest.approx(l2.m.value, rel=l2.m.error * 3)
    assert global_object.stack.undoText() == "Fitting routine"

    global_object.stack.undo()
    assert l2.m.value == m_sp
    assert l2.c.value == c_sp
    assert global_object.stack.redoText() == "Fitting routine"

    global_object.stack.redo()
    assert l2.m.value == res.p[f"p{l2.m.unique_name}"]
    assert l2.c.value == res.p[f"p{l2.c.unique_name}"]

# TODO: Check if this test is needed
# @pytest.mark.parametrize('math_funcs', [pytest.param([Parameter.__add__, float.__add__], id='Addition'),
#                                         pytest.param([Parameter.__sub__, float.__sub__], id='Subtraction')])
# def test_parameter_maths_basic(math_funcs):
#     a = 1.0
#     b = 2.0
#     sa = 0.1
#     sb = 0.2

#     p_fun = math_funcs[0]
#     f_fun = math_funcs[1]

#     result_value = f_fun(a, b)
#     result_error = (sa ** 2 + sb ** 2) ** 0.5

#     from easyscience import global_object
#     global_object.stack.enabled = True

#     # Perform basic test
#     p1 = Parameter('a', a)
#     p2 = Parameter('b', b)

#     p1 = p_fun(p1, p2)
    
#     assert p1.value == result_value
#     global_object.stack.undo()
#     assert p1.value == a
#     global_object.stack.redo()
#     assert p1.value == result_value

    # # Perform basic + error
    # p1 = Parameter('a', a, error=sa)
    # p2 = Parameter('b', b, error=sb)
    # p1 = p_fun(p1, p2)
    # assert p1.value == result_value
    # assert p1.error == result_error
    # global_object.stack.undo()
    # assert p1.value == a
    # assert p1.error == sa
    # global_object.stack.redo()
    # assert p1.value == result_value
    # assert p1.error == result_error

    # # Perform basic + units
    # p1 = Parameter('a', a, error=sa, units='m/s')
    # p2 = Parameter('b', b, error=sb, units='m/s')
    # p1 = p_fun(p1, p2)
    # assert p1.value == result_value
    # assert p1.error == result_error
    # assert str(p1.unit) == 'meter / second'
    # global_object.stack.undo()
    # assert p1.value == a
    # assert p1.error == sa
    # assert str(p1.unit) == 'meter / second'
    # global_object.stack.redo()
    # assert p1.value == result_value
    # assert p1.error == result_error
    # assert str(p1.unit) == 'meter / second'


# @pytest.mark.parametrize('math_funcs', [pytest.param([Parameter.__imul__, float.__mul__,
#                                                       'meter ** 2 / second ** 2'], id='Multiplication'),
#                                         pytest.param([Parameter.__itruediv__, float.__truediv__,
#                                                       'dimensionless'], id='Division')])
# def test_parameter_maths_advanced(math_funcs):
#     a = 4.0
#     b = 2.0
#     sa = 0.1
#     sb = 0.2
#     unit = 'meter / second'

#     p_fun = math_funcs[0]
#     f_fun = math_funcs[1]
#     u_str = math_funcs[2]

#     result_value = f_fun(a, b)
#     result_error = ((sa / a) ** 2 + (sb / b) ** 2) ** 0.5 * result_value

#     from easyscience import global_object
#     global_object.stack.enabled = True

#     # Perform basic test
#     p1 = Parameter('a', a)
#     p2 = Parameter('b', b)

#     p1 = p_fun(p1, p2)
#     assert p1.value == result_value
#     global_object.stack.undo()
#     assert p1.value == a
#     global_object.stack.redo()
#     assert p1.value == result_value

#     # Perform basic + error
#     p1 = Parameter('a', a, error=sa)
#     p2 = Parameter('b', b, error=sb)
#     p1 = p_fun(p1, p2)
#     assert p1.value == result_value
#     assert p1.error == result_error
#     global_object.stack.undo()
#     assert p1.value == a
#     assert p1.error == sa
#     global_object.stack.redo()
#     assert p1.value == result_value
#     assert p1.error == result_error

#     # Perform basic + units
#     p1 = Parameter('a', a, error=sa, units=unit)
#     p2 = Parameter('b', b, error=sb, units=unit)
#     p1 = p_fun(p1, p2)
#     assert p1.value == result_value
#     assert p1.error == result_error
#     assert str(p1.unit) == u_str
#     global_object.stack.undo()
#     assert p1.value == a
#     assert p1.error == sa
#     assert str(p1.unit) == unit
#     global_object.stack.redo()
#     assert p1.value == result_value
#     assert p1.error == result_error
#     assert str(p1.unit) == u_str
