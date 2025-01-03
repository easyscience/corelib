__author__ = "github.com/wardsimon"
__version__ = "0.1.0"

#  SPDX-FileCopyrightText: 2023 EasyScience contributors  <core@easyscience.software>
#  SPDX-License-Identifier: BSD-3-Clause
#  © 2021-2023 Contributors to the EasyScience project <https://github.com/easyScience/EasyScience

from contextlib import contextmanager
from copy import copy

from typing import ClassVar
from typing import List
from typing import Optional
from typing import Type
from typing import Union

import numpy as np
import pytest

import easyscience
from easyscience.Objects.ObjectClasses import BaseObj
from easyscience.Objects.variable import DescriptorNumber
from easyscience.Objects.variable import Parameter
from easyscience.Utils.io.dict import DictSerializer
from easyscience import global_object

@pytest.fixture
def clear():
    global_object.map._clear()

@pytest.fixture
def setup_pars():
    d = {
        "name": "test",
        "par1": Parameter("p1", 0.1, fixed=True),
        "des1": DescriptorNumber("d1", 0.1),
        "par2": Parameter("p2", 0.1),
        "des2": DescriptorNumber("d2", 0.1),
        "par3": Parameter("p3", 0.1),
    }
    return d


@contextmanager
def not_raises(
    expected_exception: Union[Type[BaseException], List[Type[BaseException]]]
):
    try:
        yield
    except expected_exception:
        raise pytest.fail(
            "Did raise exception {0} when it should not.".format(
                repr(expected_exception)
            )
        )
    except Exception as err:
        raise pytest.fail("An unexpected exception {0} raised.".format(repr(err)))


@pytest.mark.parametrize(
    "a, kw",
    [
        ([], ["par1"]),
        (["par1"], []),
        (["par1"], ["par2"]),
        (["par1", "des1"], ["par2", "des2"]), 
    ],
)
def test_baseobj_create(setup_pars: dict, a: List[str], kw: List[str]):
    name = setup_pars["name"]
    args = []
    for key in a:
        args.append(setup_pars[key])
    kwargs = {}
    for key in kw:
        kwargs[key] = setup_pars[key]
    base = BaseObj(name, None, *args, **kwargs)
    assert base.name == name
    for key in a:
        item = getattr(base, setup_pars[key].name)
        assert isinstance(item, setup_pars[key].__class__)


def test_baseobj_copy(setup_pars: dict):
    # When
    name = setup_pars["name"]
    args = []
    for key in ["par1", "des1"]:
        args.append(setup_pars[key])
    kwargs = {}
    for key in ["par2", "des2"]:
        kwargs[key] = setup_pars[key]
    base = BaseObj(name, None, *args, **kwargs)

    # Then
    base_copy = copy(base)

    # Expect
    assert base_copy.name == name
    assert base_copy.unique_name != base.unique_name

    for key in ["par1", "des1"]:
        item = getattr(base, setup_pars[key].name)
        assert isinstance(item, setup_pars[key].__class__)


def test_baseobj_get(setup_pars: dict):
    name = setup_pars["name"]
    explicit_name1 = "par1"
    explicit_name2 = "par2"
    kwargs = {
        setup_pars[explicit_name1].name: setup_pars[explicit_name1],
        setup_pars[explicit_name2].name: setup_pars[explicit_name2],
    }
    obj = BaseObj(name, **kwargs)
    with not_raises(AttributeError):
        p1: Parameter = obj.p1
    with not_raises(AttributeError):
        p2: Parameter = obj.p2


def test_baseobj_set(setup_pars: dict):
    name = setup_pars["name"]
    explicit_name1 = "par1"
    kwargs = {
        setup_pars[explicit_name1].name: setup_pars[explicit_name1],
    }
    obj = BaseObj(name, **kwargs)
    new_value = 5.0
    with not_raises([AttributeError, ValueError]):
        obj.p1 = new_value
        assert obj.p1.value == new_value


def test_baseobj_get_parameters(setup_pars: dict):
    name = setup_pars["name"]
    del setup_pars["name"]
    obj = BaseObj(name, **setup_pars)
    pars = obj.get_fit_parameters()
    assert isinstance(pars, list)
    assert len(pars) == 2
    par_names = [par.name for par in pars]
    assert "p2" in par_names
    assert "p3" in par_names


def test_baseobj_fit_objects(setup_pars: dict):
    pass


def test_baseobj_as_dict(setup_pars: dict):
    name = setup_pars["name"]
    del setup_pars["name"]
    obj = BaseObj(name, **setup_pars)
    obtained = obj.as_dict()
    assert isinstance(obtained, dict)
    expected = {
        "@module": "easyscience.Objects.ObjectClasses",
        "@class": "BaseObj",
        "@version": easyscience.__version__,
        "name": "test",
        "par1": {
            "@module": Parameter.__module__,
            "@class": Parameter.__name__,
            "@version": easyscience.__version__,
            "name": "p1",
            "value": 0.1,
            "variance": 0.0,
            "min": -np.inf,
            "max": np.inf,
            "fixed": True,
            "unit": "dimensionless",
        },
        "des1": {
            "@module": DescriptorNumber.__module__,
            "@class": DescriptorNumber.__name__,
            "@version": easyscience.__version__,
            "name": "d1",
            "value": 0.1,
            "unit": "dimensionless",
            "description": "",
            "url": "",
            "display_name": "d1",
        },
        "par2": {
            "@module": Parameter.__module__,
            "@class": Parameter.__name__,
            "@version": easyscience.__version__,
            "name": "p2",
            "value": 0.1,
            "variance": 0.0,
            "min": -np.inf,
            "max": np.inf,
            "fixed": False,
            "unit": "dimensionless",
        },
        "des2": {
            "@module": DescriptorNumber.__module__,
            "@class": DescriptorNumber.__name__,
            "@version": easyscience.__version__,
            "name": "d2",
            "value": 0.1,
            "unit": "dimensionless",
            "description": "",
            "url": "",
            "display_name": "d2",
        },
        "par3": {
            "@module": Parameter.__module__,
            "@class": Parameter.__name__,
            "@version": easyscience.__version__,
            "name": "p3",
            "value": 0.1,
            "variance": 0.0,
            "min": -np.inf,
            "max": np.inf,
            "fixed": False,
            "unit": "dimensionless",
        },
    }

    def check_dict(check, item):
        if isinstance(check, dict) and isinstance(item, dict):
            if "@module" in item.keys():
                with not_raises([ValueError, AttributeError]):
                    global_object.map._clear()
                    this_obj = DictSerializer().decode(item)

            for key in check.keys():
                assert key in item.keys()
                check_dict(check[key], item[key])
        else:
            assert isinstance(item, type(check))
            assert item == check

    check_dict(expected, obtained)


def test_baseobj_dict_roundtrip(clear, setup_pars: dict):
    # When
    name = setup_pars["name"]
    del setup_pars["name"]
    obj = BaseObj(name, **setup_pars, unique_name='special_name')
    obj_dict = obj.as_dict()

    global_object.map._clear()

    # Then
    new_obj = BaseObj.from_dict(obj_dict)

    # Expect
    new_obj_dict = new_obj.as_dict()
    assert obj_dict == new_obj_dict


def test_baseobj_dir(setup_pars):
    name = setup_pars["name"]
    del setup_pars["name"]
    obj = BaseObj(name, **setup_pars)
    expected = [
        "encode",
        "decode",
        "as_dict",
        "constraints",
        "des1",
        "des2",
        "from_dict",
        "generate_bindings",
        "get_fit_parameters",
        "get_parameters",
        "interface",
        "unique_name",
        "name",
        "par1",
        "par2",
        "par3",
        "switch_interface",
        "as_data_dict",
        "as_dict",
        "unsafe_hash",
        "user_data",
    ]
    obtained = dir(obj)
    assert len(obtained) == len(expected)
    assert obtained == sorted(obtained)
    assert len(set(expected).difference(set(obtained))) == 0


def test_baseobj_get_parameters(setup_pars):
    name = setup_pars["name"]
    del setup_pars["name"]
    obj = BaseObj(name, **setup_pars)
    pars = obj.get_parameters()
    assert len(pars) == 3


def test_baseobj_get_parameters_nested(setup_pars):
    name = setup_pars["name"]
    del setup_pars["name"]
    obj = BaseObj(name, **setup_pars)

    name2 = name + "_2"
    obj2 = BaseObj(name2, obj=obj, **setup_pars)

    pars = obj2.get_parameters()
    assert len(pars) == 6

    pars = obj.get_parameters()
    assert len(pars) == 3


def test_baseobj_get_fit_parameters(setup_pars):
    name = setup_pars["name"]
    del setup_pars["name"]
    obj = BaseObj(name, **setup_pars)
    pars = obj.get_fit_parameters()
    assert len(pars) == 2


def test_baseobj_get_fit_parameters_nested(setup_pars):
    name = setup_pars["name"]
    del setup_pars["name"]
    obj = BaseObj(name, **setup_pars)

    name2 = name + "_2"
    obj2 = BaseObj(name2, obj=obj, **setup_pars)

    pars = obj2.get_fit_parameters()
    assert len(pars) == 4

    pars = obj.get_fit_parameters()
    assert len(pars) == 2


def test_baseobj__add_component(setup_pars):
    name = setup_pars["name"]
    del setup_pars["name"]
    obj = BaseObj(name, **setup_pars)

    p = Parameter("added_par", 1)
    new_item_name = "Added"
    obj._add_component(new_item_name, p)

    assert hasattr(obj, new_item_name)
    a = getattr(obj, new_item_name)
    assert isinstance(a, Parameter)


def test_baseObj_name(setup_pars):
    name = setup_pars["name"]
    del setup_pars["name"]
    obj = BaseObj(name, **setup_pars)
    assert obj.name == name


def test_subclassing():
    from typing import ClassVar

    from easyscience.models.polynomial import Line
    from easyscience.Objects.variable import Parameter

    class L2(Line):
        diff: ClassVar[Parameter]

        def __init__(self, m: Parameter, c: Parameter, diff: Parameter):
            super(L2, self).__init__(m=m, c=c)
            self.diff = diff
            self.foo = "bar"

        @classmethod
        def from_pars(cls, m, c, diff):
            m = Parameter("m", m)
            c = Parameter("c", c)
            diff = Parameter("diff", diff)
            return cls(m, c, diff)

        def __call__(self, *args, **kwargs):
            return super(L2, self).__call__(*args, **kwargs) + self.diff.value

    l2 = L2.from_pars(1, 2, 3)

    assert l2.m.value == 1
    assert l2.c.value == 2
    assert l2.diff.value == 3

    l2.diff = 4
    assert isinstance(l2.diff, Parameter)
    assert l2.diff.value == 4

    l2.foo = "foo"
    assert l2.foo == "foo"

    x = np.linspace(0, 10, 100)
    y = l2.m.value * x + l2.c.value + l2.diff.value

    assert np.allclose(l2(x), y)


def test_Base_GETSET():
    class A(BaseObj):
        def __init__(self, a: Parameter):
            super(A, self).__init__("a", a=a)

        @classmethod
        def from_pars(cls, a: float):
            return cls(a=Parameter("a", a))

    a_start = 5
    a_end = 10
    a = A.from_pars(a_start)
    graph = a._global_object.map

    assert a.a.raw_value == a_start
    assert len(graph.get_edges(a)) == 1

    setattr(a, "a", a_end)
    assert a.a.raw_value == a_end
    assert len(graph.get_edges(a)) == 1


def test_Base_GETSET():
    class A(BaseObj):
        def __init__(self, a: Parameter):
            super(A, self).__init__("a", a=a)
            b = 0

        @classmethod
        def from_pars(cls, a: float):
            return cls(a=Parameter("a", a))

    a = A.from_pars(5)
    b_new = 10
    a.b = b_new
    assert a.b == b_new


def test_Base_GETSET_v2():
    class A(BaseObj):

        a: ClassVar[Parameter]

        def __init__(self, a: Parameter):
            super(A, self).__init__("a", a=a)

        @classmethod
        def from_pars(cls, a: float):
            return cls(a=Parameter("a", a))

    a_start = 5
    a_end = 10
    a = A.from_pars(a_start)
    graph = a._global_object.map

    assert a.a.value == a_start
    assert len(graph.get_edges(a)) == 1

    setattr(a, "a", a_end)
    assert a.a.value == a_end
    assert len(graph.get_edges(a)) == 1


def test_Base_GETSET_v3():
    class A(BaseObj):

        a: ClassVar[Parameter]

        def __init__(self, a: Parameter):
            super(A, self).__init__("a", a=a)

        @classmethod
        def from_pars(cls, a: float):
            return cls(a=Parameter("a", a))

    a_start = 5
    a_end = 10
    a = A.from_pars(a_start)
    graph = a._global_object.map

    assert a.a.value == a_start
    assert len(graph.get_edges(a)) == 1
    a_ = Parameter("a", a_end)
    assert a.a.unique_name in graph.get_edges(a)
    a__ = a.a

    setattr(a, "a", a_)
    assert a.a.value == a_end
    assert len(graph.get_edges(a)) == 1
    assert a_.unique_name in graph.get_edges(a)
    assert a__.unique_name not in graph.get_edges(a)


def test_BaseCreation():
    class A(BaseObj):
        def __init__(self, a: Optional[Union[Parameter, float]] = None):
            super(A, self).__init__("A", a=Parameter("a", 1.0))
            if a is not None:
                self.a = a

    a = A()
    assert a.a.value == 1.0
    a = A(2.0)
    assert a.a.value == 2.0
    a = A(Parameter("a", 3.0))
    assert a.a.value == 3.0
    a.a = 4.0
    assert a.a.value == 4.0

    class B(BaseObj):
        def __init__(self, b: Optional[Union[A, Parameter, float]] = None):
            super(B, self).__init__("B", b=A())
            if b is not None:
                if isinstance(b, (float, Parameter)):
                    b = A(b)
                self.b = b

    b = B()
    assert b.b.a.value == 1.0
    b = B(2.0)
    assert b.b.a.value == 2.0
    b = B(A(3.0))
    assert b.b.a.value == 3.0
    b.b.a = 4.0
    assert b.b.a.value == 4.0

def test_unique_name_generator(clear):
    # When Then
    test_obj = BaseObj(name="test")
    # Expect
    assert test_obj.unique_name == "BaseObj_0"

def test_unique_name_change(clear):
    # When
    test_obj = BaseObj(name="test")
    # Then
    test_obj.unique_name = "test"
    # Expect
    assert test_obj.unique_name == "test"

@pytest.mark.parametrize("input", [2, 2.0, [2], {2}, None])
def test_unique_name_change_exception(input):
    # When
    test_obj = BaseObj(name="test")
    # Then Expect
    with pytest.raises(TypeError):
        test_obj.unique_name = input