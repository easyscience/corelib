__author__ = "github.com/wardsimon"
__version__ = "0.1.0"

#  SPDX-FileCopyrightText: 2023 EasyScience contributors  <core@easyscience.software>
#  SPDX-License-Identifier: BSD-3-Clause
#  © 2021-2023 Contributors to the EasyScience project <https://github.com/easyScience/EasyScience

from typing import List

import pytest

import easyscience
from easyscience.Objects.Groups import BaseCollection
from easyscience.Objects.ObjectClasses import BaseObj
from easyscience.Objects.variable import DescriptorNumber
from easyscience.Objects.variable import Parameter
from easyscience import global_object

test_dict = {
    "@module": "easyscience.Objects.Groups",
    "@class": "BaseCollection",
    "@version": easyscience.__version__,
    "name": "testing",
    "data": [
        {
            "@module": DescriptorNumber.__module__,
            "@class": DescriptorNumber.__name__,
            "@version": easyscience.__version__,
            "name": "par1",
            "value": 1.0,
            "unit": "dimensionless",
            "variance": None,
            "unique_name": "BaseCollection_0",
            "description": "",
            "url": "",
            "display_name": "par1",
        }
    ],
}


class Alpha(BaseCollection):
    pass


class_constructors = [BaseCollection, Alpha]


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


@pytest.mark.parametrize("cls", class_constructors)
def test_baseCollection_from_base(cls, setup_pars):
    name = setup_pars["name"]
    del setup_pars["name"]
    coll = cls(name, **setup_pars)

    assert coll.name == name
    assert len(coll) == 5
    assert coll.user_data == {}

    for item, key in zip(coll, setup_pars.keys()):
        assert item.name == setup_pars[key].name
        assert item.value == setup_pars[key].value


@pytest.mark.parametrize("cls", class_constructors)
@pytest.mark.parametrize("value", range(1, 11))
def test_baseCollection_from_baseObj(cls, setup_pars: dict, value: int):
    name = setup_pars["name"]
    del setup_pars["name"]
    objs = {}

    prefix = "obj"
    for idx in range(value):
        objs[prefix + str(idx)] = BaseObj(prefix + str(idx), **setup_pars)

    coll = cls(name, **objs)

    assert coll.name == name
    assert len(coll) == value
    assert coll.user_data == {}

    idx = 0
    for item, key in zip(coll, objs.keys()):
        assert item.name == prefix + str(idx)
        assert isinstance(item, objs[key].__class__)
        idx += 1


@pytest.mark.parametrize("cls", class_constructors)
@pytest.mark.parametrize("value", ("abc", False, (), []))
def test_baseCollection_create_fail(cls, setup_pars, value):
    name = setup_pars["name"]
    del setup_pars["name"]
    setup_pars["to_fail"] = value

    with pytest.raises(AttributeError):
        coll = cls(name, **setup_pars)


@pytest.mark.parametrize("cls", class_constructors)
@pytest.mark.parametrize("key", ("user_data", "_kwargs", "interface"))
def test_baseCollection_create_fail2(cls, setup_pars, key):
    name = setup_pars["name"]
    del setup_pars["name"]
    setup_pars[key] = DescriptorNumber("fail_name", 0)

    with pytest.raises(AttributeError):
        coll = cls(name, **setup_pars)


@pytest.mark.parametrize("cls", class_constructors)
def test_baseCollection_append_base(cls, setup_pars):
    name = setup_pars["name"]
    del setup_pars["name"]

    new_item_name = "boo"
    new_item_value = 100
    new_item = Parameter(new_item_name, new_item_value)

    coll = cls(name, **setup_pars)
    n_before = len(coll)

    coll.append(new_item)
    assert len(coll) == n_before + 1
    assert coll[-1].name == new_item_name
    assert coll[-1].value == new_item_value


@pytest.mark.parametrize("cls", class_constructors)
@pytest.mark.parametrize("value", ("abc", False, (), []))
def test_baseCollection_append_fail(cls, setup_pars, value):
    name = setup_pars["name"]
    del setup_pars["name"]

    coll = cls(name, **setup_pars)
    with pytest.raises(AttributeError):
        coll.append(value)


@pytest.mark.parametrize("cls", class_constructors)
@pytest.mark.parametrize("value", (0, 1, 3, "par1", "des1"))
def test_baseCollection_getItem(cls, setup_pars, value):
    name = setup_pars["name"]
    del setup_pars["name"]

    coll = cls(name, **setup_pars)

    get_item = coll[value]
    if isinstance(value, str):
        key = value
    else:
        key = list(setup_pars.keys())[value]
    assert get_item.name == setup_pars[key].name


@pytest.mark.parametrize("cls", class_constructors)
@pytest.mark.parametrize("value", (False, [], (), 100, 100.4))
def test_baseCollection_getItem_type_fail(cls, setup_pars, value):
    name = setup_pars["name"]
    del setup_pars["name"]

    coll = cls(name, **setup_pars)

    with pytest.raises((IndexError, TypeError)):
        get_item = coll[value]


@pytest.mark.parametrize("cls", class_constructors)
def test_baseCollection_getItem_slice(cls, setup_pars):
    name = setup_pars["name"]
    del setup_pars["name"]

    coll = cls(name, **setup_pars)

    get_item = coll[0:2]
    assert len(get_item) == 2


@pytest.mark.parametrize("cls", class_constructors)
@pytest.mark.parametrize("value", (0, 1, 3))
def test_baseCollection_setItem(cls, setup_pars, value):
    name = setup_pars["name"]
    del setup_pars["name"]

    coll = cls(name, **setup_pars)
    n_coll = len(coll)
    name_coll_idx = coll[value].name

    new_item_value = 100

    coll[value] = new_item_value

    assert len(coll) == n_coll
    assert coll[value].name == name_coll_idx
    assert coll[value].value == new_item_value


@pytest.mark.parametrize("cls", class_constructors)
@pytest.mark.parametrize("value", ("abc", (), []))
def test_baseCollection_setItem_fail(cls, setup_pars, value):
    name = setup_pars["name"]
    del setup_pars["name"]

    coll = cls(name, **setup_pars)

    with pytest.raises(NotImplementedError):
        for idx in range(len(coll)):
            coll[idx] = value


@pytest.mark.parametrize("cls", class_constructors)
@pytest.mark.parametrize("value", (0, 1, 3))
def test_baseCollection_delItem(cls, setup_pars, value):
    name = setup_pars["name"]
    del setup_pars["name"]

    coll = cls(name, **setup_pars)
    n_coll = len(coll)
    # On del we should shift left
    name_coll_idx = coll[value].name
    name_coll_idxp = coll[value + 1].name

    del coll[value]

    assert len(coll) == n_coll - 1
    assert coll[value].name == name_coll_idxp
    assert name_coll_idx not in [col.name for col in coll]


@pytest.mark.parametrize("cls", class_constructors)
@pytest.mark.parametrize("value", (0, 1, 3))
def test_baseCollection_len(cls, setup_pars, value):
    name = setup_pars["name"]
    del setup_pars["name"]

    keys = list(setup_pars.keys())
    keys = keys[0 : (value + 1)]

    coll = cls(name, **{key: setup_pars[key] for key in keys})
    assert len(coll) == (value + 1)


@pytest.mark.parametrize("cls", class_constructors)
def test_baseCollection_get_parameters(cls, setup_pars):
    name = setup_pars["name"]
    del setup_pars["name"]
    obj = cls(name, **setup_pars)
    pars = obj.get_parameters()
    assert len(pars) == 3


@pytest.mark.parametrize("cls", class_constructors)
def test_baseCollection_get_parameters_nested(cls, setup_pars):
    name = setup_pars["name"]
    del setup_pars["name"]
    obj = BaseObj(name, **setup_pars)

    name2 = name + "_2"
    obj2 = cls(name2, obj=obj, **setup_pars)

    pars = obj2.get_parameters()
    assert len(pars) == 6


@pytest.mark.parametrize("cls", class_constructors)
def test_baseCollection_get_fit_parameters(cls, setup_pars):
    name = setup_pars["name"]
    del setup_pars["name"]
    obj = cls(name, **setup_pars)
    pars = obj.get_fit_parameters()
    assert len(pars) == 2


@pytest.mark.parametrize("cls", class_constructors)
def test_baseCollection_get_fit_parameters_nested(cls, setup_pars):
    name = setup_pars["name"]
    del setup_pars["name"]
    obj = BaseObj(name, **setup_pars)

    name2 = name + "_2"
    obj2 = cls(name2, obj=obj, **setup_pars)

    pars = obj2.get_fit_parameters()
    assert len(pars) == 4


@pytest.mark.parametrize("cls", class_constructors)
def test_baseCollection_dir(cls):
    name = "testing"
    kwargs = {"p1": DescriptorNumber("par1", 1)}
    obj = cls(name, **kwargs)
    d = set(dir(obj))

    expected = {
        "user_data",
        "reverse",
        "constraints",
        "get_fit_parameters",
        "append",
        "unique_name",
        "index",
        "as_dict",
        "clear",
        "extend",
        "encode",
        "remove",
        "as_data_dict",
        "interface",
        "from_dict",
        "name",
        "switch_interface",
        "get_parameters",
        "insert",
        "data",
        "pop",
        "count",
        "generate_bindings",
        "unsafe_hash",
        "decode",
        "encode_data",
        "sort",
    }
    assert not d.difference(expected)


@pytest.mark.parametrize("cls", class_constructors)
def test_baseCollection_as_dict(cls):
    name = "testing"
    kwargs = {"p1": DescriptorNumber("par1", 1)}
    obj = cls(name, **kwargs)
    d = obj.as_dict()

    def check_dict(dict_1: dict, dict_2: dict):
        keys_1 = list(dict_1.keys())
        keys_2 = list(dict_2.keys())
        if "unique_name" in keys_1:
            del keys_1[keys_1.index("unique_name")]
        if "unique_name" in keys_2:
            del keys_2[keys_2.index("unique_name")]

        assert not set(keys_1).difference(set(keys_2))

        def testit(item1, item2):
            if isinstance(item1, dict) and isinstance(item2, dict):
                check_dict(item1, item2)
            elif isinstance(item1, list) and isinstance(item2, list):
                for v1, v2 in zip(item1, item2):
                    testit(v1, v2)
            else:
                if isinstance(item1, str) and isinstance(item2, str):
                    assert item1 == item2
                elif isinstance(item1, float) and isinstance(item2, float):
                    assert item1 == item2
                else:
                    assert item1 is item2

        keys_1 = list(keys_1)
        keys_1.sort()
        keys_2 = list(keys_2)
        keys_2.sort()

        for k1, k2 in zip(keys_1, keys_2):
            if k1[0] == "@":
                continue
            testit(dict_1[k1], dict_2[k2])

    check_dict(d, test_dict)


@pytest.mark.parametrize("cls", class_constructors)
def test_baseCollection_from_dict(cls):
    name = "testing"
    kwargs = {"p1": DescriptorNumber("par1", 1)}
    expected = cls.from_dict(test_dict)
    ref = cls(name, **kwargs)

    assert ref.name == expected.name
    assert len(ref) == len(expected)
    for item1, item2 in zip(ref, expected):
        assert item1.name == item2.name
        assert item1.value == item2.value


@pytest.mark.parametrize("cls", class_constructors)
def test_baseCollection_constraints(cls):
    name = "test"
    p1 = Parameter("p1", 1)
    p2 = Parameter("p2", 2)

    from easyscience.Constraints import ObjConstraint

    p2.user_constraints["testing"] = ObjConstraint(p2, "2*", p1)

    obj = cls(name, p1, p2)

    cons: List[ObjConstraint] = obj.constraints
    assert len(cons) == 1


@pytest.mark.parametrize("cls", class_constructors)
def test_baseCollection_repr(cls):
    name = "test"
    p1 = Parameter("p1", 1)
    obj = cls(name, p1)
    test_str = str(obj)
    ref_str = f"{cls.__name__} `{name}` of length 1"
    assert test_str == ref_str


@pytest.mark.parametrize("cls", class_constructors)
def test_baseCollection_iterator(cls):
    name = "test"
    p1 = Parameter("p1", 1)
    p2 = Parameter("p2", 2)
    p3 = Parameter("p3", 3)
    p4 = Parameter("p4", 4)

    l_object = [p1, p2, p3, p4]

    obj = cls(name, *l_object)

    for index, item in enumerate(obj):
        assert item == l_object[index]


@pytest.mark.parametrize("cls", class_constructors)
def test_baseCollection_iterator_dict(cls):
    name = "test"
    p1 = Parameter("p1", 1)
    p2 = Parameter("p2", 2)
    p3 = Parameter("p3", 3)
    p4 = Parameter("p4", 4)

    l_object = [p1, p2, p3, p4]

    obj = cls(name, *l_object)
    d = obj.as_dict()
    global_object.map._clear()
    obj2 = cls.from_dict(d)

    for index, item in enumerate(obj2):
        assert item.value == l_object[index].value


@pytest.mark.parametrize("cls", class_constructors)
def test_baseCollection_sameName(cls):
    name = "test"
    p1 = Parameter("p1", 1)
    p2 = Parameter("p1", 2)
    p3 = Parameter("p3", 3)
    p4 = Parameter("p4", 4)

    l_object = [p1, p2, p3, p4]
    obj = cls(name, *l_object)
    assert len(l_object) == len(obj)
    for index, item in enumerate(obj):
        assert item == l_object[index]

    obj12 = obj["p1"]
    assert len(obj12) == 2
    for index, item in enumerate(obj12):
        assert item == l_object[index]


@pytest.mark.parametrize("cls", class_constructors)
def test_baseCollection_set_index(cls):
    name = "test"
    p1 = Parameter("p1", 1)
    p2 = Parameter("p1", 2)
    p3 = Parameter("p3", 3)
    p4 = Parameter("p4", 4)

    l_object = [p1, p2, p3]
    obj = cls(name, *l_object)

    idx = 1
    assert obj[idx] == p2
    obj[idx] = p4
    assert obj[idx] == p4
    edges = obj._global_object.map.get_edges(obj)
    assert len(edges) == len(obj)
    for item in obj:
        assert item.unique_name in edges
    assert p2.unique_name not in edges


@pytest.mark.parametrize("cls", class_constructors)
def test_baseCollection_set_index_based(cls):
    name = "test"
    p1 = Parameter("p1", 1)
    p2 = Parameter("p2", 2)
    p3 = Parameter("p3", 3)
    p4 = Parameter("p4", 4)
    p5 = Parameter("p5", 5)
    d = cls("testing", p1, p2)

    l_object = [p3, p4, p5]
    obj = cls(name, *l_object)

    idx = 1
    assert obj[idx] == p4
    obj[idx] = d
    assert obj[idx] == d
    edges = obj._global_object.map.get_edges(obj)
    assert len(edges) == len(obj)
    for item in obj:
        assert item.unique_name in edges
    assert p4.unique_name not in edges


@pytest.mark.parametrize("cls", class_constructors)
def test_baseCollection_sort(cls):
    name = "test"
    v = [1, 4, 3, 2, 5]
    expected = [1, 2, 3, 4, 5]
    d = cls(name, *[Parameter(f"p{i}", v[i]) for i in range(len(v))])
    d.sort(lambda x: x.value)
    for i, item in enumerate(d):
        assert item.value == expected[i]


@pytest.mark.parametrize("cls", class_constructors)
def test_baseCollection_sort_reverse(cls):
    name = "test"
    v = [1, 4, 3, 2, 5]
    expected = [1, 2, 3, 4, 5]
    expected.reverse()
    d = cls(name, *[Parameter(f"p{i}", v[i]) for i in range(len(v))])
    d.sort(lambda x: x.value, reverse=True)
    for i, item in enumerate(d):
        assert item.value == expected[i]


class Beta(BaseObj):
    pass


@pytest.mark.parametrize("cls", class_constructors)
def test_basecollectionGraph(cls):
    from easyscience import global_object

    G = global_object.map
    name = "test"
    v = [1, 2]
    p = [Parameter(f"p{i}", v[i]) for i in range(len(v))]
    p_id = [_p.unique_name for _p in p]
    bb = cls(name, *p)
    bb_id = bb.unique_name
    b = Beta("b", bb=bb)
    b_id = b.unique_name
    for _id in p_id:
        assert _id in G.get_edges(bb)
    assert len(p) == len(G.get_edges(bb))
    assert bb_id in G.get_edges(b)
    assert 1 == len(G.get_edges(b))
