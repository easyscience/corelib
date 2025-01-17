#  SPDX-FileCopyrightText: 2023 EasyScience contributors  <core@easyscience.software>
#  SPDX-License-Identifier: BSD-3-Clause
#  © 2021-2023 Contributors to the EasyScience project <https://github.com/easyScience/EasyScience

__author__ = "github.com/wardsimon"
__version__ = "0.1.0"

from copy import deepcopy
from typing import List

import numpy as np
import pytest

import scipp as sc
import easyscience
from easyscience.Utils.Exceptions import CoreSetException

from easyscience.Objects.variable import Parameter
from easyscience import global_object


def _generate_inputs():
    # These are the parameters which will always be present
    basic = {"name": "test", "value": 1}
    basic_result = {
        "name": basic["name"],
        "value": basic["value"],
    }
    # These will be the optional parameters
    advanced = {
        "unit": ["cm", "mm", "kelvin"],
        "description": "This is a test",
        "url": "https://www.whatever.com",
        "display_name": r"\Chi",
    }
    advanced_result = {
        "unit": {"name": "unit", "value": ["cm", "mm", "K"]},
        "description": {"name": "description", "value": advanced["description"]},
        "url": {"name": "url", "value": advanced["url"]},
        "display_name": {"name": "display_name", "value": advanced["display_name"]},
    }

    temp = [
        ([[basic["name"], basic["value"]], {}], basic_result),
        ([[], basic], basic_result),
    ]

    def create_entry(base, key, value, ref, ref_key=None):
        this_temp = deepcopy(base)
        for item in base:
            test, res = item
            new_opt = deepcopy(test[1])
            new_res = deepcopy(res)
            if ref_key is None:
                ref_key = key
            new_res[ref_key] = ref
            new_opt[key] = value
            this_temp.append(([test[0], new_opt], new_res))
        return this_temp

    for add_opt in advanced.keys():
        if isinstance(advanced[add_opt], list):
            for idx, item in enumerate(advanced[add_opt]):
                temp = create_entry(
                    temp,
                    add_opt,
                    item,
                    advanced_result[add_opt]["value"][idx],
                    ref_key=advanced_result[add_opt]["name"],
                )
        else:
            temp = create_entry(
                temp,
                add_opt,
                advanced[add_opt],
                advanced_result[add_opt]["value"],
                ref_key=advanced_result[add_opt]["name"],
            )
    return temp

@pytest.mark.parametrize("element, expected", _generate_inputs())
def test_item_creation(element: List, expected: dict):
    d = Parameter(*element[0], **element[1])
    for field in expected.keys():
        ref = expected[field]
        obtained = getattr(d, field)
        assert obtained == ref


@pytest.mark.parametrize(
    "element, expected",
    [
        ("", sc.scalar(1)),
        ("centimeters", sc.scalar(1, unit='cm')),
        ("mm",sc.scalar(1, unit='mm')),
        ("kelvin", sc.scalar(1, unit='K')),
    ],
)
def test_Parameter_value_get(element, expected):
    d = Parameter("test", 1, unit=element)
    assert d.full_value == expected


@pytest.mark.parametrize("debug", (True, False))
@pytest.mark.parametrize("enabled", (None, True, False))
def test_item_value_set(enabled, debug):
    global_object.debug = debug
    set_value = 2
    d = Parameter("test", 1)
    if enabled is not None:
        d.enabled = enabled
    else:
        enabled = True
    if enabled:
        d.value = set_value
        assert d.value == set_value
    else:
        if debug:
            with pytest.raises(CoreSetException):
                d.value = set_value
    d = Parameter("test", 1, unit="kelvin")
    if enabled is not None:
        d.enabled = enabled
    else:
        enabled = True

    if enabled:
        d.value = set_value
        assert d.value == set_value
        assert str(d.unit) == "K"
    else:
        if debug:
            with pytest.raises(CoreSetException):
                d.value = set_value

def test_item_convert_unit():
    d = Parameter("test", 360, unit="km/h")
    d.convert_unit("m/s")
    assert 100 == pytest.approx(d.value)


@pytest.mark.parametrize(
    "conv_unit, data_in, result",
    [
        ("m/s", {"min": 0}, {"min": 0, "value": 100}),
        ("m/s", {"max": 500}, {"max": 138.8888, "value": 100}),
        ("m/s", {"variance": 1.}, {"variance":0.07716049382, "value": 100}),
        (
            "m/s",
            {"min": 0, "max": 500, "variance": 1.},
            {"min": 0, "max": 138.8888, "variance": 0.07716049382, "value": 100},
        ),
        ("miles/h", {"min": 0}, {"min": 0, "value": 223.6936}),
        ("miles/h", {"max": 500}, {"max": 310.6855, "value": 223.6936}),
        ("miles/h", {"variance": 1.}, {"variance": 0.3861021585424, "value": 223.6936}),
        (
            "miles/h",
            {"min": 0, "max": 500, "variance": 1.},
            {"min": 0, "max": 310.6855, "variance": 0.3861021585424, "value": 223.6936},
        ),
    ],
    ids=[
        "min_m/s",
        "max_m/s",
        "variance_m/s",
        "combined_m/s",
        "min_miles/h",
        "max_miles/h",
        "variance_miles/h",
        "combined_miles/h",
    ],
)
def test_parameter_advanced_convert_unit(conv_unit: str, data_in: dict, result: dict):
    d = Parameter("test", 360, unit="km/h", **data_in)
    d.convert_unit(conv_unit)
    for key in result.keys():
        assert result[key] == pytest.approx(getattr(d, key))


def test_parameter_repr():
    d = Parameter("test", 1)
    assert repr(d) == f"<{d.__class__.__name__} 'test': 1.0000, bounds=[-inf:inf]>"
    d = Parameter("test", 1, unit="cm")
    assert repr(d) == f"<{d.__class__.__name__} 'test': 1.0000 cm, bounds=[-inf:inf]>"
    d = Parameter("test", 1, variance=0.1)
    assert repr(d) == f"<{d.__class__.__name__} 'test': 1.0000 ± 0.3162, bounds=[-inf:inf]>"

    d = Parameter("test", 1, fixed=True)
    assert (
        repr(d)
        == f"<{d.__class__.__name__} 'test': 1.0000 (fixed), bounds=[-inf:inf]>"
    )
    d = Parameter("test", 1, unit="cm", variance=0.1, fixed=True)
    assert (
        repr(d)
        == f"<{d.__class__.__name__} 'test': 1.0000 ± 0.3162 cm (fixed), bounds=[-inf:inf]>"
    )


def test_parameter_as_dict():
    d = Parameter("test", 1)
    result = d.as_dict()
    expected = {
        "@module": Parameter.__module__,
        "@class": Parameter.__name__,
        "@version": easyscience.__version__,
        "name": "test",
        "value": 1.0,
        "variance": 0.0,
        "min": -np.inf,
        "max": np.inf,
        "fixed": False,
        "unit": "dimensionless",
    }
    for key in expected.keys():
        assert result[key] == expected[key]

    # Check that additional arguments work
    d = Parameter("test", 1, unit="km", url="https://www.boo.com")
    result = d.as_dict()
    expected = {
        "@module": Parameter.__module__,
        "@class": Parameter.__name__,
        "@version": easyscience.__version__,
        "name": "test",
        "unit": "km",
        "value": 1.0,
        "variance": 0.0,
        "min": -np.inf,
        "max": np.inf,
        "fixed": False,
        "url": "https://www.boo.com",
    }
    for key in expected.keys():
        assert result[key] == expected[key]


def test_item_from_dict():
    reference = {
            "@module": Parameter.__module__,
            "@class": Parameter.__name__,
            "@version": easyscience.__version__,
            "name": "test",
            "unit": "km",
            "value": 1.0,
            "variance": 0.0,
            "min": -np.inf,
            "max": np.inf,
            "fixed": False,
            "url": "https://www.boo.com",
        }
    constructor = Parameter
    d = constructor.from_dict(reference)
    for key, item in reference.items():
        if key == "callback" or key.startswith("@"):
            continue
        obtained = getattr(d, key)
        assert obtained == item


@pytest.mark.parametrize(
    "construct",
    (
        {
            "@module": Parameter.__module__,
            "@class": Parameter.__name__,
            "@version": easyscience.__version__,
            "name": "test",
            "unit": "km",
            "value": 1.0,
            "variance": 0.0,
            "min": -np.inf,
            "max": np.inf,
            "fixed": False,
            "url": "https://www.boo.com",
        },
    ),
    ids=["Parameter"],
)
def test_item_from_Decoder(construct):

    from easyscience.Utils.io.dict import DictSerializer

    d = DictSerializer().decode(construct)
    assert d.__class__.__name__ == construct["@class"]
    for key, item in construct.items():
        if key == "callback" or key.startswith("@"):
            continue
        obtained = getattr(d, key)
        assert obtained == item


@pytest.mark.parametrize("value", (-np.inf, 0, 1.0, 2147483648, np.inf))
def test_parameter_min(value):
    d = Parameter("test", -0.1)
    if d.value < value:
        with pytest.raises(ValueError):
            d.min = value
    else:
        d.min = value
        assert d.min == value


@pytest.mark.parametrize("value", [-np.inf, 0, 1.1, 2147483648, np.inf])
def test_parameter_max(value):
    d = Parameter("test", 2147483649)
    if d.value > value:
        with pytest.raises(ValueError):
            d.max = value
    else:
        d.max = value
        assert d.max == value


@pytest.mark.parametrize("value", [True, False, 5])
def test_parameter_fixed(value):
    d = Parameter("test", -np.inf)
    if isinstance(value, bool):
        d.fixed = value
        assert d.fixed == value
    else:
        with pytest.raises(ValueError):
            d.fixed = value


@pytest.mark.parametrize("value", (-np.inf, -0.1, 0, 1.0, 2147483648, np.inf))
def test_parameter_error(value):
    d = Parameter("test", 1)
    if value >= 0:
        d.error = value
        assert d.error == value
    else:
        with pytest.raises(ValueError):
            d.error = value


def _generate_advanced_inputs():
    temp = _generate_inputs()
    # These will be the optional parameters
    advanced = {"variance": 1.0, "min": -0.1, "max": 2147483648, "fixed": False}
    advanced_result = {
        "variance": {"name": "variance", "value": advanced["variance"]},
        "min": {"name": "min", "value": advanced["min"]},
        "max": {"name": "max", "value": advanced["max"]},
        "fixed": {"name": "fixed", "value": advanced["fixed"]},
    }

    def create_entry(base, key, value, ref, ref_key=None):
        this_temp = deepcopy(base)
        for item in base:
            test, res = item
            new_opt = deepcopy(test[1])
            new_res = deepcopy(res)
            if ref_key is None:
                ref_key = key
            new_res[ref_key] = ref
            new_opt[key] = value
            this_temp.append(([test[0], new_opt], new_res))
        return this_temp

    for add_opt in advanced.keys():
        if isinstance(advanced[add_opt], list):
            for idx, item in enumerate(advanced[add_opt]):
                temp = create_entry(
                    temp,
                    add_opt,
                    item,
                    advanced_result[add_opt]["value"][idx],
                    ref_key=advanced_result[add_opt]["name"],
                )
        else:
            temp = create_entry(
                temp,
                add_opt,
                advanced[add_opt],
                advanced_result[add_opt]["value"],
                ref_key=advanced_result[add_opt]["name"],
            )
    return temp


@pytest.mark.parametrize("element, expected", _generate_advanced_inputs())
def test_parameter_advanced_creation(element, expected):
    if len(element[0]) > 0:
        value = element[0][1]
    else:
        value = element[1]["value"]
    if "min" in element[1].keys():
        if element[1]["min"] > value:
            with pytest.raises(ValueError):
                d = Parameter(*element[0], **element[1])
    elif "max" in element[1].keys():
        if element[1]["max"] < value:
            with pytest.raises(ValueError):
                d = Parameter(*element[0], **element[1])
    else:
        d = Parameter(*element[0], **element[1])
        for field in expected.keys():
            ref = expected[field]
            obtained = getattr(d, field)
            assert obtained == ref


@pytest.mark.parametrize("value", ("This is ", "a fun ", "test"))
def test_parameter_display_name(value):
    p = Parameter("test", 1, display_name=value)
    assert p.display_name == value


@pytest.mark.parametrize("value", (True, False))
def test_parameter_bounds(value):
    for fixed in (True, False):
        p = Parameter("test", 1, enabled=value, fixed=fixed)
        assert p.min == -np.inf
        assert p.max == np.inf
        assert p.fixed == fixed
        assert p.bounds == (-np.inf, np.inf)

        p.bounds = (0, 2)
        assert p.min == 0
        assert p.max == 2
        assert p.bounds == (0, 2)
        assert p.enabled is True
        assert p.fixed is False