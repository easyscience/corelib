#  SPDX-FileCopyrightText: 2023 EasyScience contributors  <core@easyscience.software>
#  SPDX-License-Identifier: BSD-3-Clause
#  © 2021-2023 Contributors to the EasyScience project <https://github.com/easyScience/EasyScience

__author__ = "github.com/wardsimon"
__version__ = "0.1.0"

from copy import deepcopy
from typing import List

import pytest
import scipp as sc

import easyscience
from easyscience.Objects.variable import DescriptorBool
from easyscience.Objects.variable import DescriptorNumber
from easyscience.Objects.variable import DescriptorStr


@pytest.fixture
def instance(request):
    def class_creation(*args, **kwargs):
        return request.param(*args, **kwargs)

    return class_creation


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
    d = DescriptorNumber(*element[0], **element[1])
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
def test_descriptor_number_value_get(element, expected):
    d = DescriptorNumber("test", 1, unit=element)
    assert d.full_value == expected

def test_item_convert_unit():
    d = DescriptorNumber("test", 360, unit="km/h")
    d.convert_unit("m/s")
    assert 100 == pytest.approx(d.value)


def test_descriptor_repr():
    d = DescriptorNumber("test", 1)
    assert repr(d) == f"<{d.__class__.__name__} 'test': 1.0000>"
    d = DescriptorNumber("test", 1, unit="cm")
    assert repr(d) == f"<{d.__class__.__name__} 'test': 1.0000 cm>"


def test_descriptor_number_as_dict():
    d = DescriptorNumber("test", 1)
    result = d.as_dict()
    expected = {
        "@module": DescriptorNumber.__module__,
        "@class": DescriptorNumber.__name__,
        "@version": easyscience.__version__,
        "name": "test",
        "value": 1,
        "unit": "dimensionless",
        "description": "",
        "url": "",
        "display_name": "test",
        "callback": None,
    }
    for key in expected.keys():
        if key == "callback":
            continue
        assert result[key] == expected[key]


@pytest.mark.parametrize(
    "reference, constructor",
    (
        [
            {
                "@module": DescriptorBool.__module__,
                "@class": DescriptorBool.__name__,
                "@version": easyscience.__version__,
                "name": "test",
                "value": False,
                "description": "",
                "url": "",
                "display_name": "test",
            },
            DescriptorBool,
        ],
        [
            {
                "@module": DescriptorNumber.__module__,
                "@class": DescriptorNumber.__name__,
                "@version": easyscience.__version__,
                "name": "test",
                "value": 1,
                "unit": "dimensionless",
                "variance": 0.0,
                "description": "",
                "url": "",
                "display_name": "test",
            },
            DescriptorNumber,
        ],
        [
            {
                "@module": DescriptorStr.__module__,
                "@class": DescriptorStr.__name__,
                "@version": easyscience.__version__,
                "name": "test",
                "value": "string",
                "description": "",
                "url": "",
                "display_name": "test",
            },
            DescriptorStr,
        ],
    ),
    ids=["DescriptorBool", "DescriptorNumber", "DescriptorStr"],
)
def test_item_from_dict(reference, constructor):
    d = constructor.from_dict(reference)
    for key, item in reference.items():
        if key.startswith("@"):
            continue
        obtained = getattr(d, key)
        assert obtained == item


@pytest.mark.parametrize("value", ("This is ", "a fun ", "test"))
def test_parameter_display_name(value):
    p = DescriptorNumber("test", 1, display_name=value)
    assert p.display_name == value


def test_item_boolean_value():
    item = DescriptorBool("test", True)
    assert item.value is True
    item.value = False
    assert item.value is False

    item = DescriptorBool("test", False)
    assert item.value is False
    item.value = True
    assert item.value is True

