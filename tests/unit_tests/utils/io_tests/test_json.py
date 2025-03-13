__author__ = "github.com/wardsimon"
__version__ = "0.0.1"

import json
from copy import deepcopy
from typing import Type

import pytest

from easyscience.Utils.io.json import JsonDataSerializer
from easyscience.Utils.io.json import JsonSerializer
from easyscience.Objects.variable import DescriptorNumber

from .test_core import A
from .test_core import check_dict
from .test_core import dp_param_dict
from .test_core import skip_dict
from easyscience import global_object


def recursive_remove(d, remove_keys: list) -> dict:
    """
    Remove keys from a dictionary.
    """
    if not isinstance(remove_keys, list):
        remove_keys = [remove_keys]
    if isinstance(d, dict):
        dd = {}
        for k in d.keys():
            if k not in remove_keys:
                dd[k] = recursive_remove(d[k], remove_keys)
        return dd
    else:
        return d


########################################################################################################################
# TESTING ENCODING
########################################################################################################################
@pytest.mark.parametrize(**skip_dict)
@pytest.mark.parametrize(**dp_param_dict)
def test_variable_DictSerializer(dp_kwargs: dict, dp_cls: Type[DescriptorNumber], skip):
    data_dict = {k: v for k, v in dp_kwargs.items() if k[0] != "@"}

    obj = dp_cls(**data_dict)

    dp_kwargs = deepcopy(dp_kwargs)

    if isinstance(skip, str):
        del dp_kwargs[skip]

    if not isinstance(skip, list):
        skip = [skip]

    enc = obj.encode(skip=skip, encoder=JsonSerializer)
    assert isinstance(enc, str)

    # We can test like this as we don't have "complex" objects yet
    dec = json.loads(enc)
    expected_keys = set(dp_kwargs.keys())
    obtained_keys = set(dec.keys())

    dif = expected_keys.difference(obtained_keys)

    assert len(dif) == 0

    check_dict(dp_kwargs, dec)


@pytest.mark.parametrize(**skip_dict)
@pytest.mark.parametrize(**dp_param_dict)
def test_variable_DataDictSerializer(dp_kwargs: dict, dp_cls: Type[DescriptorNumber], skip):
    data_dict = {k: v for k, v in dp_kwargs.items() if k[0] != "@"}

    obj = dp_cls(**data_dict)

    if isinstance(skip, str):
        del data_dict[skip]

    if not isinstance(skip, list):
        skip = [skip]

    enc = obj.encode(skip=skip, encoder=JsonDataSerializer)
    assert isinstance(enc, str)
    enc_d = json.loads(enc)

    expected_keys = set(data_dict.keys())
    obtained_keys = set(enc_d.keys())

    dif = expected_keys.difference(obtained_keys)

    assert len(dif) == 0

    check_dict(data_dict, enc_d)


@pytest.mark.parametrize(**skip_dict)
@pytest.mark.parametrize(**dp_param_dict)
def test_custom_class_DictSerializer_encode(
    dp_kwargs: dict, dp_cls: Type[DescriptorNumber], skip
):
    data_dict = {k: v for k, v in dp_kwargs.items() if k[0] != "@"}

    a_kw = {data_dict["name"]: dp_cls(**data_dict)}

    full_d = {
        "@module": A.__module__,
        "@class": A.__name__,
        "@version": None,
        "name": "A",
        dp_kwargs["name"]: deepcopy(dp_kwargs),
    }

    if not isinstance(skip, list):
        skip = [skip]

    full_d = recursive_remove(full_d, skip)

    obj = A(**a_kw)

    enc = obj.encode(skip=skip, encoder=JsonSerializer)
    assert isinstance(enc, str)

    # We can test like this as we don't have "complex" objects yet
    dec = json.loads(enc)

    expected_keys = set(full_d.keys())
    obtained_keys = set(dec.keys())

    dif = expected_keys.difference(obtained_keys)

    assert len(dif) == 0

    check_dict(full_d, dec)


@pytest.mark.parametrize(**skip_dict)
@pytest.mark.parametrize(**dp_param_dict)
def test_custom_class_DataDictSerializer(
    dp_kwargs: dict, dp_cls: Type[DescriptorNumber], skip
):
    data_dict = {k: v for k, v in dp_kwargs.items() if k[0] != "@"}

    a_kw = {data_dict["name"]: dp_cls(**data_dict)}

    full_d = {"name": "A", dp_kwargs["name"]: data_dict}

    if not isinstance(skip, list):
        skip = [skip]

    full_d = recursive_remove(full_d, skip)

    obj = A(**a_kw)

    enc = obj.encode(skip=skip, encoder=JsonDataSerializer)
    dec = json.loads(enc)

    expected_keys = set(full_d.keys())
    obtained_keys = set(dec.keys())

    dif = expected_keys.difference(obtained_keys)

    assert len(dif) == 0

    check_dict(full_d, dec)


# ########################################################################################################################
# # TESTING DECODING
# ########################################################################################################################
@pytest.mark.parametrize(**dp_param_dict)
def test_variable_DictSerializer_decode(dp_kwargs: dict, dp_cls: Type[DescriptorNumber]):
    data_dict = {k: v for k, v in dp_kwargs.items() if k[0] != "@"}

    obj = dp_cls(**data_dict)

    enc = obj.encode(encoder=JsonSerializer)
    global_object.map._clear()
    assert isinstance(enc, str)
    dec = obj.decode(enc, decoder=JsonSerializer)

    for k in data_dict.keys():
        if hasattr(obj, k) and hasattr(dec, k):
            assert getattr(obj, k) == getattr(dec, k)
        else:
            raise AttributeError(f"{k} not found in decoded object")


@pytest.mark.parametrize(**dp_param_dict)
def test_variable_DataDictSerializer_decode(dp_kwargs: dict, dp_cls: Type[DescriptorNumber]):
    data_dict = {k: v for k, v in dp_kwargs.items() if k[0] != "@"}

    obj = dp_cls(**data_dict)

    enc = obj.encode(encoder=JsonDataSerializer)
    global_object.map._clear()
    with pytest.raises(NotImplementedError):
        dec = obj.decode(enc, decoder=JsonDataSerializer)
