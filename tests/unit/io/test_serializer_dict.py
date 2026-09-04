# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

from copy import deepcopy
from typing import Type

import pytest

from easyscience import DescriptorNumber
from easyscience import ObjBase
from easyscience import global_object
from easyscience.io.serializer_dict import SerializerDict

from .test_serializer_component import check_dict
from .test_serializer_component import dp_param_dict
from .test_serializer_component import skip_dict


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
def test_variable_SerializerDict(dp_kwargs: dict, dp_cls: Type[DescriptorNumber], skip):
    data_dict = {k: v for k, v in dp_kwargs.items() if k[0] != '@'}

    obj = dp_cls(**data_dict)

    dp_kwargs = deepcopy(dp_kwargs)

    if isinstance(skip, str):
        del dp_kwargs[skip]

    if not isinstance(skip, list):
        skip = [skip]

    enc = SerializerDict().encode(obj, skip=skip)

    expected_keys = set(dp_kwargs.keys())
    obtained_keys = set(enc.keys())

    dif = expected_keys.difference(obtained_keys)

    assert len(dif) == 0

    check_dict(dp_kwargs, enc)


########################################################################################################################
# TESTING DECODING
########################################################################################################################
@pytest.mark.parametrize(**dp_param_dict)
def test_variable_SerializerDict_decode(dp_kwargs: dict, dp_cls: Type[DescriptorNumber]):
    data_dict = {k: v for k, v in dp_kwargs.items() if k[0] != '@'}

    obj = dp_cls(**data_dict)

    enc = SerializerDict().encode(obj)
    global_object.map._clear()
    dec = SerializerDict.decode(enc)

    for k in data_dict.keys():
        if hasattr(obj, k) and hasattr(dec, k):
            assert getattr(obj, k) == getattr(dec, k)
        else:
            raise AttributeError(f'{k} not found in decoded object')


@pytest.mark.parametrize(**dp_param_dict)
def test_variable_SerializerDict_from_dict(dp_kwargs: dict, dp_cls: Type[DescriptorNumber]):
    data_dict = {k: v for k, v in dp_kwargs.items() if k[0] != '@'}

    obj = dp_cls(**data_dict)

    enc = SerializerDict().encode(obj)
    global_object.map._clear()
    dec = dp_cls.from_dict(enc)

    for k in data_dict.keys():
        if hasattr(obj, k) and hasattr(dec, k):
            assert getattr(obj, k) == getattr(dec, k)
        else:
            raise AttributeError(f'{k} not found in decoded object')


def test_group_encode():
    d0 = DescriptorNumber('a', 0)
    d1 = DescriptorNumber('b', 1)

    from easyscience.base_classes import CollectionBase

    b = CollectionBase('test', d0, d1)
    d = b.as_dict()
    assert isinstance(d['data'], list)


def test_group_encode2():
    d0 = DescriptorNumber('a', 0)
    d1 = DescriptorNumber('b', 1)

    from easyscience.base_classes import CollectionBase

    b = ObjBase('outer', b=CollectionBase('test', d0, d1))
    d = b.as_dict()
    assert isinstance(d['b'], dict)
