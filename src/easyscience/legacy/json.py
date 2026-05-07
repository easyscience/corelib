# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

__author__ = 'https://github.com/materialsvirtuallab/monty/blob/master/monty/json.py'
__version__ = '3.0.0'


import json
from typing import TYPE_CHECKING
from typing import List

from .template import BaseEncoderDecoder

if TYPE_CHECKING:
    from .component_serializer import ComponentSerializer


class JsonSerializer(BaseEncoderDecoder):
    def encode(self, obj: ComponentSerializer, skip: List[str] = []) -> str:
        """
        Returns a json string representation of the ComponentSerializer
        object.
        """
        ENCODER = type(
            JsonEncoderTemplate.__name__,
            (JsonEncoderTemplate, BaseEncoderDecoder),
            {'skip': skip},
        )
        return json.dumps(obj, cls=ENCODER)

    @classmethod
    def decode(cls, data: str) -> ComponentSerializer:
        """Decode function."""
        return json.loads(data, cls=JsonDecoderTemplate)


class JsonDataSerializer(BaseEncoderDecoder):
    def encode(self, obj: ComponentSerializer, skip: List[str] = []) -> str:
        """
        Returns a json string representation of the ComponentSerializer
        object.
        """
        from .dict import DataDictSerializer

        ENCODER = type(
            JsonEncoderTemplate.__name__,
            (JsonEncoderTemplate, BaseEncoderDecoder),
            {
                'skip': skip,
                '_converter': lambda *args, **kwargs: DataDictSerializer._parse_dict(
                    DataDictSerializer._convert_to_dict(*args, **kwargs)
                ),
            },
        )

        return json.dumps(obj, cls=ENCODER)

    @classmethod
    def decode(cls, data: str) -> ComponentSerializer:
        """Decode function."""
        raise NotImplementedError(
            'It is not possible to reconstitute objects from data only objects.'
        )


class JsonEncoderTemplate(json.JSONEncoder):
    """
    A Json Encoder which supports the ComponentSerializer API, plus adds
    support for numpy arrays, datetime objects, bson ObjectIds (requires
    bson).

    Usage::

        # Add it as a *cls* keyword when using json.dump
        json.dumps(object, cls=MontyEncoder)
    """

    skip = []
    _converter = BaseEncoderDecoder._convert_to_dict

    def default(self, o) -> dict:  # pylint: disable=E0202
        """
        Overriding default method for JSON encoding.

        This method does two things: (a) If an object has a to_dict
        property, return the to_dict output. (b) If the @module and
        @class keys are not in the to_dict, add them to the output
        automatically. If the object has no to_dict property, the
        default Python json encoder default method is called.

        Parameters
        ----------
        o :
            Python object.

        Returns
        -------
        dict
        """
        return self._converter(o, self.skip, full_encode=True)


class JsonDecoderTemplate(json.JSONDecoder):
    """
    A Json Decoder which supports the ComponentSerializer API.

    By default, the decoder attempts to find a module and name
    associated with a dict. If found, the decoder will generate a
    Pymatgen as a priority.  If that fails, the original decoded
    dictionary from the string is returned. Note that nested lists and
    dicts containing pymatgen object will be decoded correctly as well.

    Usage:

        # Add it as a *cls* keyword when using json.load
        json.loads(json_string, cls=MontyDecoder)
    """

    _converter = BaseEncoderDecoder._convert_from_dict

    def decode(self, s):
        """
        Overrides decode from JSONDecoder.

        Parameters
        ----------
        s :
            String.

        Returns
        -------

            Object.
        """
        d = json.JSONDecoder.decode(self, s)
        return self.__class__._converter(d)
