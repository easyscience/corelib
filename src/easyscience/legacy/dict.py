# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

__author__ = 'https://github.com/materialsvirtuallab/monty/blob/master/monty/json.py'
__version__ = '3.0.0'


from typing import TYPE_CHECKING
from typing import Any
from typing import Dict
from typing import List
from typing import Optional

from .template import BaseEncoderDecoder

if TYPE_CHECKING:
    from .component_serializer import ComponentSerializer

_KNOWN_CORE_TYPES = ('Descriptor', 'Parameter')


class DictSerializer(BaseEncoderDecoder):
    """
    This is a serializer that can encode and decode EasyScience objects
    to a JSON encoded dictionary.
    """

    def encode(
        self,
        obj: ComponentSerializer,
        skip: Optional[List[str]] = None,
        full_encode: bool = False,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Convert an EasyScience object to a JSON encoded dictionary.

        Parameters
        ----------
        obj : ComponentSerializer
            Object to be encoded.
        skip : Optional[List[str]], default=None
            List of field names as strings to skip when forming the
            encoded object. By default, None.
        full_encode : bool, default=False
            Should the data also be JSON encoded (default False). By
            default, False.
        **kwargs : Any
            Any additional key word arguments to be passed to the
            encoder.

        Returns
        -------
        Dict[str, Any]
            Object encoded to dictionary containing all information to
            reform an EasyScience object.
        """

        return self._convert_to_dict(obj, skip=skip, full_encode=full_encode, **kwargs)

    @classmethod
    def decode(cls, d: Dict[str, Any]) -> ComponentSerializer:
        """
        Decode function.

        Parameters
        ----------
        d : Dict[str, Any]
            Dict representation.

        Returns
        -------
        ComponentSerializer
            ComponentSerializer class.
        """

        return BaseEncoderDecoder._convert_from_dict(d)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> ComponentSerializer:
        """
        From dict.

        Parameters
        ----------
        d : Dict[str, Any]
            Dict representation.

        Returns
        -------
        ComponentSerializer
            ComponentSerializer class.
        """
        return BaseEncoderDecoder._convert_from_dict(d)


class DataDictSerializer(DictSerializer):
    """
    This is a serializer that can encode the data in an EasyScience
    object to a JSON encoded dictionary.
    """

    def encode(
        self,
        obj: ComponentSerializer,
        skip: Optional[List[str]] = None,
        full_encode: bool = False,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Convert an EasyScience object to a JSON encoded data dictionary.

        Parameters
        ----------
        obj : ComponentSerializer
            Object to be encoded.
        skip : Optional[List[str]], default=None
            List of field names as strings to skip when forming the
            encoded object. By default, None.
        full_encode : bool, default=False
            Should the data also be JSON encoded (default False). By
            default, False.
        **kwargs : Any
            Any additional key word arguments to be passed to the
            encoder.

        Returns
        -------
        Dict[str, Any]
            Object encoded to data dictionary.

        Raises
        ------
        ValueError
            If ``skip`` is not a list of strings.
        """

        if skip is None:
            skip = []
        elif isinstance(skip, str):
            skip = [skip]
        if not isinstance(skip, list):
            raise ValueError('Skip must be a list of strings.')
        encoded = super().encode(obj, skip=skip, full_encode=full_encode, **kwargs)
        return self._parse_dict(encoded)

    @classmethod
    def decode(cls, d: Dict[str, Any]) -> ComponentSerializer:
        """
        This function is not implemented as a data dictionary does not
        contain the necessary information to re-form an EasyScience
        object.
        """

        raise NotImplementedError(
            'It is not possible to reconstitute objects from data only dictionary.'
        )

    @staticmethod
    def _parse_dict(in_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Strip out any non-data from a dictionary."""

        out_dict = dict()
        for key in in_dict.keys():
            if key[0] == '@':
                if key == '@class' and in_dict[key] not in _KNOWN_CORE_TYPES:
                    out_dict['name'] = in_dict[key]
                continue
            out_dict[key] = in_dict[key]
            if isinstance(in_dict[key], dict):
                out_dict[key] = DataDictSerializer._parse_dict(in_dict[key])
            elif isinstance(in_dict[key], list):
                out_dict[key] = [
                    DataDictSerializer._parse_dict(x) if isinstance(x, dict) else x
                    for x in in_dict[key]
                ]
        return out_dict
