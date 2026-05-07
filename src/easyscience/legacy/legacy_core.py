# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import json
from collections import OrderedDict
from hashlib import sha1
from typing import TYPE_CHECKING
from typing import Any
from typing import Dict
from typing import List
from typing import Optional

from ..io.dict import DataDictSerializer
from ..io.dict import DictSerializer
from ..io.json import jsanitize

if TYPE_CHECKING:
    from ..io.template import BaseEncoderDecoder


class ComponentSerializer:
    """
    This is the base class for all EasyScience objects and deals with
    the data conversion to other formats via the ``encode`` and
    ``decode`` functions.

    Shortcuts for dictionary and data dictionary encoding is also
    present.
    """

    _CORE = True

    def __deepcopy__(self, memo):
        """Deepcopy function."""
        return self.from_dict(self.as_dict())

    def encode(
        self,
        skip: Optional[List[str]] = None,
        encoder: Optional[BaseEncoderDecoder] = None,
        **kwargs,
    ) -> Any:
        """
        Use an encoder to covert an EasyScience object into another
        format. Default is to a dictionary using ``DictSerializer``.

        Parameters
        ----------
        skip : Optional[List[str]], default=None
            List of field names as strings to skip when forming the
            encoded object. By default, None.
        encoder : Optional[BaseEncoderDecoder], default=None
            The encoder to be used for encoding the data. By default,
            None.
        **kwargs :
            Any additional key word arguments to be passed to the
            encoder.

        Returns
        -------
        Any
            Encoded object containing all information to reform an
            EasyScience object.
        """
        if encoder is None:
            encoder = DictSerializer
        encoder_obj = encoder()
        return encoder_obj.encode(self, skip=skip, **kwargs)

    @classmethod
    def decode(cls, obj: Any, decoder: Optional[BaseEncoderDecoder] = None) -> Any:
        """
        Re-create an EasyScience object from the output of an encoder.
        The default decoder is ``DictSerializer``.

        Parameters
        ----------
        cls :
        obj : Any
            Encoded EasyScience object.
        decoder : Optional[BaseEncoderDecoder], default=None
            Decoder to be used to reform the EasyScience object. By
            default, None.

        Returns
        -------
        Any
            Reformed EasyScience object.
        """

        if decoder is None:
            decoder = DictSerializer
        return decoder.decode(obj)

    def as_dict(self, skip: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Convert an EasyScience object into a full dictionary using
        ``DictSerializer``. This is a shortcut for
        ```obj.encode(encoder=DictSerializer)```

        Parameters
        ----------
        skip : Optional[List[str]], optional
            List of field names as strings to skip when forming
            the dictionary. By default, None.

        Returns
        -------
        Dict[str, Any]
            Encoded object containing all information to reform an
            EasyScience object.
        """

        return self.encode(skip=skip, encoder=DictSerializer)

    @classmethod
    def from_dict(cls, obj_dict: Dict[str, Any]) -> None:
        """
        Re-create an EasyScience object from a full encoded dictionary.

        Parameters
        ----------
        cls :
        obj_dict : Dict[str, Any]
            Dictionary containing the serialized contents (from
            ``DictSerializer``) of an EasyScience object.

        Returns
        -------
        None
            Reformed EasyScience object.
        """

        return cls.decode(obj_dict, decoder=DictSerializer)

    def encode_data(
        self,
        skip: Optional[List[str]] = None,
        encoder: Optional[BaseEncoderDecoder] = None,
        **kwargs,
    ) -> Any:
        """
        Returns just the data in an EasyScience object win the format
        specified by an encoder.

        Parameters
        ----------
        skip : Optional[List[str]], default=None
            List of field names as strings to skip when forming the
            dictionary. By default, None.
        encoder : Optional[BaseEncoderDecoder], default=None
            The encoder to be used for encoding the data. By default,
            None.
        **kwargs :
            Any additional keywords to pass to the encoder when
            encoding.

        Returns
        -------
        Any
            Encoded object containing just the data of an EasyScience
            object.
        """

        if encoder is None:
            encoder = DataDictSerializer
        return self.encode(skip=skip, encoder=encoder, **kwargs)

    def as_data_dict(self, skip: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Returns a dictionary containing just the data of an EasyScience
        object.

        Parameters
        ----------
        skip : Optional[List[str]], default=None
            List of field names as strings to skip when forming the
            dictionary. By default, None.

        Returns
        -------
        Dict[str, Any]
            Dictionary containing just the data of an EasyScience
            object.
        """

        return self.encode(skip=skip, encoder=DataDictSerializer)

    def unsafe_hash(self) -> sha1:
        """
        Returns an hash of the current object.

        This uses a generic but low performance method of converting the
        object to a dictionary, flattening any nested keys, and then
        performing a hash on the resulting object
        """

        def flatten(obj, seperator='.'):
            """Flatten function."""
            # Flattens a dictionary

            flat_dict = {}
            for key, value in obj.items():
                if isinstance(value, dict):
                    flat_dict.update({
                        seperator.join([key, _key]): _value
                        for _key, _value in flatten(value).items()
                    })
                elif isinstance(value, list):
                    list_dict = {
                        '{}{}{}'.format(key, seperator, num): item
                        for num, item in enumerate(value)
                    }
                    flat_dict.update(flatten(list_dict))
                else:
                    flat_dict[key] = value

            return flat_dict

        ordered_keys = sorted(flatten(jsanitize(self.as_dict())).items(), key=lambda x: x[0])
        ordered_keys = [item for item in ordered_keys if '@' not in item[0]]
        return sha1(json.dumps(OrderedDict(ordered_keys)).encode('utf-8'))  # noqa: S324
