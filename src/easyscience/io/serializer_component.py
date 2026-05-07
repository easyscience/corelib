# SPDX-FileCopyrightText: 2025 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any
from typing import Dict
from typing import List
from typing import Optional

from .serializer_dict import SerializerDict

if TYPE_CHECKING:
    from .serializer_base import SerializerBase


class SerializerComponent:
    """This base class adds the capability of saving and loading
    (encoding/decoding, serializing/deserializing) easyscience objects
    via the `encode` and `decode` methods. The default encoder is
    `SerializerDict`, which converts the object to a dictionary.

    Shortcuts for dictionary and encoding is also present.
    """

    def __deepcopy__(self, memo):
        """Deepcopy function."""
        return self.from_dict(self.as_dict())

    def encode(
        self, skip: Optional[List[str]] = None, encoder: Optional[SerializerBase] = None, **kwargs
    ) -> Any:
        """Use an encoder to covert an EasyScience object into another
        format. Default is to a dictionary using `SerializerDict`.

        Parameters
        ----------
        skip : Optional[List[str]], optional
            List of field names as strings to skip when forming the encoded object. By default, None.
        encoder : Optional[SerializerBase], optional
            The encoder to be used for encoding the data. By default, None.
        **kwargs :
            Any additional key word arguments to be passed to the encoder.

        Returns
        -------
        Any
            Encoded object containing all information to reform an EasyScience object.
        """
        if encoder is None:
            encoder = SerializerDict
        encoder_obj = encoder()
        return encoder_obj.encode(self, skip=skip, **kwargs)

    @classmethod
    def decode(cls, obj: Any, decoder: Optional[SerializerBase] = None) -> Any:
        """Re-create an EasyScience object from the output of an
        encoder. The default decoder is `SerializerDict`.

        Parameters
        ----------
        cls :
        obj : Any
            Encoded EasyScience object.
        decoder : Optional[SerializerBase], optional
            Decoder to be used to reform the EasyScience
            object. By default, None.

        Returns
        -------
        Any
            Reformed EasyScience object.
        """

        if decoder is None:
            decoder = SerializerDict
        return decoder.decode(obj)

    def as_dict(self, skip: Optional[List[str]] = None) -> Dict[str, Any]:
        """Convert an EasyScience object into a full dictionary using
        `SerializerDict`. This is a shortcut for
        ```obj.encode(encoder=SerializerDict)```

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

        return self.encode(skip=skip, encoder=SerializerDict)

    @classmethod
    def from_dict(cls, obj_dict: Dict[str, Any]) -> None:
        """Re-create an EasyScience object from a full encoded
        dictionary.

        Parameters
        ----------
        cls :
        obj_dict : Dict[str, Any]
            Dictionary containing the serialized contents (from `SerializerDict`) of an EasyScience object.

        Returns
        -------
        None
            Reformed EasyScience object.
        """

        return cls.decode(obj_dict, decoder=SerializerDict)
