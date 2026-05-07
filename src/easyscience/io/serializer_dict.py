# SPDX-FileCopyrightText: 2025 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

__author__ = 'https://github.com/materialsvirtuallab/monty/blob/master/monty/json.py'
__version__ = '3.0.0'


from typing import TYPE_CHECKING
from typing import Dict
from typing import List
from typing import Optional

from .serializer_base import SerializerBase

if TYPE_CHECKING:
    from .serializer_component import SerializerComponent


class SerializerDict(SerializerBase):
    """
    This is a serializer that can encode and decode EasyScience objects
    to and from a dictionary.
    """

    def encode(
        self,
        obj: SerializerComponent,
        skip: Optional[List[str]] = None,
        full_encode: bool = False,
        **kwargs,
    ):
        """
        Convert an EasyScience object to a dictionary.

        Parameters
        ----------
        obj : SerializerComponent
            Object to be encoded.
        skip : Optional[List[str]], default=None
            List of field names as strings to skip when forming the
            encoded object. By default, None.
        full_encode : bool, default=False
            Should the data also be encoded (default False). By default,
            False.
        **kwargs :
            Any additional key word arguments to be passed to the
            encoder.

        Returns
        -------

            Object encoded to dictionary containing all information to
            reform an EasyScience object.
        """

        return self._convert_to_dict(obj, skip=skip, full_encode=full_encode, **kwargs)

    @classmethod
    def decode(cls, d: Dict) -> SerializerComponent:
        """
        Re-create an EasyScience object from the dictionary
        representation.

        Parameters
        ----------
        cls :
        d : Dict
            Dict representation of an EasyScience object.

        Returns
        -------
        SerializerComponent
            EasyScience object.
        """

        return SerializerBase._convert_from_dict(d)
