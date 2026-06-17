# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import sys
import xml.etree.ElementTree as ET
from numbers import Number
from typing import TYPE_CHECKING
from typing import Any
from typing import List
from typing import Optional

import numpy as np

from easyscience import global_object

from ..io.dict import DataDictSerializer
from ..io.dict import DictSerializer
from ..io.template import BaseEncoderDecoder

if TYPE_CHECKING:
    from ..io.component_serializer import ComponentSerializer


can_intent = (sys.version_info.major > 2) & (sys.version_info.minor > 8)


class XMLSerializer(BaseEncoderDecoder):
    """
    This is a serializer that can encode and decode EasyScience objects
    to a basic xml format.
    """

    def encode(
        self,
        obj: ComponentSerializer,
        skip: Optional[List[str]] = None,
        data_only: bool = False,
        fast: bool = False,
        use_header: bool = False,
        **kwargs,
    ) -> str:
        """
        Convert an EasyScience object to an XML encoded string.

        Note that for speed the ``fast`` setting can be changed to
        ``True``. An XML document with initial block *data* is returned.

        Parameters
        ----------
        obj : ComponentSerializer
            Object to be encoded.
        skip : Optional[List[str]], default=None
            List of field names as strings to skip when forming the
            encoded object. By default, None.
        data_only : bool, default=False
            Should only the object's data be encoded. By default, False.
        fast : bool, default=False
            Should the returned string be pretty? This can be turned off
            for speed. By default, False.
        use_header : bool, default=False
            Should a header of ``'?xml version="1.0"
            encoding="UTF-8"?'`` be included? By default, False.
        **kwargs :
            Any additional key-words to pass to the Dictionary
            Serializer.

        Returns
        -------
        str
            String containing the XML encoded object.
        """

        if skip is None:
            skip = []
        encoder = DictSerializer
        if data_only:
            encoder = DataDictSerializer
        if isinstance(obj, dict):
            obj_dict = obj
        else:
            obj_dict = encoder().encode(obj, skip=skip, full_encode=True, **kwargs)
        block = ET.Element('data')
        self._check_class(block, None, obj_dict, skip=skip)
        header = ''
        if use_header:
            header = '?xml version="1.0"  encoding="UTF-8"?'
        if not fast and can_intent:
            ET.indent(block)
            if use_header:
                header += '\n'
        return header + ET.tostring(block, encoding='unicode')

    @classmethod
    def decode(cls, data: str) -> ComponentSerializer:
        """
        Decode an EasyScience object which has been encoded in XML
        format.

        Parameters
        ----------
        data : str
            String containing XML encoded data.

        Returns
        -------
        ComponentSerializer
            Reformed EasyScience object.
        """

        data_xml = ET.XML(data)
        out_dict = {}
        for element in data_xml:
            XMLSerializer._element_to_dict(element, out_dict)
        return DictSerializer.decode(out_dict)

    @staticmethod
    def _element_to_dict(element, out_dict):
        """Convert an XML element to a dictionary recursively."""

        label = element.tag
        if label[0] == '_':
            label = '@' + label[2:]
        if len(element) == 0:
            out_dict[label] = XMLSerializer.string_to_variable(element.text)
        else:
            this_dict = {}
            for el in element:
                XMLSerializer._element_to_dict(el, this_dict)
            if label in out_dict.keys():
                # This object is a list. Create a list.
                old_value = out_dict[label]
                if not isinstance(old_value, list):
                    old_value = [old_value]
                old_value.append(this_dict)
                this_dict = old_value
            out_dict[label] = this_dict

    @staticmethod
    def string_to_variable(in_string: str):
        """Convert an XML encoded string to JSON form."""
        if in_string is None:
            return in_string
        in_string = in_string.strip()
        if "'" in in_string:
            in_string = in_string.replace("'", '')
        if '"' in in_string:
            in_string = in_string.replace('"', '')
        try:
            value = float(in_string)
        except ValueError:
            if in_string == 'True':
                value = True
            elif in_string == 'False':
                value = False
            elif in_string == 'None':
                value = None
            else:
                value = in_string
        return value

    def _check_class(self, element, key: str, value: Any, skip: Optional[List[str]] = None):
        """
        Add a value to an element or create a new element based on input
        type.
        """
        T_ = type(value)
        if isinstance(value, dict):
            for k, v in value.items():
                if k in skip:
                    continue
                kk = k
                if k[0] == '@':
                    kk = '__' + kk[1:]
                if not isinstance(v, list):
                    s = ET.SubElement(element, kk)
                    self._check_class(s, kk, v, skip=skip)
                else:
                    self._check_class(element, kk, v, skip=skip)
        elif isinstance(value, bool):
            element.text = str(value)
        elif isinstance(value, str):
            element.text = value
        elif isinstance(value, list):
            for i in value:
                s = ET.SubElement(element, key)
                self._check_class(s, None, i, skip=skip)
        elif value is None:
            element.text = 'None'
        elif issubclass(T_, Number):
            element.text = str(value)
        elif issubclass(T_, np.ndarray):
            element.text = str(value.tolist())
        else:
            global_object.log.getLogger('legacy.xml').error('Cannot encode %s to XML', T_)
            raise NotImplementedError
