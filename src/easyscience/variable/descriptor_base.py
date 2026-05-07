# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import abc
from typing import Any
from typing import Optional

from easyscience import global_object
from easyscience.global_object.undo_redo import property_stack
from easyscience.io import SerializerComponent


class DescriptorBase(SerializerComponent, metaclass=abc.ABCMeta):
    """
    This is the base of all variable descriptions for models.

    It contains all information to describe a single unique property of
    an object. This description includes a name and value as well as
    optionally a unit, description and url (for reference material).
    Also implemented is a callback so that the value can be read/set
    from a linked library object.

    A ``Descriptor`` is typically something which describes part of a
    model and is non-fittable and generally changes the state of an
    object.
    """

    _global_object = global_object
    # Used by serializer
    _REDIRECT = {'parent': None}

    def __init__(
        self,
        name: str,
        unique_name: Optional[str] = None,
        description: Optional[str] = None,
        url: Optional[str] = None,
        display_name: Optional[str] = None,
        parent: Optional[Any] = None,
    ):
        """
        This is the base of variables for models.

        It contains all information to describe a single unique property
        of an object. This description includes a name, description and
        url (for reference material).

        A ``Descriptor`` is typically something which describes part of
        a model and is non-fittable and generally changes the state of
        an object.

        Parameters
        ----------
        name : str
            Name of this object.
        unique_name : Optional[str], default=None
            Unique identifier for this object. By default, None.
        description : Optional[str], default=None
            A brief summary of what this object is. By default, None.
        url : Optional[str], default=None
            Lookup url for documentation/information. By default, None.
        display_name : Optional[str], default=None
            A pretty name for the object. By default, None.
        parent : Optional[Any], default=None
            The object which this descriptor is attached to. By default,
            None.

        Raises
        ------
        TypeError
            If ``name`` is not a string or if any optional string field
            has an invalid type.
        """

        if unique_name is None:
            unique_name = global_object.generate_unique_name(self.__class__.__name__)
        self._unique_name = unique_name

        if not isinstance(name, str):
            raise TypeError('Name must be a string')
        self._name: str = name

        if display_name is not None and not isinstance(display_name, str):
            raise TypeError('Display name must be a string or None')
        self._display_name: str = display_name

        if description is not None and not isinstance(description, str):
            raise TypeError('Description must be a string or None')
        if description is None:
            description = ''
        self._description: str = description

        if url is not None and not isinstance(url, str):
            raise TypeError('url must be a string')
        if url is None:
            url = ''
        self._url: str = url

        # Let the collective know we've been assimilated
        self._parent = parent
        global_object.map.add_vertex(self, obj_type='created')
        # Make the connection between self and parent
        if parent is not None:
            global_object.map.add_edge(parent, self)

    @property
    def name(self) -> str:
        """
        Get the name of the object.

        Returns
        -------
        str
            Name of the object.
        """
        return self._name

    @name.setter
    @property_stack
    def name(self, new_name: str) -> None:
        """
        Set the name.

        Parameters
        ----------
        new_name : str
            Name of the object.

        Raises
        ------
        TypeError
            If ``new_name`` is not a string.
        """
        if not isinstance(new_name, str):
            raise TypeError('Name must be a string')
        self._name = new_name

    @property
    def display_name(self) -> str:
        """
        Get a pretty display name.

        Returns
        -------
        str
            The pretty display name.
        """
        display_name = self._display_name
        if display_name is None:
            display_name = self._name
        return display_name

    @display_name.setter
    @property_stack
    def display_name(self, name: Optional[str]) -> None:
        """
        Set the pretty display name.

        Parameters
        ----------
        name : Optional[str]
            Pretty display name of the object.

        Raises
        ------
        TypeError
            If ``name`` is neither a string nor ``None``.
        """
        if name is not None and not isinstance(name, str):
            raise TypeError('Display name must be a string or None')
        self._display_name = name

    @property
    def description(self) -> str:
        """
        Get the description of the object.

        Returns
        -------
        str
            Description of the object.
        """
        return self._description

    @description.setter
    def description(self, description: Optional[str]) -> None:
        """
        Set the description of the object.

        Parameters
        ----------
        description : Optional[str]
            Description of the object.

        Raises
        ------
        TypeError
            If ``description`` is neither a string nor ``None``.
        """
        if description is not None and not isinstance(description, str):
            raise TypeError('Description must be a string or None')
        self._description = description

    @property
    def url(self) -> str:
        """
        Get the url of the object.

        Returns
        -------
        str
            Url of the object.
        """
        return self._url

    @url.setter
    def url(self, url: Optional[str]) -> None:
        """
        Set the url of the object.

        Parameters
        ----------
        url : Optional[str]
            Url of the object.

        Raises
        ------
        TypeError
            If ``url`` is neither a string nor ``None``.
        """
        if url is not None and not isinstance(url, str):
            raise TypeError('url must be a string')
        self._url = url

    @property
    def unique_name(self) -> str:
        """
        Get the unique name of this object.

        Returns
        -------
        str
            Unique name of this object.
        """
        return self._unique_name

    @unique_name.setter
    def unique_name(self, new_unique_name: str):
        """
        Set a new unique name for the object.

        The old name is still kept in the map.

        Parameters
        ----------
        new_unique_name : str
            New unique name for the object.

        Raises
        ------
        TypeError
            If ``new_unique_name`` is not a string.
        """
        if not isinstance(new_unique_name, str):
            raise TypeError('Unique name has to be a string.')
        self._unique_name = new_unique_name
        global_object.map.add_vertex(self)

    @property
    @abc.abstractmethod
    def value(self) -> Any:
        """Get the value of the object."""

    @value.setter
    @abc.abstractmethod
    def value(self, value: Any) -> None:
        """Set the value of the object."""

    @abc.abstractmethod
    def __repr__(self) -> str:
        """Return printable representation of the object."""

    def __copy__(self) -> DescriptorBase:
        """Return a copy of the object."""
        temp = self.as_dict(skip=['unique_name'])
        new_obj = self.__class__.from_dict(temp)
        return new_obj
