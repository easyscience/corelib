# SPDX-FileCopyrightText: 2025 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

from easyscience.variable.descriptor_number import DescriptorNumber

if TYPE_CHECKING:
    from typing import Any
    from typing import Dict
    from typing import List
    from typing import Optional

from ..io import SerializerBase
from ..variable import Parameter
from ..variable.descriptor_base import DescriptorBase
from .new_base import NewBase


class ModelBase(NewBase):
    """
    This is the base class for all model classes in EasyScience. It
    provides methods to get parameters for fitting and analysis as well
    as proper serialization/deserialization for
    DescriptorNumber/Parameter attributes.

    It assumes that Parameters/DescriptorNumbers are assigned as
    properties with the getters returning the parameter but the setter
    only setting the value of the parameter. e.g.
    ```python
    @property
    def my_param(self) -> Parameter:
        '''My param.'''
        return self._my_param


    @my_param.setter
    def my_param(self, new_value: float) -> None:
        '''My param.'''
        self._my_param.value = new_value
    ```
    """

    def __init__(self, unique_name: Optional[str] = None, display_name: Optional[str] = None):
        super().__init__(unique_name=unique_name, display_name=display_name)

    def get_all_variables(self) -> List[DescriptorBase]:
        """
        Get all ``Descriptor`` and ``Parameter`` objects as a list.

        Returns
        -------
        List[DescriptorBase]
            List of ``Descriptor`` and ``Parameter`` objects.
        """
        vars = []
        for attr_name in dir(self):
            attr = getattr(self, attr_name)
            if isinstance(attr, DescriptorBase):
                vars.append(attr)
            elif hasattr(attr, 'get_all_variables'):
                vars += attr.get_all_variables()
        return vars

    def get_all_parameters(self) -> List[Parameter]:
        """
        Get all ``Parameter`` objects as a list.

        Returns
        -------
        List[Parameter]
            List of ``Parameter`` objects.
        """
        return [param for param in self.get_all_variables() if isinstance(param, Parameter)]

    def get_fittable_parameters(self) -> List[Parameter]:
        """
        Get all parameters which can be fitted as a list.

        Returns
        -------
        List[Parameter]
            List of ``Parameter`` objects.
        """
        return [param for param in self.get_all_parameters() if param.independent]

    def get_free_parameters(self) -> List[Parameter]:
        """
        Get all parameters which are currently free to be fitted as a
        list.

        Returns
        -------
        List[Parameter]
            List of ``Parameter`` objects.
        """
        return [param for param in self.get_fittable_parameters() if not param.fixed]

    def get_fit_parameters(self) -> List[Parameter]:
        """
        This is an alias for ``get_free_parameters``.

        To be removed when fully moved to new base classes and minimizer
        can be changed.
        """
        return self.get_free_parameters()

    @classmethod
    def from_dict(cls, obj_dict: Dict[str, Any]) -> ModelBase:
        """
        Re-create an EasyScience object with DescriptorNumber attributes
        from a full encoded dictionary.

        Parameters
        ----------
        obj_dict : Dict[str, Any]
            Dictionary containing the serialized contents (from
            ``SerializerDict``) of an EasyScience object.

        Returns
        -------
        ModelBase
            Reformed EasyScience object.

        Raises
        ------
        SyntaxError
            If a deserialized parameter cannot be attached back to the
            class definition.
        ValueError
            If the input dictionary does not describe the expected
            class.
        """
        if not SerializerBase._is_serialized_easyscience_object(obj_dict):
            raise ValueError('Input must be a dictionary representing an EasyScience object.')
        if obj_dict['@class'] == cls.__name__:
            kwargs = SerializerBase.deserialize_dict(obj_dict)
            parameter_placeholder = {}
            for key, value in kwargs.items():
                if isinstance(value, DescriptorNumber):
                    parameter_placeholder[key] = value
                    kwargs[key] = value.value
            cls_instance = cls(**kwargs)
            for key, value in parameter_placeholder.items():
                try:
                    temp_param = getattr(cls_instance, key)
                    setattr(cls_instance, '_' + key, value)
                    cls_instance._global_object.map.prune(temp_param.unique_name)
                except Exception as e:
                    raise SyntaxError(f"""Could not set parameter {key} during `from_dict` with full deserialized variable. \n'
                            This should be fixed in the class definition. Error: {e}""") from e
            return cls_instance
        else:
            raise ValueError(
                f'Class name in dictionary does not match the expected class: {cls.__name__}.'
            )
