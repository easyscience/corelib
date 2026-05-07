# SPDX-FileCopyrightText: 2025 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import Any
from typing import Dict
from typing import List

from .parameter import Parameter


def resolve_all_parameter_dependencies(obj: Any) -> None:
    """
    Recursively find all Parameter objects in an object hierarchy and
    resolve their pending dependencies.

    This function should be called after deserializing a complex object
    that contains Parameters with dependencies to ensure all dependency
    relationships are properly established.

    Parameters
    ----------
    obj : Any
        The object to search for Parameters (can be a single Parameter,
        list, dict, or complex object).

    Raises
    ------
    ValueError
        If one or more pending dependencies cannot be resolved.
    """

    def _collect_parameters(item: Any, parameters: List[Parameter]) -> None:
        """Recursively collect all Parameter objects from an item."""
        if isinstance(item, Parameter):
            parameters.append(item)
        elif isinstance(item, dict):
            for value in item.values():
                _collect_parameters(value, parameters)
        elif isinstance(item, (list, tuple)):
            for element in item:
                _collect_parameters(element, parameters)
        elif hasattr(item, '__dict__'):
            # Check instance attributes
            for attr_name, attr_value in item.__dict__.items():
                if not attr_name.startswith('_'):  # Skip private attributes
                    _collect_parameters(attr_value, parameters)

            # Check class properties (descriptors like Parameter instances)
            for attr_name in dir(type(item)):
                if not attr_name.startswith('_'):  # Skip private attributes
                    class_attr = getattr(type(item), attr_name, None)
                    if isinstance(class_attr, property):
                        try:
                            attr_value = getattr(item, attr_name)
                            _collect_parameters(attr_value, parameters)
                        except (AttributeError, Exception):
                            # log the exception
                            print(f"Error accessing property '{attr_name}' of {item}")
                            # Skip properties that can't be accessed
                            continue

    # Collect all parameters
    all_parameters = []
    _collect_parameters(obj, all_parameters)

    # Resolve dependencies for all parameters that have pending dependencies
    resolved_count = 0
    error_count = 0
    errors = []

    for param in all_parameters:
        if hasattr(param, '_pending_dependency_string'):
            try:
                param.resolve_pending_dependencies()
                resolved_count += 1
            except Exception as e:
                error_count += 1
                serializer_id = getattr(param, '_DescriptorNumber__serializer_id', 'unknown')
                errors.append(
                    f"Failed to resolve dependencies for parameter '{param.name}'"
                    f" (unique_name: '{param.unique_name}', serializer_id: '{serializer_id}'): {e}"
                )

    # Report results
    if resolved_count > 0:
        print(f'Successfully resolved dependencies for {resolved_count} parameter(s).')

    if error_count > 0:
        error_message = (
            f'Failed to resolve dependencies for {error_count} parameter(s):\n' + '\n'.join(errors)
        )
        raise ValueError(error_message)


def get_parameters_with_pending_dependencies(obj: Any) -> List[Parameter]:
    """
    Find all Parameter objects in an object hierarchy that have pending
    dependencies.

    Parameters
    ----------
    obj : Any
        The object to search for Parameters.

    Returns
    -------
    List[Parameter]
        List of Parameters with pending dependencies.
    """
    parameters_with_pending = []

    def _collect_pending_parameters(item: Any) -> None:
        """
        Recursively collect all Parameter objects with pending
        dependencies.
        """
        if isinstance(item, Parameter):
            if hasattr(item, '_pending_dependency_string'):
                parameters_with_pending.append(item)
        elif isinstance(item, dict):
            for value in item.values():
                _collect_pending_parameters(value)
        elif isinstance(item, (list, tuple)):
            for element in item:
                _collect_pending_parameters(element)
        elif hasattr(item, '__dict__'):
            # Check instance attributes
            for attr_name, attr_value in item.__dict__.items():
                if not attr_name.startswith('_'):  # Skip private attributes
                    _collect_pending_parameters(attr_value)

            # Check class properties (descriptors like Parameter instances)
            for attr_name in dir(type(item)):
                if not attr_name.startswith('_'):  # Skip private attributes
                    class_attr = getattr(type(item), attr_name, None)
                    if isinstance(class_attr, property):
                        try:
                            attr_value = getattr(item, attr_name)
                            _collect_pending_parameters(attr_value)
                        except (AttributeError, Exception):
                            # log the exception
                            print(f"Error accessing property '{attr_name}' of {item}")
                            # Skip properties that can't be accessed
                            continue

    _collect_pending_parameters(obj)
    return parameters_with_pending


def deserialize_and_resolve_parameters(
    params_data: Dict[str, Dict[str, Any]],
) -> Dict[str, Parameter]:
    """
    Deserialize parameters from a dictionary and resolve their
    dependencies.

    This is a convenience function that combines Parameter.from_dict()
    deserialization with dependency resolution in a single call.

    Parameters
    ----------
    params_data : Dict[str, Dict[str, Any]]
        Dictionary mapping parameter names to their serialized data.

    Returns
    -------
    Dict[str, Parameter]
        Dictionary mapping parameter names to deserialized Parameters
        with resolved dependencies.
    """
    # Deserialize all parameters first
    new_params = {}
    for name, data in params_data.items():
        new_params[name] = Parameter.from_dict(data)

    # Resolve all dependencies
    resolve_all_parameter_dependencies(new_params)

    return new_params
