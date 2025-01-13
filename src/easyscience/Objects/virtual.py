#  SPDX-FileCopyrightText: 2022 EasyScience contributors  <core@easyscience.software>
#  SPDX-License-Identifier: BSD-3-Clause
#  © 2022 Contributors to the EasyScience project <https://github.com/easyScience/EasyScience>

from __future__ import annotations

__author__ = 'github.com/wardsimon'
__version__ = '0.0.1'

import inspect
import weakref
from copy import deepcopy
from typing import TYPE_CHECKING
from typing import Iterable
from typing import MutableSequence

from easyscience import global_object
from easyscience.Constraints import ObjConstraint

if TYPE_CHECKING:
    from easyscience.Objects.ObjectClasses import BV


def raise_(ex):
    raise ex


def _remover(a_obj_id: str, v_obj_id: str):
    try:
        # Try to get parent object (might be deleted)
        a_obj = global_object.map.get_item_by_key(a_obj_id)
    except ValueError:
        return
    if a_obj._constraints['virtual'].get(v_obj_id, False):
        del a_obj._constraints['virtual'][v_obj_id]


def realizer(obj: BV):
    """
    Convert component which is `Virtual` in a `Virtual` object into a `Real` component.

    :param obj: Virtual object which has the property `component`
    """
    if getattr(obj, '_is_virtual', False):
        klass = getattr(obj, '__non_virtual_class__')
        import easyscience.Objects.Variable as ec_var

        args = []
        if klass in ec_var.__dict__.values():  # is_variable check
            kwargs = obj.encode_data()
            kwargs['unique_name'] = None
            return klass(**kwargs)
        else:
            kwargs = {name: realizer(item) for name, item in obj._kwargs.items()}
            if isinstance(klass, Iterable) or issubclass(klass, MutableSequence):
                for key, value in inspect.signature(klass).parameters.items():
                    if value.kind in [
                        inspect.Parameter.POSITIONAL_ONLY,
                        inspect.Parameter.POSITIONAL_OR_KEYWORD,
                    ]:
                        args.append(getattr(obj, key))
            return klass(*args, **kwargs)
    else:
        return obj


def component_realizer(obj: BV, component: str, recursive: bool = True):
    """
    Convert component which is `Virtual` in a `Virtual` object into a `Real` component.

    :param obj: Virtual object which has the property `component`
    :param component: The name of the component to be converted
    :param recursive: Should we realize all sub-components of the component
    """

    import easyscience.Objects.Variable as ec_var

    done_mapping = True
    if not isinstance(obj, Iterable) or not issubclass(obj.__class__, MutableSequence):
        old_component = obj._kwargs[component]
        new_components = realizer(obj._kwargs[component])
        if hasattr(new_components, 'enabled'):
            new_components.enabled = True
    else:
        old_component = obj[component]
        new_components = realizer(obj[component])
        idx = obj.index(old_component)
        del obj[component]
        obj.insert(idx, new_components)
        done_mapping = False
    if not recursive:
        for key in iter(component):
            if isinstance(key, str):
                value = component._kwargs[key]
            else:
                value = key
                key = value.unique_name
            if getattr(value, '__old_class__', value.__class__) in ec_var.__dict__.values():
                continue
            component._global_object.map.prune_vertex_from_edge(component, component._kwargs[key])
            component._global_object.map.add_edge(component, old_component._kwargs[key])
            component._kwargs[key] = old_component._kwargs[key]
            done_mapping = False
    if done_mapping:
        obj._global_object.map.prune_vertex_from_edge(obj, old_component)
        obj._global_object.map.add_edge(obj, new_components)
        obj._kwargs[component] = new_components


def virtualizer(obj: BV) -> BV:
    """
    Convert a real `EasyScience` object to a virtual object.
    This means that the object returned is an exact copy which is unsettable, unchangeable
    and linked to the parent object.
    The object can be realized and returned as a copy via the `realizer` function. If you need a
    component realized in place then `relalize_component` should be called.

    :param obj:
    :type obj:
    :return:
    :rtype:
    """
    # First  check if we're already a virtual object
    if getattr(obj, '_is_virtual', False):
        new_obj = deepcopy(obj)
        old_obj = obj._global_object.map.get_item_by_key(obj._derived_from)
        constraint = ObjConstraint(new_obj, '', old_obj)
        constraint.external = True
        old_obj._constraints['virtual'][str(obj.unique_name)] = constraint
        new_obj._constraints['builtin'] = dict()
        # setattr(new_obj, "__previous_set", getattr(olobj, "__previous_set", None))
        weakref.finalize(
            new_obj,
            _remover,
            old_obj.unique_name,
            new_obj.unique_name,
        )
        return new_obj

    # The supplied class
    klass = getattr(obj, '__old_class__', obj.__class__)
    virtual_options = {
        '_is_virtual': True,
        'is_virtual': property(fget=lambda self: self._is_virtual),
        '_derived_from': property(fget=lambda self: obj.unique_name),
        '__non_virtual_class__': klass,
        'realize': realizer,
        'relalize_component': component_realizer,
    }

    import easyscience.Objects.variable as ec_var

    if klass in ec_var.__dict__.values():  # is_variable check
        virtual_options['fixed'] = property(
            fget=lambda self: self._fixed,
            fset=lambda self, value: raise_(AttributeError('Virtual parameters cannot be fixed')),
        )
    # Generate a new class
    cls = type('Virtual' + klass.__name__, (klass,), virtual_options)
    # Determine what to do next.
    args = []
    # If `obj` is a parameter or descriptor etc, then simple mods.
    if hasattr(obj, '_constructor'):
        # All Variables are based on the Descriptor.
        d = obj.encode_data()
        if hasattr(d, 'fixed'):
            d['fixed'] = True
        d['unique_name'] = None
        v_p = cls(**d)
        v_p._enabled = False
        constraint = ObjConstraint(v_p, '', obj)
        constraint.external = True
        obj._constraints['virtual'][v_p.unique_name] = constraint
        v_p._constraints['builtin'] = dict()
        setattr(v_p, '__previous_set', getattr(obj, '__previous_set', None))
        weakref.finalize(
            v_p,
            _remover,
            obj.unique_name,
            v_p.unique_name,
        )
    else:
        # In this case, we need to be recursive.
        kwargs = {name: virtualizer(item) for name, item in obj._kwargs.items()}
        if isinstance(klass, Iterable) or issubclass(klass, MutableSequence):
            for key, value in inspect.signature(cls).parameters.items():
                if value.kind in [
                    inspect.Parameter.POSITIONAL_ONLY,
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                ]:
                    args.append(getattr(obj, key))
        v_p = cls(*args, **kwargs)
    return v_p
