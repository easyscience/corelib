# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

from ..legacy.obj_base import ObjBase


class TheoreticalModelBase(ObjBase):
    """This virtual class allows for the creation of technique-specific
    Theory objects.
    """

    def __init__(self, name: str, *args, **kwargs):
        self._name = name
        super().__init__(name, *args, **kwargs)

    # required dunder methods
    def __str__(self):
        raise NotImplementedError('Copy not implemented')

    def as_dict(self, skip: list = []) -> dict:
        this_dict = super().as_dict(skip=skip)
        return this_dict
