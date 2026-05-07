# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

from ..utils.classUtils import singleton
from .hugger.hugger import ScriptManager
from .logger import Logger
from .map import Map


@singleton
class GlobalObject:
    """
    GlobalObject is the assimilated knowledge of ``EasyScience``.

    Every class based on ``EasyScience`` gets brought into the
    collective.
    """

    __log = Logger()
    __map = Map()
    __stack = None
    __debug = False

    def __init__(self):
        # Logger. This is so there's a unified logging interface
        self.log: Logger = self.__log
        # Debug. Global debugging level
        self.debug: bool = self.__debug
        # Stack. This is where the undo/redo operations are stored.
        self.stack = self.__stack
        #
        self.script: ScriptManager = ScriptManager()
        # Map. This is the conduit database between all global object species
        self.map: Map = self.__map

    def instantiate_stack(self):
        """
        The undo/redo stack references the collective.

        Hence it has to be imported after initialization.

        Returns
        -------
        noneType
            None.
        """
        from easyscience.global_object.undo_redo import UndoStack

        self.stack = UndoStack()

    def generate_unique_name(self, name_prefix: str) -> str:
        """
        Generate a generic unique name for the object using the class
        name and a global iterator. Names are in the format
        ``name_prefix_0``, ``name_prefix_1``, ``name_prefix_2``, etc.

        Parameters
        ----------
        name_prefix : str
            The prefix to be used for the name.
        """
        names_with_prefix = [
            name for name in self.map.vertices() if name.startswith(name_prefix + '_')
        ]
        if names_with_prefix:
            name_with_prefix_count = [0]
            for name in names_with_prefix:
                # Strip away the prefix and trailing _
                name_without_prefix = name.replace(name_prefix + '_', '')
                if name_without_prefix.isdecimal():
                    name_with_prefix_count.append(int(name_without_prefix))
            unique_name = f'{name_prefix}_{max(name_with_prefix_count) + 1}'
        else:
            unique_name = f'{name_prefix}_0'
        return unique_name
