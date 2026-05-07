# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

from abc import ABCMeta
from abc import abstractmethod

from ..legacy.obj_base import ObjBase
from .analysis import AnalysisBase
from .experiment import ExperimentBase
from .theoreticalmodel import TheoreticalModelBase


class JobBase(ObjBase, metaclass=ABCMeta):
    """This virtual class allows for the creation of technique-specific
    Job objects.
    """

    def __init__(self, name: str, *args, **kwargs):
        super(JobBase, self).__init__(name, *args, **kwargs)
        self.name = name
        self._theory = None
        self._experiment = None
        self._analysis = None
        self._summary = None
        self._info = None

    """
    JobBase consists of Theory, Experiment, Analysis virtual classes.
    Summary and Info classes are included to store additional information.
    """

    @property
    def theorerical_model(self):
        return self._theory

    @theorerical_model.setter
    @abstractmethod
    def theoretical_model(self, theory: TheoreticalModelBase):
        raise NotImplementedError('theory setter not implemented')

    @property
    def experiment(self):
        return self._experiment

    @experiment.setter
    @abstractmethod
    def experiment(self, experiment: ExperimentBase):
        raise NotImplementedError('experiment setter not implemented')

    @property
    def analysis(self):
        return self._analysis

    @analysis.setter
    @abstractmethod
    def analysis(self, analysis: AnalysisBase):
        raise NotImplementedError('analysis setter not implemented')

    # TODO: extend derived classes to include Summary and Info
    # @property
    # def summary(self):
    #     return self._summary

    # @summary.setter
    # @abstractmethod
    # def summary(self, summary: SummaryBase):
    #     raise NotImplementedError("summary setter not implemented")

    # @property
    # def info(self):
    #     return self._info

    # @info.setter
    # @abstractmethod
    # def info(self, info: InfoBase):
    #     raise NotImplementedError("info setter not implemented")

    @abstractmethod
    def calculate_theory(self, *args, **kwargs):
        raise NotImplementedError('calculate_theory not implemented')

    @abstractmethod
    def fit(self, *args, **kwargs):
        raise NotImplementedError('fit not implemented')
