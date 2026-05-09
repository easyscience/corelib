# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

import copy
import warnings
from typing import Callable
from typing import List

import numpy as np
from bumps.fitters import FIT_AVAILABLE_IDS
from bumps.fitters import FITTERS
from bumps.fitters import FitDriver
from bumps.names import Curve
from bumps.names import FitProblem
from bumps.parameter import Parameter as BumpsParameter

# causes circular import when Parameter is imported
# from easyscience.base_classes import ObjBase
from easyscience.variable import Parameter

from ..available_minimizers import AvailableMinimizers
from .bumps_utils import BumpsProgressMonitor
from .bumps_utils import EvalCounter
from .minimizer_base import MINIMIZER_PARAMETER_PREFIX
from .minimizer_base import MinimizerBase
from .utils import FitError
from .utils import FitResults

FIT_AVAILABLE_IDS_FILTERED = copy.copy(FIT_AVAILABLE_IDS)
# Considered experimental
FIT_AVAILABLE_IDS_FILTERED.remove('pt')


class Bumps(MinimizerBase):
    """
    This is a wrapper to Bumps: https://bumps.readthedocs.io/
    It allows for the Bumps fitting engine to use parameters declared in an `EasyScience.base_classes.ObjBase`.
    """

    package = 'bumps'

    def __init__(
        self,
        obj,  #: ObjBase,
        fit_function: Callable,
        minimizer_enum: AvailableMinimizers | None = None,
    ):  # todo after constraint changes, add type hint: obj: ObjBase  # noqa: E501
        """Initialize the fitting engine with a `ObjBase` and an
        arbitrary fitting function.

        :param obj: Object containing elements of the `Parameter` class
        :type obj: ObjBase
        :param fit_function: function that when called returns y values. 'x' must be the first
                            and only positional argument. Additional values can be supplied by
                            keyword/value pairs
        :type fit_function: Callable
        """
        super().__init__(obj=obj, fit_function=fit_function, minimizer_enum=minimizer_enum)
        self._p_0 = {}
        self._eval_counter: EvalCounter | None = None

    @staticmethod
    def all_methods() -> List[str]:
        return FIT_AVAILABLE_IDS_FILTERED

    @staticmethod
    def supported_methods() -> List[str]:
        # only a small subset
        methods = ['amoeba', 'newton', 'lm']
        return methods

    def fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
        weights: np.ndarray,
        model: Callable | None = None,
        parameters: List[Parameter] | None = None,
        method: str | None = None,
        tolerance: float | None = None,
        max_evaluations: int | None = None,
        progress_callback: Callable[[dict], bool | None] | None = None,
        abort_test: Callable[[], bool] | None = None,
        minimizer_kwargs: dict | None = None,
        engine_kwargs: dict | None = None,
        **kwargs,
    ) -> FitResults:
        """Perform a fit using the BUMPS engine.

        :param x: points to be calculated at
        :type x: np.ndarray
        :param y: measured points
        :type y: np.ndarray
        :param weights: Weights for supplied measured points
        :type weights: np.ndarray
        :param model: Optional Model which is being fitted to
        :param parameters: Optional parameters for the fit
        :type parameters: List[BumpsParameter]
        :param method: Method for minimization
        :type method: str
        :param max_evaluations: Maximum number of optimizer steps. Forwarded to BUMPS as
            its ``steps`` parameter. If ``None``, the default value defined by the
            selected BUMPS fitter (``fitclass.settings``) is used.
        :type max_evaluations: int | None
        :param progress_callback: Optional callback for progress updates. The payload
            field ``iteration`` carries the BUMPS optimizer step index.
        :type progress_callback: Callable

        .. note::
            The :class:`FitResults` field ``n_evaluations`` reports the number of
            **objective-function evaluations** consumed by the fit, for cross-backend
            consistency with LMFit (``nfev``) and DFO-LS (``nf``). For BUMPS this is
            distinct from the optimizer **step count** that ``max_evaluations`` (i.e.
            BUMPS ``steps``) is budgeted against and returned as
            :class:`FitResults.iterations`; a single step may trigger several objective
            evaluations, so ``n_evaluations`` can legitimately exceed
            ``max_evaluations``. The budget-exhaustion check is performed against
            ``iterations``, not ``n_evaluations``.

        :return: Fit results
        :rtype: FitResults
        """
        method_dict = self._get_method_kwargs(method)

        x, y, weights = np.asarray(x), np.asarray(y), np.asarray(weights)

        if y.shape != x.shape:
            raise ValueError('x and y must have the same shape.')

        if weights.shape != x.shape:
            raise ValueError('Weights must have the same shape as x and y.')

        if not np.isfinite(weights).all():
            raise ValueError('Weights cannot be NaN or infinite.')

        if (weights <= 0).any():
            raise ValueError('Weights must be strictly positive and non-zero.')

        if engine_kwargs is None:
            engine_kwargs = {}

        if minimizer_kwargs is None:
            minimizer_kwargs = {}
        minimizer_kwargs.update(engine_kwargs)

        method_str = method_dict.get('method', self._method)
        fitclass = self._resolve_fitclass(method_str)

        # Resolve BUMPS-native defaults so the budget reported back to the caller (and
        # used by the budget-exhaustion check in `_gen_fit_results`) reflects the values
        # actually consumed by the fitter, even when the caller passes None.
        fitter_settings = dict(fitclass.settings)
        if max_evaluations is None:
            max_evaluations = fitter_settings.get('steps')
        if tolerance is None:
            ftol = fitter_settings.get('ftol')
            xtol = fitter_settings.get('xtol')
            tols = [t for t in (ftol, xtol) if t is not None]
            tolerance = min(tols) if tols else None

        if tolerance is not None:
            minimizer_kwargs['ftol'] = tolerance  # tolerance for change in function value
            minimizer_kwargs['xtol'] = (
                tolerance  # tolerance for change in parameter value, could be an independent value
            )
        if max_evaluations is not None:
            minimizer_kwargs['steps'] = max_evaluations

        if model is None:
            model_function = self._make_model(parameters=parameters)
            model = model_function(x, y, weights)
        self._cached_model = model

        self._p_0 = {f'p{key}': self._cached_pars[key].value for key in self._cached_pars.keys()}

        problem = FitProblem(model)

        monitors = []
        if progress_callback is not None:
            if not callable(progress_callback):
                raise ValueError('progress_callback must be callable')
            monitors.append(
                BumpsProgressMonitor(problem, progress_callback, self._build_progress_payload)
            )

        driver = FitDriver(
            fitclass=fitclass,
            problem=problem,
            monitors=monitors,
            abort_test=abort_test or (lambda: False),
            **minimizer_kwargs,
            **kwargs,
        )
        driver.clip()

        # Why do we do this? Because a fitting template has to have global_object instantiated outside pre-runtime
        from easyscience import global_object

        stack_status = global_object.stack.enabled
        global_object.stack.enabled = False

        try:
            # Drive the fit through the local FitDriver instance so the supplied
            # `monitors` (including the optional progress callback monitor) are
            # invoked. `bumps.fitters.fit` constructs its own driver.
            x, fx = driver.fit()
            from scipy.optimize import OptimizeResult

            # BUMPS' `MonitorRunner.history.step` is populated by the driver itself
            # (independently of any user-supplied monitors) and exposes the canonical
            # last-step index reached by the fitter, so we use it as `nit`.
            history_step = getattr(getattr(driver, 'monitor_runner', None), 'history', None)
            nit_value = int(history_step.step[0]) if history_step is not None else None
            model_results = OptimizeResult(
                x=x,
                dx=driver.stderr(),
                fun=fx,
                success=True,
                status=0,
                message='successful termination',
                nit=nit_value,
            )
            model_results.state = driver.fitter.state
            self._set_parameter_fit_result(model_results, stack_status, problem._parameters)
            results = self._gen_fit_results(
                model_results,
                max_evaluations=max_evaluations,
                tolerance=tolerance,
            )
        except Exception as e:
            self._restore_parameter_values()
            raise FitError(e)
        finally:
            global_object.stack.enabled = stack_status
        return results

    @staticmethod
    def _resolve_fitclass(method: str):
        for fitclass in FITTERS:
            if fitclass.id == method:
                return fitclass
        raise FitError(f'Unknown BUMPS fitting method: {method}')

    def _build_progress_payload(
        self, problem, iteration: int, point: np.ndarray, nllf: float
    ) -> dict:
        # Use the nllf already computed by the fitter to avoid a costly
        # model re-evaluation, and let BUMPS apply its own chisq scaling.
        chi2 = float(problem.chisq(nllf=nllf, norm=False))
        reduced_chi2 = float(problem.chisq(nllf=nllf, norm=True))

        parameter_values = self._current_parameter_snapshot(problem, point)

        return {
            'iteration': iteration,
            'chi2': chi2,
            'reduced_chi2': reduced_chi2,
            'parameter_values': parameter_values,
            'refresh_plots': False,
            'finished': False,
        }

    def _current_parameter_snapshot(self, problem, point: np.ndarray) -> dict:
        labels = problem.labels()
        values = problem.getp() if point is None else point
        snapshot = {}
        for label, value in zip(labels, values):
            dict_name = label[len(MINIMIZER_PARAMETER_PREFIX) :]
            snapshot[dict_name] = float(value)
        return snapshot

    def convert_to_pars_obj(self, par_list: List[Parameter] | None = None) -> List[BumpsParameter]:
        """Create a container with the `Parameters` converted from the
        base object.

        :param par_list: If only a single/selection of parameter is
            required. Specify as a list
        :type par_list: List[str]
        :return: bumps Parameters list
        :rtype: List[BumpsParameter]
        """
        if par_list is None:
            # Assume that we have a ObjBase for which we can obtain a list
            par_list = self._object.get_fit_parameters()
        pars_obj = [self.__class__.convert_to_par_object(obj) for obj in par_list]
        return pars_obj

    # For some reason I have to double staticmethod :-/
    @staticmethod
    def convert_to_par_object(obj) -> BumpsParameter:
        """Convert an `EasyScience.variable.Parameter` object to a bumps
        Parameter object.

        :return: bumps Parameter compatible object.
        :rtype: BumpsParameter
        """

        value = obj.value

        return BumpsParameter(
            name=MINIMIZER_PARAMETER_PREFIX + obj.unique_name,
            value=value,
            bounds=[obj.min, obj.max],
            fixed=obj.fixed,
        )

    def _make_model(self, parameters: List[BumpsParameter] | None = None) -> Callable:
        """Generate a bumps model from the supplied `fit_function` and
        parameters in the base object. Note that this makes a callable
        as it needs to be initialized with *x*, *y*, *weights*

        Weights are converted to dy (standard deviation of y).

        :return: Callable to make a bumps Curve model
        :rtype: Callable
        """
        fit_func = EvalCounter(self._generate_fit_function())
        self._eval_counter = fit_func

        def _outer(obj):
            def _make_func(x, y, weights):
                bumps_pars = {}
                if not parameters:
                    for name, par in obj._cached_pars.items():
                        bumps_pars[MINIMIZER_PARAMETER_PREFIX + str(name)] = (
                            obj.convert_to_par_object(par)
                        )
                else:
                    for par in parameters:
                        bumps_pars[MINIMIZER_PARAMETER_PREFIX + par.unique_name] = (
                            obj.convert_to_par_object(par)
                        )
                return Curve(fit_func, x, y, dy=1 / weights, **bumps_pars)

            return _make_func

        return _outer(self)

    def sample(
        self,
        x: np.ndarray,
        y: np.ndarray,
        weights: np.ndarray,
        samples: int = 10000,
        burn: int = 2000,
        thin: int = 10,
        chains: int | None = None,
        population: int | None = None,
        seed: int | None = None,
        sampler_kwargs: dict | None = None,
        progress_callback: Callable[[dict], bool | None] | None = None,
        abort_test: Callable[[], bool] | None = None,
    ) -> dict:
        """Run Bayesian MCMC sampling using the BUMPS DREAM sampler.

        Builds a BUMPS :class:`~bumps.names.FitProblem` from the current
        model and runs the DREAM sampler.  This is the public minimizer-level
        entry point for Bayesian sampling; the higher-level
        :meth:`easyscience.fitting.multi_fitter.MultiFitter.sample` delegates
        to this method after flattening multi-dataset arrays.

        :param x: Flattened independent variable array.
        :param y: Flattened dependent variable array.
        :param weights: Flattened weight array.
        :param samples: Number of retained DREAM samples requested from BUMPS.
        :param burn: Burn-in steps.
        :param thin: Thinning interval.
        :param chains: User-friendly alias for BUMPS DREAM population count.
        :param population: BUMPS DREAM population count for advanced users.
        :param seed: Best-effort random seed.
        :param sampler_kwargs: Additional keyword arguments forwarded to
            :func:`bumps.fitters.fit`.
        :param progress_callback: Optional callback for progress updates during
            sampling.  The payload dict includes ``iteration`` (DREAM generation
            number) and ``sampling: True``.
        :return: Dictionary with keys ``'draws'``, ``'param_names'``, ``'state'`",
            and ``'logp'``.
        """
        from bumps.fitters import DreamFit
        from bumps.names import FitProblem

        # Build the BUMPS Curve model using the minimizer's existing machinery
        model_func = self._make_model()
        x_flat = np.linspace(0, y.size - 1, y.size)
        curve = model_func(x_flat, y, weights)
        problem = FitProblem(curve)

        # Best-effort seed: sets numpy's global RNG state just before DREAM starts.
        if seed is not None:
            np.random.seed(seed)

        # Resolve population parameter
        if chains is not None and population is not None:
            if chains != population:
                raise ValueError(
                    f'Conflicting population arguments: chains={chains}, population={population}. '
                    'Only provide one.'
                )
            pop = chains
        elif chains is not None:
            pop = chains
        elif population is not None:
            pop = population
        else:
            pop = None

        # Build DREAM kwargs
        dream_kwargs: dict = {'samples': samples, 'burn': burn, 'thin': thin}
        if pop is not None:
            dream_kwargs['pop'] = pop
        if sampler_kwargs:
            dream_kwargs.update(sampler_kwargs)

        # Build monitors (same pattern as classical Bumps.fit())
        monitors = []
        if progress_callback is not None:
            if not callable(progress_callback):
                raise ValueError('progress_callback must be callable')
            # Compute total DREAM steps for progress display (burn + sampling generations)
            pop_val = pop if pop else 10
            _total_steps = burn + (samples + pop_val - 1) // pop_val
            monitors.append(
                BumpsProgressMonitor(
                    problem,
                    progress_callback,
                    lambda problem, iteration, point, nllf: {
                        **self._build_sample_progress_payload(problem, iteration, point, nllf),
                        'total_steps': _total_steps,
                    },
                )
            )

        driver = FitDriver(
            fitclass=DreamFit,
            problem=problem,
            monitors=monitors,
            abort_test=abort_test or (lambda: False),
            **dream_kwargs,
        )
        driver.clip()

        from easyscience import global_object

        stack_status = global_object.stack.enabled
        global_object.stack.enabled = False

        try:
            x_opt, fx = driver.fit()
            result_state = getattr(driver.fitter, 'state', None)
            if result_state is None:
                raise FitError('Sampling aborted by user')
        except Exception:
            self._restore_parameter_values()
            raise
        finally:
            global_object.stack.enabled = stack_status

        draws = result_state.draw().points
        param_names = [p.name[len(MINIMIZER_PARAMETER_PREFIX) :] for p in problem._parameters]
        logp = getattr(result_state, 'logp', None)

        return {
            'draws': draws,
            'param_names': param_names,
            'state': result_state,
            'logp': logp,
        }

    def _build_sample_progress_payload(
        self, problem, iteration: int, point: np.ndarray, nllf: float
    ) -> dict:
        """Build a progress payload for Bayesian DREAM sampling steps.

        Called by :class:`BumpsProgressMonitor` at each DREAM generation.
        The payload includes ``sampling: True`` so downstream consumers can
        distinguish sampling progress from classical fitting progress.
        """
        parameter_values = self._current_parameter_snapshot(problem, point)
        return {
            'iteration': iteration,
            'chi2': float(problem.chisq(nllf=nllf, norm=False)),
            'reduced_chi2': float(problem.chisq(nllf=nllf, norm=True)),
            'parameter_values': parameter_values,
            'refresh_plots': False,
            'finished': False,
            'sampling': True,
        }

    def _set_parameter_fit_result(
        self,
        fit_result,
        stack_status: bool,
        par_list: List[BumpsParameter],
    ):
        """Update parameters to their final values and assign a std
        error to them.

        :param fit_result: BUMPS OptimizeResult containing best-fit
            values and errors
        :param stack_status: Whether the undo stack was enabled
        :param par_list: List of BUMPS parameter objects
        """
        from easyscience import global_object

        pars = self._cached_pars
        x_result = np.asarray(fit_result.x)
        stderr = np.asarray(fit_result.dx)

        if stack_status:
            self._restore_parameter_values()
            global_object.stack.enabled = True
            global_object.stack.beginMacro('Fitting routine')

        for index, name in enumerate([par.name for par in par_list]):
            dict_name = name[len(MINIMIZER_PARAMETER_PREFIX) :]
            pars[dict_name].value = x_result[index]
            pars[dict_name].error = stderr[index]
        if stack_status:
            global_object.stack.endMacro()

    def _gen_fit_results(
        self,
        fit_results,
        max_evaluations: int | None = None,
        tolerance: float | None = None,
        **kwargs,
    ) -> FitResults:
        """Convert fit results into the unified `FitResults` format.

        :param x_result: Optimized parameter values from FitDriver
        :param fx: Final objective function value
        :param driver: The FitDriver instance
        :param n_evaluations: Number of iterations completed
        :param max_evaluations: Maximum evaluations budget (if set)
        :return: fit results container
        :rtype: FitResults
        """
        results = FitResults()

        for name, value in kwargs.items():
            if getattr(results, name, False):
                setattr(results, name, value)
        n_evaluations = None if self._eval_counter is None else self._eval_counter.count
        # BUMPS exposes `nit` as the last reported optimizer step index rather than the
        # total number of objective calls. We keep `n_evaluations` as objective-call
        # count for cross-backend consistency with LMFit (`nfev`) and DFO-LS (`nf`).
        n_iterations = getattr(fit_results, 'nit', None)
        # Convert the zero-based step index into the number of optimizer steps that have
        # actually been consumed against the configured BUMPS `steps` budget.
        n_steps_used = None if n_iterations is None else n_iterations + 1
        stopped_on_budget = max_evaluations is not None and (
            # For BUMPS, `max_evaluations` is forwarded as `steps`, so budget
            # exhaustion must be checked against consumed optimizer steps, not raw
            # objective evaluations, which can legitimately exceed the step budget.
            (n_steps_used is not None and n_steps_used >= max_evaluations)
            or (
                n_iterations is None
                and n_evaluations is not None
                and n_evaluations >= max_evaluations
            )
        )

        results.success = fit_results.success and not stopped_on_budget
        pars = self._cached_pars
        item = {}
        for index, name in enumerate(self._cached_model.pars.keys()):
            dict_name = name[len(MINIMIZER_PARAMETER_PREFIX) :]
            item[name] = pars[dict_name].value

        results.p0 = self._p_0
        results.p = item
        results.x = self._cached_model.x
        results.y_obs = self._cached_model.y
        results.y_calc = self.evaluate(results.x, minimizer_parameters=results.p)
        results.y_err = self._cached_model.dy
        results.n_evaluations = n_evaluations
        results.iterations = n_steps_used
        results.message = ''
        if stopped_on_budget:
            results.message = (
                f'Fit stopped: reached maximum optimizer steps ({max_evaluations}); '
                f'objective evaluated {n_evaluations} times'
            )
        if stopped_on_budget:
            if tolerance is None:
                warnings.warn(
                    f'Fit did not converge within the maximum optimizer steps of {max_evaluations} '
                    f'({n_evaluations} objective evaluations). '
                    'Consider increasing the maximum number of evaluations or adjusting the tolerance.',
                    UserWarning,
                )
            else:
                warnings.warn(
                    f'Fit did not reach the desired tolerance of {tolerance} within the maximum optimizer steps of {max_evaluations} '
                    f'({n_evaluations} objective evaluations). '
                    'Consider increasing the maximum number of evaluations or adjusting the tolerance.',
                    UserWarning,
                )

        # results.residual = results.y_obs - results.y_calc
        # results.goodness_of_fit = np.sum(results.residual**2)
        results.minimizer_engine = self.__class__
        results.fit_args = None
        results.engine_result = fit_results
        return results
