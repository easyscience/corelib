# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import copy
import warnings
from typing import TYPE_CHECKING
from typing import Any
from typing import Callable

import numpy as np
from bumps.fitters import FIT_AVAILABLE_IDS
from bumps.fitters import FITTERS
from bumps.fitters import FitDriver
from bumps.names import FitProblem
from bumps.parameter import Parameter as BumpsParameter

# causes circular import when Parameter is imported
# from easyscience.base_classes import ObjBase
from easyscience.variable import Parameter

from ..available_minimizers import AvailableMinimizers
from .bumps_utils import BumpsProgressMonitor
from .bumps_utils import EvalCounter
from .bumps_utils import build_curve_problem
from .bumps_utils import parameter_snapshot
from .bumps_utils import to_bumps_parameter
from .bumps_utils import validate_arrays
from .minimizer_base import MINIMIZER_PARAMETER_PREFIX
from .minimizer_base import MinimizerBase
from .utils import FitError
from .utils import FitResults

if TYPE_CHECKING:
    from bumps.dream.state import MCMCDraw

FIT_AVAILABLE_IDS_FILTERED = copy.copy(FIT_AVAILABLE_IDS)
# Considered experimental
FIT_AVAILABLE_IDS_FILTERED.remove('pt')


class Bumps(MinimizerBase):
    """
    This is a wrapper to Bumps: https://bumps.readthedocs.io/ It allows
    for the Bumps fitting engine to use parameters declared in an
    ``EasyScience.base_classes.ObjBase``.
    """

    package = 'bumps'

    def __init__(
        self,
        obj: object,  #: ObjBase,
        fit_function: Callable,
        minimizer_enum: AvailableMinimizers | None = None,
    ):  # todo after constraint changes, add type hint: obj: ObjBase  # noqa: E501
        """
        Initialize the fitting engine.

        Parameters
        ----------
        obj : object
            Object containing the ``Parameter`` instances to fit.
        fit_function : Callable
            Callable returning model y values for the supplied x values.
        minimizer_enum : AvailableMinimizers | None, default=None
            Selected BUMPS minimizer configuration. By default, None.
        """
        super().__init__(obj=obj, fit_function=fit_function, minimizer_enum=minimizer_enum)
        self._p_0 = {}
        self._eval_counter: EvalCounter | None = None

    @staticmethod
    def all_methods() -> list[str]:
        return FIT_AVAILABLE_IDS_FILTERED

    @staticmethod
    def supported_methods() -> list[str]:
        # only a small subset
        methods = ['amoeba', 'newton', 'lm']
        return methods

    def fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
        weights: np.ndarray,
        model: Callable | None = None,
        parameters: list[Parameter] | None = None,
        method: str | None = None,
        tolerance: float | None = None,
        max_evaluations: int | None = None,
        progress_callback: Callable[[dict], bool | None] | None = None,
        abort_test: Callable[[], bool] | None = None,
        minimizer_kwargs: dict | None = None,
        engine_kwargs: dict | None = None,
        **kwargs: Any,
    ) -> FitResults:
        """
        Perform a fit using the BUMPS engine.

        Parameters
        ----------
        x : np.ndarray
            Points to be calculated at.
        y : np.ndarray
            Measured points.
        weights : np.ndarray
            Weights for supplied measured points.
        model : Callable | None, default=None
            Optional Model which is being fitted to. By default, None.
        parameters : list[Parameter] | None, default=None
            Optional parameters for the fit. By default, None.
        method : str | None, default=None
            Method for minimization. By default, None.
        tolerance : float | None, default=None
            Requested optimizer tolerance. By default, None.
        max_evaluations : int | None, default=None
            Maximum number of optimizer steps. Forwarded to BUMPS as its
            ``steps`` parameter. If ``None``, the default value defined
            by the selected BUMPS fitter (``fitclass.settings``) is
            used. By default, None.
        progress_callback : Callable[[dict], bool | None] | None, default=None
            Optional callback for progress updates. The payload field
            ``iteration`` carries the BUMPS optimizer step index. By
            default, None.
        abort_test : Callable[[], bool] | None, default=None
            Optional callback that returns ``True`` to signal that the
            fit should be aborted.  Called periodically during the BUMPS
            optimizer iteration loop.
        minimizer_kwargs : dict | None, default=None
            Additional keyword arguments passed to the BUMPS minimizer.
            By default, None.
        engine_kwargs : dict | None, default=None
            Additional engine keyword arguments. By default, None.
        **kwargs : Any
            Additional keyword arguments passed to ``FitDriver``.

        Returns
        -------
        FitResults
            Fit results.

        Raises
        ------
        FitError
            If the BUMPS fit fails.
        ValueError
            If the input shapes or weights are invalid.
        """
        method_dict = self._get_method_kwargs(method)

        x, y, weights = np.asarray(x), np.asarray(y), np.asarray(weights)

        validate_arrays(x, y, weights, check_finite_xy=False)

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
            # The Curve comes back directly from the helper: do NOT read it
            # from ``problem.fitness``, which is deprecated in BUMPS and warns.
            problem, self._eval_counter, model = build_curve_problem(
                self, x, y, weights, parameters=parameters
            )
        else:
            problem = FitProblem(model)
        self._cached_model = model

        self._p_0 = {f'p{key}': self._cached_pars[key].value for key in self._cached_pars.keys()}

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
            abort_test=abort_test if abort_test is not None else (lambda: False),
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

        parameter_values = parameter_snapshot(problem, point)

        return {
            'iteration': iteration,
            'chi2': chi2,
            'reduced_chi2': reduced_chi2,
            'parameter_values': parameter_values,
            'refresh_plots': False,
            'finished': False,
        }

    def convert_to_pars_obj(self, par_list: list[Parameter] | None = None) -> list[BumpsParameter]:
        """
        Create a container with the ``Parameters`` converted from the
        base object.

        Parameters
        ----------
        par_list : list[Parameter] | None, default=None
            If only a single/selection of parameter is required. Specify
            as a list. By default, None.

        Returns
        -------
        list[BumpsParameter]
            Bumps Parameters list.
        """
        if par_list is None:
            # Assume that we have a ObjBase for which we can obtain a list
            par_list = self._object.get_fit_parameters()
        pars_obj = [self.__class__.convert_to_par_object(obj) for obj in par_list]
        return pars_obj

    # For some reason I have to double staticmethod :-/
    @staticmethod
    def convert_to_par_object(obj: Parameter) -> BumpsParameter:
        """
        Convert an ``EasyScience.variable.Parameter`` object to a bumps
        Parameter object.

        Parameters
        ----------
        obj : Parameter
            EasyScience parameter to convert.

        Returns
        -------
        BumpsParameter
            Bumps Parameter compatible object.
        """
        return to_bumps_parameter(obj)

    def mcmc_sample(
        self,
        x: np.ndarray,
        y: np.ndarray,
        weights: np.ndarray,
        samples: int = 10000,
        burn: int = 2000,
        thin: int = 10,
        population: int | None = None,
        resume_state: MCMCDraw | None = None,
        sampler_kwargs: dict | None = None,
        progress_callback: Callable[[dict], bool | None] | None = None,
        abort_test: Callable[[], bool] | None = None,
    ) -> dict:
        """
        Run Bayesian MCMC sampling using the BUMPS DREAM sampler.

        Parameters
        ----------
        x : np.ndarray
            Flattened independent variable array.
        y : np.ndarray
            Flattened dependent variable array.
        weights : np.ndarray
            Flattened weight array.
        samples : int, default=10000
            Number of raw samples to draw across all chains, before thinning.
            A guaranteed minimum, not an exact count: DREAM advances in
            blocks of 10 generations (one generation = one draw per chain)
            and stops at the first block boundary at or past ``samples``.
        burn : int, default=2000
            Burn-in generations to discard. BUMPS counts ``burn`` in
            generations while ``samples`` counts raw draws, so ``burn=500``
            discards ``500 * n_chains`` raw samples.
        thin : int, default=10
            Thinning interval — only every ``thin``-th generation is stored.
        population : int | None, default=None
            BUMPS DREAM population count per parameter (number of parallel
            chains): BUMPS creates ``ceil(population * n_parameters)`` chains.
        resume_state : MCMCDraw | None, default=None
            A BUMPS ``MCMCDraw`` state object from a previous
            ``mcmc_sample()`` call (e.g. ``PosteriorResults.sampler_state``).
            When provided, DREAM **continues** the saved chain instead of
            starting cold.  The population, parameter count, and parameter
            names must match the current model — a ``ValueError`` is raised
            otherwise.

            ``samples`` must be the **total** number of raw samples, not an
            increment: to extend an existing chain of ``N`` raw samples by
            ``M``, pass ``samples=N + M`` (DREAM keeps only the last
            ``samples`` draws in its buffer). The `Sampler.extend` helper
            computes this for you.

            ``burn`` is forced to 0 on resume: a previously-converged chain is
            never re-burned.

            The ``population`` and ``initializer`` parameters
            have **no effect** when ``resume_state`` is provided — they
            are determined by the saved state.

            Resuming against *different* data is undefined behaviour (the
            chain's likelihood changes underneath it).
        sampler_kwargs : dict | None, default=None
            Additional keyword arguments forwarded to
            ``bumps.fitters.fit``.
        progress_callback : Callable[[dict], bool | None] | None, default=None
            Optional callback for progress updates during sampling.  The
            payload dict includes ``iteration`` (DREAM generation
            number) and ``sampling: True``.
        abort_test : Callable[[], bool] | None, default=None
            Optional callback that returns ``True`` to signal that
            sampling should be aborted. Called periodically during the
            DREAM sampling loop.

        Returns
        -------
        dict
            Dictionary with keys ``'draws'``, ``'param_names'``,
            ``'internal_bumps_object'``, and ``'logp'``.

        Raises
        ------
        ValueError
            If the input shapes or weights are invalid, if
            ``progress_callback`` is not callable, or if ``resume_state``
            is incompatible with the current model (parameter count,
            names/order, or population mismatch).
        FitError
            If DREAM sampling was aborted by the user (via
            ``abort_test``).
        Exception
            Re-raised from DREAM fitting if any unexpected error occurs
            (parameter values are restored beforehand).
        """
        warnings.warn(
            'Bumps.mcmc_sample() is deprecated. Use easyscience.fitting.Sampler '
            '(which no longer requires a BUMPS minimizer) instead.',
            DeprecationWarning,
            stacklevel=2,
        )
        from ..samplers.sampler_dream import DreamSampler

        engine = DreamSampler(self._object, self._original_fit_function)
        return engine.run(
            x=x,
            y=y,
            weights=weights,
            samples=samples,
            burn=burn,
            thin=thin,
            population=population,
            resume_state=resume_state,
            sampler_kwargs=sampler_kwargs,
            progress_callback=progress_callback,
            abort_test=abort_test,
        )

    def _set_parameter_fit_result(
        self,
        fit_result: Any,
        stack_status: bool,
        par_list: list[BumpsParameter],
    ) -> None:
        """
        Update parameters to their final values and assign a std error
        to them.

        Parameters
        ----------
        fit_result : Any
            BUMPS OptimizeResult containing best-fit values and errors.
        stack_status : bool
            Whether the undo stack was enabled.
        par_list : list[BumpsParameter]
            List of BUMPS parameter objects.
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
        fit_results: Any,
        max_evaluations: int | None = None,
        tolerance: float | None = None,
        **kwargs: Any,
    ) -> FitResults:
        """
        Convert fit results into the unified ``FitResults`` format.

        Parameters
        ----------
        fit_results : Any
            Native BUMPS fit result object.
        max_evaluations : int | None, default=None
            Maximum evaluations budget (if set). By default, None.
        tolerance : float | None, default=None
            Requested optimizer tolerance. By default, None.
        **kwargs : Any
            Additional result attributes to copy onto ``FitResults``.

        Returns
        -------
        FitResults
            Fit results container.
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
            from easyscience import global_object

            if tolerance is None:
                global_object.log.getLogger('fitting.bumps').warning(
                    f'Fit did not converge within the maximum optimizer steps of {max_evaluations} '
                    f'({n_evaluations} objective evaluations). '
                    'Consider increasing the maximum number of evaluations or adjusting the tolerance.'
                )
            else:
                global_object.log.getLogger('fitting.bumps').warning(
                    f'Fit did not reach the desired tolerance of {tolerance} within the maximum optimizer steps of {max_evaluations} '
                    f'({n_evaluations} objective evaluations). '
                    'Consider increasing the maximum number of evaluations or adjusting the tolerance.'
                )

        # results.residual = results.y_obs - results.y_calc
        # results.goodness_of_fit = np.sum(results.residual**2)
        results.minimizer_engine = self.__class__
        results.fit_args = None
        results.engine_result = fit_results
        return results
