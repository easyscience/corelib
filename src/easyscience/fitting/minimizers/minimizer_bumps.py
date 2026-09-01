# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any
from typing import Callable
from typing import cast

import numpy as np
from bumps.fitters import FIT_AVAILABLE_IDS
from bumps.fitters import FITTERS
from bumps.fitters import FitDriver
from bumps.names import FitProblem
from bumps.parameter import Parameter as BumpsParameter
from scipy.optimize import OptimizeResult

# causes circular import when Parameter is imported
# from easyscience.base_classes import ObjBase
from easyscience.variable import Parameter

from ..available_minimizers import AvailableMinimizers
from .bumps_utils import BumpsProgressMonitor
from .bumps_utils import EvalCounter
from .bumps_utils import build_curve_problem
from .bumps_utils import parameter_names
from .bumps_utils import parameter_snapshot
from .bumps_utils import to_bumps_parameter
from .bumps_utils import validate_arrays
from .minimizer_base import MINIMIZER_PARAMETER_PREFIX
from .minimizer_base import MinimizerBase
from .utils import FitError
from .utils import FitResults

if TYPE_CHECKING:
    from bumps.fitters import FitBase

# 'pt' (parallel tempering) is considered experimental and is not exposed.
FIT_AVAILABLE_IDS_FILTERED = [fit_id for fit_id in FIT_AVAILABLE_IDS if fit_id != 'pt']


class Bumps(MinimizerBase):
    """
    This is a wrapper to Bumps: https://bumps.readthedocs.io/ It allows
    for the Bumps fitting engine to use parameters declared in an
    ``EasyScience.base_classes.ObjBase``.
    """

    package = 'bumps'

    def __init__(
        self,
        obj: object,
        fit_function: Callable,
        minimizer_enum: AvailableMinimizers | None = None,
    ):
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
        # Copy so callers cannot mutate the module-level list in place.
        return list(FIT_AVAILABLE_IDS_FILTERED)

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
        progress_callback: Callable[[dict], None] | None = None,
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
            Optional BUMPS ``Curve`` which is being fitted to. When
            omitted, one is built from ``fit_function`` and the object's
            fit parameters. A supplied ``Curve`` must expose ``pars``,
            ``x``, ``y`` and ``dy``, since the results are assembled from
            them. By default, None.
        parameters : list[Parameter] | None, default=None
            Optional parameters for the fit. Ignored when ``model`` is
            supplied. By default, None.
        method : str | None, default=None
            Method for minimization. By default, None.
        tolerance : float | None, default=None
            Requested optimizer tolerance. By default, None.
        max_evaluations : int | None, default=None
            Maximum number of optimizer steps. Forwarded to BUMPS as its
            ``steps`` parameter. If ``None``, the default value defined
            by the selected BUMPS fitter (``fitclass.settings``) is
            used. By default, None.
        progress_callback : Callable[[dict], None] | None, default=None
            Optional callback for progress updates. The field
            ``iteration`` carries the BUMPS optimizer step index. The
            return value is ignored: use ``abort_test`` to stop a
            running fit. By default, None.
        abort_test : Callable[[], bool] | None, default=None
            Optional callback that returns ``True`` to signal that the
            fit should be aborted.  Called periodically during the BUMPS
            optimizer iteration loop, and once more after the optimizer
            returns in order to distinguish an aborted fit from a
            converged one, so it must be side-effect free. An aborted fit
            returns ``FitResults(success=False)`` rather than raising.
        minimizer_kwargs : dict | None, default=None
            Additional keyword arguments passed to the BUMPS minimizer.
            The mapping is copied before use, so it is never mutated. By
            default, None.
        engine_kwargs : dict | None, default=None
            Additional engine keyword arguments. By default, None.
        **kwargs : Any
            Additional keyword arguments passed to ``FitDriver``.

        Returns
        -------
        FitResults
            Fit results. ``FitResults.iterations`` is the number of BUMPS
            *optimizer steps* consumed (the last reported step index plus
            one), which is what ``max_evaluations`` tests against; it is
            not comparable to LMFit's ``nfev`` or DFO-LS' ``nf``. The
            objective-call count is reported separately as
            ``FitResults.n_evaluations``, which is the cross-backend
            consistent figure. Note that BUMPS derives the step index from
            whatever the selected fitter reports to its monitors, so the
            granularity of a "step" varies between fitters.

        Raises
        ------
        FitError
            If the BUMPS fit raises. A fit that merely fails to converge
            is reported as ``FitResults(success=False)`` instead.
        ValueError
            If the input shapes or weights are invalid, or if
            ``progress_callback`` is not callable.
        """
        method_dict = self._get_method_kwargs(method)

        x, y, weights = np.asarray(x), np.asarray(y), np.asarray(weights)

        validate_arrays(x, y, weights, check_finite_xy=False)

        if progress_callback is not None and not callable(progress_callback):
            raise ValueError('progress_callback must be callable')

        if engine_kwargs is None:
            engine_kwargs = {}

        # Copy rather than mutate: `ftol`/`xtol`/`steps` are injected below, and a
        # caller reusing the same mapping for a second fit would otherwise silently
        # inherit the settings resolved for the first one.
        minimizer_kwargs = {} if minimizer_kwargs is None else dict(minimizer_kwargs)
        minimizer_kwargs.update(engine_kwargs)

        method_str = method_dict.get('method', self._method)
        fitclass = self._resolve_fitclass(method_str)

        # Reset the per-fit evaluation counter. A caller-supplied `model` bypasses
        # `build_curve_problem`, which is what installs the counter, so without this
        # the results would carry the previous fit's objective-call count.
        self._eval_counter = None

        # Only values the caller supplied explicitly are pushed back into
        # `minimizer_kwargs`. BUMPS pairs an independent `ftol`/`xtol` default per
        # fitter (`newton` combines ftol=1e-6 with xtol=1e-12, `amoeba` ftol=1e-8 with
        # xtol=1e-6).
        fitter_settings = dict(fitclass.settings)

        if max_evaluations is not None:
            minimizer_kwargs['steps'] = max_evaluations
        else:
            max_evaluations = fitter_settings.get('steps')

        if tolerance is not None:
            minimizer_kwargs['ftol'] = tolerance  # tolerance for change in function value
            minimizer_kwargs['xtol'] = (
                tolerance  # tolerance for change in parameter value, could be an independent value
            )
        else:
            # Report the stricter of the two BUMPS defaults; nothing is written back.
            tols = [
                t
                for t in (fitter_settings.get('ftol'), fitter_settings.get('xtol'))
                if t is not None
            ]
            tolerance = min(tols) if tols else None

        if model is None:
            # The Curve comes back directly from the helper.
            problem, self._eval_counter, model = build_curve_problem(
                self, x, y, weights, parameters=parameters
            )
        else:
            # A caller-supplied model bypasses `build_curve_problem`, which is also
            # what populates the parameter cache that `_p_0`,
            # `_set_parameter_fit_result` and `_gen_fit_results` all read. Build the
            # wrapped fit function here purely for that side effect, so the cache
            # describes the current object rather than being empty or left over from
            # an earlier fit.
            self._fit_function = self._generate_fit_function()
            problem = FitProblem(model)
        self._cached_model = model

        self._p_0 = {f'p{key}': self._cached_pars[key].value for key in self._cached_pars.keys()}

        monitors = []
        if progress_callback is not None:
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
            #
            # Named `best_x` rather than `x` so the caller's independent-variable
            # array stays intact for the rest of the method.
            best_x, fx = driver.fit()

            # BUMPS signals a failed optimization by returning `None` in place of a
            # parameter vector (e.g. Levenberg-Marquardt landing on non-finite
            # values); `FitDriver.fit` skips `problem.setp` in that case. Poll
            # `abort_test` once more to tell a user-cancelled run apart from a
            # converged one, since BUMPS stops quietly either way.
            if best_x is None:
                success = False
                message = 'BUMPS returned no solution; the fit did not converge'
            elif abort_test is not None and abort_test():
                success = False
                message = 'Fit aborted before convergence'
            else:
                success = True
                message = 'successful termination'

            # BUMPS' `MonitorRunner.history.step` is populated by the driver itself
            # (independently of any user-supplied monitors) and exposes the canonical
            # last-step index reached by the fitter, so we use it as `nit`. `Trace`
            # indexes into an internal list, so an empty trace raises `IndexError`
            # rather than returning a default — that happens when the fit is aborted
            # before the fitter reports its first step.
            history = getattr(getattr(driver, 'monitor_runner', None), 'history', None)
            step_trace = getattr(history, 'step', None)
            nit_value = int(step_trace[0]) if step_trace is not None and len(step_trace) else None

            model_results = OptimizeResult(
                # `driver.stderr()` derives the errors from the covariance at the
                # solution, so it cannot be evaluated without one.
                x=best_x,
                dx=driver.stderr() if best_x is not None else None,
                fun=fx,
                success=success,
                status=0 if success else 1,
                message=message,
                nit=nit_value,
            )
            model_results.state = driver.fitter.state

            if best_x is None:
                self._restore_parameter_values()
            else:
                self._set_parameter_fit_result(
                    model_results, stack_status, parameter_names(problem)
                )
            results = self._gen_fit_results(
                model_results,
                max_evaluations=max_evaluations,
                tolerance=tolerance,
            )
        except Exception as e:
            self._restore_parameter_values()
            raise FitError(e) from e
        finally:
            global_object.stack.enabled = stack_status
        return results

    @staticmethod
    def _resolve_fitclass(method: str) -> type[FitBase]:
        """
        Look up the BUMPS fitter class registered under ``method``.

        Parameters
        ----------
        method : str
            A BUMPS fitter id, e.g. ``'amoeba'``.

        Returns
        -------
        type[FitBase]
            The matching BUMPS fitter class.

        Raises
        ------
        FitError
            If no registered fitter carries that id.
        """
        # Built per call rather than cached at import time so that fitters
        # registered into `FITTERS` after import are still resolvable.
        fitclass = {fitclass.id: fitclass for fitclass in FITTERS}.get(method)
        if fitclass is None:
            raise FitError(f'Unknown BUMPS fitting method: {method}')
        # BUMPS annotates `FITTERS` as `List[FitBase]`, but it holds the fitter
        # *classes* — `FitDriver` instantiates them as `self.fitclass(problem)`.
        return cast('type[FitBase]', fitclass)

    def _build_progress_payload(
        self, problem: FitProblem, iteration: int, point: np.ndarray, nllf: float
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
        pars_obj = [self.convert_to_par_object(obj) for obj in par_list]
        return pars_obj

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

    def _set_parameter_fit_result(
        self,
        fit_result: Any,
        stack_status: bool,
        par_names: list[str],
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
        par_names : list[str]
            Cached-parameter names in BUMPS problem order, already
            stripped of ``MINIMIZER_PARAMETER_PREFIX``. As seen in
            :func:`~easyscience.fitting.minimizers.bumps_utils.parameter_names`.
        """
        from easyscience import global_object

        pars = self._cached_pars
        x_result = np.asarray(fit_result.x)
        # Some BUMPS fitters cannot produce a covariance and hand back no errors;
        # report those parameters as having no uncertainty rather than failing,
        # matching what the LMFit minimizer does when `errorbars` is False.
        stderr = None if fit_result.dx is None else np.asarray(fit_result.dx)

        if stack_status:
            self._restore_parameter_values()
            global_object.stack.enabled = True
            global_object.stack.beginMacro('Fitting routine')

        for index, name in enumerate(par_names):
            pars[name].value = x_result[index]
            pars[name].error = 0.0 if stderr is None else stderr[index]
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

        # `hasattr`, not a truthiness test: every `FitResults` field starts out
        # falsy, so testing the current value would silently discard every kwarg.
        for name, value in kwargs.items():
            if hasattr(results, name):
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
        # Costs one extra model evaluation beyond those the optimizer consumed, and
        # deliberately so: it runs through the uncounted `self._fit_function`, keeping
        # `n_evaluations` a faithful count of optimizer-driven objective calls.
        results.y_calc = self.evaluate(results.x, minimizer_parameters=results.p)
        results.y_err = self._cached_model.dy
        results.n_evaluations = n_evaluations
        results.iterations = n_steps_used
        # A successful fit carries no message; anything else reports why it stopped.
        results.message = (
            '' if fit_results.success else (getattr(fit_results, 'message', '') or '')
        )

        if stopped_on_budget:
            from easyscience import global_object

            results.message = (
                f'Fit stopped: reached maximum optimizer steps ({max_evaluations}); '
                f'objective evaluated {n_evaluations} times'
            )
            if tolerance is None:
                reason = 'Fit did not converge within'
            else:
                reason = f'Fit did not reach the desired tolerance of {tolerance} within'
            global_object.log.getLogger('fitting.bumps').warning(
                f'{reason} the maximum optimizer steps of {max_evaluations} '
                f'({n_evaluations} objective evaluations). '
                'Consider increasing the maximum number of evaluations or adjusting the tolerance.'
            )

        results.minimizer_engine = self.__class__
        results.fit_args = None
        results.engine_result = fit_results
        return results
