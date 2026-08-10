# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause
"""The BUMPS DREAM MCMC engine — ``DreamSampler``.

One file per sampling backend, mirroring the one-file-per-minimizer
layout under ``fitting/minimizers/``. When a second MCMC backend
arrives, its ``run()`` signature is formalized as a ``SamplerBase`` ABC
and dispatched via a factory (see discussion easyscience/core#280).
"""

from __future__ import annotations

import copy
import math
from typing import TYPE_CHECKING
from typing import Callable

import numpy as np
from bumps.fitters import FitDriver

from ..engine_base import PARAMETER_PREFIX
from ..engine_base import EngineBase
from ..minimizers.bumps_utils import BumpsProgressMonitor
from ..minimizers.bumps_utils import build_curve_problem
from ..minimizers.bumps_utils import parameter_names
from ..minimizers.bumps_utils import parameter_snapshot
from ..minimizers.bumps_utils import validate_arrays
from ..minimizers.bumps_utils import validate_run_settings
from ..minimizers.utils import FitError

if TYPE_CHECKING:
    from bumps.dream.state import MCMCDraw
    from bumps.names import FitProblem


class DreamSampler(EngineBase):
    """
    BUMPS DREAM MCMC engine. Runs and resumes chains for one
    ``(obj, fit_function)`` binding.

    This is the minimizer-independent home of Bayesian sampling: it
    builds its own BUMPS ``FitProblem`` via the shared ``bumps_utils``
    helpers, so sampling no longer requires the ``Fitter``'s active
    minimizer to be BUMPS — only an installed ``bumps`` package.

    ``DreamSampler`` is internal machinery; the public entry point is
    :class:`easyscience.fitting.Sampler`.
    """

    package = 'bumps'

    def __init__(
        self,
        obj: object,  #: ObjBase,
        fit_function: Callable,
    ):  # todo after constraint changes, add type hint: obj: ObjBase  # noqa: E501
        """
        Initialize the sampling engine.

        Parameters
        ----------
        obj : object
            Object containing the ``Parameter`` instances to sample.
        fit_function : Callable
            Callable returning model y values for the supplied x values.
        """
        super().__init__(obj=obj, fit_function=fit_function)

    def run(
        self,
        x: np.ndarray,
        y: np.ndarray,
        weights: np.ndarray | None,
        samples: int = 10000,
        burn: int = 2000,
        thin: int = 10,
        population: int | None = None,
        resume_state: MCMCDraw | None = None,
        sampler_kwargs: dict | None = None,
        progress_callback: Callable[[dict], None] | None = None,
        abort_test: Callable[[], bool] | None = None,
    ) -> dict:
        """
        Run Bayesian MCMC sampling using the BUMPS DREAM sampler.

        Builds a BUMPS ``FitProblem`` from the bound object and fit
        function and runs the DREAM sampler.  This is the engine-level
        entry point for Bayesian sampling; the public
        :class:`~easyscience.fitting.Sampler` delegates to this method
        after flattening its bound data.

        Parameters
        ----------
        x : np.ndarray
            Flattened independent variable array.
        y : np.ndarray
            Flattened dependent variable array.
        weights : np.ndarray | None
            Flattened weight array. Must not be ``None`` — sampling has
            no default weighting; a clear ``ValueError`` is raised.
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
            A BUMPS ``MCMCDraw`` state object from a previous ``run()``
            call. When provided, DREAM **continues** the saved chain
            instead of starting cold.  The population, parameter count,
            and parameter names must match the current model — a
            ``ValueError`` is raised otherwise.

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
        progress_callback : Callable[[dict], None] | None, default=None
            Optional callback for progress updates during sampling.  The
            payload dict includes ``iteration`` (DREAM generation
            number) and ``sampling: True``. Any return value is ignored.
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
        from bumps.fitters import DreamFit

        if weights is None:
            raise ValueError(
                'weights must not be None for Bayesian sampling. Pass '
                'measurement weights (e.g. ``1 / sigma``) matching x and y.'
            )
        x, y, weights = np.asarray(x), np.asarray(y), np.asarray(weights)

        validate_run_settings(samples, burn, thin)
        validate_arrays(x, y, weights, check_finite_xy=True)

        # Build the BUMPS Curve model around the engine's wrapped fit function
        problem, _, _ = build_curve_problem(self, x, y, weights)

        pop = population
        if resume_state is not None:
            pop, burn = self._validate_resume_state(problem, resume_state, population, burn)

        # Build DREAM kwargs. Use the resolved ``pop``, not the raw
        # ``population`` argument — on resume ``pop`` is the negative
        # absolute chain count that reproduces the saved state's
        # population, which BUMPS requires to match.
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
            # Compute total DREAM steps for progress display (burn + sampling generations).
            # BUMPS DREAM default population count is 10 when not specified by the user.
            # A negative ``pop`` (resume) is an absolute chain count.
            _dream_default_pop = 10
            pop_val = abs(pop) if pop is not None else _dream_default_pop
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
            abort_test=abort_test if abort_test is not None else (lambda: False),
            **dream_kwargs,
        )
        driver.clip()

        from easyscience import global_object

        stack_status = global_object.stack.enabled
        global_object.stack.enabled = False

        try:
            fit_kwargs = {}
            if resume_state is not None:
                # Defensive copy: BUMPS mutates the state object in-place
                # (via MCMCDraw.resize() — see bumps/dream/core.py allocate_state)
                # during resume.  Without a copy, the caller's original state
                # object is silently altered, making it impossible to compare
                # pre- and post-resume state (shape mismatch).  See
                # https://github.com/easyscience/core/pull/257
                fit_kwargs['fit_state'] = copy.deepcopy(resume_state)
            x_opt, fx = driver.fit(**fit_kwargs)
            result_state = getattr(driver.fitter, 'state', None)
            if result_state is None:
                raise FitError('Sampling aborted by user')
        except Exception:
            self._restore_parameter_values()
            raise
        finally:
            global_object.stack.enabled = stack_status

        _draw = result_state.draw()

        return {
            'draws': _draw.points,
            'param_names': parameter_names(problem),
            'internal_bumps_object': result_state,
            'logp': _draw.logp,
        }

    def _validate_resume_state(
        self,
        problem: FitProblem,
        resume_state: MCMCDraw,
        population: int | None,
        burn: int,
    ) -> tuple[int, int]:
        """Check that ``resume_state`` is compatible with ``problem`` and
        resolve the population and burn values to use when resuming.

        Parameters
        ----------
        problem : FitProblem
            The freshly built BUMPS ``FitProblem`` for the current model.
        resume_state : MCMCDraw
            The saved chain state to resume from.
        population : int | None
            The caller-supplied population scale factor, or ``None``.
        burn : int
            The caller-supplied burn-in, ignored (with a warning) on resume.

        Returns
        -------
        tuple[int, int]
            ``(population, burn)`` to pass to DREAM. The population is
            returned as a **negative** number, which BUMPS'
            ``initpop.generate`` reads as an absolute chain count, exactly
            reproducing the saved state's population. ``burn`` is always 0:
            a previously converged chain is never re-burned.

        Raises
        ------
        ValueError
            If ``resume_state`` is incompatible with the current model
            (parameter count, names/order, or population mismatch).
        """
        from easyscience import global_object

        logger = global_object.log.getLogger('fitting.bumps')

        # Parameter count
        n_params = len(problem._parameters)
        if n_params != resume_state.Nvar:
            raise ValueError(
                f'resume_state has {resume_state.Nvar} parameters but the current '
                f'model has {n_params}. The model must have the same '
                f'number of fitted parameters as when the saved chain was created.'
            )

        prefix = PARAMETER_PREFIX
        fresh_names = [(p.name or '')[len(prefix) :] for p in problem._parameters]
        state_labels = list(resume_state.labels)
        if state_labels and all(lbl.startswith(prefix) for lbl in state_labels):
            state_names = [lbl[len(prefix) :] for lbl in state_labels]
            if fresh_names != state_names:
                raise ValueError(
                    f'Parameter names/order mismatch between the current model '
                    f'and resume_state.\n'
                    f'  Current model : {fresh_names}\n'
                    f'  resume_state  : {state_names}'
                )
        else:
            logger.warning(
                'resume_state does not carry parameter names (it was most '
                'likely reloaded from disk, where BUMPS does not preserve '
                'labels). Parameter-name validation is skipped; the saved '
                'chain is matched to the current model by parameter order. '
                'Ensure this is the same model, with parameters in the same '
                'order, used to create the chain.'
            )

        # Population. BUMPS creates ``ceil(population * n_params)`` chains
        # and requires the resumed state's chain count to match.
        if population is not None:
            expected_npop = math.ceil(population * n_params)
            if expected_npop != resume_state.Npop:
                raise ValueError(
                    f'Requested population ({population}) would produce '
                    f'{expected_npop} chains but the saved state has '
                    f'{resume_state.Npop} chains. The population cannot '
                    f'be changed on resume.'
                )
        if burn > 0:
            logger.warning(
                f'burn={burn} ignored on resume: a previously converged '
                f'chain is not re-burned. Forcing burn=0.'
            )

        # A negative ``pop`` is read by ``bumps.initpop.generate`` as an
        # absolute chain count, exactly reproducing the saved population
        # without having to recover the original scale factor.
        return -int(resume_state.Npop), 0

    def _build_sample_progress_payload(
        self, problem, iteration: int, point: np.ndarray, nllf: float
    ) -> dict:
        """
        Build a progress payload for Bayesian DREAM sampling steps.

        Called by :class:`BumpsProgressMonitor` at each DREAM
        generation. The payload includes ``sampling: True`` so
        downstream consumers can distinguish sampling progress from
        classical fitting progress; the remaining keys match the
        classical-fit payload built by the minimizers.
        """
        # Use the nllf already computed by the sampler to avoid a costly
        # model re-evaluation, and let BUMPS apply its own chisq scaling.
        chi2 = float(problem.chisq(nllf=nllf, norm=False))
        reduced_chi2 = float(problem.chisq(nllf=nllf, norm=True))

        return {
            'iteration': iteration,
            'chi2': chi2,
            'reduced_chi2': reduced_chi2,
            'parameter_values': parameter_snapshot(problem, point),
            'refresh_plots': False,
            'finished': False,
            'sampling': True,
        }
