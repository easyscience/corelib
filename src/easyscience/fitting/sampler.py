# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause
"""Bayesian MCMC sampling — the ``Sampler`` class and persistence helpers."""

from __future__ import annotations

import hashlib
import json
import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING
from typing import Callable

import numpy as np

from .minimizers.minimizer_base import MINIMIZER_PARAMETER_PREFIX

if TYPE_CHECKING:  # avoid import cycles; only needed for type hints
    from bumps.dream.state import MCMCDraw

    from .fitter import Fitter

# A sidecar is a small JSON file written next to the BUMPS chain files
# (``<path>.params.json``). The BUMPS chain format stores neither our
# parameter names nor any provenance, so we keep them alongside it.
#
# The schema is owned by easyscience from version 2 onwards. Version 1
# sidecars (written by easyreflectometry's save_posterior) carry only
# 'param_names' and are still accepted on load.
_SIDECAR_SCHEMA_VERSION = 2
_ACCEPTED_SIDECAR_SCHEMA_VERSIONS = (1, 2)


def _data_fingerprint(
    x_list: list,
    y_list: list,
    w_list: list,
) -> str | None:
    """Return a SHA-256 hex digest of concatenated (x|y|weights), or None."""
    try:
        h = hashlib.sha256()
        for arr in list(x_list) + list(y_list) + list(w_list):
            h.update(np.ascontiguousarray(arr, dtype=np.float64).tobytes())
        return h.hexdigest()
    except Exception:
        return None


def _load_bumps_state(path: str, skip: int = 0) -> MCMCDraw:
    """Read a BUMPS chain from disk, working around a BUMPS >= 1.0.4 bug.

    ``bumps.dream.state.load_state`` reads the saved buffers with
    ``numpy.loadtxt``, which collapses a single-row file to a 1-D array. A
    short chain stores a single CR-weight update row, so ``load_state``'s
    subsequent ``stats[:, 0]`` indexing raises ``IndexError: too many indices
    for array``. We coerce each buffer read back to 2-D before ``load_state``
    consumes it. Older BUMPS releases (< 1.0.4) used a custom 2-D-safe parser
    and are read unchanged.

    Parameters
    ----------
    path : str
        File path prefix that was passed to ``save_chain``.
    skip : int, default=0
        Discard the first ``skip`` saved generations, forwarded to
        ``bumps.dream.state.load_state``.

    Returns
    -------
    MCMCDraw
        The reloaded BUMPS chain state.
    """
    from bumps.dream import state as bumps_state

    loader = getattr(bumps_state, 'loadtxt_with_fallback', None)
    if loader is None:
        # BUMPS < 1.0.4: the custom parser is already 2-D safe.
        return bumps_state.load_state(path, skip=skip)

    def _loadtxt_2d(file, report: int = 0) -> np.ndarray:
        arr = loader(file, report=report)
        # A single saved row is read back as 1-D; restore it to a (1, N) row.
        return np.atleast_2d(arr) if getattr(arr, 'ndim', 2) == 1 else arr

    bumps_state.loadtxt_with_fallback = _loadtxt_2d
    try:
        return bumps_state.load_state(path, skip=skip)
    finally:
        bumps_state.loadtxt_with_fallback = loader


def save_chain(
    state: MCMCDraw,
    param_names: list[str] | None,
    path: str,
    data_fingerprint: str | None = None,
) -> None:
    """Persist a DREAM chain state plus a metadata sidecar.

    Writes the BUMPS native files (``<path>-chain.mc.gz``, ``<path>-point.mc.gz``
    and ``<path>-stats.mc.gz``) plus a ``<path>.params.json`` sidecar holding
    the parameter names and metadata.

    Parameters
    ----------
    state : MCMCDraw
        The BUMPS chain state to persist.
    param_names : list[str] | None
        Parameter names (one per chain column), stored in the sidecar.
    path : str
        File path prefix. BUMPS appends its own suffixes.
    data_fingerprint : str | None, default=None
        SHA-256 fingerprint of the data the chain was sampled against,
        verified (with a warning) on reload.
    """
    from bumps.dream.state import save_state

    from easyscience import __version__

    save_state(state, str(path))

    sidecar = {
        'schema_version': _SIDECAR_SCHEMA_VERSION,
        'param_names': param_names,
        'easyscience_version': __version__,
        'data_fingerprint': data_fingerprint,
    }
    with open(f'{path}.params.json', 'w') as f:
        json.dump(sidecar, f, indent=2)


def load_chain(path: str, skip: int = 0) -> tuple[MCMCDraw, list[str] | None, dict]:
    """Reload a DREAM chain state saved by ``save_chain``.

    Parameter names are restored from the sidecar when available (schema
    versions 1 and 2), falling back to the state's labels with the minimizer
    prefix stripped.

    Parameters
    ----------
    path : str
        File path prefix used when saving.
    skip : int, default=0
        Discard the first ``skip`` saved generations on load, forwarded to
        ``bumps.dream.state.load_state``.

    Returns
    -------
    tuple[MCMCDraw, list[str] | None, dict]
        The reloaded BUMPS chain state, the parameter names (or ``None`` if
        neither sidecar nor labels yielded them), and the raw sidecar dict
        (empty if absent/unreadable).
    """
    state = _load_bumps_state(str(path), skip=skip)

    sidecar: dict = {}
    param_names: list[str] | None = None
    try:
        with open(f'{path}.params.json', 'r') as f:
            sidecar = json.load(f)
        if sidecar.get('schema_version') in _ACCEPTED_SIDECAR_SCHEMA_VERSIONS:
            param_names = sidecar.get('param_names')
    except (FileNotFoundError, json.JSONDecodeError):
        sidecar = {}

    if param_names is None:
        # Fallback: strip the minimizer prefix from state.labels. Note BUMPS
        # save_state/load_state does not preserve labels, so a reloaded state
        # typically carries default labels like ['P0', 'P1', ...].
        param_names = [
            lbl[len(MINIMIZER_PARAMETER_PREFIX) :]
            if lbl.startswith(MINIMIZER_PARAMETER_PREFIX)
            else lbl
            for lbl in state.labels
        ]

    return state, param_names, sidecar


@dataclass
class SamplingResults:
    """Structured result of an MCMC sampling run (analogous to ``FitResults``).

    Attributes
    ----------
    draws : np.ndarray
        Posterior samples, shape ``(n_samples, n_params)``. For results from
        ``Sampler.sample()``/``Sampler.extend()`` this is a *trimmed,
        outlier-filtered* view of the chain, not the raw buffer, so it holds
        fewer rows than ``samples / thin`` — see the notes on ``Sampler``.
    param_names : list[str]
        Parameter names (one per column of ``draws``).
    logp : np.ndarray
        Log-posterior values, shape ``(n_samples,)``.
    state : MCMCDraw
        The raw BUMPS chain state, holding the full untrimmed chain.
    """

    draws: np.ndarray
    param_names: list[str]
    logp: np.ndarray
    state: MCMCDraw

    def to_legacy_dict(self) -> dict:
        """Return the legacy dict shape produced by the deprecated
        ``mcmc_sample()`` APIs."""
        return {
            'draws': self.draws,
            'param_names': self.param_names,
            'internal_bumps_object': self.state,
            'logp': self.logp,
        }


class Sampler:
    """Bayesian MCMC sampler for one dataset, backed by a Fitter's BUMPS minimizer.

    One ``Sampler`` instance represents one chain over one ``(x, y, weights)``
    dataset. The data is bound at construction; ``sample()`` and ``extend()``
    take no data arguments, so a chain can never be extended against different
    data (undefined behaviour in BUMPS).

    Construct directly with a configured ``Fitter`` (or ``MultiFitter``) whose
    minimizer has been switched to ``AvailableMinimizers.Bumps``. **Running a
    fit first is not required** — the ``Fitter`` supplies the model and the
    minimizer, not a fit result, and sampling from the initial parameter values
    works fine.

    It is often worth fitting first anyway. DREAM seeds its whole starting
    population inside a tiny ball around the parameters' *current* values
    (BUMPS' default ``init='eps'``), so sampling from fitted values starts the
    chain in the right region and shortens the burn-in needed to reach the
    typical set. From a poor initial guess, expect to burn for longer.

    The sampler is BUMPS/DREAM-specific for now: the BUMPS check in ``_run()``
    is the seam where another backend would plug in.

    Parameters
    ----------
    fitter : Fitter
        A configured ``Fitter`` (or ``MultiFitter``) whose minimizer has been
        switched to ``AvailableMinimizers.Bumps``.
    x : np.ndarray | list[np.ndarray]
        Independent variable array (or list of arrays for ``MultiFitter``).
    y : np.ndarray | list[np.ndarray]
        Dependent variable array (or list of arrays for ``MultiFitter``).
    weights : np.ndarray | list[np.ndarray | None] | None, default=None
        Weight array (or list of arrays for ``MultiFitter``).
    vectorized : bool, default=False
        When ``True``, each x array may be multi-dimensional (e.g. an
        ``(N, M, 2)`` grid for a 2D model) and is left as-is.
    sampler_kwargs : dict | None, default=None
        Per-instance default keyword arguments forwarded to the BUMPS DREAM
        sampler on every run, merged with (and overridden by) per-call
        ``sampler_kwargs``.

    Notes
    -----
    **The retained draws are a trimmed view, not the whole chain.** BUMPS'
    DREAM sampler defaults to ``trim=True``: once sampling finishes it runs a
    convergence-based burn-point detector over the chain and returns only the
    portion after that point. ``state.draw()`` additionally drops chains
    flagged as outliers. So ``results.draws`` is usually *smaller* than the
    chain, and its length is not deterministic — the detector re-runs from
    scratch on every ``sample()`` and ``extend()`` call and may place the burn
    point differently each time. Read the count off the array; do not predict
    it.

    Nor is the chain itself exactly ``samples / thin`` rows: DREAM allocates
    its ring buffer in whole generations, so the true size is ``samples /
    thin`` rounded *up* to a generation boundary. ``samples`` is a floor, not
    a promise.

    The trimming is only a *view* — nothing is dropped from the buffer. To
    read the full untrimmed chain::

        results = sampler.sample(samples=10000, burn=500, thin=2)
        full = results.state.draw(portion=1.0, outliers=True)
        full.points  # (n_chain_rows, n_params)
        full.logp

    To switch the automatic trimming off entirely, pass BUMPS' own ``trim``
    option straight through ``sampler_kwargs``; ``results.draws`` then holds
    the whole chain::

        sampler = Sampler(
            fitter, x, y, weights=w, sampler_kwargs={'trim': False}
        )

    Note also that trimming does not survive a ``save()``/``load_state()``
    round-trip: BUMPS does not persist the trim point, so a reloaded chain
    comes back untrimmed and ``load_state()`` reports *more* draws than the
    ``sample()`` call that created it. The chain itself is identical.
    """

    def __init__(
        self,
        fitter: 'Fitter',
        x: np.ndarray | list[np.ndarray],
        y: np.ndarray | list[np.ndarray],
        weights: np.ndarray | list[np.ndarray | None] | None = None,
        vectorized: bool = False,
        sampler_kwargs: dict | None = None,
    ):
        self._fitter = fitter
        self._x = x
        self._y = y
        self._weights = weights
        self._vectorized = vectorized
        self._default_sampler_kwargs = dict(sampler_kwargs or {})
        self._state: MCMCDraw | None = None  # current chain state
        self._results: SamplingResults | None = None

    @property
    def state(self) -> MCMCDraw | None:
        """Raw BUMPS MCMCDraw state (None before first sample/load_state)."""
        return self._state

    @property
    def results(self) -> SamplingResults | None:
        """Results of the most recent sample/extend/load_state call."""
        return self._results

    @property
    def draws(self) -> np.ndarray | None:
        """Posterior draws from the most recent run (or None)."""
        return self._results.draws if self._results is not None else None

    @property
    def param_names(self) -> list[str] | None:
        """Parameter names from the most recent run (or None)."""
        return self._results.param_names if self._results is not None else None

    @property
    def logp(self) -> np.ndarray | None:
        """Log-posterior values from the most recent run (or None)."""
        return self._results.logp if self._results is not None else None

    def _fingerprint(self) -> str | None:
        """SHA-256 fingerprint of the bound (x, y, weights) data, or None."""
        x_list = list(self._x) if isinstance(self._x, (list, tuple)) else [self._x]
        y_list = list(self._y) if isinstance(self._y, (list, tuple)) else [self._y]
        if self._weights is None:
            w_list = []
        elif isinstance(self._weights, (list, tuple)):
            w_list = [w for w in self._weights if w is not None]
        else:
            w_list = [self._weights]
        return _data_fingerprint(x_list, y_list, w_list)

    def _run(
        self,
        samples: int,
        burn: int,
        thin: int,
        population: int | None,
        resume_state: MCMCDraw | None,
        sampler_kwargs: dict | None,
        progress_callback: Callable[[dict], bool | None] | None,
        abort_test: Callable[[], bool] | None,
    ) -> SamplingResults:
        """Shared sampling engine for ``sample()`` and ``extend()``.

        Argument validation for ``samples``/``burn``/``thin`` lives in
        ``Bumps.mcmc_sample`` (single source of truth).
        """
        # Check the minimizer is BUMPS *before* mutating the fitter — a
        # non-BUMPS fitter must not be needlessly rebuilt.
        minimizer = self._fitter.minimizer
        if not (hasattr(minimizer, 'package') and minimizer.package == 'bumps'):
            raise RuntimeError(
                'Bayesian sampling requires a BUMPS minimizer. '
                'Use ``fitter.switch_minimizer(AvailableMinimizers.Bumps)`` first.'
            )

        x_fit, x_new, y_new, w_new, dims = self._fitter._precompute_reshaping(
            self._x, self._y, self._weights, self._vectorized
        )
        self._fitter._dependent_dims = dims
        wrapped = self._fitter._fit_function_wrapper(x_new, flatten=True)

        merged_kwargs = {**self._default_sampler_kwargs, **(sampler_kwargs or {})}

        original_fit_func = self._fitter.fit_function
        # Assigning fit_function triggers _update_minimizer() and *rebuilds*
        # the minimizer object — it must be re-fetched after this assignment.
        self._fitter.fit_function = wrapped
        try:
            minimizer = self._fitter.minimizer
            result = minimizer.mcmc_sample(
                x=x_fit,
                y=y_new,
                weights=w_new,
                samples=samples,
                burn=burn,
                thin=thin,
                population=population,
                resume_state=resume_state,
                sampler_kwargs=merged_kwargs or None,
                progress_callback=progress_callback,
                abort_test=abort_test,
            )
        finally:
            self._fitter.fit_function = original_fit_func

        results = SamplingResults(
            draws=result['draws'],
            param_names=result['param_names'],
            logp=result['logp'],
            state=result['internal_bumps_object'],
        )
        self._state = results.state
        self._results = results
        return results

    def sample(
        self,
        samples: int = 10000,
        burn: int = 2000,
        thin: int = 10,
        population: int | None = None,
        sampler_kwargs: dict | None = None,
        progress_callback: Callable[[dict], bool | None] | None = None,
        abort_test: Callable[[], bool] | None = None,
    ) -> SamplingResults:
        """Run fresh Bayesian MCMC sampling on the bound data.

        Calling ``sample()`` on a sampler that already holds a chain starts a
        **fresh** chain — the previous state and results are replaced. Use
        ``extend()`` to continue an existing chain.

        Parameters
        ----------
        samples : int, default=10000
            Number of retained DREAM samples requested from BUMPS.
        burn : int, default=2000
            Burn-in steps to discard before collecting samples.
        thin : int, default=10
            Thinning interval — only every ``thin``-th sample is kept,
            which reduces autocorrelation between consecutive draws.
        population : int | None, default=None
            DREAM population **scale factor** (not an absolute chain count):
            BUMPS creates ``ceil(population * n_parameters)`` parallel chains.
        sampler_kwargs : dict | None, default=None
            Additional keyword arguments forwarded to the BUMPS DREAM
            sampler (merged over the instance defaults).
        progress_callback : Callable[[dict], bool | None] | None, default=None
            Optional callback invoked at each DREAM generation. The payload
            dict includes ``iteration`` and ``sampling: True``.
        abort_test : Callable[[], bool] | None, default=None
            Optional callable that returns ``True`` to abort sampling early.

        Returns
        -------
        SamplingResults
            Structured sampling results (also stored on ``results``).

        Notes
        -----
        ``results.draws`` is a trimmed, outlier-filtered view of the chain,
        so it is usually smaller than the chain and its length is not
        predictable from ``samples``/``thin``. See the ``Sampler`` class notes
        for how to read the full chain or switch trimming off.

        Exceptions propagate from the sampling engine: ``ValueError`` if
        ``samples``, ``burn``, or ``thin`` are invalid, and ``RuntimeError``
        if the active minimizer is not a BUMPS instance.
        """
        return self._run(
            samples=samples,
            burn=burn,
            thin=thin,
            population=population,
            resume_state=None,
            sampler_kwargs=sampler_kwargs,
            progress_callback=progress_callback,
            abort_test=abort_test,
        )

    def extend(
        self,
        additional_samples: int = 5000,
        thin: int = 10,
        total_samples: int | None = None,
        sampler_kwargs: dict | None = None,
        progress_callback: Callable[[dict], bool | None] | None = None,
        abort_test: Callable[[], bool] | None = None,
    ) -> SamplingResults:
        """Continue the existing chain with additional samples.

        DREAM stores draws in a fixed-size ring buffer sized to its
        ``samples`` parameter; this method does the ring-buffer arithmetic
        for you (``samples = stored_generations * population +
        additional_samples``) so no existing draws are dropped from the
        buffer, regardless of the thinning interval. Runs with ``burn=0`` —
        re-burning a converged chain is usually a mistake, and BUMPS forces
        it to 0 on resume in any case. The DREAM population is recovered from
        the saved state and cannot be changed on extend.

        Parameters
        ----------
        additional_samples : int, default=5000
            Number of additional DREAM samples to draw, in the same units
            as ``samples`` in ``sample()``. This grows the ring buffer by
            exactly ``additional_samples``; how many of the new draws become
            *visible* in ``results.draws`` depends on the thinning interval
            and on BUMPS' automatic trimming (see Notes).
        thin : int, default=10
            Thinning interval for the retained draws.
        total_samples : int | None, default=None
            Advanced: total retained samples requested from the ring buffer,
            **overriding** the ``additional_samples`` arithmetic. With
            ``total_samples=N``, only the last N draws are retained.
        sampler_kwargs : dict | None, default=None
            Additional keyword arguments forwarded to the BUMPS DREAM
            sampler (merged over the instance defaults).
        progress_callback : Callable[[dict], bool | None] | None, default=None
            Optional callback invoked at each DREAM generation.
        abort_test : Callable[[], bool] | None, default=None
            Optional callable that returns ``True`` to abort sampling early.

        Returns
        -------
        SamplingResults
            Structured sampling results for the full (extended) chain.

        Raises
        ------
        RuntimeError
            If there is no chain to extend (call ``sample()`` or
            ``load_state()`` first), or the minimizer is not BUMPS.

        Notes
        -----
        ``results.draws`` will **not** grow by exactly
        ``additional_samples / thin``. Two things get in the way, neither of
        them a re-applied burn-in: BUMPS re-runs its burn-point detector over
        the whole extended chain and re-trims the returned view (so the
        visible count can even shrink), and the ring buffer is allocated in
        whole generations, so it grows by *at least*
        ``additional_samples / thin``, rounded up. To see the chain itself,
        compare ``results.state.draw(portion=1.0, outliers=True)`` before and
        after, or run with ``sampler_kwargs={'trim': False}``. See the
        ``Sampler`` class notes.
        """
        if self._state is None:
            raise RuntimeError('No chain to extend. Call sample() or load_state() first.')
        if total_samples is None:
            # The DREAM ring buffer holds Ngen generations of Npop points
            # each. Sizing the new buffer from the stored generations (rather
            # than the retained-draw count, which is divided by the thinning
            # interval) guarantees no existing draws are dropped for any
            # ``thin`` value.
            old_samples = int(self._state.Ngen) * int(self._state.Npop)
            total_samples = old_samples + int(additional_samples)
        return self._run(
            samples=total_samples,
            burn=0,
            thin=thin,
            population=None,
            resume_state=self._state,
            sampler_kwargs=sampler_kwargs,
            progress_callback=progress_callback,
            abort_test=abort_test,
        )

    def save(self, path: str) -> None:
        """Persist the chain state and metadata to disk.

        Writes BUMPS native files (``<path>-chain.mc.gz`` etc.) plus a
        ``<path>.params.json`` sidecar with parameter names, the easyscience
        version, and a fingerprint of the bound data (verified with a warning
        on ``load_state()``).

        Parameters
        ----------
        path : str
            File path prefix. BUMPS appends its own suffixes.

        Raises
        ------
        RuntimeError
            If no chain state exists yet (call ``sample()`` first).
        """
        if self._state is None:
            raise RuntimeError('No chain state to save. Call sample() first.')
        save_chain(
            self._state,
            self._results.param_names if self._results is not None else None,
            path,
            data_fingerprint=self._fingerprint(),
        )

    def load_state(self, path: str, skip: int = 0) -> SamplingResults:
        """Load a previously saved chain into this sampler.

        The sampler must be constructed with the same fitter and data used to
        create the chain — ``extend()`` then continues the saved chain. If the
        sidecar carries a data fingerprint and it does not match this
        sampler's bound data, a ``UserWarning`` is emitted (extending a chain
        against different data is undefined behaviour).

        Populates ``state`` and ``results`` (draws, log-posterior and
        parameter names) from the saved chain, so summaries and
        ``PosteriorResults.from_sampler()`` work without resampling.

        Parameters
        ----------
        path : str
            File path prefix used in ``save()``.
        skip : int, default=0
            Discard the first ``skip`` saved generations on load. Useful for
            trimming additional burn-in without re-sampling.

        Returns
        -------
        SamplingResults
            The reloaded chain results (also stored on ``results``).

        Notes
        -----
        BUMPS does not persist its automatic trim point, so a reloaded chain
        comes back **untrimmed**: this method reports more draws than the
        ``sample()`` call that wrote the file, even though the chain is
        identical. Use ``skip`` to discard leading generations explicitly.
        See the ``Sampler`` class notes.
        """
        state, param_names, sidecar = load_chain(path, skip=skip)

        saved_fingerprint = sidecar.get('data_fingerprint')
        if saved_fingerprint is not None:
            current_fingerprint = self._fingerprint()
            if current_fingerprint is not None and current_fingerprint != saved_fingerprint:
                warnings.warn(
                    'The data bound to this Sampler does not match the data '
                    'fingerprint stored with the saved chain. Extending a '
                    'chain against different data is undefined behaviour.',
                    UserWarning,
                    stacklevel=2,
                )

        _draw = state.draw()
        results = SamplingResults(
            draws=_draw.points,
            param_names=param_names,
            logp=_draw.logp,
            state=state,
        )
        self._state = state
        self._results = results
        return results
