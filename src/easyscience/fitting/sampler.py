# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause
"""Bayesian MCMC sampling — the ``Sampler`` class and persistence helpers."""

from __future__ import annotations

import hashlib
import json
import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING
from typing import Any
from typing import Callable
from typing import List
from typing import Optional
from typing import Union

import numpy as np

from .minimizers.minimizer_base import MINIMIZER_PARAMETER_PREFIX

if TYPE_CHECKING:  # avoid import cycles; Fitter is only needed for type hints
    from .fitter import Fitter

# Sidecar schema owned by easyscience from version 2 onwards. Version 1
# sidecars (written by easyreflectometry's save_posterior) carry only
# 'param_names' and are still accepted on load.
_SIDECAR_SCHEMA_VERSION = 2
_ACCEPTED_SIDECAR_SCHEMA_VERSIONS = (1, 2)


def _easyscience_version() -> str:
    """Return the installed easyscience version string."""
    try:
        from importlib.metadata import version as _v

        return _v('easyscience')
    except Exception:
        return 'unknown'


def _data_fingerprint(
    x_list: list,
    y_list: list,
    w_list: list,
) -> Optional[str]:
    """Return a SHA-256 hex digest of concatenated (x|y|weights), or None."""
    try:
        h = hashlib.sha256()
        for arr in list(x_list) + list(y_list) + list(w_list):
            h.update(np.ascontiguousarray(arr, dtype=np.float64).tobytes())
        return h.hexdigest()
    except Exception:
        return None


def save_chain(
    state: Any,
    param_names: Optional[List[str]],
    path: str,
    data_fingerprint: Optional[str] = None,
) -> None:
    """Persist a DREAM chain state plus a metadata sidecar.

    Writes BUMPS native files (``<path>-chain.mc.gz`` etc. via
    ``bumps.dream.state.save_state``) plus a ``<path>.params.json`` sidecar
    holding parameter names and metadata. Used by :meth:`Sampler.save` and by
    the deprecated ``easyreflectometry`` ``save_posterior`` shim.

    Parameters
    ----------
    state : Any
        The BUMPS ``MCMCDraw`` object to persist.
    param_names : Optional[List[str]]
        Parameter names (one per chain column), stored in the sidecar.
    path : str
        File path prefix. BUMPS appends its own suffixes.
    data_fingerprint : Optional[str], default=None
        SHA-256 fingerprint of the data the chain was sampled against,
        verified (with a warning) on reload.
    """
    from .minimizers.minimizer_bumps import save_sampler_state

    save_sampler_state(state, str(path))

    sidecar = {
        'schema_version': _SIDECAR_SCHEMA_VERSION,
        'param_names': param_names,
        'easyscience_version': _easyscience_version(),
        'data_fingerprint': data_fingerprint,
    }
    with open(f'{path}.params.json', 'w') as f:
        json.dump(sidecar, f, indent=2)


def load_chain(path: str, skip: int = 0) -> tuple[Any, Optional[List[str]], dict]:
    """Reload a DREAM chain state saved by :func:`save_chain`.

    Uses the patched loader (``load_sampler_state``), which carries the
    BUMPS >= 1.0.4 single-row ``loadtxt`` workaround. Parameter names are
    restored from the sidecar when available (schema versions 1 and 2),
    falling back to the state's labels with the minimizer prefix stripped.
    Used by :meth:`Sampler.load_state` and by the deprecated
    ``easyreflectometry`` ``load_posterior`` shim.

    Parameters
    ----------
    path : str
        File path prefix used when saving.
    skip : int, default=0
        Discard the first ``skip`` saved generations on load, forwarded to
        ``bumps.dream.state.load_state``.

    Returns
    -------
    tuple[Any, Optional[List[str]], dict]
        The reloaded BUMPS ``MCMCDraw`` state, the parameter names (or
        ``None`` if neither sidecar nor labels yielded them), and the raw
        sidecar dict (empty if absent/unreadable).
    """
    from .minimizers.minimizer_bumps import load_sampler_state

    state = load_sampler_state(str(path), skip=skip)

    sidecar: dict = {}
    param_names: Optional[List[str]] = None
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
        Posterior samples, shape ``(n_samples, n_params)``.
    param_names : list[str]
        Parameter names (one per column of ``draws``).
    logp : np.ndarray
        Log-posterior values, shape ``(n_samples,)``.
    state : Any
        The raw BUMPS ``MCMCDraw`` chain state.
    """

    draws: np.ndarray
    param_names: list[str]
    logp: np.ndarray
    state: Any

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
    dataset. The data is bound at construction; :meth:`sample` and
    :meth:`extend` take no data arguments, so a chain can never be extended
    against different data (undefined behaviour in BUMPS).

    Create via ``fitter.create_sampler(x, y, weights)`` (or, in
    reflectometry-lib, ``fitter.create_sampler(data)``).

    The sampler is BUMPS/DREAM-specific for now: the BUMPS check in
    :meth:`_run` is the seam where another backend would plug in.

    Parameters
    ----------
    fitter : Fitter
        A configured ``Fitter`` (or ``MultiFitter``) whose minimizer has been
        switched to ``AvailableMinimizers.Bumps``.
    x : Union[np.ndarray, List[np.ndarray]]
        Independent variable array (or list of arrays for ``MultiFitter``).
    y : Union[np.ndarray, List[np.ndarray]]
        Dependent variable array (or list of arrays for ``MultiFitter``).
    weights : Union[Optional[np.ndarray], List[Optional[np.ndarray]]], default=None
        Weight array (or list of arrays for ``MultiFitter``).
    vectorized : bool, default=False
        When ``True``, each x array may be multi-dimensional (e.g. an
        ``(N, M, 2)`` grid for a 2D model) and is left as-is.
    sampler_kwargs : Optional[dict], default=None
        Per-instance default keyword arguments forwarded to the BUMPS DREAM
        sampler on every run, merged with (and overridden by) per-call
        ``sampler_kwargs``.
    """

    def __init__(
        self,
        fitter: 'Fitter',
        x: Union[np.ndarray, List[np.ndarray]],
        y: Union[np.ndarray, List[np.ndarray]],
        weights: Union[Optional[np.ndarray], List[Optional[np.ndarray]]] = None,
        vectorized: bool = False,
        sampler_kwargs: Optional[dict] = None,
    ):
        self._fitter = fitter
        self._x = x
        self._y = y
        self._weights = weights
        self._vectorized = vectorized
        self._default_sampler_kwargs = dict(sampler_kwargs or {})
        self._state: Any = None  # BUMPS MCMCDraw (current chain state)
        self._results: Optional[SamplingResults] = None

    @property
    def state(self) -> Any:
        """Raw BUMPS MCMCDraw state (None before first sample/load_state)."""
        return self._state

    @property
    def results(self) -> Optional[SamplingResults]:
        """Results of the most recent sample/extend/load_state call."""
        return self._results

    @property
    def draws(self) -> Optional[np.ndarray]:
        """Posterior draws from the most recent run (or None)."""
        return self._results.draws if self._results is not None else None

    @property
    def param_names(self) -> Optional[List[str]]:
        """Parameter names from the most recent run (or None)."""
        return self._results.param_names if self._results is not None else None

    @property
    def logp(self) -> Optional[np.ndarray]:
        """Log-posterior values from the most recent run (or None)."""
        return self._results.logp if self._results is not None else None

    def _fingerprint(self) -> Optional[str]:
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
        population: Optional[int],
        resume_state: Optional[Any],
        sampler_kwargs: Optional[dict],
        progress_callback: Optional[Callable[[dict], Optional[bool]]],
        abort_test: Optional[Callable[[], bool]],
    ) -> SamplingResults:
        """Shared sampling engine for :meth:`sample`, :meth:`extend` and the
        deprecated ``Fitter.mcmc_sample`` shim.

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

        x_fit, y_new, w_new, wrapped = self._fitter._prepare_sampling(
            self._x, self._y, self._weights, self._vectorized
        )

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
        population: Optional[int] = None,
        sampler_kwargs: Optional[dict] = None,
        progress_callback: Optional[Callable[[dict], Optional[bool]]] = None,
        abort_test: Optional[Callable[[], bool]] = None,
    ) -> SamplingResults:
        """Run fresh Bayesian MCMC sampling on the bound data.

        Calling ``sample()`` on a sampler that already holds a chain starts a
        **fresh** chain — the previous state and results are replaced. Use
        :meth:`extend` to continue an existing chain.

        Parameters
        ----------
        samples : int, default=10000
            Number of retained DREAM samples requested from BUMPS.
        burn : int, default=2000
            Burn-in steps to discard before collecting samples.
        thin : int, default=10
            Thinning interval — only every ``thin``-th sample is kept,
            which reduces autocorrelation between consecutive draws.
        population : Optional[int], default=None
            DREAM population **scale factor** (not an absolute chain count):
            BUMPS creates ``ceil(population * n_parameters)`` parallel chains.
        sampler_kwargs : Optional[dict], default=None
            Additional keyword arguments forwarded to the BUMPS DREAM
            sampler (merged over the instance defaults).
        progress_callback : Optional[Callable[[dict], Optional[bool]]], default=None
            Optional callback invoked at each DREAM generation. The payload
            dict includes ``iteration`` and ``sampling: True``.
        abort_test : Optional[Callable[[], bool]], default=None
            Optional callable that returns ``True`` to abort sampling early.

        Returns
        -------
        SamplingResults
            Structured sampling results (also stored on :attr:`results`).

        Notes
        -----
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
        total_samples: Optional[int] = None,
        sampler_kwargs: Optional[dict] = None,
        progress_callback: Optional[Callable[[dict], Optional[bool]]] = None,
        abort_test: Optional[Callable[[], bool]] = None,
    ) -> SamplingResults:
        """Continue the existing chain with additional samples.

        DREAM stores draws in a fixed-size ring buffer sized to its
        ``samples`` parameter; this method does the ring-buffer arithmetic
        for you (``samples = stored_generations * population +
        additional_samples``) so no existing draws are lost, regardless of
        the thinning interval. Runs with ``burn=0`` — re-burning a converged
        chain is usually a mistake. The DREAM population is recovered from
        the saved state and cannot be changed on extend.

        Parameters
        ----------
        additional_samples : int, default=5000
            Number of additional DREAM samples to draw, in the same units
            as ``samples`` in :meth:`sample`. With thinning, approximately
            ``additional_samples / thin`` new draws are retained.
        thin : int, default=10
            Thinning interval for the retained draws.
        total_samples : Optional[int], default=None
            Advanced: total retained samples requested from the ring buffer,
            **overriding** the ``additional_samples`` arithmetic. With
            ``total_samples=N``, only the last N draws are retained.
        sampler_kwargs : Optional[dict], default=None
            Additional keyword arguments forwarded to the BUMPS DREAM
            sampler (merged over the instance defaults).
        progress_callback : Optional[Callable[[dict], Optional[bool]]], default=None
            Optional callback invoked at each DREAM generation.
        abort_test : Optional[Callable[[], bool]], default=None
            Optional callable that returns ``True`` to abort sampling early.

        Returns
        -------
        SamplingResults
            Structured sampling results for the full (extended) chain.

        Raises
        ------
        RuntimeError
            If there is no chain to extend (call :meth:`sample` or
            :meth:`load_state` first), or the minimizer is not BUMPS.
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
        on :meth:`load_state`).

        Parameters
        ----------
        path : str
            File path prefix. BUMPS appends its own suffixes.

        Raises
        ------
        RuntimeError
            If no chain state exists yet (call :meth:`sample` first).
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
        create the chain — :meth:`extend` then continues the saved chain. If
        the sidecar carries a data fingerprint and it does not match this
        sampler's bound data, a ``UserWarning`` is emitted (extending a chain
        against different data is undefined behaviour).

        Populates :attr:`state` and :attr:`results` (draws, log-posterior and
        parameter names) from the saved chain, so summaries and
        ``PosteriorResults.from_sampler()`` work without resampling.

        Parameters
        ----------
        path : str
            File path prefix used in :meth:`save`.
        skip : int, default=0
            Discard the first ``skip`` saved generations on load. Useful for
            trimming additional burn-in without re-sampling.

        Returns
        -------
        SamplingResults
            The reloaded chain results (also stored on :attr:`results`).
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
