# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause
"""Quick benchmark: Bayesian DREAM sampling with multiprocessing.

Runs the same small sampling problem sequentially and with process workers,
printing wall-clock times so you can judge the speedup.
"""

import time
import warnings

import numpy as np

from easyscience import ObjBase
from easyscience import Parameter
from easyscience.fitting.multi_fitter import MultiFitter

# -- simple test model --------------------------------------------------------

# Simulate an expensive model by adding a configurable CPU burn per evaluation.
# Set to 0.0 for the trivial model; try 0.02–0.1 to see multiprocessing speedup.
_MODEL_DELAY = 0.09  # seconds of CPU work per model call


class Line(ObjBase):
    m: Parameter
    c: Parameter

    def __init__(self, m_val: float, c_val: float):
        super().__init__(
            'line',
            m=Parameter('m', m_val),
            c=Parameter('c', c_val),
        )

    def __call__(self, x: np.ndarray) -> np.ndarray:
        if _MODEL_DELAY > 0:
            # burn CPU to simulate a real physics model
            t0 = time.perf_counter()
            while time.perf_counter() - t0 < _MODEL_DELAY:
                _ = np.sum(np.sin(x) ** 2 + np.cos(x) ** 2)
        return self.m.value * x + self.c.value

# -- helpers ------------------------------------------------------------------

def run_sample(n_workers: int | None, **sample_kwargs) -> tuple[dict, float]:
    """Run one DREAM sampling call and return (result_dict, wall_seconds)."""
    x = np.linspace(0, 10, 60)
    y_true = 2.5 * x + 1.3
    rng = np.random.default_rng(42)
    y = y_true + rng.normal(0, 0.3, size=x.shape)
    weights = np.full_like(x, 1.0 / 0.3)

    model = Line(2.0, 1.0)
    model.m.fixed = False
    model.c.fixed = False

    fitter = MultiFitter([model], [model])
    fitter.switch_minimizer('Bumps')

    t0 = time.perf_counter()
    result = fitter.mcmc_sample(
        x=[x],
        y=[y],
        weights=[weights],
        n_workers=n_workers,
        **sample_kwargs,
    )
    elapsed = time.perf_counter() - t0
    return result, elapsed

def summarise(label: str, result: dict, elapsed: float) -> None:
    draws = result['draws']
    print(f'  {label:>12s}  {elapsed:6.2f} s  '
          f'draws shape {draws.shape}  '
          f'params: {result["param_names"]}')

# -- main ---------------------------------------------------------------------

def main() -> None:
    sample_kwargs = dict(samples=200, burn=50, thin=2, population=5, seed=123)

    print('Bayesian multiprocessing quick test')
    print('-----------------------------------')
    print(f'  config: {sample_kwargs}')
    print()

    # 1. sequential (default)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        res_seq, t_seq = run_sample(n_workers=None, **sample_kwargs)
    summarise('sequential', res_seq, t_seq)

    # 2. n_workers=1 (same as sequential, but explicit)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        res_w1, t_w1 = run_sample(n_workers=1, **sample_kwargs)
    summarise('n_workers=1', res_w1, t_w1)

    # 3. n_workers=2 (actual multiprocessing)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        res_w2, t_w2 = run_sample(n_workers=2, **sample_kwargs)
    summarise('n_workers=2', res_w2, t_w2)

    # 4. n_workers=4
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        res_w4, t_w4 = run_sample(n_workers=4, **sample_kwargs)
    summarise('n_workers=4', res_w4, t_w4)

    print()
    for label, t_val in [('n_workers=2', t_w2), ('n_workers=4', t_w4)]:
        ratio = t_seq / t_val
        tag = f'{ratio:.1f}× speedup' if ratio > 1 else f'{1/ratio:.1f}× slower'
        print(f'  {label:>12s}  {tag}  (seq {t_seq:.2f}s → {t_val:.2f}s)')

if __name__ == '__main__':
    main()
