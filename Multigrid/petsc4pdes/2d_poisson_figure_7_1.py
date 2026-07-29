"""
Reproduces p4pdes Figure 7.1: condition number kappa(M^{-1}A) vs h, read from
the last "max/min" value printed by -ksp_monitor_singular_value once KSP has
converged.
"""
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
RUNNER = HERE / "solver_figure_7_1.py"

Lx = 1.0

# n = 17, 33, 65, 129, 257, 513, 1025
n_values = [2**k + 1 for k in range(4, 11)]
h = np.array([Lx/(n-1) for n in n_values])


def last_condition_number(text):
    matches = re.findall(r'max/min\s+([\d.eE+-]+)', text)
    if not matches:
        raise RuntimeError(
            'no "max/min" values found; is -ksp_monitor_singular_value active?'
        )
    return float(matches[-1])


def run_series(solver_type):
    condition_numbers = []
    for n in n_values:
        print(f"running n={n} ({solver_type}) ...", flush=True)
        result = subprocess.run(
            [sys.executable, str(RUNNER), str(n), solver_type],
            cwd=HERE, capture_output=True, text=True,
        )
        if result.returncode != 0:
            print(result.stdout)
            print(result.stderr, file=sys.stderr)
            raise RuntimeError(
                f"run failed for n={n}, solver_type={solver_type} "
                f"(exit {result.returncode})"
            )

        kappa = last_condition_number(result.stdout)
        condition_numbers.append(kappa)

        iters_match = re.search(r'iters=(\d+)', result.stdout)
        iters = int(iters_match.group(1)) if iters_match else None

        print(f"  kappa(M^-1 A) = {kappa:.6e}, KSP iterations = {iters}")
    return condition_numbers


condition_numbers_mg = run_series('mg')
condition_numbers_none = run_series('none')


# Plot: condition number vs h (p4pdes Figure 7.1)
plt.figure(figsize=(6, 5))
plt.loglog(h, condition_numbers_none, 'o', markerfacecolor='none',
           markeredgecolor='black', label='none')
plt.loglog(h, condition_numbers_mg, '*', markerfacecolor='none',
           markeredgecolor='tab:blue', label='mg')
plt.xlabel(r'$h$')
plt.ylabel(r'condition number of $M^{-1}A$')
plt.ylim(5e-1, 1e6)
plt.title('Condition number vs. h')
plt.legend()
plt.tight_layout()
plt.savefig("petsc4pdes_figure_7_1.png", dpi=200)
plt.show()
