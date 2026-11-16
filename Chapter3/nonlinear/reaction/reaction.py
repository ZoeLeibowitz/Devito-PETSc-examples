import os
import numpy as np
from sympy import sqrt

from devito import Grid, Function, Eq, Operator, Border, configuration

from devito.petsc import petscsolve, EssentialBC
from devito.petsc.initialize import PetscInitialize

import matplotlib.pyplot as plt

configuration['compiler'] = 'custom'
os.environ['CC'] = 'mpicc'


# 1D nonlinear reaction diffusion problem
# Solving -u'' + rho*sqrt(u) = 0 on [0, 1]
# Dirichlet BCs: u(0) = alpha, u(1) = beta
# rho = 10, M = (rho/12)^2, alpha = M, beta = 16M
# Exact solution: u(x) = M(x + 1)^4
# ref - https://github.com/bueler/p4pdes/blob/master/c/ch4/reaction.c

PetscInitialize()

rho = np.float64(10.)
M = (rho/12.)**2
alpha, beta = M, 16.*M


def exact(x):
    return M*(x + 1.)**4


# The residual is nonlinear in the target, so petscsolve's defaults (single
# linear solve, snes_type ksponly) do not apply.
# These options select Newton's method, with the action of the Jacobian approximated by
# finite differences of the generated residual (JFNK, -snes_mf) and CG without
# a preconditioner

# Ofc, un-preconditioned CG is not a good idea for a real problem - see ed bueler ref page 87
params = {'snes_type': 'newtonls', 'snes_mf': None,
        'ksp_type': 'cg', 'pc_type': 'none',
        'snes_converged_reason': None, 'ksp_converged_reason': None}


def solve(n, solver_parameters=None):
    grid = Grid(shape=(n,), extent=(1.,), dtype=np.float64)

    u = Function(name='u', grid=grid, space_order=2)
    bc = Function(name='bc', grid=grid, space_order=2)

    X = np.linspace(0, 1, n).astype(np.float64)

    # Initial guess.. the line between the boundary
    # values (same as in reaction.c)
    u.data[:] = alpha*(1. - X) + beta*X

    bc.data[0] = alpha  # u(0) = alpha
    bc.data[-1] = beta  # u(1) = beta

    eqn = Eq(-u.laplace + rho*sqrt(u), 0, subdomain=grid.interior)
    bcs = [EssentialBC(u, bc, subdomain=Border(grid, 1))]

    petsc = petscsolve(
        [eqn] + bcs, target=u,
        solver_parameters=solver_parameters,
        options_prefix='reaction',
        get_info=['SNESGetIterationNumber']
    )

    op = Operator(petsc, language='petsc')
    summary = op.apply()

    newton_iters = summary.petsc[('section0', 'reaction')].SNESGetIterationNumber
    infinity_norm = np.linalg.norm(u.data[:] - exact(X), ord=np.inf)
    return infinity_norm, newton_iters


# Convergence test
# n = 9, 17, 33, 65, 129, 257, 513, 1025, 2049
n_values = [2**k + 1 for k in range(3, 12)]
h = np.array([1./(n - 1) for n in n_values])

infinity_norms = []
newton_iters = []
for n in n_values:
    e, its = solve(n, params)
    infinity_norms.append(e)
    newton_iters.append(its)

slope, intercept = np.polyfit(np.log(h), np.log(infinity_norms), 1)
print(infinity_norms)
print(newton_iters)
assert slope > 1.9
assert slope < 2.1

# Convergence Plot
plt.figure(figsize=(6, 5))
plt.loglog(h, infinity_norms, 'o-', label=rf'Observed order $\hat{{q}} = {slope:.3f}$', color='orange')
plt.loglog(h, infinity_norms[0]*(h/h[0])**2, 'k--', lw=0.8,
           label='Reference line of slope 2')
plt.xlabel(r'Grid spacing $h$')
plt.ylabel(r'$\|e\|_\infty$')
plt.legend()
plt.tight_layout()
plt.savefig("reaction.png", dpi=200)
plt.show()


# petscsolve defaults treat the problem as linear. The solve then reports
# convergence... but the answer is wrong
e, its = solve(33)
print(f"Default solver options, n=33: error {e:.3e}")
