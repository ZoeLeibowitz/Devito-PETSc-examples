import os
import numpy as np

from devito import (Grid, Function, TimeFunction, Eq, Operator, switchconfig,
                    configuration, SubDomain, norm)
from devito.symbolics import retrieve_functions, INT

from devito.petsc import PETScSolve, EssentialBC
from devito.petsc.initialize import PetscInitialize

import matplotlib.pyplot as plt

configuration['compiler'] = 'custom'
os.environ['CC'] = 'mpicc'


# TODO: which solver?
# python3 1d_heat_explicit.py -ksp_converged_reason -ksp_type gmres -ksp_rtol 1e-12 -pc_type none

# 1D test
# Solving u.dt = alpha * u.laplace + f(x)
# Dirichlet BCs: u(0, t) = 0, u(L, t) = 0
# Manufactured solution: u(x, t) = 5tx(L - x), with corresponding RHS f(x) = 5x(L - x) + 10*alpha*t
# ref - https://hplgit.github.io/fdm-book/doc/pub/book/pdf/fdm-book-4print.pdf

PetscInitialize()

# Subdomains to implement BCs
class SubLeft(SubDomain):
    name = 'subleft'

    def define(self, dimensions):
        x, = dimensions
        return {x: ('left', 1)}


class SubRight(SubDomain):
    name = 'subright'

    def define(self, dimensions):
        x, = dimensions
        return {x: ('right', 1)}
    

def neumann_right(eq, subdomain):
    lhs, rhs = eq.evaluate.args

    # Get horizontal subdimension and its parent
    xfs = subdomain.dimensions[0]
    x = xfs.parent

    # Functions present in stencil
    funcs = retrieve_functions(lhs-rhs)

    mapper = {}
    for f in funcs:
        # Get the x index
        xind = f.indices[-1]
        if (xind - x).as_coeff_Mul()[0] > 0:
            tmp = x - INT(abs(x.symbolic_max - xind))
            mapper.update({f: f.subs({xind: tmp})})

    return Eq(lhs.subs(mapper), rhs.subs(mapper), subdomain=subdomain)


sub1 = SubLeft()
sub2 = SubRight()

subdomains = (sub1, sub2,)


def exact(x):
    return -np.float64(np.exp(x))


Lx = np.float64(1.)

# n = 9, 17, 33, 65, 129, 257, 513, 1025, 2049, 4097, 8193
n_values = [2**k + 1 for k in range(3, 14)]
n_values = [51]

dx = np.array([Lx/(n-1) for n in n_values])

alpha = 1.22e-3
# sigma = 0.5
sigma = 5.0
dt = sigma * dx[0]**2 / alpha
nt = 100

infinity_norms = []
discrete_l2_norms = []
ksp_iters = []

for n in n_values:
    grid = Grid(
        shape=(n,), extent=(Lx,), subdomains=subdomains, dtype=np.float64
    )

    u = TimeFunction(name='u', grid=grid, space_order=2)
    bc = Function(name='bc', grid=grid, space_order=2)

    u.data[0][0] = 100.  # Initial condition

    eqn = Eq(u.dt, alpha*u.forward.laplace, subdomain=grid.interior)

    X = np.linspace(0, Lx, n).astype(np.float64)

    bc.data[0] = np.float64(100.0)

    # Create boundary condition expressions using subdomains
    bcs = [EssentialBC(u.forward, bc, subdomain=sub1)]
    bcs += [neumann_right(eqn, sub2)]

    exprs = [eqn] + bcs
    petsc = PETScSolve(exprs, target=u.forward, solver_parameters={'ksp_rtol': 1e-12})

    with switchconfig(log_level='DEBUG'):
        op = Operator(petsc, language='petsc')
        summary = op.apply(time=nt, dt=dt)

    # u_exact = Function(name='u_exact', grid=grid, space_order=2)
    # u_exact.data[:] = exact(X)

    # diff = Function(name='diff', grid=grid, space_order=2)
    # diff.data[:] = u_exact.data[:] - u.data[:]

#     # Compute infinity norm using numpy
#     # TODO: Figure out how to compute the infinity norm using Devito
#     infinity_norm = np.linalg.norm(diff.data[:].ravel(), ord=np.inf)
#     infinity_norms.append(infinity_norm)

#     # Compute discrete L2 norm (RMS error)
#     n_interior = np.prod([s - 1 for s in grid.shape])
#     discrete_l2_norm = norm(diff) / np.sqrt(n_interior)
#     discrete_l2_norms.append(discrete_l2_norm)


from matplotlib import pyplot

# Set the font family and size to use for Matplotlib figures.
pyplot.rcParams['font.family'] = 'serif'
pyplot.rcParams['font.size'] = 16


# Plot the temperature along the rod.
pyplot.figure(figsize=(6.0, 4.0))
pyplot.xlabel('Distance [m]')
pyplot.ylabel('Temperature [C]')
pyplot.grid()
pyplot.plot(X, u.data[-1], color='C0', linestyle='-', linewidth=2)
pyplot.xlim(0.0, 1.)
pyplot.ylim(0.0, 100.0)

# save fig
fig_path = '1d_heat_explicit.png'
pyplot.savefig(fig_path, bbox_inches='tight', dpi=300)