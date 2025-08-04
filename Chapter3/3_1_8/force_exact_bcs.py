import os
import numpy as np

from devito import (Grid, Function, Eq, Operator, switchconfig,
                    configuration, SubDomain, norm)
from devito.finite_differences.differentiable import EvalDerivative
from devito.symbolics import retrieve_functions, INT

from devito.petsc import PETScSolve, EssentialBC
from devito.petsc.initialize import PetscInitialize

import matplotlib.pyplot as plt

configuration['compiler'] = 'custom'
os.environ['CC'] = 'mpicc'


# python3 force_exact_bcs.py -ksp_converged_reason -ksp_type cg -ksp_rtol 1e-13 -pc_type none
# modify the equations near boundaries -> 4th order discretisation


# solve for inner 
# set the analytical solution on the outer 2 layers

PetscInitialize()


def exact(x, y, k1=1., k2=1.):
    tmp1 = np.float64(np.pi/8.0) * np.float64(np.pi/8.0)
    tmp2 = k1**2 + k2**2
    tmp3 = tmp1 * tmp2
    tmp4 = -1.0/tmp3
    tmp5 = np.float64(np.sin((np.pi*x*k1)/8.0))
    tmp6 = np.float64(np.sin((np.pi*y*k2)/8.0))
    return tmp4*tmp5*tmp6

Lx = np.float64(16.)
Ly = np.float64(16.)


n_values = [900, 950, 1000, 1050, 1100, 1150, 1200, 1250, 1300]

h = np.array([Lx/(n-1) for n in n_values])
infinity_norms = []
discrete_l2_norms = []
ksp_iters = []

so = 4

for n in n_values:

    # Subdomains to implement BCs
    class SubTopBC(SubDomain):
        name = 'subtop'

        def define(self, dimensions):
            x, y = dimensions
            return {x: x, y: ('right', 1)}

    class SubBottomBC(SubDomain):
        name = 'subbottom'

        def define(self, dimensions):
            x, y = dimensions
            return {x: x, y: ('left', 1)}

    class SubLeftBC(SubDomain):
        name = 'subleft'

        def define(self, dimensions):
            x, y = dimensions
            return {x: ('left', 1), y: ('middle', 1, 1)}

    class SubRightBC(SubDomain):
        name = 'subright'

        def define(self, dimensions):
            x, y = dimensions
            return {x: ('right', 1), y: ('middle', 1, 1)}
        
    ##################### modified stencil subdomains #####################

    class SubMain(SubDomain):
        name = 'submain'

        def define(self, dimensions):
            x, y = dimensions
            return {x: ('middle', 2, 2), y: ('middle', 2, 2)}
        

    class SubTopInner(SubDomain):
        name = 'subtopinner'

        def __init__(self, N):
            super().__init__()
            self.N = N

        def define(self, dimensions):
            x, y = dimensions
            return {x: ('middle', 1, 1), y: ('middle', self.N - 2, 1)}
        

    class SubBottomInner(SubDomain):
        name = 'subbottominner'

        def __init__(self, N):
            super().__init__()
            self.N = N

        def define(self, dimensions):
            x, y = dimensions
            return {x: ('middle', 1, 1), y: ('middle', 1, self.N - 2)}
        

    class SubLeftInner(SubDomain):
        name = 'subleftinner'

        def __init__(self, N):
            super().__init__()
            self.N = N

        def define(self, dimensions):
            x, y = dimensions
            return {x: ('middle', 1, self.N - 2), y: ('middle', 2, 2)}
    

    class SubRightInner(SubDomain):
        name = 'subrightinner'

        def __init__(self, N):
            super().__init__()
            self.N = N

        def define(self, dimensions):
            x, y = dimensions
            return {x: ('middle', self.N - 2, 1), y: ('middle', 2, 2)}


    sub1 = SubTopBC()
    sub2 = SubBottomBC()
    sub3 = SubLeftBC()
    sub4 = SubRightBC()

    sub5 = SubMain()
    sub6 = SubTopInner(n)
    sub7 = SubBottomInner(n)
    sub8 = SubLeftInner(n)
    sub9 = SubRightInner(n)


    subdomains = (sub1, sub2, sub3, sub4, sub5, sub6, sub7, sub8, sub9)

    grid = Grid(
        shape=(n, n), extent=(Lx, Ly), subdomains=subdomains, dtype=np.float64
    )

    u = Function(name='u', grid=grid, space_order=so)
    f = Function(name='f', grid=grid, space_order=so)
    bc = Function(name='bc', grid=grid, space_order=so)

    # eqn = Eq(u.dx2(weights=[-0.08333333333333333, 1.3333333333333333, -2.5, 1.3333333333333333, -0.08333333333333333]) + u.dy2(weights=[-0.08333333333333333, 1.3333333333333333, -2.5, 1.3333333333333333, -0.08333333333333333]), f, subdomain=sub5)

    eqn = Eq(u.laplace, f, subdomain=sub5)

    tmpx = np.linspace(0, Lx, n).astype(np.float64)
    tmpy = np.linspace(0, Ly, n).astype(np.float64)

    # Y, X = np.meshgrid(tmpx, tmpy)

    X, Y = np.meshgrid(tmpx, tmpy, indexing='ij')

    k1, k2 = 1., 1.

    f.data[:] = np.float64(np.sin((np.pi*X*k1)/8.0)) * np.float64(np.sin((np.pi*Y*k2)/8.0))

    u_exact = Function(name='u_exact', grid=grid, space_order=so)
    u_exact.data[:] = exact(X, Y)

    bc.data[:] = u_exact.data[:]
    bc.data[2:-2, 2:-2] = 0.

    # Create boundary condition expressions using subdomains
    bcs = [EssentialBC(u, bc, subdomain=sub1)] # top boundary
    bcs += [EssentialBC(u, bc, subdomain=sub2)] # bottom boundary
    bcs += [EssentialBC(u, bc, subdomain=sub3)] # left boundary
    bcs += [EssentialBC(u, bc, subdomain=sub4)] # right boundary

    bcs += [EssentialBC(u, bc, subdomain=sub7)]  # Left modify
    bcs += [EssentialBC(u, bc, subdomain=sub9)]  # Right modify
    bcs += [EssentialBC(u, bc, subdomain=sub8)]  # Bottom modify
    bcs += [EssentialBC(u, bc, subdomain=sub6)]  # Top modify

    exprs = [eqn] + bcs

    # u.data[:] = 0.001

    petsc = PETScSolve(exprs, target=u, solver_parameters={'ksp_rtol': 1e-12})

    # with switchconfig(log_level='DEBUG'):
    op = Operator(petsc, language='petsc')
    summary = op.apply()

    # from IPython import embed; embed()

    # iters = summary.petsc[('section0', None)].KSPGetIterationNumber
    # ksp_iters.append(iters)

    diff = Function(name='diff', grid=grid, space_order=so)
    diff.data[:] = u_exact.data[:] - u.data[:]

    # Compute infinity norm using numpy
    infinity_norm = np.linalg.norm(diff.data[2:-2, 2:-2].ravel(), ord=np.inf)
    infinity_norms.append(infinity_norm)
    print(infinity_norm)

    # Compute discrete L2 norm (RMS error)
    n_interior = np.prod([s - 1 for s in grid.shape])
    discrete_l2_norm = norm(diff) / np.sqrt(n_interior)
    discrete_l2_norms.append(discrete_l2_norm)
    print(discrete_l2_norm)


print(infinity_norms)
slope, intercept = np.polyfit(np.log(h), np.log(infinity_norms), 1)
# print(op.ccode)
# print(op.arguments())
# print(slope)
# # assert slope > 3.9
# # assert slope < 4.1

# Plot
plt.figure(figsize=(6, 5))
plt.loglog(h, infinity_norms, 'o-', label=f'Observed rate ≈ {slope:.3f}', color='orange')
plt.loglog(
    h, np.exp(intercept) * h**4,
    'k--',
    label=r'Reference slope $O(h^4)$'
)
plt.xlabel(r'Grid spacing h')
plt.ylabel(r'$\infty$-norm error')
plt.title('Convergence Plot')
plt.legend()
plt.tight_layout()
plt.savefig("3_1_8_forced.png", dpi=200)
plt.show()







# for comparison, plot the pure petsc code:
# ./fish -ksp_converged_reason -ksp_type cg -ksp_rtol 1e-12 -pc_type none
# infinity_norms = [3.004e-09,
#                   ]
plt.figure(figsize=(6, 5))
plt.loglog(h, infinity_norms, 'o-', label=f'Observed rate ≈ {slope:.3f}', color='orange')
plt.loglog(
    h, np.exp(intercept) * h**4,
    'k--',
    label=r'Reference slope $O(h^4)$'
)
plt.xlabel(r'Grid spacing h')
plt.ylabel(r'$\infty$-norm error')
plt.title('Convergence Plot')
plt.legend()
plt.tight_layout()
plt.savefig("pure_petsc.png", dpi=200)
plt.show()

