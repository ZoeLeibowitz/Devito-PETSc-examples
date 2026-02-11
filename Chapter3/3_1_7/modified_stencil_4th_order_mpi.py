import os
import numpy as np

import matplotlib
matplotlib.use("Agg")  # Fully deterministic non-interactive backend
import matplotlib.pyplot as plt

from devito import (Grid, Function, Eq, Operator,
                    configuration, SubDomain)
from devito.symbolics import retrieve_functions

from devito.petsc import petscsolve, EssentialBC
from devito.petsc.initialize import PetscInitialize
from devito.mpi.distributed import MPI


configuration['compiler'] = 'custom'
os.environ['CC'] = 'mpicc'

# Increase the precision of finite difference weights to avoid numerical issues
# since we are using a 4th order stencil
import devito.finite_differences.finite_difference as fdiff
fdiff._PRECISION = 18

# python3 modified_stencil_4th_order.py -ksp_converged_reason -ksp_type cg -ksp_rtol 1e-13 -pc_type none
# modify the equations near boundaries -> 4th order discretisation


PetscInitialize()


def modified_left(eq, subdomain):
    lhs, rhs = eq.evaluate.args

    # Get horizontal subdimension and its parent
    xfs = subdomain.dimensions[0]
    x = xfs.parent

    yfs = subdomain.dimensions[1]
    y = yfs.parent

    # Functions present in stencil
    funcs = retrieve_functions(lhs-rhs)

    for f in funcs:
        xind = f.indices[-2]
        yind = f.indices[-1]
        h_x = f.grid.spacing_symbols[0]
        if xind == x + h_x:
            u_2_h_x = f
        elif xind == x and yind == y:
            u_h_x = f
        else:
            continue
    mapper = {}
    for f in funcs:
        # Get the x index
        xind = f.indices[-2]
        h_x = f.grid.spacing_symbols[0]
        if xind == x - 2*h_x:
            a3 = (u_2_h_x - 2.*u_h_x) / (6.*h_x**3)
            a1 = (u_h_x - a3*(h_x**3)) / h_x

            new = -1.*a1*h_x - a3*(h_x**3)

            # from IPython import embed; embed()  # Debugging lin
            mapper.update({f: -f.subs({xind: u_h_x.indices[0]})})
            # mapper.update({f: -f})

    return Eq(lhs.subs(mapper), rhs.subs(mapper), subdomain=subdomain)



def modified_right(eq, subdomain):
    lhs, rhs = eq.evaluate.args

    # Get horizontal subdimension and its parent
    xfs = subdomain.dimensions[0]
    x = xfs.parent

    yfs = subdomain.dimensions[1]
    y = yfs.parent

    # Functions present in stencil
    funcs = retrieve_functions(lhs-rhs)

    for f in funcs:
        xind = f.indices[-2]
        yind = f.indices[-1]
        h_x = f.grid.spacing_symbols[0]
        if xind == x and yind == y:
            u_minus_h_x = f
        else:
            continue
    mapper = {}
    for f in funcs:
        # Get the x index
        xind = f.indices[-2]
        h_x = f.grid.spacing_symbols[0]
        if xind == x + 2*h_x:

            new = u_minus_h_x

            mapper.update({f: -f.subs({xind: u_minus_h_x.indices[0]})})
            # mapper.update({f: -f})
    return Eq(lhs.subs(mapper), rhs.subs(mapper), subdomain=subdomain)


def modified_bottom(eq, subdomain):
    lhs, rhs = eq.evaluate.args

    # Get horizontal subdimension and its parent
    xfs = subdomain.dimensions[0]
    x = xfs.parent

    yfs = subdomain.dimensions[1]
    y = yfs.parent

    # Functions present in stencil
    funcs = retrieve_functions(lhs-rhs)

    for f in funcs:
        xind = f.indices[-2]
        yind = f.indices[-1]
        h_y = f.grid.spacing_symbols[1]
        if xind == x and yind == y:
            tmp = f
        else:
            continue
    mapper = {}
    for f in funcs:
        # Get the x index
        yind = f.indices[-1]
        h_y = f.grid.spacing_symbols[1]
        if yind == y - 2*h_y:
            new = -tmp

            mapper.update({f: -f.subs({yind: tmp.indices[-1]})})
            # mapper.update({f: -f})
    return Eq(lhs.subs(mapper), rhs.subs(mapper), subdomain=subdomain)



def modified_top(eq, subdomain):
    lhs, rhs = eq.evaluate.args

    # Get horizontal subdimension and its parent
    xfs = subdomain.dimensions[0]
    x = xfs.parent

    yfs = subdomain.dimensions[1]
    y = yfs.parent

    # Functions present in stencil
    funcs = retrieve_functions(lhs-rhs)

    for f in funcs:
        xind = f.indices[-2]
        yind = f.indices[-1]
        h_x = f.grid.spacing_symbols[0]
        if xind == x and yind == y:
            tmp = f
        else:
            continue
    mapper = {}
    for f in funcs:
        # Get the y index
        yind = f.indices[-1]
        h_y = f.grid.spacing_symbols[1]
        if yind == y + 2*h_y:

            mapper.update({f: -f.subs({yind: tmp.indices[-1]})})
            # mapper.update({f: -f})
    return Eq(lhs.subs(mapper), rhs.subs(mapper), subdomain=subdomain)


def exact(x, y, k1=1, k2=1):
    tmp1 = np.float64(np.pi/8.0) * np.float64(np.pi/8.0)
    tmp2 = k1**2 + k2**2
    tmp3 = tmp1 * tmp2
    tmp4 = -1.0/tmp3
    tmp5 = np.float64(np.sin((np.pi*x*k1)/8.0))
    tmp6 = np.float64(np.sin((np.pi*y*k2)/8.0))
    return tmp4*tmp5*tmp6

Lx = np.float64(16.)
Ly = np.float64(16.)


# n = 9, 17, 33, 65, 129, 257, 513
n_values = [2**k + 1 for k in range(3, 10)]

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
        

    class SubTopModify(SubDomain):
        name = 'subtopmodify'

        def __init__(self, N):
            super().__init__()
            self.N = N

        def define(self, dimensions):
            x, y = dimensions
            return {x: ('middle', 2, 2), y: ('middle', self.N - 2, 1)}
        

    class SubLeftModify(SubDomain):
        name = 'subleftmodify'

        def __init__(self, N):
            super().__init__()
            self.N = N

        def define(self, dimensions):
            x, y = dimensions
            return {x: ('middle', 1, self.N - 2), y: ('middle', 2, 2)}
        

    class SubBottomModify(SubDomain):
        name = 'subbottommodify'

        def __init__(self, N):
            super().__init__()
            self.N = N

        def define(self, dimensions):
            x, y = dimensions
            return {x: ('middle', 2, 2), y: ('middle', 1, self.N - 2)}
        

    class SubRightModify(SubDomain):
        name = 'subrightmodify'

        def __init__(self, N):
            super().__init__()
            self.N = N

        def define(self, dimensions):
            x, y = dimensions
            return {x: ('middle', self.N - 2, 1), y: ('middle', 2, 2)}
        

    class SubTopLeftModify(SubDomain):
        name = 'subtopleftmodify'

        def __init__(self, N):
            super().__init__()
            self.N = N

        def define(self, dimensions):
            x, y = dimensions
            return {x: ('middle', 1, self.N - 2), y: ('middle', self.N - 2, 1)}
        

    class SubBottomLeftModify(SubDomain):
        name = 'subbottomleftmodify'

        def __init__(self, N):
            super().__init__()
            self.N = N

        def define(self, dimensions):
            x, y = dimensions
            return {x: ('middle', 1, self.N - 2), y: ('middle', 1, self.N - 2)}
        
    
    class SubTopRightModify(SubDomain):
        name = 'subtoprightmodify'

        def __init__(self, N):
            super().__init__()
            self.N = N

        def define(self, dimensions):
            x, y = dimensions
            return {x: ('middle', self.N - 2, 1), y: ('middle', self.N - 2, 1)}
        


    class SubBottomRightModify(SubDomain):
        name = 'subbottomrightmodify'

        def __init__(self, N):
            super().__init__()
            self.N = N

        def define(self, dimensions):
            x, y = dimensions
            return {x: ('middle', self.N - 2, 1), y: ('middle', 1, self.N - 2)}


    sub1 = SubTopBC()
    sub2 = SubBottomBC()
    sub3 = SubLeftBC()
    sub4 = SubRightBC()

    sub5 = SubMain()
    sub6 = SubTopModify(n)
    sub7 = SubLeftModify(n)
    sub8 = SubBottomModify(n)
    sub9 = SubRightModify(n)

    sub10 = SubTopLeftModify(n)
    sub11 = SubBottomLeftModify(n)
    sub12 = SubTopRightModify(n)
    sub13 = SubBottomRightModify(n)


    subdomains = (sub1, sub2, sub3, sub4, sub5, sub6, sub7, sub8, sub9, sub10, sub11, sub12, sub13)

    grid = Grid(
        shape=(n, n), extent=(Lx, Ly), subdomains=subdomains, dtype=np.float64
    )

    u = Function(name='u', grid=grid, space_order=so)
    f = Function(name='f', grid=grid, space_order=so)
    bc = Function(name='bc', grid=grid, space_order=so)

    eqn = Eq(u.laplace, f, subdomain=sub5)

    tmpx = np.linspace(0, Lx, n).astype(np.float64)
    tmpy = np.linspace(0, Ly, n).astype(np.float64)

    Y, X = np.meshgrid(tmpx, tmpy)

    k1, k2 = 1., 1.

    f.data[:] = np.float64(np.sin((np.pi*X*k1)/8.0)) * np.float64(np.sin((np.pi*Y*k2)/8.0))

    bc.data[:, 0] = 0.
    bc.data[:, -1] = 0.
    bc.data[0, :] = 0.
    bc.data[-1, :] = 0.

    # Create boundary condition expressions using subdomains
    bcs = [EssentialBC(u, bc, subdomain=sub1)] # top boundary
    bcs += [EssentialBC(u, bc, subdomain=sub2)] # bottom boundary
    bcs += [EssentialBC(u, bc, subdomain=sub3)] # left boundary
    bcs += [EssentialBC(u, bc, subdomain=sub4)] # right boundary

    bcs += [modified_left(eqn, sub7)]  # Left modify
    bcs += [modified_right(eqn, sub9)]  # Right modify
    bcs += [modified_bottom(eqn, sub8)]  # Bottom modify
    bcs += [modified_top(eqn, sub6)]  # Top modify

    bcs += [modified_left(modified_top(eqn, sub10), sub10)]  # Top left modify
    bcs += [modified_right(modified_top(eqn, sub12), sub12)]  # Top right modify
    bcs += [modified_left(modified_bottom(eqn, sub11), sub11)]  # Bottom left modify
    bcs += [modified_right(modified_bottom(eqn, sub13), sub13)]  # Bottom right modify

    exprs = [eqn] + bcs

    # Can play around with initial guess -> if it's zero then cg just converges in 1 iteration because
    # the rhs is an eigenvector of the matrix -> I think?
    # u.data[:] = 0.001

    petsc = petscsolve(
        exprs, target=u,
        solver_parameters={'ksp_rtol': 1e-13, 'ksp_type': 'cg', 'pc_type': 'none'},
        options_prefix='modified_4th_order'
    )

    op = Operator(petsc, language='petsc')
    summary = op.apply()

    u_exact = Function(name='u_exact', grid=grid, space_order=so)
    u_exact.data[:] = exact(X, Y)

    diff = Function(name='diff', grid=grid, space_order=so)
    diff.data[:] = u_exact.data[:] - u.data[:]

    gathered = diff.data._gather()
    comm = grid.comm

    if comm is not None and configuration['mpi']:
        if comm != MPI.COMM_NULL and comm.rank == 0:
            infinity_norm_mpi = np.linalg.norm(np.asarray(gathered).ravel(), ord=np.inf)
        else:
            infinity_norm_mpi = None
    else:
        infinity_norm_mpi = None

    infinity_norms.append(infinity_norm_mpi)


size = comm.size
if comm.rank == 0:
    # print(infinity_norms)
    slope, intercept = np.polyfit(np.log(h), np.log(infinity_norms), 1)

    print(slope)
    assert slope > 3.9
    assert slope < 4.1

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
    plt.title(f'Convergence Plot (MPI processes = {size})')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"3_1_7_mpi_procs{size}.png", dpi=200, metadata={})
    plt.show()
