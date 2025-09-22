import os
import numpy as np

from devito import (Grid, Function, Eq, Operator, switchconfig,
                    configuration, SubDomain, norm)
from devito.symbolics import retrieve_functions, INT

from devito.petsc import petscsolve, EssentialBC
from devito.petsc.initialize import PetscInitialize

import matplotlib.pyplot as plt

configuration['compiler'] = 'custom'
os.environ['CC'] = 'mpicc'

# CHECK THIS -> what solver to use?
# check symmetry etc
# python3 2d_poisson_neumann.py -ksp_converged_reason -ksp_type gmres -ksp_rtol 1e-12 -pc_type none

# 2D test
# Solving -u_xx - u_yy = 8pi**2 * cos(2pix) * cos(2piy)
# Homogeneous Neumann boundary conditions
# Manufactured solution: u(x,y) = cos(2pix) * cos(2piy)
# ref - https://people.tamu.edu/~guermond//M610_SPRING_2016/ca03.pdf

PetscInitialize()

# TODO: explore optimisation using BOUNDARY_MIRROR
so = 2

class SubTop(SubDomain):
    name = 'subtop'

    def __init__(self, S_O):
        super().__init__()
        self.S_O = S_O

    def define(self, dimensions):
        x, y = dimensions
        return {x: ('middle', 1, 1), y: ('right', self.S_O//2)}


class SubBottom(SubDomain):
    name = 'subbottom'

    def __init__(self, S_O):
        super().__init__()
        self.S_O = S_O

    def define(self, dimensions):
        x, y = dimensions
        return {x: ('middle', 1, 1), y: ('left', self.S_O//2)}


class SubLeft(SubDomain):
    name = 'subleft'

    def __init__(self, S_O):
        super().__init__()
        self.S_O = S_O

    def define(self, dimensions):
        x, y = dimensions
        return {x: ('left', self.S_O//2), y: ('middle', 1, 1)}


class SubRight(SubDomain):
    name = 'subright'

    def __init__(self, S_O):
        super().__init__()
        self.S_O = S_O

    def define(self, dimensions):
        x, y = dimensions
        return {x: ('right', self.S_O//2), y: ('middle', 1, 1)}


class SubPointBottomLeft(SubDomain):
    name = 'subpointbottomleft'

    def define(self, dimensions):
        x, y = dimensions
        return {x: ('left', 1), y: ('left', 1)}


class SubPointBottomRight(SubDomain):
    name = 'subpointbottomright'

    def define(self, dimensions):
        x, y = dimensions
        return {x: ('right', 1), y: ('left', 1)}


class SubPointTopLeft(SubDomain):
    name = 'subpointtopleft'

    def define(self, dimensions):
        x, y = dimensions
        return {x: ('left', 1), y: ('right', 1)}


class SubPointTopRight(SubDomain):
    name = 'subpointtopright'

    def define(self, dimensions):
        x, y = dimensions
        return {x: ('right', 1), y: ('right', 1)}


def neumann_bottom(eq, subdomain):
    lhs, rhs = eq.evaluate.args

    # Get vertical subdimension and its parent
    yfs = subdomain.dimensions[-1]
    y = yfs.parent

    # Functions present in stencil
    funcs = retrieve_functions(lhs-rhs)

    mapper = {}
    for f in funcs:
        # Get the y index
        yind = f.indices[-1]
        if (yind - y).as_coeff_Mul()[0] < 0:
            mapper.update({f: f.subs({yind: INT(abs(yind))})})

    return Eq(lhs.subs(mapper), rhs.subs(mapper), subdomain=subdomain)


def neumann_top(eq, subdomain):
    lhs, rhs = eq.evaluate.args

    # Get vertical subdimension and its parent
    yfs = subdomain.dimensions[-1]
    y = yfs.parent

    # Functions present in stencil
    funcs = retrieve_functions(lhs-rhs)

    mapper = {}
    for f in funcs:
        # Get the y index
        yind = f.indices[-1]
        if (yind - y).as_coeff_Mul()[0] > 0:
            # Symmetric mirror
            tmp = y - INT(abs(y.symbolic_max - yind))
            mapper.update({f: f.subs({yind: tmp})})

    return Eq(lhs.subs(mapper), rhs.subs(mapper), subdomain=subdomain)


def neumann_left(eq, subdomain):
    lhs, rhs = eq.evaluate.args

    # Get horizontal subdimension and its parent
    xfs = subdomain.dimensions[0]
    x = xfs.parent

    # Functions present in stencil
    funcs = retrieve_functions(lhs-rhs)

    mapper = {}
    for f in funcs:
        # Get the x index
        xind = f.indices[-2]
        if (xind - x).as_coeff_Mul()[0] < 0:
            # Symmetric mirror
            # Substitute where index is negative for +ve where
            # index is positive
            mapper.update({f: f.subs({xind: INT(abs(xind))})})

    return Eq(lhs.subs(mapper), rhs.subs(mapper), subdomain=subdomain)


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
        xind = f.indices[-2]
        if (xind - x).as_coeff_Mul()[0] > 0:
            tmp = x - INT(abs(x.symbolic_max - xind))
            mapper.update({f: f.subs({xind: tmp})})

    return Eq(lhs.subs(mapper), rhs.subs(mapper), subdomain=subdomain)

sub1 = SubTop(so)
sub2 = SubBottom(so)
sub3 = SubLeft(so)
sub4 = SubRight(so)
sub5 = SubPointBottomLeft()
sub6 = SubPointBottomRight()
sub7 = SubPointTopLeft()
sub8 = SubPointTopRight()

subdomains = (sub1, sub2, sub3, sub4, sub5, sub6, sub7, sub8)

def exact(x, y):
    return np.cos(2*np.pi*x)*np.cos(2*np.pi*y)

Lx = np.float64(1.)
Ly = np.float64(1.)

# n = 9, 17, 33, 65, 129, 257, 513, 1025
n_values = [2**k + 1 for k in range(3, 15)]
n_values = [17]
h = np.array([Lx/(n-1) for n in n_values])
infinity_norms = []
discrete_l2_norms = []
ksp_iters = []

for n in n_values:
    grid = Grid(
        shape=(n, n), extent=(Lx, Ly), subdomains=subdomains, dtype=np.float64
    )

    u = Function(name='u', grid=grid, space_order=so, dtype=np.float64)
    f = Function(name='f', grid=grid, space_order=so, dtype=np.float64)

    tmpx = np.linspace(0, Lx, n).astype(np.float64)
    tmpy = np.linspace(0, Ly, n).astype(np.float64)
    Y, X = np.meshgrid(tmpx, tmpy)
    f.data[:] = (8.*(np.pi**2))*np.cos(2.*np.pi*X)*np.cos(2.*np.pi*Y)

    eqn = Eq(-u.laplace, f, subdomain=grid.interior)
    
    # Scaling to preserve symmetry of matrix
    # See https://www.math.uci.edu/~chenlong/226/FDM.pdf
    bcs1 = neumann_top(eqn, sub1)
    bcs1 = Eq(0.5*bcs1.lhs, 0.5*bcs1.rhs, subdomain=sub1)
    bcs2 = neumann_bottom(eqn, sub2)
    bcs2 = Eq(0.5*bcs2.lhs, 0.5*bcs2.rhs, subdomain=sub2)
    bcs3 = neumann_left(eqn, sub3)
    bcs3 = Eq(0.5*bcs3.lhs, 0.5*bcs3.rhs, subdomain=sub3)
    bcs4 = neumann_right(eqn, sub4)
    bcs4 = Eq(0.5*bcs4.lhs, 0.5*bcs4.rhs, subdomain=sub4)
    bcs5 = neumann_left(neumann_bottom(eqn, sub5), sub5)
    bcs5 = Eq(0.25*bcs5.lhs, 0.25*bcs5.rhs, subdomain=sub5)
    bcs6 = neumann_right(neumann_bottom(eqn, sub6), sub6)
    bcs6 = Eq(0.25*bcs6.lhs, 0.25*bcs6.rhs, subdomain=sub6)
    bcs7 = neumann_left(neumann_top(eqn, sub7), sub7)
    bcs7 = Eq(0.25*bcs7.lhs, 0.25*bcs7.rhs, subdomain=sub7)
    bcs8 = neumann_right(neumann_top(eqn, sub8), sub8)
    bcs8 = Eq(0.25*bcs8.lhs, 0.25*bcs8.rhs, subdomain=sub8)
    bcs = [bcs1, bcs2, bcs3, bcs4, bcs5, bcs6, bcs7, bcs8]

    # u.data[:] = 0.1
    exprs = [eqn] + bcs

    petsc = petscsolve(exprs, target=u, solver_parameters={'ksp_rtol': 1e-12})

    with switchconfig(log_level='DEBUG'):
        op = Operator(petsc, language='petsc')
        summary = op.apply()

    iters = summary.petsc[('section0', None)].KSPGetIterationNumber
    ksp_iters.append(iters)

    u_exact = Function(name='u_exact', grid=grid, space_order=2)
    u_exact.data[:] = exact(X, Y)

    diff = Function(name='diff', grid=grid, space_order=2)
    diff.data[:] = u_exact.data[:] - u.data[:]

    # Compute infinity norm using numpy
    # TODO: Figure out how to compute the infinity norm using Devito
    infinity_norm = np.linalg.norm(diff.data[:].ravel(), ord=np.inf)
    infinity_norms.append(infinity_norm)

    # Compute discrete L2 norm (RMS error)
    n_interior = np.prod([s - 1 for s in grid.shape])
    discrete_l2_norm = norm(diff) / np.sqrt(n_interior)
    discrete_l2_norms.append(discrete_l2_norm)

print(op.ccode)
# print(op.arguments())

# print(u_exact.data[:])
# print(infinity_norms)
# slope, intercept = np.polyfit(np.log(h), np.log(infinity_norms), 1)

# assert slope > 1.9
# assert slope < 2.1

# # Plot
# plt.figure(figsize=(6, 5))
# plt.loglog(h, infinity_norms, 'o-', label=f'Observed rate ≈ {slope:.3f}', color='orange')
# plt.loglog(
#     h, np.exp(intercept) * h**2,
#     'k--',
#     label=r'Reference slope $O(h^2)$'
# )
# plt.xlabel(r'Grid spacing h')
# plt.ylabel(r'$\infty$-norm error')
# plt.title('Convergence Plot')
# plt.legend()
# plt.tight_layout()
# plt.savefig("3_1_5.png", dpi=200)
# plt.show()


# Error vs iterations plot
plt.figure(figsize=(6, 5))
plt.semilogy(ksp_iters, infinity_norms, 'o-', color='darkgreen')
plt.xlabel('KSP Iterations')
plt.ylabel(r'Infinity Norm Error')
plt.title('Error vs. KSP Iteration Count')
plt.grid(True, which='both', ls='--')
plt.tight_layout()
plt.savefig("error_vs_iterations.png", dpi=200)
plt.show()
