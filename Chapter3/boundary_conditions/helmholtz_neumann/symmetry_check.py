import os
import sys
import shutil
import tempfile
import numpy as np

from devito.symbolics import retrieve_functions, INT
from devito import configuration, Operator, Eq, Grid, Function, SubDomain
from devito.petsc import petscsolve
from devito.petsc.initialize import PetscInitialize
configuration['compiler'] = 'custom'
os.environ['CC'] = 'mpicc'

# Symmetry of the Jacobian that petscsolve generates for the modified
# Helmholtz problem with Neumann conditions (2d_modified_helmholtz.py),
# on a 33 x 33 grid.

PetscInitialize()

Lx = 1.
Ly = Lx
so = 2
n = 33
tmpdir = tempfile.mkdtemp()
matrix_file = os.path.join(tmpdir, 'helmholtz_jacobian.bin')


class SubTop(SubDomain):
    name = 'subtop'

    def __init__(self, S_O, grid=None):
        self.S_O = S_O
        super().__init__(grid=grid)

    def define(self, dimensions):
        x, y = dimensions
        return {x: ('middle', 1, 1), y: ('right', self.S_O//2)}


class SubBottom(SubDomain):
    name = 'subbottom'

    def __init__(self, S_O, grid=None):
        self.S_O = S_O
        super().__init__(grid=grid)

    def define(self, dimensions):
        x, y = dimensions
        return {x: ('middle', 1, 1), y: ('left', self.S_O//2)}


class SubLeft(SubDomain):
    name = 'subleft'

    def __init__(self, S_O, grid=None):
        self.S_O = S_O
        super().__init__(grid=grid)

    def define(self, dimensions):
        x, y = dimensions
        return {x: ('left', self.S_O//2), y: ('middle', 1, 1)}


class SubRight(SubDomain):
    name = 'subright'

    def __init__(self, S_O, grid=None):
        self.S_O = S_O
        super().__init__(grid=grid)

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
    yfs = subdomain.dimensions[-1]
    y = yfs.parent
    funcs = retrieve_functions(lhs-rhs)
    mapper = {}
    for f in funcs:
        yind = f.indices[-1]
        if (yind - y).as_coeff_Mul()[0] < 0:
            mapper.update({f: f.subs({yind: INT(abs(yind))})})
    return Eq(lhs.subs(mapper), rhs.subs(mapper), subdomain=subdomain)


def neumann_top(eq, subdomain):
    lhs, rhs = eq.evaluate.args
    yfs = subdomain.dimensions[-1]
    y = yfs.parent
    funcs = retrieve_functions(lhs-rhs)
    mapper = {}
    for f in funcs:
        yind = f.indices[-1]
        if (yind - y).as_coeff_Mul()[0] > 0:
            tmp = y - INT(abs(y.symbolic_max - yind))
            mapper.update({f: f.subs({yind: tmp})})
    return Eq(lhs.subs(mapper), rhs.subs(mapper), subdomain=subdomain)


def neumann_left(eq, subdomain):
    lhs, rhs = eq.evaluate.args
    xfs = subdomain.dimensions[0]
    x = xfs.parent
    funcs = retrieve_functions(lhs-rhs)
    mapper = {}
    for f in funcs:
        xind = f.indices[-2]
        if (xind - x).as_coeff_Mul()[0] < 0:
            mapper.update({f: f.subs({xind: INT(abs(xind))})})
    return Eq(lhs.subs(mapper), rhs.subs(mapper), subdomain=subdomain)


def neumann_right(eq, subdomain):
    lhs, rhs = eq.evaluate.args
    xfs = subdomain.dimensions[0]
    x = xfs.parent
    funcs = retrieve_functions(lhs-rhs)
    mapper = {}
    for f in funcs:
        xind = f.indices[-2]
        if (xind - x).as_coeff_Mul()[0] > 0:
            tmp = x - INT(abs(x.symbolic_max - xind))
            mapper.update({f: f.subs({xind: tmp})})
    return Eq(lhs.subs(mapper), rhs.subs(mapper), subdomain=subdomain)


grid = Grid(shape=(n, n), extent=(Lx, Ly), dtype=np.float64)
top = SubTop(so, grid=grid)
bottom = SubBottom(so, grid=grid)
left = SubLeft(so, grid=grid)
right = SubRight(so, grid=grid)
bottom_left = SubPointBottomLeft(grid=grid)
bottom_right = SubPointBottomRight(grid=grid)
top_left = SubPointTopLeft(grid=grid)
top_right = SubPointTopRight(grid=grid)

u = Function(name='u', grid=grid, space_order=so, dtype=np.float64)
f = Function(name='f', grid=grid, space_order=so, dtype=np.float64)

tmpx = np.linspace(0, Lx, n).astype(np.float64)
tmpy = np.linspace(0, Ly, n).astype(np.float64)
Y, X = np.meshgrid(tmpx, tmpy)
f.data[:] = (1.+(8.*(np.pi**2)))*np.cos(2.*np.pi*X)*np.cos(2.*np.pi*Y)

eqn = Eq(-u.laplace+u, f, subdomain=grid.interior)

# Boundary rows scaled by one half on the sides and one quarter at the
# corners, as in 2d_modified_helmholtz.py
bc_top = neumann_top(eqn, top)
bc_top = Eq(0.5*bc_top.lhs, 0.5*bc_top.rhs, subdomain=top)
bc_bottom = neumann_bottom(eqn, bottom)
bc_bottom = Eq(0.5*bc_bottom.lhs, 0.5*bc_bottom.rhs, subdomain=bottom)
bc_left = neumann_left(eqn, left)
bc_left = Eq(0.5*bc_left.lhs, 0.5*bc_left.rhs, subdomain=left)
bc_right = neumann_right(eqn, right)
bc_right = Eq(0.5*bc_right.lhs, 0.5*bc_right.rhs, subdomain=right)
bc_bottom_left = neumann_left(neumann_bottom(eqn, bottom_left), bottom_left)
bc_bottom_left = Eq(0.25*bc_bottom_left.lhs, 0.25*bc_bottom_left.rhs, subdomain=bottom_left)
bc_bottom_right = neumann_right(neumann_bottom(eqn, bottom_right), bottom_right)
bc_bottom_right = Eq(0.25*bc_bottom_right.lhs, 0.25*bc_bottom_right.rhs, subdomain=bottom_right)
bc_top_left = neumann_left(neumann_top(eqn, top_left), top_left)
bc_top_left = Eq(0.25*bc_top_left.lhs, 0.25*bc_top_left.rhs, subdomain=top_left)
bc_top_right = neumann_right(neumann_top(eqn, top_right), top_right)
bc_top_right = Eq(0.25*bc_top_right.lhs, 0.25*bc_top_right.rhs, subdomain=top_right)
bcs = [bc_top, bc_bottom, bc_left, bc_right,
       bc_bottom_left, bc_bottom_right, bc_top_left, bc_top_right]

# -ksp_view_mat_explicit forms the entries of the Jacobian from its products
# and writes them to matrix_file after the solve
solver = petscsolve(
    [eqn]+bcs, target=u,
    solver_parameters={'ksp_rtol': 1e-8, 'ksp_type': 'cg', 'pc_type': 'none',
                       'ksp_view_mat_explicit': f'binary:{matrix_file}'},
    options_prefix='helmholtz_2d'
)

op = Operator(solver, language='petsc')
op.apply()

# Read the matrix back and check SPD
sys.path.insert(0, os.path.join(os.environ['PETSC_DIR'], 'lib', 'petsc', 'bin'))
import PetscBinaryIO

(A,) = PetscBinaryIO.PetscBinaryIO().readBinaryFile(matrix_file, mattype='scipy.sparse')
A = A.toarray()
shutil.rmtree(tmpdir)
asymmetry = np.abs(A - A.T).max()
eigenvalues = np.linalg.eigvalsh(A)
print(f'Jacobian of size {A.shape[0]} x {A.shape[1]} on a {n} x {n} grid')
print(f'max |A - A^T| = {asymmetry:.1e}')
print(f'smallest eigenvalue = {eigenvalues.min():.4f}, '
      f'largest eigenvalue = {eigenvalues.max():.4f}')

assert asymmetry == 0.
assert eigenvalues.min() > 0.
