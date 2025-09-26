import os
import numpy as np

from devito import (Grid, TimeFunction, Constant, Eq,
                    Operator, SubDomain, switchconfig, configuration)
from devito.symbolics import retrieve_functions, INT

from devito.petsc import petscsolve, EssentialBC
from devito.petsc.initialize import PetscInitialize
configuration['compiler'] = 'custom'
os.environ['CC'] = 'mpicc'


PetscInitialize()

# Chorin's projection method
# Explicit time-stepping

# Physical parameters
rho = Constant(name='rho', dtype=np.float64)
nu = Constant(name='nu', dtype=np.float64)

rho.data = np.float64(1.)
nu.data = np.float64(1./10.)

Lx = 1.
Ly = Lx

# Number of grid points in each direction
nx = 41
ny = 41

# mesh spacing
dx = Lx/(nx-1)
dy = Ly/(ny-1)
so = 2


# Use subdomains just for pressure field for now
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
            if f.name == 'pn1':
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
            if f.name == 'pn1':
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
            # Substitute where index is negative for +ve
            # where index is positive
            if f.name == 'pn1':
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
            if f.name == 'pn1':
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

grid = Grid(
    shape=(nx, ny), extent=(Lx, Ly), subdomains=subdomains, dtype=np.float64
)
time = grid.time_dim
t = grid.stepping_dim
x, y = grid.dimensions

# time stepping parameters
dt = 1e-3
t_end = 1.
ns = int(t_end/dt)

u1 = TimeFunction(name='u1', grid=grid, space_order=2, dtype=np.float64)
v1 = TimeFunction(name='v1', grid=grid, space_order=2, dtype=np.float64)
pn1 = TimeFunction(name='pn1', grid=grid, space_order=2, dtype=np.float64)

eq_pn1 = Eq(pn1.forward.laplace, rho*(1./dt*(u1.forward.dxc+v1.forward.dyc)),
            subdomain=grid.interior)


bc_pn1 = [neumann_top(eq_pn1, sub1)]
bc_pn1 += [neumann_bottom(eq_pn1, sub2)]
bc_pn1 += [neumann_left(eq_pn1, sub3)]
bc_pn1 += [neumann_right(eq_pn1, sub4)]
bc_pn1 += [EssentialBC(pn1.forward, 0., subdomain=sub5)]
bc_pn1 += [neumann_right(neumann_bottom(eq_pn1, sub6), sub6)]
bc_pn1 += [neumann_left(neumann_top(eq_pn1, sub7), sub7)]
bc_pn1 += [neumann_right(neumann_top(eq_pn1, sub8), sub8)]


eqn_p = petscsolve([eq_pn1]+bc_pn1, pn1.forward, options_prefix='pressure_solve', solver_parameters={'ksp_type': 'cg'})

eq_u1 = Eq(u1.dt + u1*u1.dxc + v1*u1.dyc, nu*u1.laplace, subdomain=grid.interior)
eq_v1 = Eq(v1.dt + u1*v1.dxc + v1*v1.dyc, nu*v1.laplace, subdomain=grid.interior)

update_u = Eq(u1.forward, u1.forward - (dt/rho)*(pn1.forward.dxc),
              subdomain=grid.interior)

update_v = Eq(v1.forward, v1.forward - (dt/rho)*(pn1.forward.dyc),
              subdomain=grid.interior)

# TODO: Can drop due to initial guess CB
u1.data[0, :, -1] = np.float64(1.)
u1.data[1, :, -1] = np.float64(1.)


# Create Dirichlet BC expressions for velocity
bc_u = [EssentialBC(u1.forward, 1., subdomain=sub1)]  # top
bc_u += [EssentialBC(u1.forward, 0., subdomain=sub3)]  # left
bc_u += [EssentialBC(u1.forward, 0., subdomain=sub4)]  # right
bc_u += [EssentialBC(u1.forward, 0., subdomain=sub2)]  # bottom
bc_u += [EssentialBC(u1.forward, 0., subdomain=sub5)]  # bottom left
bc_u += [EssentialBC(u1.forward, 0., subdomain=sub6)]  # bottom right
bc_u += [EssentialBC(u1.forward, 0., subdomain=sub7)]  # top left
bc_u += [EssentialBC(u1.forward, 0., subdomain=sub8)]  # top right


bc_v = [EssentialBC(v1.forward, 0., subdomain=sub3)]  # left
bc_v += [EssentialBC(v1.forward, 0., subdomain=sub4)]  # right
bc_v += [EssentialBC(v1.forward, 0., subdomain=sub1)]  # top
bc_v += [EssentialBC(v1.forward, 0., subdomain=sub2)]  # bottom
bc_v += [EssentialBC(v1.forward, 0., subdomain=sub5)]  # bottom left
bc_v += [EssentialBC(v1.forward, 0., subdomain=sub6)]  # bottom right
bc_v += [EssentialBC(v1.forward, 0., subdomain=sub7)]  # top left
bc_v += [EssentialBC(v1.forward, 0., subdomain=sub8)]  # top right


tentu = petscsolve([eq_u1]+bc_u, u1.forward)
tentv = petscsolve([eq_v1]+bc_v, v1.forward)

exprs = [tentu, tentv, eqn_p, update_u, update_v] + bc_u + bc_v

with switchconfig(language='petsc'):
    op = Operator(exprs)
    op.apply(time_m=0, time_M=ns-1, dt=dt)

# Pressure norm check
tol = 1e-3
assert np.sum((pn1.data[0]-pn1.data[1])**2/np.maximum(pn1.data[0]**2, 1e-10)) < tol



# Import u values at x=L/2 (table 6, column 2 rows 12-26) in Marchi et al.
Marchi_Re10_u = np.array([[0.0625, -3.85425800e-2],
                          [0.125,  -6.96238561e-2],
                          [0.1875, -9.6983962e-2],
                          [0.25,   -1.22721979e-1],
                          [0.3125, -1.47636199e-1],
                          [0.375,  -1.71260757e-1],
                          [0.4375, -1.91677043e-1],
                          [0.5,    -2.05164738e-1],
                          [0.5625, -2.05770198e-1],
                          [0.625,  -1.84928116e-1],
                          [0.6875, -1.313892353e-1],
                          [0.75,   -3.1879308e-2],
                          [0.8125,  1.26912095e-1],
                          [0.875,   3.54430364e-1],
                          [0.9375,  6.50529292e-1]])
# Import v values at y=L/2 (table 6, column 2 rows 27-41) in Marchi et al.
Marchi_Re10_v = np.array([[0.0625, 9.2970121e-2],
                          [0.125,  1.52547843e-1],
                          [0.1875, 1.78781456e-1],
                          [0.25,   1.76415100e-1],
                          [0.3125, 1.52055820e-1],
                          [0.375,  1.121477612e-1],
                          [0.4375, 6.21048147e-2],
                          [0.5,    6.3603620e-3],
                          [0.5625,-5.10417285e-2],
                          [0.625, -1.056157259e-1],
                          [0.6875,-1.51622101e-1],
                          [0.75,  -1.81633561e-1],
                          [0.8125,-1.87021651e-1],
                          [0.875, -1.59898186e-1],
                          [0.9375,-9.6409942e-2]])


# make plot comparing to Marchi et al.


#NBVAL_IGNORE_OUTPUT
# Check results with Marchi et al 2009.

import matplotlib.pyplot as pyplot
nx = 41
ny = nx
npgrid=[nx,ny]

x_coord = np.linspace(0, 1, npgrid[0])
y_coord = np.linspace(0, 1, npgrid[1])

fig = pyplot.figure(figsize=(12, 6))
ax1 = fig.add_subplot(121)
ax1.plot(u1.data[-1, int(npgrid[0]/2),:],y_coord[:], 'black',label="Devito + PETSc")
ax1.plot(Marchi_Re10_u[:,1],Marchi_Re10_u[:,0],'o', color='red', label="Marchi et al. 2009")
ax1.set_xlabel('$u$')
ax1.set_ylabel('$y$')
ax1.legend()
ax1 = fig.add_subplot(122)
ax1.plot(x_coord[:],v1.data[-1,:,int(npgrid[1]/2)], 'black', label="Devito + PETSc")
ax1.plot(Marchi_Re10_v[:,0],Marchi_Re10_v[:,1],'o', color='red', label="Marchi et al. 2009")
ax1.set_xlabel('$x$')
ax1.set_ylabel('$v$')
ax1.legend()

pyplot.savefig('chorins_projection.png', format='jpg', dpi=300)
