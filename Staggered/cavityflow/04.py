import numpy as np
import os

from matplotlib import pyplot, cm
from devito import Grid, TimeFunction, Function, Eq, solve, Operator, configuration, SubDomain, NODE, switchconfig, Constant
from devito.symbolics import retrieve_functions, INT

from devito.petsc import petscsolve, EssentialBC
from devito.petsc.initialize import PetscInitialize
configuration['compiler'] = 'custom'
os.environ['CC'] = 'mpicc'


PetscInitialize()

######## same as 03.py but using PETSc for pressure solve...
#### also, using a single operator instead of needing two

ny = 41
nx = 41


x_coord = np.linspace(0, 1, nx)
y_coord = np.linspace(0, 1, ny)


Y, X = np.meshgrid(x_coord, y_coord)

# Physical parameters
rho = Constant(name='rho', dtype=np.float64)
nu = Constant(name='nu', dtype=np.float64)

rho.data = np.float64(1.)
nu.data = np.float64(1./10.)


dx = 1.0 / (nx - 1)                                                                                                                         
dt = 0.45 * dx**2 / (4 * nu.data)                                                                               
nt = int(2.0 / dt)
so = 2
    

# Use subdomains just for pressure field for now
class Sub1(SubDomain):
    name = 'sub1'
    def __init__(self, S_O):
        super().__init__()
        self.S_O = S_O

    def define(self, dimensions):
        x, y = dimensions
        return {x: ('left', 1), y: ('middle', ny-2, 1)}
    

class Sub2(SubDomain):
    name = 'sub2'
    def __init__(self, S_O):
        super().__init__()
        self.S_O = S_O

    def define(self, dimensions):
        x, y = dimensions
        return {x: ('middle', 1, 2), y: ('middle', ny-2, 1)}
    

class Sub3(SubDomain):
    name = 'sub3'
    def __init__(self, S_O):
        super().__init__()
        self.S_O = S_O

    def define(self, dimensions):
        x, y = dimensions
        return {x: ('middle', nx-2, 1), y: ('middle', ny-2, 1)}
    


class Sub4(SubDomain):
    name = 'sub4'
    def __init__(self, S_O):
        super().__init__()
        self.S_O = S_O

    def define(self, dimensions):
        x, y = dimensions
        return {x: ('left', 1), y: ('middle', 1, 2)}



class Sub5(SubDomain):
    name = 'sub5'
    def __init__(self, S_O):
        super().__init__()
        self.S_O = S_O

    def define(self, dimensions):
        x, y = dimensions
        return {x: ('middle', 1, 2), y: ('middle', 1, 2)}
    


class Sub6(SubDomain):
    name = 'sub6'
    def __init__(self, S_O):
        super().__init__()
        self.S_O = S_O

    def define(self, dimensions):
        x, y = dimensions
        return {x: ('middle', nx-2, 1), y: ('middle', 1, 2)}
    

class Sub7(SubDomain):
    name = 'sub7'
    def __init__(self, S_O):
        super().__init__()
        self.S_O = S_O

    def define(self, dimensions):
        x, y = dimensions
        return {x: ('left', 1), y: ('left', 1)}
    

class Sub8(SubDomain):
    name = 'sub8'
    def __init__(self, S_O):
        super().__init__()
        self.S_O = S_O

    def define(self, dimensions):
        x, y = dimensions
        return {x: ('middle', 1, 2), y: ('left', 1)}
    


class Sub9(SubDomain):
    name = 'sub9'
    def __init__(self, S_O):
        super().__init__()
        self.S_O = S_O

    def define(self, dimensions):
        x, y = dimensions
        return {x: ('middle', nx-2, 1), y: ('left', 1)}


class Sub10(SubDomain):
    name = 'sub10'
    def __init__(self, S_O):
        super().__init__()
        self.S_O = S_O

    def define(self, dimensions):
        x, y = dimensions
        return {x: x, y: ('right', 1)}
    

class Sub11(SubDomain):
    name = 'sub11'
    def __init__(self, S_O):
        super().__init__()
        self.S_O = S_O

    def define(self, dimensions):
        x, y = dimensions
        return {x: ('right', 1), y: ('left', ny-1)}
    


def neumann_bottom(eq, t, subdomain):
    lhs, rhs = eq.evaluate.args

    # Functions present in stencil
    funcs = retrieve_functions(lhs-rhs)

    yind_target = t.indices[-1]

    mapper = {}
    for f in funcs:
        yind = f.indices[-1]
        if (yind - yind_target).as_coeff_Mul()[0] < 0:
            if f.name == 'p':
                mapper.update({f: f.subs({yind: yind_target})})

    return Eq(lhs.subs(mapper), rhs.subs(mapper), subdomain=subdomain)


def neumann_top(eq, t, subdomain):
    lhs, rhs = eq.evaluate.args

    # Functions present in stencil
    funcs = retrieve_functions(lhs-rhs)

    yind_target = t.indices[-1]

    mapper = {}
    for f in funcs:
        # Get the y index
        yind = f.indices[-1]
        # from IPython import embed; embed()
        if (yind - yind_target).as_coeff_Mul()[0] > 0:
            # Symmetric mirror: ghost maps to yind_target (last physical p-cell).
            # For staggered p at y+h_y/2 the top wall sits exactly at the midpoint
            # between yind_target and the ghost, so the Neumann mirror is yind_target.
            if f.name == 'p':
                mapper.update({f: f.subs({yind: yind_target})})

    return Eq(lhs.subs(mapper), rhs.subs(mapper), subdomain=subdomain)


def neumann_left(eq, t, subdomain):
    lhs, rhs = eq.evaluate.args

    # Functions present in stencil
    funcs = retrieve_functions(lhs-rhs)

    xind_target = t.indices[-2]

    mapper = {}
    for f in funcs:
        xind = f.indices[-2]
        if (xind - xind_target).as_coeff_Mul()[0] < 0:
            if f.name == 'p':
                mapper.update({f: f.subs({xind: xind_target})})

    return Eq(lhs.subs(mapper), rhs.subs(mapper), subdomain=subdomain)


def neumann_right(eq, t, subdomain):
    lhs, rhs = eq.evaluate.args

    # Functions present in stencil
    funcs = retrieve_functions(lhs-rhs)

    xind_target = t.indices[-2]

    mapper = {}
    for f in funcs:
        xind = f.indices[-2]
        if (xind - xind_target).as_coeff_Mul()[0] > 0:
            if f.name == 'p':
                mapper.update({f: f.subs({xind: xind_target})})

    return Eq(lhs.subs(mapper), rhs.subs(mapper), subdomain=subdomain)


sub1 = Sub1(so)
sub2 = Sub2(so)
sub3 = Sub3(so)
sub4 = Sub4(so)
sub5 = Sub5(so)
sub6 = Sub6(so)
sub7 = Sub7(so)
sub8 = Sub8(so)
sub9 = Sub9(so)
sub10 = Sub10(so)
sub11 = Sub11(so)

subdomains = (sub1, sub2, sub3, sub4, sub5, sub6, sub7, sub8, sub9, sub10, sub11)

grid = Grid(shape=(nx, ny), extent=(1, 1.), subdomains=subdomains, dtype=np.float64)
x, y = grid.dimensions
t = grid.stepping_dim

# Staggered MAC grid:
#   u - staggered in y
#   v - staggered in x
#   p - cell centres (staggered in both x and y)
u = TimeFunction(name='u', grid=grid, space_order=2, staggered=y)
v = TimeFunction(name='v', grid=grid, space_order=2, staggered=x)
p = Function(name='p', grid=grid, space_order=2, staggered=(x, y))


# u-momentum:
# p.dxc is forcing the non staggering so it is not correct - that is why I use p.dx instead
eq_u = Eq(u.dt + u*u.dxc + v*u.dyc, -1./rho * p.dx + nu*(u.dx2 + u.dy2))
stencil_u = solve(eq_u, u.forward)
update_u = Eq(u.forward, stencil_u)

# v-momentum:
eq_v = Eq(v.dt + u*v.dxc + v*v.dyc, -1./rho * p.dy + nu*(v.dx2 + v.dy2))
stencil_v = solve(eq_v, v.forward)
update_v = Eq(v.forward, stencil_v)

# manual edit to x0 since using uf.dx and vf.dy alone does not seem to be correct? it seems to be left staggered?
ux_cc = u.dx(x0=x + x.spacing/2)
vy_cc = v.dy(x0=y + y.spacing/2)
# uy_cc = (u[x, y+1] + u[x+1, y+1] - u[x, y-1] - u[x+1, y-1]) / (4 * y.spacing)
# vx_cc = (v[x+1, y] + v[x+1, y+1] - v[x-1, y] - v[x-1, y+1]) / (4 * x.spacing)

eq_p = Eq(p.laplace,
          rho * (1./dt * (ux_cc + vy_cc) - ux_cc**2 - 2*u.dy*v.dx - vy_cc**2),
          subdomain=grid.subdomains['sub5'])

# u BCs
# Left/right walls: u is not x-staggered so nodes sit exactly on the walls.
# Bottom wall - average across the wall is zero, so ghost point is negative of first interior point

# Top lid (y=1): value at the wall equals U_lid=1: (u[ny-2] + u[ny-1])/2 = 1
# u[ny-1] = 2 - u[ny-2]
bc_u  = [Eq(u[t+1, 0, y], 0)] # left
bc_u += [Eq(u[t+1, nx-1, y], 0)] # right
bc_u += [Eq(u[t+1, x, -1], -u[t+1, x, 0])] # bottom
bc_u += [Eq(u[t+1, x, ny-1], 2 - u[t+1, x, ny-2])]  # lid: u=1 at y=1

bc_v  = [Eq(v[t+1, -1, y], -v[t+1, 0, y])] # left
bc_v += [Eq(v[t+1, nx-1, y], -v[t+1, nx-2, y])]  # ghost beyond right wall: interp to 0 at x=1
bc_v += [Eq(v[t+1, x, ny-1], 0)] # top
bc_v += [Eq(v[t+1, x, 0], 0)] # bottom

# p is at cell centres
bc_p = [neumann_left(neumann_top(eq_p, p, sub1), p, sub1)]
bc_p += [neumann_top(eq_p, p, sub2)]
bc_p += [neumann_right(neumann_top(eq_p, p, sub3), p, sub3)]
bc_p += [neumann_left(eq_p, p, sub4)]
bc_p += [neumann_right(eq_p, p, sub6)]
bc_p += [neumann_bottom(eq_p, p, sub8)]
bc_p += [neumann_right(neumann_bottom(eq_p, p, sub9), p, sub9)]

bc_tmp1 = Function(name='bc_tmp1', grid=grid, space_order=2, staggered=(x, y))
bc_tmp1.data[:] = 0.
bc_p += [EssentialBC(p, bc_tmp1, subdomain=grid.subdomains['sub7'], constrain=True)] # pin pressure at corner

# These two will be automated in the compiler
bc_p += [EssentialBC(p, bc_tmp1, subdomain=grid.subdomains['sub10'], constrain=True)]
bc_p += [EssentialBC(p, bc_tmp1, subdomain=grid.subdomains['sub11'], constrain=True)]


# TODO: check symmetry of matrix before using CG..
pressure_solve = petscsolve([eq_p]+bc_p, p, options_prefix='pressure_solve', solver_parameters={'ksp_type': 'cg'})
# pressure_solve = petscsolve([eq_p]+bc_p, p, options_prefix='pressure_solve', solver_parameters={'ksp_type': 'gmres'})

exprs = [pressure_solve] + [update_u, update_v] + bc_u + bc_v

with switchconfig(language='petsc'):
    op = Operator(exprs)
    # print(op.ccode)
    op.apply(time_M=nt, dt=dt)


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

# run trivial operators to interpolate the staggered fields back onto original grid "nodes" for plotting and analysis

plotfunc_u = Function(name='plotfunc_u', grid=grid, space_order=2, staggered=NODE)
plotfunc_v = Function(name='plotfunc_v', grid=grid, space_order=2, staggered=NODE)
plotfunc_p = Function(name='plotfunc_p', grid=grid, space_order=2, staggered=NODE)

Operator(Eq(plotfunc_u, u))(time_M=0)
Operator(Eq(plotfunc_v, v))(time_M=0)
Operator(Eq(plotfunc_p, p))()


fig = pyplot.figure(figsize=(11, 7), dpi=100)
pyplot.contourf(X, Y, plotfunc_p.data[:], alpha=0.5, cmap=cm.viridis)
pyplot.colorbar()
pyplot.contour(X, Y, plotfunc_p.data[:], cmap=cm.viridis)
pyplot.quiver(X[::2, ::2], Y[::2, ::2], plotfunc_u.data[::2, ::2], plotfunc_v.data[::2, ::2])
pyplot.xlabel('X')
pyplot.ylabel('Y')
pyplot.savefig('04.png', dpi=100, bbox_inches='tight')
pyplot.show()

#NBVAL_IGNORE_OUTPUT
# Again, check results with Marchi et al 2009.
fig = pyplot.figure(figsize=(12, 6))
ax1 = fig.add_subplot(121)
ax1.plot(plotfunc_u.data[int(grid.shape[0]/2),:],y_coord[:])
ax1.plot(Marchi_Re10_u[:,1],Marchi_Re10_u[:,0],'ro')
ax1.set_xlabel('$u$')
ax1.set_ylabel('$y$')
ax1 = fig.add_subplot(122)
ax1.plot(x_coord[:],plotfunc_v.data[:,int(grid.shape[0]/2)])
ax1.plot(Marchi_Re10_v[:,0],Marchi_Re10_v[:,1],'ro')
ax1.set_xlabel('$x$')
ax1.set_ylabel('$v$')
pyplot.savefig('04_comparison.png', dpi=100, bbox_inches='tight')
pyplot.show()
