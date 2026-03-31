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

######## using a predictor corrector approach (EXPLICIT time stepping for momentum equations)
######### using Kim & Moin paper 1985
# https://pdf.sciencedirectassets.com/272570/1-s2.0-S0021999100X02513/1-s2.0-0021999185901482/main.pdf?X-Amz-Security-Token=IQoJb3JpZ2luX2VjEGsaCXVzLWVhc3QtMSJGMEQCIBS1F7mGptLOMbJ6PunezwOAjZYzhWrZuMY%2B8lzItRNMAiAGFG%2FEVxQoL1h%2BBvzDij%2Bv14K7ycj6ZxBVeOeuxf8izyqyBQgzEAUaDDA1OTAwMzU0Njg2NSIMNMMwndYgdsGt%2FCfzKo8FQACH4ZJmbQo32vbvrQn6sTRdUTCazmZnmy3RUDkBoMP6u5O8EndwsFvmZ3njNpCReNgu%2F7WWGcDcxzdJLS50DstxG7Ul2LjZJxTWZ%2FoTow6yOdlPgbwgCXeMXLWnridULXEg%2BHp%2BL8HnW43%2BRXBIg5sH%2B2Q7p85mIIGtjI8xZxzcZSNQdw2GYJQ%2FxScu%2FH5dv1LEgeqtaJo50CzNUDVAFJiYQyfnn2SqHDs86GwTKaZnNjI3KyDf8GBAHeuHJv1CSgC6y7p3r0QAyQzB28jwp%2BHL6BoTo6Y8XC4NZCRHcLyucetWSPJZ%2BNh56PtXErzjzUiQtOJr%2BLQhNMFXUsAwLpKv%2BgLFQqHhNXgb76hgrv5kGq3IgTvqRldgKSVTAxYF9i6Ac%2Bpc3k8Gdyp31%2BKmYc1qdLRl1gWxujliDlFjLubmva4CfCkGfINYtyqsgsY98Dz8uYR90%2Fe8Ys2%2FsfBQcgb%2FGCP15UQM6uC4Mz3P1UZ89ccfMIDLxo36SBzHXdO%2BaYXTsaxhAGqdB6qN%2BSwPZ81E8AlIhvlYTHRH9xzRS5v7YMZ010Ne0%2BRIysG28LZhvo36sNPtagDh%2BpdotIfo6hh0SsJM07Mn0xYwwRuKZcfnbuJ6kGkBWAlTM9GdIQOs5JchjcjvXRp%2B%2F%2FyhoeEd7tY4lOK7QQ52JPKIu%2FG4IPoFlHXMvDcIWiY4VPwG9KJpwOXT4hvmxBma3FSfeNow9dvAYo%2Bbt9LgDZM9QQQh9SWFuRlYfzdr5NazVeyHj5Z1yN0XofTWwjEoP7GMZrvZkTn%2F3MKE5TibD5V4Q%2BRtVCDOjtr1%2BL0FUHLD6t6aWLoYdaFxQFXulCiNX9sPUX5ePWMju6BpxHkjzZT3gqgtYjCAvfTNBjqyAaW1XqS2gtHvAzgbOHZ%2B07ql73%2B0k7UtAfA8ZIeO1EgGwVkcSw3nmgnc3sjw3qg8dRWE%2F5KmxEN2Yjx4j5qO4ZJB7VbugAkOlkzPhW%2B5WRSGOWg6nr3WtctxBYtAOKUBeOUmRg069ncsOIlk%2BfwqOKzg%2Bc9%2BpUyV6%2F0f3ZVQlPb7xRALBJQQk%2FKyC%2Bx6Z8J5inmovtflrhOL0Cvm1mNRaQpVclNDAGYSzpOUamdK%2FlJE%2FQE%3D&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Date=20260320T110721Z&X-Amz-SignedHeaders=host&X-Amz-Expires=300&X-Amz-Credential=ASIAQ3PHCVTYTXKHTLAE%2F20260320%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Signature=a4302a5660eef862d1845af92f4e5db2995cf0e523b59e15f99a3d72941f825a&hash=3fe0c26f74abb9f667c9d80ad4bf2664e1dd49869b1e0445ce629b1d60e03d53&host=68042c943591013ac2b2430a89b270f6af2c76d8dfd086a07176afe7c76c2c61&pii=0021999185901482&tid=spdf-8999359f-9965-4281-ad4e-09e6ad2782bf&sid=6631e4b18440084e6799914-00d95cf11814gxrqb&type=client&tsoh=d3d3LnNjaWVuY2VkaXJlY3QuY29t&rh=d3d3LnNjaWVuY2VkaXJlY3QuY29t&ua=02055c0a045455540c56&rr=9df440b17fc4631d&cc=gb
# paper is - Application of a fractional step method to incompressible Navier-Stokes equations, Kim & Moin 1985

# also closely linked to 'Computational fluid dynamics by example' textbook - biringen and chow


# using petsc for the momentum equations (still explicit) and the pressure solve

ny = 97
nx = 97


x_coord = np.linspace(0, 1, nx)
y_coord = np.linspace(0, 1, ny)


Y, X = np.meshgrid(x_coord, y_coord)

re = Constant(name='re', dtype=np.float64)
re.data = np.float64(400.)

# dx = 1.0 / (nx - 1)                                                                                                                         
# dt = (0.45 * dx**2 * re.data )/ 4 
dt = 0.01                                                                           
nt = int(50.0 / dt)
# from IPython import embed; embed()
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


class Sub12(SubDomain):
    name = 'sub12'
    def __init__(self, S_O):
        super().__init__()
        self.S_O = S_O

    def define(self, dimensions):
        x, y = dimensions
        return {x: ('middle', 1, 1), y: ('left', 1)}


class Sub13(SubDomain):
    name = 'sub13'
    def __init__(self, S_O):
        super().__init__()
        self.S_O = S_O

    def define(self, dimensions):
        x, y = dimensions
        return {x: ('middle', 1, 1), y: ('middle', ny-2, 1)}
    

class Sub14(SubDomain):
    name = 'sub14'
    def __init__(self, S_O):
        super().__init__()
        self.S_O = S_O

    def define(self, dimensions):
        x, y = dimensions
        return {x: ('left', 1), y: ('left', ny-1)}
    

class Sub15(SubDomain):
    name = 'sub15'
    def __init__(self, S_O):
        super().__init__()
        self.S_O = S_O

    def define(self, dimensions):
        x, y = dimensions
        return {x: ('middle', 1, 1), y: ('middle', 1, 2)}
    

class Sub16(SubDomain):
    name = 'sub16'
    def __init__(self, S_O):
        super().__init__()
        self.S_O = S_O

    def define(self, dimensions):
        x, y = dimensions
        return {x: ('middle', nx-2, 1), y: ('middle', 1, 1)}


class Sub17(SubDomain):
    name = 'sub17'
    def __init__(self, S_O):
        super().__init__()
        self.S_O = S_O

    def define(self, dimensions):
        x, y = dimensions
        return {x: ('middle', 1, 2), y: ('middle', 1, 1)}
    

class Sub18(SubDomain):
    name = 'sub18'
    def __init__(self, S_O):
        super().__init__()
        self.S_O = S_O

    def define(self, dimensions):
        x, y = dimensions
        return {x: ('left', nx-1), y: ('right', 1)}
    

class Sub19(SubDomain):
    name = 'sub19'
    def __init__(self, S_O):
        super().__init__()
        self.S_O = S_O

    def define(self, dimensions):
        x, y = dimensions
        return {x: ('left', nx-1), y: ('left', 1)}


class Sub20(SubDomain):
    name = 'sub20'
    def __init__(self, S_O):
        super().__init__()
        self.S_O = S_O

    def define(self, dimensions):
        x, y = dimensions
        return {x: ('left', 1), y: ('middle', 1, 1)}
    

class Sub21(SubDomain):
    name = 'sub21'
    def __init__(self, S_O):
        super().__init__()
        self.S_O = S_O

    def define(self, dimensions):
        x, y = dimensions
        return {x: ('right', 1), y: y}



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
            if f.name == 'u':
                mapper.update({f: -f.subs({yind: yind_target})})

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

            if f.name == 'u':
                mapper.update({f: 2-f.subs({yind: yind_target})})

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

            if f.name == 'v':
                mapper.update({f: -f.subs({xind: xind_target})})

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

            if f.name == 'v':
                mapper.update({f: -f.subs({xind: xind_target})})

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
sub12 = Sub12(so)
sub13 = Sub13(so)
sub14 = Sub14(so)
sub15 = Sub15(so)
sub16 = Sub16(so)
sub17 = Sub17(so)
sub18 = Sub18(so)
sub19 = Sub19(so)
sub20 = Sub20(so)
sub21 = Sub21(so)

subdomains = (sub1, sub2, sub3, sub4, sub5, sub6, sub7, sub8, sub9, sub10, sub11, sub12, sub13, sub14, sub15, sub16, sub17, sub18, sub19, sub20, sub21)

grid = Grid(shape=(nx, ny), extent=(1, 1.), subdomains=subdomains, dtype=np.float64)
x, y = grid.dimensions
t = grid.stepping_dim


# Staggered MAC grid:
u = TimeFunction(name='u', grid=grid, space_order=2, staggered=y)
v = TimeFunction(name='v', grid=grid, space_order=2, staggered=x)
p = Function(name='p', grid=grid, space_order=2, staggered=(x, y))


# u-momentum:
eq_u_tent = Eq(u.dt + u*u.dxc + v*u.dyc, (1./re)*(u.dx2 + u.dy2), subdomain=grid.subdomains['sub15'])

# v-momentum:
eq_v_tent = Eq(v.dt + u*v.dxc + v*v.dyc, (1./re)*(v.dx2 + v.dy2), subdomain=grid.subdomains['sub17'])


# manual edit to x0 since using uf.dx and vf.dy alone does not seem to be correct? it seems to be left staggered?
ux_cc = u.forward.dx(x0=x + x.spacing/2)
vy_cc = v.forward.dy(x0=y + y.spacing/2)


eq_p = Eq(p.laplace, (1./dt)*(ux_cc+vy_cc), subdomain=grid.subdomains['sub5'])

# u BCs
bc_u_tent = [EssentialBC(u.forward, 0, subdomain=grid.subdomains['sub14'])] # left
bc_u_tent += [EssentialBC(u.forward, 0, subdomain=grid.subdomains['sub11'])] # right
# NOTE: don't acc need to modify these equations with the explicit scheme since I set them with bc_u_halo after
# but just setting it up this way since I THINK it may be needed for the implicit scheme
bc_u_tent += [neumann_bottom(eq_u_tent, u, subdomain=grid.subdomains['sub12'])] # bottom
bc_u_tent += [neumann_top(eq_u_tent, u, subdomain=grid.subdomains['sub13'])] # top


# This will be automated by the compiler
bc_tmp_u = Function(name='bc_tmp_u', grid=grid, space_order=2, staggered=y)
bc_u_tent += [EssentialBC(u.forward, bc_tmp_u, subdomain=grid.subdomains['sub10'], constrain=True)]


u_tent_solve = petscsolve([eq_u_tent]+bc_u_tent, u.forward, options_prefix='utent_solve', solver_parameters={'ksp_type': 'cg'})


# TODO: can you use subdomains instead of index notation to set the halo region here?
bc_u_halo = [Eq(u[t+1, x, ny-1], 2 - u[t+1, x, ny-2])]  # lid: u=1 at y=1
bc_u_halo += [Eq(u[t+1, x, -1], -u[t+1, x, 0])] # bottom


# v BCs
bc_v_tent = [EssentialBC(v.forward, 0, subdomain=grid.subdomains['sub18'])] # top
bc_v_tent += [EssentialBC(v.forward, 0, subdomain=grid.subdomains['sub19'])] # bottom
bc_v_tent += [neumann_left(eq_v_tent, v, subdomain=grid.subdomains['sub20'])] # left
bc_v_tent += [neumann_right(eq_v_tent, v, subdomain=grid.subdomains['sub16'])] # right


# This will be automated by the compiler
bc_tmp_v = Function(name='bc_tmp_v', grid=grid, space_order=2, staggered=y)
bc_v_tent += [EssentialBC(v.forward, bc_tmp_v, subdomain=grid.subdomains['sub21'], constrain=True)]

v_tent_solve = petscsolve([eq_v_tent]+bc_v_tent, v.forward, options_prefix='vtent_solve', solver_parameters={'ksp_type': 'cg'})


bc_v_halo = [Eq(v[t+1, nx-1, y], -v[t+1, nx-2, y])]
bc_v_halo += [Eq(v[t+1, -1, y], -v[t+1, 0, y])]


# p is at cell centres
bc_p = [neumann_left(neumann_top(eq_p, p, sub1), p, sub1)]
bc_p += [neumann_top(eq_p, p, sub2)]
bc_p += [neumann_right(neumann_top(eq_p, p, sub3), p, sub3)]
bc_p += [neumann_left(eq_p, p, sub4)]
bc_p += [neumann_right(eq_p, p, sub6)]
bc_p += [neumann_bottom(eq_p, p, sub8)]
bc_p += [neumann_right(neumann_bottom(eq_p, p, sub9), p, sub9)]

bc_tmp_p = Function(name='bc_tmp_p', grid=grid, space_order=2, staggered=(x, y))
bc_tmp_p.data[:] = 0.
bc_p += [EssentialBC(p, bc_tmp_p, subdomain=grid.subdomains['sub7'])] # pin pressure at corner


# These two will be automated in the compiler
bc_p += [EssentialBC(p, bc_tmp_p, subdomain=grid.subdomains['sub10'], constrain=True)]
bc_p += [EssentialBC(p, bc_tmp_p, subdomain=grid.subdomains['sub11'], constrain=True)]


# TODO: check symmetry of matrix before using CG..
pressure_solve = petscsolve([eq_p]+bc_p, p, options_prefix='pressure_solve', solver_parameters={'ksp_type': 'cg'})


# Velocity correction
update_u = Eq(u.forward, u.forward - dt*p.dx,
              subdomain=grid.subdomains['sub15'])

update_v = Eq(v.forward, v.forward - dt*p.dy,
              subdomain=grid.subdomains['sub17'])


bc_u = [EssentialBC(u.forward, 0, subdomain=grid.subdomains['sub14'])] # left
bc_u += [EssentialBC(u.forward, 0, subdomain=grid.subdomains['sub11'])] # right
# NOTE: don't acc need to modify these equations with the explicit scheme since I set them with bc_u_halo after
# but just setting it up this way since I THINK it may be needed for the implicit scheme
bc_u += [neumann_bottom(update_u, u, subdomain=grid.subdomains['sub12'])] # bottom
bc_u += [neumann_top(update_u, u, subdomain=grid.subdomains['sub13'])] # top


bc_v = [EssentialBC(v.forward, 0, subdomain=grid.subdomains['sub18'])] # top
bc_v += [EssentialBC(v.forward, 0, subdomain=grid.subdomains['sub19'])] # bottom
bc_v += [neumann_left(update_v, v, subdomain=grid.subdomains['sub20'])] # left
bc_v += [neumann_right(update_v, v, subdomain=grid.subdomains['sub16'])] # right

exprs = [u_tent_solve] + bc_u_halo + [v_tent_solve] + bc_v_halo + [pressure_solve] + [update_u] + bc_u + bc_u_halo + [update_v] + bc_v + bc_v_halo

with switchconfig(language='petsc'):
    op = Operator(exprs)
    # print(op.ccode)
    op.apply(time_M=nt, dt=dt)

# run trivial operators to interpolate the staggered fields back onto original grid "nodes" for plotting and analysis

plotfunc_u = Function(name='plotfunc_u', grid=grid, space_order=2, staggered=NODE)
plotfunc_v = Function(name='plotfunc_v', grid=grid, space_order=2, staggered=NODE)
plotfunc_p = Function(name='plotfunc_p', grid=grid, space_order=2, staggered=NODE)

Operator(Eq(plotfunc_u, u))(time_M=0)
Operator(Eq(plotfunc_v, v))(time_M=0)
Operator(Eq(plotfunc_p, p))()


omega = Function(name='vorticity', grid=grid, space_order=2, staggered=NODE)
vorticity_eqn = Eq(omega, plotfunc_v.dx - plotfunc_u.dy)
# from IPython import embed; embed()
op_vorticity = Operator(vorticity_eqn)
op_vorticity.apply()



#NBVAL_IGNORE_OUTPUT
# Again, check results with Marchi et al 2009.
fig = pyplot.figure(figsize=(12, 6))
ax1 = fig.add_subplot(121)
ax1.plot(plotfunc_u.data[int(grid.shape[0]/2),:], y_coord[:])
ax1.set_xlabel('$u$')
ax1.set_ylabel('$y$')
pyplot.savefig('fig6.png', dpi=100, bbox_inches='tight')
pyplot.show()

vort_data = np.array(omega.data).T  # shape (ny, nx) for matplotlib

# Use percentile-based range so interior contours are visible regardless of grid resolution
# (fine grids resolve very large wall vorticity which would otherwise dominate the scale)
vort_neg_scale = float(np.percentile(vort_data[vort_data < 0], 2))
vort_pos_scale = float(np.percentile(vort_data[vort_data > 0], 98)) if np.any(vort_data > 0) else 1.0

# ~20 negative levels (dashed) and ~10 positive levels (solid), matching paper style
levels_neg = np.linspace(vort_neg_scale, 0, 11)[:-1]
levels_pos = np.linspace(0, vort_pos_scale, 6)[1:]

fig_vort, ax_vort = pyplot.subplots(figsize=(6, 6))
ax_vort.contour(x_coord, y_coord, vort_data, levels=levels_neg,
                colors='k', linestyles='dashed', linewidths=0.6)
ax_vort.contour(x_coord, y_coord, vort_data, levels=levels_pos,
                colors='k', linestyles='solid', linewidths=0.6)
ax_vort.set_aspect('equal')
ax_vort.set_xlim(0, 1)
ax_vort.set_ylim(0, 1)
ax_vort.set_xticks([])
ax_vort.set_yticks([])
pyplot.savefig('vorticity.png', dpi=150, bbox_inches='tight')
pyplot.show()

# Velocity vector plot (normalized: parallel to flow direction, equal length arrows)
U = plotfunc_u.data.T   # shape (ny, nx)
V = plotfunc_v.data.T
mag = np.sqrt(U**2 + V**2)
mag[mag == 0] = 1.0     # avoid division by zero at no-slip walls

fig_vec, ax_vec = pyplot.subplots(figsize=(6, 6))
ax_vec.quiver(x_coord, y_coord, U / mag, V / mag,
              scale=nx, scale_units='width',
              headwidth=2, headlength=2, headaxislength=2,
              width=0.002, color='k')
ax_vec.set_aspect('equal')
ax_vec.set_xlim(0, 1)
ax_vec.set_ylim(0, 1)
ax_vec.set_xticks([])
ax_vec.set_yticks([])
pyplot.savefig('velocity_vectors.png', dpi=150, bbox_inches='tight')
pyplot.show()
