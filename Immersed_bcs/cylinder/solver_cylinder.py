from operator import sub
import os
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt

from devito import (Grid, TimeFunction, Function, Eq, Operator, Border,
                    configuration, SubDomain, NODE, switchconfig, Constant, solve)
from devito.symbolics import retrieve_functions, retrieve_derivatives
from devito.petsc import petscsolve, EssentialBC
from devito.petsc.initialize import PetscInitialize
from devito.mpi.distributed import MPI

from schism import BoundaryConditions , BoundaryGeometry , Boundary

configuration['compiler'] = 'custom'
os.environ['CC'] = 'mpicc'

PetscInitialize()


def _neumann_bottom(eq, t, subdomain):


    lhs, rhs = eq.args
    lhs = lhs._eval_at(t).evaluate
    rhs = rhs._eval_at(t).evaluate
    
    funcs = retrieve_functions(lhs - rhs)
    yind_target = t.indices[-1]
    mapper = {}
    for f in funcs:
        yind = f.indices[-1]
        if (yind - yind_target).as_coeff_Mul()[0] < 0:
            if f.name == 'p':
                mapper[f] = f.subs({yind: yind_target})
            if f.name == 'u' or f.name == 'plotfunc_u':
                mapper[f] = -f.subs({yind: yind_target})
    return Eq(lhs.subs(mapper), rhs.subs(mapper), subdomain=subdomain)


def _neumann_top(eq, t, subdomain):
    # lhs, rhs = eq.evaluate.args

    lhs, rhs = eq.args
    lhs = lhs._eval_at(t).evaluate
    rhs = rhs._eval_at(t).evaluate

    funcs = retrieve_functions(lhs - rhs)
    yind_target = t.indices[-1]
    mapper = {}
    for f in funcs:
        yind = f.indices[-1]
        if (yind - yind_target).as_coeff_Mul()[0] > 0:
            if f.name == 'p':
                mapper[f] = f.subs({yind: yind_target})
            if f.name == 'u':
                mapper[f] = -f.subs({yind: yind_target})
    return Eq(lhs.subs(mapper), rhs.subs(mapper), subdomain=subdomain)


def _neumann_left(eq, t, subdomain):
    # lhs, rhs = eq.evaluate.args

    lhs, rhs = eq.args
    lhs = lhs._eval_at(t).evaluate
    rhs = rhs._eval_at(t).evaluate

    funcs = retrieve_functions(lhs - rhs)
    xind_target = t.indices[-2]
    mapper = {}
    for f in funcs:
        xind = f.indices[-2]
        if (xind - xind_target).as_coeff_Mul()[0] < 0:
            if f.name == 'p':
                mapper[f] = f.subs({xind: xind_target})
            if f.name == 'v' or f.name == 'plotfunc_v':
                mapper[f] = -f.subs({xind: xind_target})
    return Eq(lhs.subs(mapper), rhs.subs(mapper), subdomain=subdomain)


def _neumann_right(eq, t, subdomain):
    # lhs, rhs = eq.evaluate.args

    lhs, rhs = eq.args
    lhs = lhs._eval_at(t).evaluate
    rhs = rhs._eval_at(t).evaluate

    funcs = retrieve_functions(lhs - rhs)
    xind_target = t.indices[-2]
    mapper = {}
    for f in funcs:
        xind = f.indices[-2]
        if (xind - xind_target).as_coeff_Mul()[0] > 0:
            if f.name == 'p':
                mapper[f] = -f.subs({xind: xind_target})
            if f.name == 'v':
                mapper[f] = f.subs({xind: xind_target})
            if f.name == 'u':
                x_dim, spacing = xind.args
                mapper[f] = f.subs({xind: x_dim - spacing})
    return Eq(lhs.subs(mapper), rhs.subs(mapper), subdomain=subdomain)


def make_circle(x_msh, y_msh, centre_x, centre_y, radius):
    return np.sqrt((x_msh - centre_x) **2 + (y_msh - centre_y) **2) - radius


def setup_immersed_bcs(grid, u=None, v=None, p=None, derivs=None, radius=None, centre=None, nu=None):

    # create a sdf for each "staggered grid"

    x, y = grid.dimensions

    # dummy - oversight in schism
    sdf = Function(name='sdf', grid=grid, space_order=2, staggered=NODE)

    sdf_x = Function(name='sdf_x', grid=grid, space_order=2, staggered=x)
    sdf_y = Function(name='sdf_y', grid=grid, space_order=2, staggered=y)
    sdf_x_y = Function(name='sdf_x_y', grid=grid, space_order=2, staggered=(x,y))

    x_coords = np.linspace(0, grid.extent[0], grid.shape[0])
    y_coords = np.linspace(0, grid.extent[1], grid.shape[1])
    x_msh , y_msh = np.meshgrid(x_coords, y_coords, indexing='ij')

    centre_x, centre_y = centre
    sdf.data[:] = make_circle(x_msh, y_msh, centre_x, centre_y, radius)

    sdf_x.data[:] = make_circle(x_msh+grid.spacing[0]/2, y_msh, centre_x, centre_y, radius)

    sdf_y.data[:] = make_circle(x_msh, y_msh+grid.spacing[1]/2, centre_x, centre_y, radius)

    sdf_x_y.data[:] = make_circle(x_msh+grid.spacing[0]/2, y_msh+grid.spacing[1]/2, centre_x, centre_y, radius)


    # # save each field to file
    # plt.imshow(sdf_x.data.T, origin='lower', extent=(0, grid.extent[0], 0, grid.extent[1]))
    # plt.contour(x_msh.T+grid.spacing[0]/2, y_msh.T, sdf_x.data.T, levels=[0], colors='black', linewidths=2)
    # plt.savefig('sdf_x.png')
    # plt.close()

    # Setup bcs
    zero = sp.core.numbers.Zero()

    # cutoff is how close a point can get to the boundary before being excluded
    # from the fluid solve entirely
    cutoff = {(grid.dimensions[0].spacing/2, grid.dimensions[1].spacing/2): 0.1,
              (grid.dimensions[0].spacing/2, zero): 0.05,
              (zero, grid.dimensions[1].spacing/2): 0.05}
    bg = BoundaryGeometry((sdf, sdf_x, sdf_y, sdf_x_y), cutoff=cutoff)

    bcs = BoundaryConditions([Eq(u, 0), Eq(v, 0), Eq(p.dx, 0), Eq(p.dy, 0)], funcs=(u, v, p))


    # bcs = BoundaryConditions([Eq(u, 0), Eq(v, 0), Eq(p.dx - nu*u.laplace, 0), Eq(p.dy - nu*v.laplace, 0)], funcs=(u, v, p))

    # bcs = BoundaryConditions(
    #     [Eq(u, 0), Eq(v, 0), Eq(-bg.n[0]*p.dx - bg.n[1]*p.dy, 0)],
    #     funcs=(u, v, p)
    # )

    boundary = Boundary(bcs, bg)

    subs = boundary.substitutions(tuple(derivs))

    return subs


def make_solver(ny, nx=None, ab2=False, implicit_diffusion=False, u_max=0.3):

    # ab2 - Adams-Bashforth 2 for convection, otherwise forward Euler
    # implicit_diffusion - Crank-Nicolson for diffusion, otherwise forward Euler
    # u_max - peak parabolic inflow velocity (DFG 2D-1: 0.3, DFG 2D-2/2D-3: 1.5)
    so = 2

    # DFG benchmark's own physical units: D=0.1, channel 2.2 x 0.41, ν=0.001
    x_extent = 2.2
    y_extent = 0.41
    radius = 0.05
    centre_x, centre_y = (0.2, 0.2)
    if nx is None:
        nx = int(round((x_extent / y_extent) * (ny - 1))) + 1
    dx_phys = x_extent / (nx - 1)
    dy_phys = y_extent / (ny - 1)
    print(f'Grid: nx={nx}, ny={ny}, dx={dx_phys:.6f}, dy={dy_phys:.6f}')

    class Sub1(SubDomain):
        name = 'sub1'
        def define(self, dimensions):
            x, y = dimensions
            return {x: ('left', 1), y: ('middle', ny-2, 1)}

    class Sub2(SubDomain):
        name = 'sub2'
        def define(self, dimensions):
            x, y = dimensions
            return {x: ('middle', 1, 2), y: ('middle', ny-2, 1)}

    class Sub3(SubDomain):
        name = 'sub3'
        def define(self, dimensions):
            x, y = dimensions
            return {x: ('middle', nx-2, 1), y: ('middle', ny-2, 1)}

    class Sub4(SubDomain):
        name = 'sub4'
        def define(self, dimensions):
            x, y = dimensions
            return {x: ('left', 1), y: ('middle', 1, 2)}

    class Sub5(SubDomain):
        name = 'sub5'
        def define(self, dimensions):
            x, y = dimensions
            return {x: ('middle', 1, 2), y: ('middle', 1, 2)}

    class Sub6(SubDomain):
        name = 'sub6'
        def define(self, dimensions):
            x, y = dimensions
            return {x: ('middle', nx-2, 1), y: ('middle', 1, 2)}

    class Sub7(SubDomain):
        name = 'sub7'
        def define(self, dimensions):
            x, y = dimensions
            return {x: ('left', 1), y: ('left', 1)}

    class Sub8(SubDomain):
        name = 'sub8'
        def define(self, dimensions):
            x, y = dimensions
            return {x: ('middle', 1, 2), y: ('left', 1)}

    class Sub9(SubDomain):
        name = 'sub9'
        def define(self, dimensions):
            x, y = dimensions
            return {x: ('middle', nx-2, 1), y: ('left', 1)}

    class Sub10(SubDomain):
        name = 'sub10'
        def define(self, dimensions):
            x, y = dimensions
            return {x: x, y: ('right', 1)}

    class Sub11(SubDomain):
        name = 'sub11'
        def define(self, dimensions):
            x, y = dimensions
            return {x: ('right', 1), y: ('left', ny-1)}

    class Sub12(SubDomain):
        name = 'sub12'
        def define(self, dimensions):
            x, y = dimensions
            return {x: ('middle', 1, 1), y: ('left', 1)}

    class Sub13(SubDomain):
        name = 'sub13'
        def define(self, dimensions):
            x, y = dimensions
            return {x: ('middle', 1, 1), y: ('middle', ny-2, 1)}

    class Sub15(SubDomain):
        name = 'sub15'
        def define(self, dimensions):
            x, y = dimensions
            return {x: ('middle', 1, 1), y: ('middle', 1, 2)}

    class Sub16(SubDomain):
        name = 'sub16'
        def define(self, dimensions):
            x, y = dimensions
            return {x: ('middle', nx-2, 1), y: ('middle', 1, 1)}

    class Sub17(SubDomain):
        name = 'sub17'
        def define(self, dimensions):
            x, y = dimensions
            return {x: ('middle', 1, 2), y: ('middle', 1, 1)}

    class Sub18(SubDomain):
        name = 'sub18'
        def define(self, dimensions):
            x, y = dimensions
            return {x: ('left', nx-1), y: ('right', 1)}

    class Sub19(SubDomain):
        name = 'sub19'
        def define(self, dimensions):
            x, y = dimensions
            return {x: ('left', nx-1), y: ('left', 1)}

    class Sub20(SubDomain):
        name = 'sub20'
        def define(self, dimensions):
            x, y = dimensions
            return {x: ('left', 1), y: ('middle', 1, 1)}

    class Sub21(SubDomain):
        name = 'sub21'
        def define(self, dimensions):
            x, y = dimensions
            return {x: ('right', 1), y: y}

    class Sub22(SubDomain):
        name = 'sub22'
        def define(self, dimensions):
            x, y = dimensions
            return {x: ('left', 1), y: ('left', ny-1)}

    subdomains = tuple(cls() for cls in [
        Sub1, Sub2, Sub3, Sub4, Sub5, Sub6, Sub7, Sub8, Sub9, Sub10,
        Sub11, Sub12, Sub13, Sub15, Sub16, Sub17, Sub18, Sub19,
        Sub20, Sub21, Sub22
    ])

    grid = Grid(shape=(nx, ny), extent=(x_extent, y_extent), subdomains=subdomains,
                dtype=np.float64)

    x, y = grid.dimensions
    t = grid.stepping_dim

    nu = Constant(name='nu', dtype=np.float64)
    dt_c = Constant(name='dt_c', dtype=np.float64)

    time_order = 2 if ab2 else 1

    u = TimeFunction(name='u', grid=grid, space_order=so, time_order=time_order, staggered=y)
    v = TimeFunction(name='v', grid=grid, space_order=so, time_order=time_order, staggered=x)
    p = Function(name='p', grid=grid, space_order=so, staggered=(x, y))

    # parabolic inflow u across the full channel height (no step)
    u_inflow = Function(name='u_inflow', grid=grid, space_order=so, staggered=y)
    for j in range(ny - 1):
        y_j = j * dy_phys + dy_phys / 2  # staggered y position for u
        u_inflow.data[0, j] = 4.0 * u_max * y_j * (y_extent - y_j) / y_extent**2


    # Convection
    if ab2:
        conv_u = (3./2.)*(u*u.dxc + v*u.dyc) - (1./2.)*(u.backward*u.backward.dxc + v.backward*u.backward.dyc)
        conv_v = (3./2.)*(u*v.dxc + v*v.dyc) - (1./2.)*(u.backward*v.backward.dxc + v.backward*v.backward.dyc)
    else:
        conv_u = u*u.dxc + v*u.dyc
        conv_v = u*v.dxc + v*v.dyc

    # Diffusion
    if implicit_diffusion:
        diff_u = (nu/2.)*(u.dx2 + u.dy2) + (nu/2.)*(u.forward.dx2 + u.forward.dy2)
        diff_v = (nu/2.)*(v.dx2 + v.dy2) + (nu/2.)*(v.forward.dx2 + v.forward.dy2)
    else:
        diff_u = nu*(u.dx2 + u.dy2)
        diff_v = nu*(v.dx2 + v.dy2)

    eq_u_tent = Eq(u.dt + conv_u, diff_u, subdomain=grid.subdomains['sub15'])
    eq_v_tent = Eq(v.dt + conv_v, diff_v, subdomain=grid.subdomains['sub17'])

    ux_cc = u.forward.dx(x0=x + x.spacing/2)
    vy_cc = v.forward.dy(x0=y + y.spacing/2)

    eq_p = Eq(p.laplace, (1./dt_c)*(ux_cc + vy_cc), subdomain=grid.subdomains['sub5'])

    # u BCs (tent)
    bc_u_tent = [EssentialBC(u.forward, u_inflow, subdomain=grid.subdomains['sub22'])]
    bc_u_tent += [_neumann_right(eq_u_tent, u, subdomain=grid.subdomains['sub11'])]
    bc_u_tent += [_neumann_bottom(eq_u_tent, u, subdomain=grid.subdomains['sub12'])]
    bc_u_tent += [_neumann_top(eq_u_tent, u, subdomain=grid.subdomains['sub13'])]
    
    # This can be done in postprocessing before doing the interpolation but just doing it here for ease
    # Essentially, I apply this mapping already when updating the nodes for u along sub13
    bc_u_halo = [Eq(u.forward, -u.forward.subs({y + y.spacing/2: y - y.spacing/2}), subdomain=grid.subdomains['sub10'])]
    # pretty sure this can be dropped, potentially want to add it in post processing before interpolation but it doesn't need to be part of the solve
    # since i modify the u equation along sub12 to account for this


    # v BCs (tent)
    bc_v_tent = [EssentialBC(v.forward, 0, subdomain=grid.subdomains['sub18'])]
    bc_v_tent += [EssentialBC(v.forward, 0, subdomain=grid.subdomains['sub19'])]
    bc_v_tent += [_neumann_left(eq_v_tent, v, subdomain=grid.subdomains['sub20'])]
    bc_v_tent += [_neumann_right(eq_v_tent, v, subdomain=grid.subdomains['sub16'])]

    bc_v_halo = [Eq(v.forward, v.forward.subs({x+x.spacing/2: x-x.spacing/2}), subdomain=grid.subdomains['sub21'])]


    # p BCs
    bc_tmp_p = Function(name='bc_tmp_p', grid=grid, space_order=so, staggered=(x, y))
    bc_tmp_p.data[:] = 0.

    # sub = subdomains
    bc_p = [_neumann_left(_neumann_top(eq_p, p, grid.subdomains['sub1']), p, grid.subdomains['sub1'])]
    bc_p += [_neumann_top(eq_p, p, grid.subdomains['sub2'])]
    bc_p += [_neumann_right(_neumann_top(eq_p, p, grid.subdomains['sub3']), p, grid.subdomains['sub3'])]
    bc_p += [_neumann_left(eq_p, p, grid.subdomains['sub4'])]
    bc_p += [_neumann_right(eq_p, p, grid.subdomains['sub6'])]
    bc_p += [_neumann_bottom(eq_p, p, grid.subdomains['sub8'])]
    bc_p += [_neumann_right(_neumann_bottom(eq_p, p, grid.subdomains['sub9']), p, grid.subdomains['sub9'])]

    # Velocity correction
    # p.dx/p.dy need x0 pinned explicitly to the unstaggered location for schism to work correctly with staggering
    update_u = Eq(u.forward, u.forward - dt_c*p.dx(x0={x: x}), subdomain=grid.subdomains['sub15'])
    update_v = Eq(v.forward, v.forward - dt_c*p.dy(x0={y: y}), subdomain=grid.subdomains['sub17'])

    bc_u = [Eq(u.forward, u_inflow, subdomain=grid.subdomains['sub22'])]
    # Simple halo copy for outflow: u[nx] = u[nx-1] (first-order Neumann, no dx2 stencil here)
    bc_u += [Eq(u.forward, u.forward.subs({x: x - x.spacing}), subdomain=grid.subdomains['sub11'])]
    bc_u += [_neumann_bottom(update_u, u, subdomain=grid.subdomains['sub12'])]
    bc_u += [_neumann_top(update_u, u, subdomain=grid.subdomains['sub13'])]

    bc_v = [Eq(v.forward, 0, subdomain=grid.subdomains['sub18'])]
    bc_v += [Eq(v.forward, 0, subdomain=grid.subdomains['sub19'])]
    bc_v += [_neumann_left(update_v, v, subdomain=grid.subdomains['sub20'])]
    bc_v += [_neumann_right(update_v, v, subdomain=grid.subdomains['sub16'])]
    

    derivs = retrieve_derivatives([eq_u_tent] + [eq_v_tent] + [eq_p] + [update_u] + [update_v], mode='unique')
    derivs = [d for d in derivs if grid.stepping_dim not in d.dims]
    print([derivs])

    ib_subs = setup_immersed_bcs(grid, u, v, p, derivs, radius=radius, centre=(centre_x, centre_y), nu=nu)

    pressure_solve = petscsolve([eq_p.subs(ib_subs)] + bc_p, p,
                                options_prefix='pressure_solve',
                                solver_parameters={'ksp_type': 'cg', 'ksp_rtol': 1e-4})

    u_tent_solve = petscsolve([eq_u_tent.subs(ib_subs)] + bc_u_tent, u.forward,
                              options_prefix='utent_solve',
                              solver_parameters={'ksp_type': 'cg', 'ksp_rtol': 1e-7})

    v_tent_solve = petscsolve([eq_v_tent.subs(ib_subs)] + bc_v_tent, v.forward,
                              options_prefix='vtent_solve',
                              solver_parameters={'ksp_type': 'cg', 'ksp_rtol': 1e-7})

    exprs = ([u_tent_solve] + bc_u_halo +
             [v_tent_solve] + bc_v_halo +
             [pressure_solve] +
             [update_u.subs(ib_subs)] + bc_u + bc_u_halo +
             [update_v.subs(ib_subs)] + bc_v + bc_v_halo)

    with switchconfig(language='petsc'):
        op = Operator(exprs)

    # then gather to rank 0 here?

    plotfunc_u = Function(name='plotfunc_u', grid=grid, space_order=so,
                          staggered=NODE)
    plotfunc_v = Function(name='plotfunc_v', grid=grid, space_order=so,
                          staggered=NODE)
    plotfunc_p = Function(name='plotfunc_p', grid=grid, space_order=so,
                          staggered=NODE)

    vorticity = Function(name='vorticity', grid=grid, space_order=so,
                        staggered=NODE)

    stream = Function(name='psi', grid=grid, space_order=so,
                         staggered=NODE)

    psi_bc = Function(name='psi_bc', grid=grid, space_order=so, staggered=NODE)
    for j in range(ny):
        y_j = j * dy_phys
        psi_bc.data[:, j] = (4.0*u_max/y_extent**2) * (y_extent*y_j**2/2.0 - y_j**3/3.0)

    eq_interp_u = Eq(plotfunc_u, u)
    op_interp_u = Operator([eq_interp_u])

    eq_interp_v = Eq(plotfunc_v, v)
    op_interp_v = Operator([eq_interp_v])

    eq_interp_p = Eq(plotfunc_p, p)
    op_interp_p = Operator([eq_interp_p])

    vorticity_eqn = Eq(vorticity, v.dx - u.dy)
    op_vorticity = Operator(vorticity_eqn)

    border = Border(grid, 1)
    stream_bc = [
        EssentialBC(stream, psi_bc, subdomain=border),
        Eq(stream, stream.subs({x: x - x.spacing}), subdomain=grid.subdomains['sub21']),
    ]
    stream_eqn = Eq(stream.laplace, -(v.dx - u.dy), subdomain=grid.interior)
    stream_solver = petscsolve([stream_eqn]+stream_bc, stream, options_prefix='stream_solve')

    with switchconfig(language='petsc'):
        op_stream = Operator([stream_solver])


    x_coord = np.linspace(0, x_extent, nx)
    y_coord = np.linspace(0, y_extent, ny)

    def run_cavity_flow(re_val, t_end=400.0, tol=1e-3, check_every=200, fixed=False):

        # If fixed is True, run for exactly t_end time in a single op.apply(), otherwise chunk the run
        # and check convergence every check_every steps.
        # Initial condition per Griebel/Dornseifer/Neunhoeffer ch.5 "Flow Past
        # an Obstacle", scaled to the free-stream velocity: UI = u_max, VI = 0.0, PI = 0.0
        u.data[:] = u_max
        v.data[:] = 0.
        p.data[:] = 0.

        # nu is derived from the target Re and the actual physical D/Ubar, rather
        # than set independently, so Re can never silently drift out of sync with
        # the geometry/inflow scale (DFG 2D-1: D=0.1, Ubar=0.2, Re=20 -> nu=0.001,
        # exactly the benchmark's own viscosity).
        D = 2. * radius
        u_bar = (2./3.) * u_max
        nu_val = u_bar * D / re_val
        dt_conv = 0.5 * min(dx_phys, dy_phys) / u_max

        if implicit_diffusion:
            dt_val = dt_conv
        else:
            dt_diff = (0.5 * min(dx_phys, dy_phys)**2) / (4.0 * nu_val)
            dt_val = min(dt_conv, dt_diff)

        nu.data = np.float64(nu_val)
        dt_c.data = np.float64(dt_val)
        max_steps = int(t_end/dt_val)

        print(f'Re={re_val} (nu={nu_val}): dt={dt_c.data}, max_steps={max_steps}')

        n_slots = time_order + 1

        if fixed:
            op.apply(time_m=0, time_M=max_steps - 1, dt=dt_val)

            print(f'completed t_end={t_end} ({max_steps} steps)')
            tb = max_steps % n_slots
        else:
            norm0 = None
            norm = float('inf')
            step = 0

            while step < max_steps:
                chunk = min(check_every, max_steps - step)

                op.apply(time_m=step, time_M=step + chunk - 1, dt=dt_val)
                step += chunk

                du = u.data[0] - u.data[1]
                dv = v.data[0] - v.data[1]

                # If I use the chunking method then need to fix this in parallel...
                norm = float(np.sqrt(np.sum(du**2) + np.sum(dv**2)))

                if norm0 is None:
                    norm0 = norm if norm > 0 else 1.0

                print(f't={step*dt_val} step={step} norm={norm} rel={norm/norm0}')

                if norm / norm0 < tol:
                    print(f'converged at t={step*dt_val} '
                          f'(step {step}/{max_steps}), rel={norm/norm0}')
                    break
            else:
                print(f'reached t_end={t_end}, rel={norm/norm0}')

            # Refresh halos on the final buffer before post-processing
            op.apply(time_m=step, time_M=step, dt=dt_val)
            tb = step % n_slots

        op_interp_u(time_M=tb)
        op_interp_v(time_M=tb)
        op_interp_p()
        op_vorticity.apply(time_M=tb)
        op_stream.apply(time_M=tb)

        u_g = u.data_gather(rank=0)
        v_g = v.data_gather(rank=0)
        p_g = p.data_gather(rank=0)

        comm = grid.comm
        if comm is not None and configuration['mpi']:
            _my_rank = comm.rank
            if comm != MPI.COMM_NULL and _my_rank == 0:
                u_snap = u_g[tb]
                v_snap = v_g[tb]
            else:
                u_snap = None
                v_snap = None
        else:
            _my_rank = 0
            u_snap = u.data[tb]
            v_snap = v.data[tb]

        delta_p = None
        if _my_rank == 0:
            def extrap_p(field, x_boundary, y_line, direction, npts=2):
                j_lo = int(np.floor((y_line - dy_phys/2) / dy_phys))
                i = int(np.floor((x_boundary - dx_phys/2) / dx_phys)) \
                    + (1 if direction > 0 else 0)
                found = []
                while len(found) < npts:
                    val = 0.5*(field[i, j_lo] + field[i, j_lo + 1])
                    if val != 0.0:
                        xi = i*dx_phys + dx_phys/2
                        found.append((xi, val))
                    i += direction
                # Lagrange interpolation/extrapolation through `found`,
                # evaluated at x_boundary
                total = 0.0
                for k, (xk, yk) in enumerate(found):
                    term = yk
                    for m, (xm, _) in enumerate(found):
                        if m != k:
                            term *= (x_boundary - xm)/(xk - xm)
                    total += term
                return total

            # DFG benchmark points (0.15, 0.2) and (0.25, 0.2), in the solver's
            # own units now that geometry/velocity match the benchmark exactly
            p_front = extrap_p(p_g, 0.15, 0.2, direction=-1, npts=2)
            p_rear = extrap_p(p_g, 0.25, 0.2, direction=1, npts=2)
            delta_p = float(p_front - p_rear)
            print(f'Delta p (front - rear) = {delta_p}  '
                  f'(DFG 2D-1 target: 0.11752016697)')

            p_front_q = extrap_p(p_g, 0.15, 0.2, direction=-1, npts=3)
            p_rear_q = extrap_p(p_g, 0.25, 0.2, direction=1, npts=3)
            delta_p_q = float(p_front_q - p_rear_q)
            print(f'Delta p (quadratic, 3-point extrapolation) = {delta_p_q}  '
                  f'(DFG 2D-1 target: 0.11752016697)')


        return (x_coord.copy(), y_coord.copy(),
                plotfunc_u.data_gather(rank=0),
                plotfunc_v.data_gather(rank=0),
                vorticity.data_gather(rank=0),
                stream.data_gather(rank=0),
                _my_rank,
                u_snap,
                v_snap,
                delta_p,
                plotfunc_p.data_gather(rank=0))

    return run_cavity_flow
