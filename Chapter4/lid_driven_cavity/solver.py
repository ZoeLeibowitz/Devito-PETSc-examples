import os
import numpy as np

from devito import (Grid, TimeFunction, Function, Eq, Operator, Border,
                    configuration, SubDomain, NODE, Derivative, switchconfig, Constant, solve)
from devito.symbolics import retrieve_functions
from devito.petsc import petscsolve, EssentialBC, NullSpace
from devito.petsc.initialize import PetscInitialize
from devito.mpi.distributed import MPI

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
            if f.name == 'phi':
                mapper[f] = f.subs({yind: yind_target})
            if f.name == 'u' or f.name == 'plotfunc_u':
                mapper[f] = -f.subs({yind: yind_target})
    return Eq(lhs.subs(mapper), rhs.subs(mapper), subdomain=subdomain)


def _neumann_top(eq, t, subdomain):

    lhs, rhs = eq.args
    lhs = lhs._eval_at(t).evaluate
    rhs = rhs._eval_at(t).evaluate

    funcs = retrieve_functions(lhs - rhs)
    yind_target = t.indices[-1]
    mapper = {}
    for f in funcs:
        yind = f.indices[-1]
        if (yind - yind_target).as_coeff_Mul()[0] > 0:
            if f.name == 'phi':
                mapper[f] = f.subs({yind: yind_target})
            if f.name == 'u':
                mapper[f] = 2 - f.subs({yind: yind_target})
    return Eq(lhs.subs(mapper), rhs.subs(mapper), subdomain=subdomain)


def _neumann_left(eq, t, subdomain):

    lhs, rhs = eq.args
    lhs = lhs._eval_at(t).evaluate
    rhs = rhs._eval_at(t).evaluate

    funcs = retrieve_functions(lhs - rhs)
    xind_target = t.indices[-2]
    mapper = {}
    for f in funcs:
        xind = f.indices[-2]
        if (xind - xind_target).as_coeff_Mul()[0] < 0:
            if f.name == 'phi':
                mapper[f] = f.subs({xind: xind_target})
            if f.name == 'v' or f.name == 'plotfunc_v':
                mapper[f] = -f.subs({xind: xind_target})
    return Eq(lhs.subs(mapper), rhs.subs(mapper), subdomain=subdomain)

# not sure if these are acc quite right for neumann pressure?  maybe it's not 2nd order?
def _neumann_right(eq, t, subdomain):

    lhs, rhs = eq.args
    lhs = lhs._eval_at(t).evaluate
    rhs = rhs._eval_at(t).evaluate

    funcs = retrieve_functions(lhs - rhs)
    xind_target = t.indices[-2]
    mapper = {}
    for f in funcs:
        xind = f.indices[-2]
        if (xind - xind_target).as_coeff_Mul()[0] > 0:
            if f.name == 'phi':
                mapper[f] = f.subs({xind: xind_target})
            if f.name == 'v':
                mapper[f] = -f.subs({xind: xind_target})
    return Eq(lhs.subs(mapper), rhs.subs(mapper), subdomain=subdomain)


def make_solver(nx, ny, ab2=False, implicit_diffusion=False, conservative=True):

    # ab2 - Adams-Bashforth 2 for convection, otherwise forward Euler
    # implicit_diffusion - Crank-Nicolson for diffusion, otherwise forward Euler
    so = 2

    class PhiTopLeft(SubDomain):
        name = 'phi_top_left'
        def define(self, dimensions):
            x, y = dimensions
            return {x: ('left', 1), y: ('middle', ny-2, 1)}

    class PhiTopRow(SubDomain):
        name = 'phi_top_row'
        def define(self, dimensions):
            x, y = dimensions
            return {x: ('middle', 1, 2), y: ('middle', ny-2, 1)}

    class PhiTopRight(SubDomain):
        name = 'phi_top_right'
        def define(self, dimensions):
            x, y = dimensions
            return {x: ('middle', nx-2, 1), y: ('middle', ny-2, 1)}

    class PhiLeftColumn(SubDomain):
        name = 'phi_left_column'
        def define(self, dimensions):
            x, y = dimensions
            return {x: ('left', 1), y: ('middle', 1, 2)}

    class PhiInterior(SubDomain):
        name = 'phi_interior'
        def define(self, dimensions):
            x, y = dimensions
            return {x: ('middle', 1, 2), y: ('middle', 1, 2)}

    class PhiRightColumn(SubDomain):
        name = 'phi_right_column'
        def define(self, dimensions):
            x, y = dimensions
            return {x: ('middle', nx-2, 1), y: ('middle', 1, 2)}

    class PhiBottomLeft(SubDomain):
        name = 'phi_bottom_left'
        def define(self, dimensions):
            x, y = dimensions
            return {x: ('left', 1), y: ('left', 1)}

    class PhiBottomRow(SubDomain):
        name = 'phi_bottom_row'
        def define(self, dimensions):
            x, y = dimensions
            return {x: ('middle', 1, 2), y: ('left', 1)}

    class PhiBottomRight(SubDomain):
        name = 'phi_bottom_right'
        def define(self, dimensions):
            x, y = dimensions
            return {x: ('middle', nx-2, 1), y: ('left', 1)}

    class UGhost(SubDomain):
        name = 'u_ghost'
        def define(self, dimensions):
            x, y = dimensions
            return {x: x, y: ('right', 1)}

    class URightWall(SubDomain):
        name = 'u_right_wall'
        def define(self, dimensions):
            x, y = dimensions
            return {x: ('right', 1), y: ('left', ny-1)}

    class UBottomRow(SubDomain):
        name = 'u_bottom_row'
        def define(self, dimensions):
            x, y = dimensions
            return {x: ('middle', 1, 1), y: ('left', 1)}

    class UTopRow(SubDomain):
        name = 'u_top_row'
        def define(self, dimensions):
            x, y = dimensions
            return {x: ('middle', 1, 1), y: ('middle', ny-2, 1)}

    class ULeftWall(SubDomain):
        name = 'u_left_wall'
        def define(self, dimensions):
            x, y = dimensions
            return {x: ('left', 1), y: ('left', ny-1)}

    class UInterior(SubDomain):
        name = 'u_interior'
        def define(self, dimensions):
            x, y = dimensions
            return {x: ('middle', 1, 1), y: ('middle', 1, 2)}

    class VRightColumn(SubDomain):
        name = 'v_right_column'
        def define(self, dimensions):
            x, y = dimensions
            return {x: ('middle', nx-2, 1), y: ('middle', 1, 1)}

    class VInterior(SubDomain):
        name = 'v_interior'
        def define(self, dimensions):
            x, y = dimensions
            return {x: ('middle', 1, 2), y: ('middle', 1, 1)}

    class VTopWall(SubDomain):
        name = 'v_top_wall'
        def define(self, dimensions):
            x, y = dimensions
            return {x: ('left', nx-1), y: ('right', 1)}

    class VBottomWall(SubDomain):
        name = 'v_bottom_wall'
        def define(self, dimensions):
            x, y = dimensions
            return {x: ('left', nx-1), y: ('left', 1)}

    class VLeftColumn(SubDomain):
        name = 'v_left_column'
        def define(self, dimensions):
            x, y = dimensions
            return {x: ('left', 1), y: ('middle', 1, 1)}

    class VGhost(SubDomain):
        name = 'v_ghost'
        def define(self, dimensions):
            x, y = dimensions
            return {x: ('right', 1), y: y}

    grid = Grid(shape=(nx, ny), extent=(1., 1.), dtype=np.float64)

    # SubDomains of phi
    phi_interior = PhiInterior(grid=grid)
    phi_top_row = PhiTopRow(grid=grid)
    phi_bottom_row = PhiBottomRow(grid=grid)
    phi_left_column = PhiLeftColumn(grid=grid)
    phi_right_column = PhiRightColumn(grid=grid)
    phi_top_left = PhiTopLeft(grid=grid)
    phi_top_right = PhiTopRight(grid=grid)
    phi_bottom_right = PhiBottomRight(grid=grid)
    phi_bottom_left = PhiBottomLeft(grid=grid)

    # SubDomains of u
    u_interior = UInterior(grid=grid)
    u_left_wall = ULeftWall(grid=grid)
    u_right_wall = URightWall(grid=grid)
    u_bottom_row = UBottomRow(grid=grid)
    u_top_row = UTopRow(grid=grid)
    u_ghost = UGhost(grid=grid)

    # SubDomains of v
    v_interior = VInterior(grid=grid)
    v_bottom_wall = VBottomWall(grid=grid)
    v_top_wall = VTopWall(grid=grid)
    v_left_column = VLeftColumn(grid=grid)
    v_right_column = VRightColumn(grid=grid)
    v_ghost = VGhost(grid=grid)

    x, y = grid.dimensions
    t = grid.stepping_dim

    re = Constant(name='re', dtype=np.float64)
    dt_c = Constant(name='dt_c', dtype=np.float64)

    time_order = 2 if ab2 else 1

    u = TimeFunction(name='u', grid=grid, space_order=so, time_order=time_order, staggered=y)
    v = TimeFunction(name='v', grid=grid, space_order=so, time_order=time_order, staggered=x)
    phi = Function(name='phi', grid=grid, space_order=so, staggered=(x, y))

    # Convection
    if conservative:
        # Conservative form ... Each velocity is
        # interpolated to the cell centres (CELL, where phi is stored) and to the grid
        # nodes (NODE). A zeroth order Derivative with x0 does this.
        hx, hy = x.spacing, y.spacing

        def flux(u, v):
            uc = Derivative(u, (x, 0), x0={x: x + hx/2})
            vc = Derivative(v, (y, 0), x0={y: y + hy/2})
            un = Derivative(u, (y, 0), x0={y: y})
            vn = Derivative(v, (x, 0), x0={x: x})
            uu, vv, uv = uc**2, vc**2, un*vn
            hu = (uu - uu.subs({x: x - hx}))/hx + (uv.subs({y: y + hy}) - uv)/hy
            hv = (uv.subs({x: x + hx}) - uv)/hx + (vv - vv.subs({y: y - hy}))/hy
            return hu, hv

        hu, hv = flux(u, v)
        if ab2:
            hu_b, hv_b = flux(u.backward, v.backward)
            conv_u = (3./2.)*hu - (1./2.)*hu_b
            conv_v = (3./2.)*hv - (1./2.)*hv_b
        else:
            conv_u, conv_v = hu, hv
    else:
        if ab2:
            conv_u = (3./2.)*(u*u.dxc + v*u.dyc) - (1./2.)*(u.backward*u.backward.dxc + v.backward*u.backward.dyc)
            conv_v = (3./2.)*(u*v.dxc + v*v.dyc) - (1./2.)*(u.backward*v.backward.dxc + v.backward*v.backward.dyc)
        else:
            conv_u = u*u.dxc + v*u.dyc
            conv_v = u*v.dxc + v*v.dyc

    # Diffusion
    if implicit_diffusion:
        diff_u = (1./(2.*re))*(u.dx2 + u.dy2) + (1./(2.*re))*(u.forward.dx2 + u.forward.dy2)
        diff_v = (1./(2.*re))*(v.dx2 + v.dy2) + (1./(2.*re))*(v.forward.dx2 + v.forward.dy2)
    else:
        diff_u = (1./re)*(u.dx2 + u.dy2)
        diff_v = (1./re)*(v.dx2 + v.dy2)

    eq_u_tent = Eq(u.dt + conv_u, diff_u, subdomain=u_interior)
    eq_v_tent = Eq(v.dt + conv_v, diff_v, subdomain=v_interior)

    ux_cc = u.forward.dx(x0=x + x.spacing/2)
    vy_cc = v.forward.dy(x0=y + y.spacing/2)

    eq_phi = Eq(phi.laplace, (1./dt_c)*(ux_cc + vy_cc), subdomain=phi_interior)

    # u BCs (tent)
    bc_tmp_u = Function(name='bc_tmp_u', grid=grid, space_order=so, staggered=y)
    bc_u_tent = [EssentialBC(u.forward, 0, subdomain=u_left_wall)]
    bc_u_tent += [EssentialBC(u.forward, 0, subdomain=u_right_wall)]
    bc_u_tent += [_neumann_bottom(eq_u_tent, u, subdomain=u_bottom_row)]
    bc_u_tent += [_neumann_top(eq_u_tent, u, subdomain=u_top_row)]

    u_tent_solve = petscsolve([eq_u_tent] + bc_u_tent, u.forward,
                              options_prefix='utent_solve',
                              solver_parameters={'ksp_type': 'cg', 'ksp_rtol': 1e-7})

    # v BCs (tent)
    bc_v_tent = [EssentialBC(v.forward, 0, subdomain=v_top_wall)]
    bc_v_tent += [EssentialBC(v.forward, 0, subdomain=v_bottom_wall)]
    bc_v_tent += [_neumann_left(eq_v_tent, v, subdomain=v_left_column)]
    bc_v_tent += [_neumann_right(eq_v_tent, v, subdomain=v_right_column)]

    v_tent_solve = petscsolve([eq_v_tent] + bc_v_tent, v.forward,
                              options_prefix='vtent_solve',
                              solver_parameters={'ksp_type': 'cg', 'ksp_rtol': 1e-7})

    # phi BCs, homogeneous Neumann on every wall. The system is then singular, with the
    # constants as its null space, which is passed to the solver
    bc_phi = [_neumann_left(_neumann_top(eq_phi, phi, phi_top_left), phi, phi_top_left)]
    bc_phi += [_neumann_top(eq_phi, phi, phi_top_row)]
    bc_phi += [_neumann_right(_neumann_top(eq_phi, phi, phi_top_right), phi, phi_top_right)]
    bc_phi += [_neumann_left(eq_phi, phi, phi_left_column)]
    bc_phi += [_neumann_right(eq_phi, phi, phi_right_column)]
    bc_phi += [_neumann_bottom(eq_phi, phi, phi_bottom_row)]
    bc_phi += [_neumann_right(_neumann_bottom(eq_phi, phi, phi_bottom_right), phi, phi_bottom_right)]
    bc_phi += [_neumann_left(_neumann_bottom(eq_phi, phi, phi_bottom_left), phi, phi_bottom_left)]

    pressure_solve = petscsolve([eq_phi] + bc_phi, phi,
                                options_prefix='pressure_solve',
                                nullspace=NullSpace(constant=True),
                                solver_parameters={'ksp_type': 'cg', 'ksp_rtol': 1e-7})

    # Velocity correction
    update_u = Eq(u.forward, u.forward - dt_c*phi.dx, subdomain=u_interior)
    update_v = Eq(v.forward, v.forward - dt_c*phi.dy, subdomain=v_interior)

    bc_u = [Eq(u.forward, 0, subdomain=u_left_wall)]
    bc_u += [Eq(u.forward, 0, subdomain=u_right_wall)]
    bc_u += [_neumann_bottom(update_u, u, subdomain=u_bottom_row)]
    bc_u += [_neumann_top(update_u, u, subdomain=u_top_row)]

    bc_v = [Eq(v.forward, 0, subdomain=v_top_wall)]
    bc_v += [Eq(v.forward, 0, subdomain=v_bottom_wall)]
    bc_v += [_neumann_left(update_v, v, subdomain=v_left_column)]
    bc_v += [_neumann_right(update_v, v, subdomain=v_right_column)]
    
    # from IPython import embed; embed()
    exprs = ([u_tent_solve] +
             [v_tent_solve] +
             [pressure_solve] +
             [update_u] + bc_u +
             [update_v] + bc_v)

    with switchconfig(language='petsc'):
        op = Operator(exprs)
        # print(op.ccode)

    # then gather to rank 0 here?

    plotfunc_u = Function(name='plotfunc_u', grid=grid, space_order=so,
                          staggered=NODE)
    plotfunc_v = Function(name='plotfunc_v', grid=grid, space_order=so,
                          staggered=NODE)
    
    vorticity = Function(name='vorticity', grid=grid, space_order=so,
                        staggered=NODE)
    
    stream = Function(name='psi', grid=grid, space_order=so,
                         staggered=NODE)
    

    border_u = Border(grid, 1, dims={x: x, y: 'left'}, name='border_u')
    eq_interp_u = Eq(plotfunc_u, u, subdomain=grid.interior)
    border_lid = Border(grid, 1, dims={y: 'right'}, name='border_lid')
    eq_interp_u = [eq_interp_u] + [Eq(plotfunc_u, 0, subdomain=border_u)] + [Eq(plotfunc_u, 1, subdomain=border_lid)]
    op_interp_u = Operator(eq_interp_u)

    border_v = Border(grid, 1, name='border_v')
    eq_interp_v = Eq(plotfunc_v, v, subdomain=grid.interior)
    eq_interp_v = [eq_interp_v] + [Eq(plotfunc_v, 0, subdomain=border_v)]
    op_interp_v = Operator(eq_interp_v)

    # Ghost values outside the cavity, 2 - u in the row of u above the lid and -v in the
    # column of v beyond the right wall. Only needed for vorticity post processing
    eq_ghost_u = Eq(u, 2 - u.subs({y + y.spacing/2: y - y.spacing/2}),
                 subdomain=u_ghost)
    eq_ghost_v = Eq(v, -v.subs({x + x.spacing/2: x - x.spacing/2}),
                 subdomain=v_ghost)

    # TODO: fix the wall vorticity on the left and bottom walls. Fix with the same
    # antisymmetric mirror on the left and bottom wall nodes.
    vorticity_eqn = Eq(vorticity, v.dx - u.dy)
    op_vorticity = Operator([eq_ghost_u, eq_ghost_v, vorticity_eqn])

    border = Border(grid, 1, name='border_psi')
    stream_bc = [EssentialBC(stream, 0., subdomain=border)]
    stream_eqn = Eq(stream.laplace, -(v.dx - u.dy), subdomain=grid.interior)
    stream_solver = petscsolve([stream_eqn]+stream_bc, stream, options_prefix='stream_solve',
                               solver_parameters={'ksp_type': 'cg', 'ksp_rtol': 1e-12})

    with switchconfig(language='petsc'):
        op_stream = Operator([stream_solver])


    x_coord = np.linspace(0, 1, nx)
    y_coord = np.linspace(0, 1, ny)

    def run_cavity_flow(re_val, t_end=400.0, tol=1e-3, check_every=200, fixed=False):

        # If fixed is True, run for exactly t_end time in a single op.apply(), otherwise chunk the run
        # and check convergence every check_every steps.
        u.data[:] = 0.
        v.data[:] = 0.
        phi.data[:] = 0.

        dx = 1./(nx - 1)
        dt_conv = 0.5 * dx

        if implicit_diffusion:
            dt_val = dt_conv
        else:
            dt_diff = (0.5 * re_val * dx**2) / 4.0
            dt_val = min(dt_conv, dt_diff)

        re.data = np.float64(re_val)
        dt_c.data = np.float64(dt_val)
        max_steps = int(t_end/dt_val)

        print(f'Re={re_val}: dt={dt_val}, max_steps={max_steps}')

        n_slots = time_order + 1

        if fixed:
            op.apply(time_m=0, time_M=max_steps - 1, dt=dt_val)

            # op.apply(time_m=max_steps, time_M=max_steps, dt=dt_val)
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
        op_vorticity.apply(time_M=tb)
        op_stream.apply(time_M=tb)
    
        u_g = u.data_gather(rank=0)
        v_g = v.data_gather(rank=0)

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

        # Steady state residual, the change in each velocity over the last time step
        # divided by the step

        res = None
        if _my_rank == 0:
            prev = (tb - 1) % n_slots
            res = max(np.abs(u_g[tb] - u_g[prev]).max(),
                      np.abs(v_g[tb] - v_g[prev]).max())/dt_val

        return (x_coord.copy(), y_coord.copy(),
                plotfunc_u.data_gather(rank=0),
                plotfunc_v.data_gather(rank=0),
                vorticity.data_gather(rank=0),
                stream.data_gather(rank=0),
                _my_rank,
                u_snap,
                v_snap,
                res)

    return run_cavity_flow
