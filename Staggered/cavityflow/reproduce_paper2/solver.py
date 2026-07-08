"""
Lid-driven cavity flow solver (Kim & Moin 1985, fractional step method).

Usage:
    from solver import make_solver
    run = make_solver(nx=65, ny=65)   # compiles operator once
    x, y, u, v, omega = run(re_val=400)
"""
import os
import numpy as np

from devito import (Grid, TimeFunction, Function, Eq, Operator, Border,
                    configuration, SubDomain, NODE, switchconfig, Constant)
from devito.symbolics import retrieve_functions
from devito.petsc import petscsolve, EssentialBC
from devito.petsc.initialize import PetscInitialize

configuration['compiler'] = 'custom'
os.environ['CC'] = 'mpicc'

PetscInitialize()


def _neumann_bottom(eq, t, subdomain):
    lhs, rhs = eq.evaluate.args
    funcs = retrieve_functions(lhs - rhs)
    yind_target = t.indices[-1]
    mapper = {}
    for f in funcs:
        yind = f.indices[-1]
        if (yind - yind_target).as_coeff_Mul()[0] < 0:
            if f.name == 'p':
                mapper[f] = f.subs({yind: yind_target})
            if f.name == 'u':
                mapper[f] = -f.subs({yind: yind_target})
    return Eq(lhs.subs(mapper), rhs.subs(mapper), subdomain=subdomain)


def _neumann_top(eq, t, subdomain):
    lhs, rhs = eq.evaluate.args
    funcs = retrieve_functions(lhs - rhs)
    yind_target = t.indices[-1]
    mapper = {}
    for f in funcs:
        yind = f.indices[-1]
        if (yind - yind_target).as_coeff_Mul()[0] > 0:
            if f.name == 'p':
                mapper[f] = f.subs({yind: yind_target})
            if f.name == 'u':
                mapper[f] = 2 - f.subs({yind: yind_target})
    return Eq(lhs.subs(mapper), rhs.subs(mapper), subdomain=subdomain)


def _neumann_left(eq, t, subdomain):
    lhs, rhs = eq.evaluate.args
    funcs = retrieve_functions(lhs - rhs)
    xind_target = t.indices[-2]
    mapper = {}
    for f in funcs:
        xind = f.indices[-2]
        if (xind - xind_target).as_coeff_Mul()[0] < 0:
            if f.name == 'p':
                mapper[f] = f.subs({xind: xind_target})
            if f.name == 'v':
                mapper[f] = -f.subs({xind: xind_target})
    return Eq(lhs.subs(mapper), rhs.subs(mapper), subdomain=subdomain)


def _neumann_right(eq, t, subdomain):
    lhs, rhs = eq.evaluate.args
    funcs = retrieve_functions(lhs - rhs)
    xind_target = t.indices[-2]
    mapper = {}
    for f in funcs:
        xind = f.indices[-2]
        if (xind - xind_target).as_coeff_Mul()[0] > 0:
            if f.name == 'p':
                mapper[f] = f.subs({xind: xind_target})
            if f.name == 'v':
                mapper[f] = -f.subs({xind: xind_target})
    return Eq(lhs.subs(mapper), rhs.subs(mapper), subdomain=subdomain)


def make_solver(nx, ny, ab2=False, implicit_diffusion=False, rtol=1e-7):
    """
    Parameters
    ----------
    ab2 : bool
        Use Adams-Bashforth 2 for convection (more stable at high Re).
        Default False uses fully explicit forward Euler.
    implicit_diffusion : bool
        Use Crank-Nicolson (semi-implicit) for diffusion.
        Default False keeps diffusion fully explicit.
    rtol : float
        Relative tolerance for the PETSc solver.
    """
    so = 2

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

    class Sub14(SubDomain):
        name = 'sub14'
        def define(self, dimensions):
            x, y = dimensions
            return {x: ('left', 1), y: ('left', ny-1)}

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

    subdomains = tuple(cls() for cls in [
        Sub1, Sub2, Sub3, Sub4, Sub5, Sub6, Sub7, Sub8, Sub9, Sub10,
        Sub11, Sub12, Sub13, Sub14, Sub15, Sub16, Sub17, Sub18, Sub19,
        Sub20, Sub21,
    ])

    grid = Grid(shape=(nx, ny), extent=(1., 1.), subdomains=subdomains,
                dtype=np.float64)

    x, y = grid.dimensions
    t = grid.stepping_dim

    re = Constant(name='re', dtype=np.float64)
    dt_c = Constant(name='dt_c', dtype=np.float64)

    time_order = 2 if ab2 else 1

    u = TimeFunction(name='u', grid=grid, space_order=so, time_order=time_order, staggered=y)
    v = TimeFunction(name='v', grid=grid, space_order=so, time_order=time_order, staggered=x)
    p = Function(name='p', grid=grid, space_order=so, staggered=(x, y))

    # Convection
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

    eq_u_tent = Eq(u.dt + conv_u, diff_u, subdomain=grid.subdomains['sub15'])
    eq_v_tent = Eq(v.dt + conv_v, diff_v, subdomain=grid.subdomains['sub17'])

    ux_cc = u.forward.dx(x0=x + x.spacing/2)
    vy_cc = v.forward.dy(x0=y + y.spacing/2)

    eq_p = Eq(p.laplace, (1./dt_c)*(ux_cc + vy_cc), subdomain=grid.subdomains['sub5'])

    # u BCs (tent)
    bc_tmp_u = Function(name='bc_tmp_u', grid=grid, space_order=so, staggered=y)
    bc_u_tent = [EssentialBC(u.forward, 0, subdomain=grid.subdomains['sub14'])]
    bc_u_tent += [EssentialBC(u.forward, 0, subdomain=grid.subdomains['sub11'])]
    bc_u_tent += [_neumann_bottom(eq_u_tent, u, subdomain=grid.subdomains['sub12'])]
    bc_u_tent += [_neumann_top(eq_u_tent, u, subdomain=grid.subdomains['sub13'])]
    bc_u_tent += [EssentialBC(u.forward, bc_tmp_u, subdomain=grid.subdomains['sub10'], constrain=True)]

    u_tent_solve = petscsolve([eq_u_tent] + bc_u_tent, u.forward,
                              options_prefix='utent_solve',
                              solver_parameters={'ksp_type': 'cg', 'ksp_rtol': rtol})

    bc_u_halo = [Eq(u[t+1, x, ny-1], 2 - u[t+1, x, ny-2])]
    bc_u_halo += [Eq(u[t+1, x, -1], -u[t+1, x, 0])]

    # v BCs (tent)
    bc_tmp_v = Function(name='bc_tmp_v', grid=grid, space_order=so, staggered=y)
    bc_v_tent = [EssentialBC(v.forward, 0, subdomain=grid.subdomains['sub18'])]
    bc_v_tent += [EssentialBC(v.forward, 0, subdomain=grid.subdomains['sub19'])]
    bc_v_tent += [_neumann_left(eq_v_tent, v, subdomain=grid.subdomains['sub20'])]
    bc_v_tent += [_neumann_right(eq_v_tent, v, subdomain=grid.subdomains['sub16'])]
    bc_v_tent += [EssentialBC(v.forward, bc_tmp_v,
                              subdomain=grid.subdomains['sub21'], constrain=True)]

    v_tent_solve = petscsolve([eq_v_tent] + bc_v_tent, v.forward,
                              options_prefix='vtent_solve',
                              solver_parameters={'ksp_type': 'cg', 'ksp_rtol': rtol})

    bc_v_halo = [Eq(v[t+1, nx-1, y], -v[t+1, nx-2, y])]
    bc_v_halo += [Eq(v[t+1, -1, y], -v[t+1, 0, y])]

    # p BCs
    bc_tmp_p = Function(name='bc_tmp_p', grid=grid, space_order=so, staggered=(x, y))
    bc_tmp_p.data[:] = 0.

    sub = subdomains
    # TODO: fix this for clarity just use STRINGS WITH the actual subdomain numbers
    bc_p = [_neumann_left(_neumann_top(eq_p, p, sub[0]), p, sub[0])]
    bc_p += [_neumann_top(eq_p, p, sub[1])]
    bc_p += [_neumann_right(_neumann_top(eq_p, p, sub[2]), p, sub[2])]
    bc_p += [_neumann_left(eq_p, p, sub[3])]
    bc_p += [_neumann_right(eq_p, p, sub[5])]
    bc_p += [_neumann_bottom(eq_p, p, sub[7])]
    bc_p += [_neumann_right(_neumann_bottom(eq_p, p, sub[8]), p, sub[8])]
    bc_p += [EssentialBC(p, bc_tmp_p, subdomain=grid.subdomains['sub7'])]
    bc_p += [EssentialBC(p, bc_tmp_p, subdomain=grid.subdomains['sub10'],
                         constrain=True)]
    bc_p += [EssentialBC(p, bc_tmp_p, subdomain=grid.subdomains['sub11'],
                         constrain=True)]

    pressure_solve = petscsolve([eq_p] + bc_p, p,
                                options_prefix='pressure_solve',
                                solver_parameters={'ksp_type': 'cg', 'ksp_rtol': rtol})

    # Velocity correction
    update_u = Eq(u.forward, u.forward - dt_c*p.dx, subdomain=grid.subdomains['sub15'])
    update_v = Eq(v.forward, v.forward - dt_c*p.dy, subdomain=grid.subdomains['sub17'])

    bc_u = [EssentialBC(u.forward, 0, subdomain=grid.subdomains['sub14'])]
    bc_u += [EssentialBC(u.forward, 0, subdomain=grid.subdomains['sub11'])]
    bc_u += [_neumann_bottom(update_u, u, subdomain=grid.subdomains['sub12'])]
    bc_u += [_neumann_top(update_u, u, subdomain=grid.subdomains['sub13'])]

    bc_v = [EssentialBC(v.forward, 0, subdomain=grid.subdomains['sub18'])]
    bc_v += [EssentialBC(v.forward, 0, subdomain=grid.subdomains['sub19'])]
    bc_v += [_neumann_left(update_v, v, subdomain=grid.subdomains['sub20'])]
    bc_v += [_neumann_right(update_v, v, subdomain=grid.subdomains['sub16'])]

    exprs = ([u_tent_solve] + bc_u_halo +
             [v_tent_solve] + bc_v_halo +
             [pressure_solve] +
             [update_u] + bc_u + bc_u_halo +
             [update_v] + bc_v + bc_v_halo)

    with switchconfig(language='petsc'):
        op = Operator(exprs)
        print(op.ccode)

    plotfunc_u = Function(name='plotfunc_u', grid=grid, space_order=so,
                          staggered=NODE)
    plotfunc_v = Function(name='plotfunc_v', grid=grid, space_order=so,
                          staggered=NODE)
    vorticity = Function(name='vorticity', grid=grid, space_order=so,
                         staggered=NODE)
    stream = Function(name='psi', grid=grid, space_order=so,
                         staggered=NODE)
    op_interp_u = Operator(Eq(plotfunc_u, u))
    op_interp_v = Operator(Eq(plotfunc_v, v))

    vorticity_eqn = Eq(vorticity, v.dx - u.dy)
    op_vorticity = Operator(vorticity_eqn)

    border = Border(grid, 1)
    stream_bc = [EssentialBC(stream, 0., subdomain=border)]
    stream_eqn = Eq(stream.laplace, -(v.dx - u.dy), subdomain=grid.interior)
    stream_solver = petscsolve([stream_eqn]+stream_bc, stream, options_prefix='stream_solve')

    with switchconfig(language='petsc'):
        op_stream = Operator([stream_solver])


    x_coord = np.linspace(0, 1, nx)
    y_coord = np.linspace(0, 1, ny)

    # run the solver at given reynolds number, for the given grid size
    # TODO: just run it to t_end=50 (100 is excessive?)
    def run_cavity_flow(re_val, t_end=400.0, tol=1e-3, check_every=200):

        u.data[:] = 0.
        v.data[:] = 0.
        p.data[:] = 0.

        dx = 1./(nx - 1)
        # NOTE: this is if both explicit.... but when i edit and move to semi-implicit this can be changed
        dt_conv = 0.5 * dx
        # dt_diff = (0.5 * re_val * dx**2) / 4.0
        # dt_val = min(dt_conv, dt_diff)

        # worded from paper:
        # Implicit treatment of the viscous terms
        # eliminates the numerical viscous stability restriction. This restriction is particularly
        # severe for low-Reynolds-number flows


        # from textbook: steady-state at low re should be assumed to be converged when the error norm
        # decreases by at least 3 orders of magnitude.

        if implicit_diffusion:
            # Diffusion is unconditionally stable — no constraint
            dt_val = dt_conv
        else:
            # Explicit diffusion adds its own stability limit
            dt_diff = (0.5 * re_val * dx**2) / 4.0
            dt_val = min(dt_conv, dt_diff)

        re.data = np.float64(re_val)
        dt_c.data = np.float64(dt_val)
        max_steps = int(t_end/dt_val)

        print(f'Re={re_val}: dt={dt_val}, max_steps={max_steps}')

        norm0 = None
        norm = float('inf')
        step = 0

        # Stops early when max-norm of velocity change drops to tol times
        # its initial value (3 orders of magnitude by default).
        while step < max_steps:
            chunk = min(check_every, max_steps - step)

            with switchconfig(language='petsc'):
                op.apply(time_m=step, time_M=step + chunk - 1, dt=dt_val)
            step += chunk

            du = u.data[0] - u.data[1]
            dv = v.data[0] - v.data[1]
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

        op_interp_u(time_M=0)
        op_interp_v(time_M=0)
        op_vorticity.apply(time_M=0)
        op_stream.apply(time_M=0)

        return (x_coord.copy(), y_coord.copy(),
                np.array(plotfunc_u.data), np.array(plotfunc_v.data),
                np.array(vorticity.data),
                np.array(stream.data))

    return run_cavity_flow
