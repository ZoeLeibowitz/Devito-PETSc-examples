# 2D heat equation solved with petscsolve for three time stepping schemes (FTCS, BTCS, CN)

# Grid spacing and time step refined together:
# space: dt = 16 h^2 (FTCS, BTCS, CN), so the time error is O(h^2) for every scheme
# time: h = dt/4 (BTCS, CN)
import os
import numpy as np
import matplotlib.pyplot as plt

from devito import Grid, TimeFunction, Eq, Operator, Border, configuration
from devito.petsc import petscsolve, EssentialBC
from devito.petsc.initialize import PetscInitialize

configuration['compiler'] = 'custom'
os.environ['CC'] = 'mpicc'

PetscInitialize()

alpha = 0.1
T = 2.0
solver_parameters = {'ksp_type': 'cg', 'pc_type': 'none', 'ksp_rtol': 1e-12}


def heat_operator(n, scheme):
    grid = Grid(shape=(n, n), extent=(1., 1.), dtype=np.float64)
    u = TimeFunction(name='u', grid=grid, space_order=2)

    # set time stepping scheme
    if scheme == 'FTCS':
        eqn = Eq(u.dt, alpha**2 * (u.dx2 + u.dy2), subdomain=grid.interior)
    elif scheme == 'BTCS':
        eqn = Eq(u.dt, alpha**2 * (u.forward.dx2 + u.forward.dy2),
                 subdomain=grid.interior)
    elif scheme == 'CN':
        eqn = Eq(u.dt, alpha**2 * (u.dx2 + u.dy2 + u.forward.dx2 + u.forward.dy2)/2,
                 subdomain=grid.interior)

    bc = EssentialBC(u.forward, 0., subdomain=Border(grid, 1))
    solve_u = petscsolve([eqn, bc], target=u.forward,
                         solver_parameters=solver_parameters)
    op = Operator(solve_u, language='petsc')

    x = np.linspace(0, 1., n)
    X, Y = np.meshgrid(x, x, indexing='ij')
    u.data[0] = np.sin(np.pi*X) * np.sin(np.pi*Y)
    return u, op, X, Y


def exact(X, Y, t):
    return np.exp(-2*np.pi**2*alpha**2*t) * np.sin(np.pi*X) * np.sin(np.pi*Y)


def max_error(n, scheme, nt):
    u, op, X, Y = heat_operator(n, scheme)
    op.apply(dt=T/nt, time_M=nt-1)
    return np.abs(u.data[nt % 2] - exact(X, Y, T)).max()


def observed_order(steps, errors):
    return np.polyfit(np.log(steps), np.log(errors), 1)[0]


theoretical_order = {'space': {'FTCS': 2, 'BTCS': 2, 'CN': 2},
                     'time': {'BTCS': 1, 'CN': 2}}

# Convergence in space, dt = T/nt = 16 h^2
space = {}
for scheme in ('FTCS', 'BTCS', 'CN'):
    ns = np.array([9, 17, 33, 65, 129])
    hs = 1./(ns - 1)
    errors = [max_error(n, scheme, (n-1)**2 // 8) for n in ns]
    space[scheme] = (hs, errors)
    slope = observed_order(hs, errors)
    print('space %-4s ||e||_inf %s  order %.3f'
          % (scheme, ' '.join('%.3e' % e for e in errors), slope))
    assert abs(slope - theoretical_order['space'][scheme]) < 0.1

# Convergence in time, h = dt/4, so dt = T/nt with nt = (n - 1)/2
time = {}
for scheme in ('BTCS', 'CN'):
    ns = np.array([17, 33, 65, 129, 257])
    dts = T/((ns - 1) // 2)
    errors = [max_error(n, scheme, (n-1) // 2) for n in ns]
    time[scheme] = (dts, errors)
    slope = observed_order(dts, errors)
    print('time  %-4s ||e||_inf %s  order %.3f'
          % (scheme, ' '.join('%.3e' % e for e in errors), slope))
    assert abs(slope - theoretical_order['time'][scheme]) < 0.1


# Stability
n, dt, nt = 41, 0.05, 40
for scheme in ('FTCS', 'BTCS', 'CN'):
    u, op, X, Y = heat_operator(n, scheme)
    op.apply(dt=dt, time_M=nt-1)
    print('dt = %g, %-4s max|u(T)| = %.3e, exact %.3e'
          % (dt, scheme, np.abs(u.data[nt % 2]).max(), exact(X, Y, nt*dt).max()))



#### plotting
# One figure for the orders in space and one for the orders in time. Each scheme
# keeps the same default colour in both
colour = {'FTCS': 'C0', 'BTCS': 'C1', 'CN': 'C2'}

for name, results, xlabel in (('space', space, r'Grid spacing $h$'),
                              ('time', time, r'Time step $\Delta t$')):
    plt.figure(figsize=(6, 5))
    for k, (scheme, (steps, errors)) in enumerate(results.items()):
        slope = observed_order(steps, errors)
        plt.loglog(steps, errors, 'o-', color=colour[scheme],
                   label=rf'{scheme}, $\hat{{q}} = {slope:.3f}$')
        q = theoretical_order[name][scheme]
        plt.loglog(steps, errors[0]*(steps/steps[0])**q, 'k--', lw=0.8,
                   label=r'Reference lines of slope $q$' if k == len(results) - 1 else None)
    plt.xlabel(xlabel)
    plt.ylabel(r'$\|e\|_\infty$')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'heat_2d_{name}.png', dpi=200)
