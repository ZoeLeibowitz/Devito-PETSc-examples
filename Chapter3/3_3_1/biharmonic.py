# ref - https://github.com/bueler/p4pdes/blob/master/c/ch7/biharm.c


import os
import numpy as np

from devito import (Grid, Function, Eq, Operator, switchconfig,
                    configuration, SubDomain)

from devito.petsc import petscsolve, EssentialBC
from devito.petsc.initialize import PetscInitialize

import matplotlib.pyplot as plt

configuration['compiler'] = 'custom'
os.environ['CC'] = 'mpicc'


PetscInitialize()


# Subdomains to implement BCs

class SubTop(SubDomain):
    name = 'subtop'

    def define(self, dimensions):
        x, y = dimensions
        return {x: ('middle', 1, 1), y: ('right', 1)}


class SubBottom(SubDomain):
    name = 'subbottom'

    def define(self, dimensions):
        x, y = dimensions
        return {x: ('middle', 1, 1), y: ('left', 1)}


class SubLeft(SubDomain):
    name = 'subleft'

    def define(self, dimensions):
        x, y = dimensions
        return {x: ('left', 1), y: y}


class SubRight(SubDomain):
    name = 'subright'

    def define(self, dimensions):
        x, y = dimensions
        return {x: ('right', 1), y: y}


def c(x):
    return x**3 * (1 - x)**3


def ddc(x):
    return 6.0 * x * (1 - x) * (1 - 5.0 * x + 5.0 * x**2)


def d4c(x):
    return -72.0 * (1 - 5.0 * x + 5.0 * x**2)


def u_exact_fcn(x, y):
    return c(x) * c(y)


def lap_u_exact_fcn(x, y):
    return -ddc(x) * c(y) - c(x) * ddc(y)


def f_fcn(x, y):
    return d4c(x) * c(y) + 2.0 * ddc(x) * ddc(y) + c(x) * d4c(y)


sub1 = SubTop()
sub2 = SubBottom()
sub3 = SubLeft()
sub4 = SubRight()

subdomains = (sub1, sub2, sub3, sub4)

Lx = np.float64(1.)
Ly = np.float64(1.)

# n = 33, 65, 129, 257
n_values = [2**k + 1 for k in range(5, 9)]
dx = np.array([Lx/(n-1) for n in n_values])

u_errors = []
v_errors = []

for n in n_values:
    grid = Grid(
        shape=(n, n), extent=(Lx, Ly), subdomains=subdomains, dtype=np.float64
    )

    u = Function(name='u', grid=grid, space_order=2)
    v = Function(name='v', grid=grid, space_order=2)
    f = Function(name='f', grid=grid, space_order=2)

    u_exact = Function(name='u_exact', grid=grid, space_order=2)
    lap_u = Function(name='lap_u', grid=grid, space_order=2)

    eqn1 = Eq(-v.laplace, f, subdomain=grid.interior)
    eqn2 = Eq(-u.laplace, v, subdomain=grid.interior)

    tmpx = np.linspace(0, Lx, n).astype(np.float64)
    tmpy = np.linspace(0, Ly, n).astype(np.float64)
    X, Y = np.meshgrid(tmpx, tmpy)

    f.data[:] = f_fcn(X, Y)

    # # Create boundary condition expressions using subdomains
    # TODO: add initial guess callback for mixed systems
    bc_u = [EssentialBC(u, 0., subdomain=sub1)]
    bc_u += [EssentialBC(u, 0., subdomain=sub2)]
    bc_u += [EssentialBC(u, 0., subdomain=sub3)]
    bc_u += [EssentialBC(u, 0., subdomain=sub4)]
    bc_v = [EssentialBC(v, 0., subdomain=sub1)]
    bc_v += [EssentialBC(v, 0., subdomain=sub2)]
    bc_v += [EssentialBC(v, 0., subdomain=sub3)]
    bc_v += [EssentialBC(v, 0., subdomain=sub4)]

    # T (see ref) is nonsymmetric so set default KSP type to GMRES
    params = {'ksp_rtol': 1e-10, 'ksp_type': 'gmres', 'pc_type': 'none'}
    petsc = petscsolve({v: [eqn1]+bc_v, u: [eqn2]+bc_u}, solver_parameters=params)

    with switchconfig(language='petsc'):
        op = Operator(petsc)
        op.apply()

    u_exact.data[:] = u_exact_fcn(X, Y)
    lap_u.data[:] = lap_u_exact_fcn(X, Y)

    # Compute infinity norm for u
    u_diff = u_exact.data[:] - u.data[:]
    u_diff_norm = np.linalg.norm(u_diff.ravel(), ord=np.inf)
    u_error = u_diff_norm / np.linalg.norm(u_exact.data[:].ravel(), ord=np.inf)
    u_errors.append(u_error)

    # Compute infinity norm for lap_u
    v_diff = lap_u.data[:] - v.data[:]
    v_diff_norm = np.linalg.norm(v_diff.ravel(), ord=np.inf)
    v_error = v_diff_norm / np.linalg.norm(lap_u.data[:].ravel(), ord=np.inf)
    v_errors.append(v_error)

u_slope, _ = np.polyfit(np.log(dx), np.log(u_errors), 1)
v_slope, _ = np.polyfit(np.log(dx), np.log(v_errors), 1)

print(u_errors)
print(v_errors)
print(f"u slope: {u_slope}")
print(f"v slope: {v_slope}")

assert u_slope > 1.9
assert u_slope < 2.1

assert v_slope > 1.9
assert v_slope < 2.1


# Assuming dx, u_errors, v_errors are already defined
dx = np.array(dx)
u_errors = np.array(u_errors)
v_errors = np.array(v_errors)

# Plot
plt.figure(figsize=(8, 6))
plt.loglog(dx, u_errors, 'o-', label=r"$u$ error", linewidth=2, markersize=6)
plt.loglog(dx, v_errors, 's--', label=r"$v = -\nabla^{2} u$ error", linewidth=2, markersize=6)

# Reference line for 2nd order
ref_line = 3.2 * dx**2  # adjust scaling factor as needed
plt.loglog(dx, ref_line, 'k:', label=r"$\mathcal{O}(h^2)$", linewidth=1.5)

# Labels and legend
plt.xlabel("Grid spacing, h", fontsize=16)
plt.ylabel("$\|\cdot\|_\infty$", fontsize=16)
plt.title("Convergence of the infinity norm for $u$ and $v$", fontsize=16)
plt.legend(fontsize=12)
plt.tight_layout()

# Save plot
plt.savefig("3_3_1_convergence.png", dpi=300, transparent=True)
print(u_errors)
print(v_errors)


# Use the highest resolution for plotting
n_plot = n_values[-1]
tmpx = np.linspace(0, Lx, n_plot)
tmpy = np.linspace(0, Ly, n_plot)
X, Y = np.meshgrid(tmpx, tmpy)


plt.figure(figsize=(12, 10))

plt.subplot(2, 2, 1)
cp1 = plt.contourf(X, Y, u.data[:], levels=50, cmap='plasma')
plt.colorbar(cp1)
plt.title(f'$u$ (numerical), n={n_plot}')
plt.xlabel('$x$')
plt.ylabel('$y$')

plt.subplot(2, 2, 3)
cp2 = plt.contourf(X, Y, u_exact.data[:], levels=50, cmap='plasma')
plt.colorbar(cp2)
plt.title(f'$u$ (exact), n={n_plot}')
plt.xlabel('$x$')
plt.ylabel('$y$')

plt.subplot(2, 2, 2)
cp3 = plt.contourf(X, Y, v.data[:], levels=50, cmap='plasma')
plt.colorbar(cp3)
plt.title(f'$v$ (numerical), n={n_plot}')
plt.xlabel('$x$')
plt.ylabel('$y$')

plt.subplot(2, 2, 4)
cp4 = plt.contourf(X, Y, lap_u.data[:], levels=50, cmap='plasma')
plt.colorbar(cp4)
plt.title(r'$v = -\nabla^{2} u$ (exact)' + f'\n$n = {n_plot}$', fontsize=12)
plt.xlabel('$x$')
plt.ylabel('$y$')

plt.tight_layout()
plt.savefig("3_3_1_solution.png", dpi=300, transparent=True)
plt.show()







