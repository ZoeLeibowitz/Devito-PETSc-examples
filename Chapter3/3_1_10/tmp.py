import os
import numpy as np

from devito import (Grid, Function, Eq, Operator, switchconfig,
                    configuration, SubDomain, norm, mmax)
from devito.symbolics import retrieve_functions, INT

from devito.petsc import PETScSolve, EssentialBC
from devito.petsc.initialize import PetscInitialize

import matplotlib.pyplot as plt

configuration['compiler'] = 'custom'
os.environ['CC'] = 'mpicc'


# python3 tmp.py -ksp_converged_reason -ksp_type gmres -pc_type none -ksp_max_it 50000 -ksp_rtol 1e-10


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


class SubMain(SubDomain):
    name = 'submain'

    def define(self, dimensions):
        x, y = dimensions
        return {x: ('middle', 1, 1), y: y}
    

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


sub1 = SubTop()
sub2 = SubBottom()
sub3 = SubLeft()
sub4 = SubRight()
# sub5 = SubMain()

subdomains = (sub1, sub2, sub3, sub4)

def exact(x, y):
    tmp1 = np.float64(np.sinh(2.0*np.float64(np.pi)))
    tmp2 = np.float64(np.cos(np.float64(np.pi)*y))
    tmp3 = np.float64(np.sinh(np.float64(np.pi)*x))
    tmp4 = np.float64(1.)/tmp1

    return np.float64(tmp4*tmp2*tmp3)

Lx = np.float64(2.)
Ly = np.float64(1.)


k_vals = range(5, 11)
nx_values = [2**k + 1 for k in k_vals]
ny_values = [int(0.5 * (nx - 1) + 1) for nx in nx_values]

print("nx_values =", nx_values)
print("ny_values =", ny_values)

hx = np.array([Lx/(n-1) for n in nx_values])
hy = np.array([Ly/(n-1) for n in ny_values])
print("hx =", hx)
print("hy =", hy)


h = np.array([Lx/(n-1) for n in nx_values])
infinity_norms = []
discrete_l2_norms = []
ksp_iters = []

for nx, ny in zip(nx_values, ny_values):
    grid = Grid(
        shape=(nx, ny), extent=(Lx, Ly), subdomains=subdomains, dtype=np.float64
    )

    u = Function(name='u', grid=grid, space_order=2)
    f = Function(name='f', grid=grid, space_order=2)
    bc = Function(name='bc', grid=grid, space_order=2)

    eqn = Eq(u.laplace, f, subdomain=grid.interior)

    tmpx = np.linspace(0, Lx, nx).astype(np.float64)
    tmpy = np.linspace(0, Ly, ny).astype(np.float64)

    X, Y = np.meshgrid(tmpx, tmpy, indexing='ij')

    f.data[:] = 0.0

    bc.data[0, :] = np.float64(0.) # u(0,y) = 0
    bc.data[nx-1, :] = np.float64(np.cos(np.float64(np.pi * tmpy))) # u(2,y) = cos(pi*y)

    # Create boundary condition expressions using subdomains
    bcs = [EssentialBC(u, bc, subdomain=sub3)]
    bcs += [EssentialBC(u, bc, subdomain=sub4)]
    bcs += [neumann_bottom(eqn, sub2)]
    bcs += [neumann_top(eqn, sub1)]

    exprs = [eqn] + bcs
    petsc = PETScSolve(exprs, target=u, solver_parameters={'ksp_rtol': 1e-10})

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

    print(infinity_norm)
    # from IPython import embed; embed()
    # print(op.ccode)
    
print(infinity_norms)
# print(ksp_iters)
slope, intercept = np.polyfit(np.log(h), np.log(infinity_norms), 1)
print(slope)
# assert slope > 1.9
# assert slope < 2.1

# Plot
plt.figure(figsize=(6, 5))
plt.loglog(h, infinity_norms, 'o-', label=f'Observed rate ≈ {slope:.3f}', color='orange')
plt.loglog(
    h, np.exp(intercept) * h**2,
    'k--',
    label=r'Reference slope $O(h^2)$'
)
plt.xlabel(r'Grid spacing h')
plt.ylabel(r'$\infty$-norm error')
plt.title('Convergence Plot')
plt.legend()
plt.tight_layout()
plt.savefig("3_1_10.png", dpi=200)
plt.show()


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



import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec



plt.rcParams.update({
    'font.size': 75, 
    'axes.titlesize': 90,
    'axes.labelsize': 85,
    'xtick.labelsize': 60,
    'ytick.labelsize': 60,   
    'legend.fontsize': 70  
})


# Create large figure
fig = plt.figure(figsize=(75, 35))  # Massive size

# Use GridSpec for layout
gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 0.05], wspace=0.25)

# Subplots
ax0 = plt.subplot(gs[0])
ax1 = plt.subplot(gs[1])
cax = plt.subplot(gs[2])  # For colorbar

# Devito solution
c1 = ax0.contourf(X, Y, u.data[:], levels=100, cmap='viridis')
ax0.set_title('Devito Solution')
ax0.set_xlabel('x')
ax0.set_ylabel('y')

# Analytical solution
c2 = ax1.contourf(X, Y, u_exact.data[:], levels=100, cmap='viridis')
ax1.set_title('Analytical Solution')
ax1.set_xlabel('x')
ax1.set_ylabel('y')

# Sync color scales
# vmin = min(u.data[:].min(), u_exact.data[:].min())
# vmax = max(u.data[:].max(), u_exact.data[:].max())
# c1.set_clim(vmin, vmax)
# c2.set_clim(vmin, vmax)

# Colorbar
cb = fig.colorbar(c2, cax=cax)
cb.set_label('Field u')


for ax in [ax0, ax1]:
    ax.tick_params(axis='x', pad=20)
    ax.tick_params(axis='y', pad=20)

# Layout adjustment
plt.subplots_adjust(left=0.02, right=0.95, top=0.92, bottom=0.12, wspace=0.25)

# Save output
plt.savefig("compare.png", dpi=200, bbox_inches='tight', pad_inches=0.2)
plt.show()




# plot the difference on a single plot

# Create large figure
fig = plt.figure(figsize=(75, 35))  # Massive size

# Use GridSpec for layout
gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 0.05], wspace=0.25)

# Subplots
ax0 = plt.subplot(gs[0])
ax1 = plt.subplot(gs[1])
cax = plt.subplot(gs[2])  # For colorbar

# Devito solution
c1 = ax0.contourf(X, Y, diff.data[:], levels=100, cmap='viridis')
ax0.set_title('diff')
ax0.set_xlabel('x')
ax0.set_ylabel('y')

# Analytical solution
c2 = ax1.contourf(X, Y, diff.data[:], levels=100, cmap='viridis')
ax1.set_title('diff')
ax1.set_xlabel('x')
ax1.set_ylabel('y')

# Sync color scales
# vmin = min(u.data[:].min(), u_exact.data[:].min())
# vmax = max(u.data[:].max(), u_exact.data[:].max())
# c1.set_clim(vmin, vmax)
# c2.set_clim(vmin, vmax)

# Colorbar
cb = fig.colorbar(c2, cax=cax)
cb.set_label('Field u')


for ax in [ax0, ax1]:
    ax.tick_params(axis='x', pad=20)
    ax.tick_params(axis='y', pad=20)

# Layout adjustment
plt.subplots_adjust(left=0.02, right=0.95, top=0.92, bottom=0.12, wspace=0.25)

# Save output
plt.savefig("diff.png", dpi=200, bbox_inches='tight', pad_inches=0.2)
plt.show()


import matplotlib.pyplot as plt
import numpy as np

from matplotlib import cm

from mpl_toolkits.mplot3d import Axes3D

plt.style.use('_mpl-gallery')

Z = u_exact.data[:]

fig = plt.figure(figsize=(20, 15))  # Adjust as needed
ax = fig.add_subplot(111, projection='3d')

surf = ax.plot_surface(X, Y, Z, cmap=cm.Blues, edgecolor='k', linewidth=0.5)

# Axis labels
ax.set_xlabel('x', labelpad=20, fontsize=30)
ax.set_ylabel('y', labelpad=20, fontsize=30)
ax.set_zlabel('u(x, y)', labelpad=20, fontsize=30)

# Tick font size
ax.tick_params(axis='both', labelsize=20)

# Optional: add a color bar
fig.colorbar(surf, shrink=0.5, aspect=10, pad=0.1)

# Save
plt.savefig("exact_solution.png", dpi=200, bbox_inches='tight', pad_inches=0.2)
plt.show()



# plot devito solution

Z = u.data[:]

fig = plt.figure(figsize=(20, 15))  # Adjust as needed
ax = fig.add_subplot(111, projection='3d')

surf = ax.plot_surface(X, Y, Z, cmap=cm.Blues, edgecolor='k', linewidth=0.5)

# Axis labels
ax.set_xlabel('x', labelpad=20, fontsize=30)
ax.set_ylabel('y', labelpad=20, fontsize=30)
ax.set_zlabel('u(x, y)', labelpad=20, fontsize=30)

# Tick font size
ax.tick_params(axis='both', labelsize=20)

# Optional: add a color bar
fig.colorbar(surf, shrink=0.5, aspect=10, pad=0.1)

# Save
plt.savefig("devito_solution.png", dpi=200, bbox_inches='tight', pad_inches=0.2)
plt.show()


Z = diff.data[:]

fig = plt.figure(figsize=(20, 15))  # Adjust as needed
ax = fig.add_subplot(111, projection='3d')

surf = ax.plot_surface(X, Y, Z, cmap=cm.Blues, edgecolor='k', linewidth=0.5)

# Axis labels
ax.set_xlabel('x', labelpad=20, fontsize=30)
ax.set_ylabel('y', labelpad=20, fontsize=30)
ax.set_zlabel('u(x, y)', labelpad=20, fontsize=30)

# Tick font size
ax.tick_params(axis='both', labelsize=20)

# Optional: add a color bar
fig.colorbar(surf, shrink=0.5, aspect=10, pad=0.1)

# Save
plt.savefig("diff_contour.png", dpi=200, bbox_inches='tight', pad_inches=0.2)
plt.show()

