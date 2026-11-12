import os
import numpy as np

from devito import (Grid, Function, Eq, Operator, switchconfig,
                    configuration, SubDomain, norm)

from devito.petsc import petscsolve, EssentialBC
from devito.petsc.initialize import PetscInitialize

import matplotlib.pyplot as plt

configuration['compiler'] = 'custom'
os.environ['CC'] = 'mpicc'



# python3 poisson_collatz.py -ksp_converged_reason -ksp_type cg -ksp_rtol 1e-13 -pc_type none
# ref - Gupta (1984), A fourth-order Poisson solver, J. Comput. Phys. 55(1), https://doi.org/10.1016/0021-9991(84)90022-6
# -(d2p/dx2 + d2p/dy2) = f on [0, 16]^2, p = 0 on the boundary, with
# f = -sin(pi x/8) sin(pi y/8) and p = -[2 (pi/8)^2]^(-1) sin(pi x/8) sin(pi y/8)


# The nine point stencil is written as a sum of shifted Functions (the "f() notation").
# Custom weights (p.dx2(weights=...)) act along one dimension, so cannot reach the
# diagonal points, and nested derivatives (p.dx2.dy2) are not yet supported inside
# petscsolve.

PetscInitialize()

# Subdomains to implement BCs
class SubTop(SubDomain):
    name = 'subtop'

    def define(self, dimensions):
        x, y = dimensions
        return {x: x, y: ('right', 1)}


class SubBottom(SubDomain):
    name = 'subbottom'

    def define(self, dimensions):
        x, y = dimensions
        return {x: x, y: ('left', 1)}


class SubLeft(SubDomain):
    name = 'subleft'

    def define(self, dimensions):
        x, y = dimensions
        return {x: ('left', 1), y: ('middle', 1, 1)}


class SubRight(SubDomain):
    name = 'subright'

    def define(self, dimensions):
        x, y = dimensions
        return {x: ('right', 1), y: ('middle', 1, 1)}


sub1 = SubTop()
sub2 = SubBottom()
sub3 = SubLeft()
sub4 = SubRight()

subdomains = (sub1, sub2, sub3, sub4)


def exact(x, y, k1=1, k2=1):
    tmp1 = np.float64(np.pi/8.0) * np.float64(np.pi/8.0)
    tmp2 = k1**2 + k2**2
    tmp3 = tmp1 * tmp2
    tmp4 = -1.0/tmp3
    tmp5 = np.float64(np.sin((np.pi*x*k1)/8.0))
    tmp6 = np.float64(np.sin((np.pi*y*k2)/8.0))
    return tmp4*tmp5*tmp6

Lx = np.float64(16.)
Ly = np.float64(16.)


# n = 9, 17, 33, 65, 129, 257, 513
n_values = [2**k + 1 for k in range(3, 10)]


h = np.array([Lx/(n-1) for n in n_values])
infinity_norms = []
discrete_l2_norms = []
ksp_iters = []

# The stencil is written by hand, so space_order only sets the halo
so = 1


for n in n_values:
    grid = Grid(
        shape=(n, n), extent=(Lx, Ly), subdomains=subdomains, dtype=np.float64
    )

    p = Function(name='p', grid=grid, space_order=so)
    f = Function(name='f', grid=grid, space_order=so)
    bc = Function(name='bc', grid=grid, space_order=so)

    ##### 9-point stencil #####
    x, y = grid.dimensions
    h_x = x.spacing
    h_y = y.spacing

    top_left = p.subs({x: x - h_x, y: y + h_y})
    top_middle = p.subs({x: x, y: y + h_y})
    top_right = p.subs({x: x + h_x, y: y + h_y})

    middle_left = p.subs({x: x - h_x, y: y})
    middle_middle = p.subs({x: x, y: y})
    middle_right = p.subs({x: x + h_x, y: y})

    bottom_left = p.subs({x: x - h_x, y: y - h_y})
    bottom_middle = p.subs({x: x, y: y - h_y})
    bottom_right = p.subs({x: x + h_x, y: y - h_y})
    points = [
        top_left, top_middle, top_right,
        middle_left, middle_middle, middle_right,
        bottom_left, bottom_middle, bottom_right
    ]
    weights = [1, 4, 1, 4, -20, 4, 1, 4, 1]
    # Could use EvalDerivative instead of sum
    nine_point_stencil_lhs = -sum(w*pt for w, pt in zip(weights, points))/(6*h_x**2)

    top_left = f.subs({x: x - h_x, y: y + h_y})
    top_middle = f.subs({x: x, y: y + h_y})
    top_right = f.subs({x: x + h_x, y: y + h_y})

    middle_left = f.subs({x: x - h_x, y: y})
    middle_middle = f.subs({x: x, y: y})
    middle_right = f.subs({x: x + h_x, y: y})

    bottom_left = f.subs({x: x - h_x, y: y - h_y})
    bottom_middle = f.subs({x: x, y: y - h_y})
    bottom_right = f.subs({x: x + h_x, y: y - h_y})
    points = [
        top_left, top_middle, top_right,
        middle_left, middle_middle, middle_right,
        bottom_left, bottom_middle, bottom_right
    ]
    weights = [0., 1./12., 0., 1./12., 8./12., 1./12., 0., 1./12., 0.]
    nine_point_stencil_rhs = sum(w*pt for w, pt in zip(weights, points))

    eqn = Eq(nine_point_stencil_lhs, nine_point_stencil_rhs, subdomain=grid.interior)

    tmpx = np.linspace(0, Lx, n).astype(np.float64)
    tmpy = np.linspace(0, Ly, n).astype(np.float64)

    Y, X = np.meshgrid(tmpx, tmpy)

    k1, k2 = 1., 1.

    f.data[:] = -np.float64(np.sin((np.pi*X*k1)/8.0)) * np.float64(np.sin((np.pi*Y*k2)/8.0))

    bc.data[:, 0] = 0.
    bc.data[:, -1] = 0.
    bc.data[0, :] = 0.
    bc.data[-1, :] = 0.

    # Create boundary condition expressions using subdomains
    bcs = [EssentialBC(p, bc, subdomain=sub1)]
    bcs += [EssentialBC(p, bc, subdomain=sub2)]
    bcs += [EssentialBC(p, bc, subdomain=sub3)]
    bcs += [EssentialBC(p, bc, subdomain=sub4)]

    exprs = [eqn] + bcs

    # A nonzero initial guess, so that CG iterates. From zero it converges in
    # one iteration, since f is an eigenvector of the discrete operator.
    p.data[:] = 0.001

    petsc = petscsolve(
        exprs, target=p,
        solver_parameters={'ksp_rtol': 1e-13, 'ksp_type': 'cg', 'pc_type': 'none'},
        options_prefix='poisson_collatz'
    )

    with switchconfig(log_level='DEBUG'):
        op = Operator(petsc, language='petsc')
        summary = op.apply()

    iters = summary.petsc[('section0', 'poisson_collatz')].KSPGetIterationNumber
    ksp_iters.append(iters)

    p_exact = Function(name='p_exact', grid=grid, space_order=so)
    p_exact.data[:] = exact(X, Y)

    diff = Function(name='diff', grid=grid, space_order=so)
    diff.data[:] = p_exact.data[:] - p.data[:]

    # Compute infinity norm using numpy
    infinity_norm = np.linalg.norm(diff.data[:].ravel(), ord=np.inf)
    infinity_norms.append(infinity_norm)
    print(infinity_norm)

    # Compute discrete L2 norm (RMS error)
    n_interior = np.prod([s - 1 for s in grid.shape])
    discrete_l2_norm = norm(diff) / np.sqrt(n_interior)
    discrete_l2_norms.append(discrete_l2_norm)
    print(discrete_l2_norm)


print(infinity_norms)
slope, intercept = np.polyfit(np.log(h), np.log(infinity_norms), 1)


assert slope > 3.9
assert slope < 4.1

# Plot
plt.figure(figsize=(6, 5))
plt.loglog(h, infinity_norms, 'o-', label=rf'Observed order $\hat{{q}} = {slope:.3f}$', color='orange')
plt.loglog(
    h, np.exp(intercept) * h**4,
    'k--',
    label='Reference line of slope 4'
)
plt.xlabel(r'Grid spacing $h$')
plt.ylabel(r'$\|e\|_\infty$')
plt.legend()
plt.tight_layout()
plt.savefig("compact_fourth_order.png", dpi=200)
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
c1 = ax0.contourf(X, Y, p.data[:], levels=100, cmap='viridis')
ax0.set_title('Devito Solution')
ax0.set_xlabel('x')
ax0.set_ylabel('y')

# Exact solution
c2 = ax1.contourf(X, Y, p_exact.data[:], levels=100, cmap='viridis')
ax1.set_title('Exact Solution')
ax1.set_xlabel('x')
ax1.set_ylabel('y')

# Sync color scales
vmin = min(p.data[:].min(), p_exact.data[:].min())
vmax = max(p.data[:].max(), p_exact.data[:].max())
c1.set_clim(vmin, vmax)
c2.set_clim(vmin, vmax)

# Colorbar
cb = fig.colorbar(c2, cax=cax)
cb.set_label('Field p')


for ax in [ax0, ax1]:
    ax.tick_params(axis='x', pad=20)
    ax.tick_params(axis='y', pad=20)

# Layout adjustment
plt.subplots_adjust(left=0.02, right=0.95, top=0.92, bottom=0.12, wspace=0.25)

# Save output
plt.savefig("poisson_collatz_compare.png", dpi=200, bbox_inches='tight', pad_inches=0.2)
plt.show()
