import os
import numpy as np

from devito import (Grid, Function, Eq, Operator, switchconfig,
                    configuration, SubDomain, norm, mmax)

from devito.petsc import PETScSolve, EssentialBC
from devito.petsc.initialize import PetscInitialize

import matplotlib.pyplot as plt

configuration['compiler'] = 'custom'
os.environ['CC'] = 'mpicc'


# python3 laplace_4th_order.py -ksp_converged_reason -ksp_type cg -ksp_rtol 1e-12 -pc_type none

# 2D test
# Solving u_xx + u_yy = 0
# Dirichlet BCs: u(0,y) = 0, u(1,y)=0, u(x,0) = sin(pix), u(x,1)=e^(-pi)*sin(pix)
# ref - https://www.scirp.org/journal/paperinformation?paperid=113731#f2
# example 2 -> note they wrote u(x,1) bc wrong, it should be u(x,1) = e^-pi*sin(pix)
# Analytical solution: u(x,y) = e^(-pi*y)*sin(pi*x)

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

# def exact(x, y):
#     return np.float64(np.exp(-y*np.pi)) * np.float64(np.sin(np.pi*x))

def exact(x, y):
    return (np.float64(np.sinh(y*np.pi)) * np.float64(np.sin(np.pi*x))) / np.float64(np.sinh(np.pi))

Lx = np.float64(1.)
Ly = np.float64(1.)

# n = 9, 17, 33, 65, 129, 257, 513, 1025, 2049, 4097
n_values = [2**k + 1 for k in range(3, 13)]

# n_values = [2**k + 1 for k in range(3, 10)]
n_values = [13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35, 37, 39, 41, 43, 45, 47, 49, 51, 53, 55, 57, 59, 61, 63]
# n_values = [123]

# n = []
h = np.array([Lx/(n-1) for n in n_values])
infinity_norms = []
discrete_l2_norms = []
ksp_iters = []

# Not acc really used
so = 2


import numpy as np
from devito import Grid, TimeFunction, Eq, Operator, solve
from devito.finite_differences.differentiable import EvalDerivative


for n in n_values:
    grid = Grid(
        shape=(n, n), extent=(Lx, Ly), subdomains=subdomains, dtype=np.float64
    )

    u = Function(name='u', grid=grid, space_order=so)
    f = Function(name='f', grid=grid, space_order=so)
    bc = Function(name='bc', grid=grid, space_order=so)

    ##### 9-point stencil #####

    x, y = grid.dimensions
    h_x = x.spacing
    h_x = x.spacing
    h_y = y.spacing

    top_left = u.subs({x: x - h_x, y: y + h_y})
    top_middle = u.subs({x: x, y: y + h_y})
    top_right = u.subs({x: x + h_x, y: y + h_y})

    middle_left = u.subs({x: x - h_x, y: y})
    middle_middle = u.subs({x: x, y: y})
    middle_right = u.subs({x: x + h_x, y: y})

    bottom_left = u.subs({x: x - h_x, y: y - h_y})
    bottom_middle = u.subs({x: x, y: y - h_y})
    bottom_right = u.subs({x: x + h_x, y: y - h_y})
    points = [
        top_left, top_middle, top_right,
        middle_left, middle_middle, middle_right,
        bottom_left, bottom_middle, bottom_right
    ]
    weights = [1./6., 4./6., 1./6, 4./6., -20./6., 4./6., 1./6., 4./6., 1./6.]
    nine_point_stencil = EvalDerivative(*[w*p/h_x**2 for w, p in zip(weights, points)], base=u)
    # from IPython import embed; embed()
    eqn = Eq(nine_point_stencil, f, subdomain=grid.interior)

    tmpx = np.linspace(0, Lx, n).astype(np.float64)
    tmpy = np.linspace(0, Ly, n).astype(np.float64)

    Y, X = np.meshgrid(tmpx, tmpy)

    f.data[:] = 0.0


    bc.data[:, 0] = 0.
    bc.data[:, -1] = np.float64(np.sin(np.pi*tmpx))
    bc.data[0, :] = 0.
    bc.data[-1, :] = 0.

    # Create boundary condition expressions using subdomains
    bcs = [EssentialBC(u, bc, subdomain=sub1)]
    bcs += [EssentialBC(u, bc, subdomain=sub2)]
    bcs += [EssentialBC(u, bc, subdomain=sub3)]
    bcs += [EssentialBC(u, bc, subdomain=sub4)]

    exprs = [eqn] + bcs

    petsc = PETScSolve(exprs, target=u, solver_parameters={'ksp_rtol': 1e-8})

    with switchconfig(log_level='DEBUG'):
        op = Operator(petsc, language='petsc')
        summary = op.apply()

    iters = summary.petsc[('section0', None)].KSPGetIterationNumber
    ksp_iters.append(iters)

    u_exact = Function(name='u_exact', grid=grid, space_order=so)
    u_exact.data[:] = exact(X, Y)

    diff = Function(name='diff', grid=grid, space_order=so)
    diff.data[:] = u_exact.data[:] - u.data[:]

    # Compute infinity norm using numpy
    # TODO: Figure out how to compute the infinity norm using Devito
    infinity_norm = np.linalg.norm(diff.data[:].ravel(), ord=np.inf)
    infinity_norms.append(infinity_norm)
    print(infinity_norm)

    # Compute discrete L2 norm (RMS error)
    n_interior = np.prod([s - 1 for s in grid.shape])
    discrete_l2_norm = norm(diff) / np.sqrt(n_interior)
    discrete_l2_norms.append(discrete_l2_norm)
    print(discrete_l2_norm)

# print(op.ccode)
print(infinity_norms)
# print(ksp_iters)
slope, intercept = np.polyfit(np.log(h), np.log(infinity_norms), 1)
# from IPython import embed; embed()

print(slope)

# print(slope)
# assert slope > 3.9
# assert slope < 4.1

# # Plot
# plt.figure(figsize=(6, 5))
# plt.loglog(h, infinity_norms, 'o-', label=f'Observed rate ≈ {slope:.3f}', color='orange')
# plt.loglog(
#     h, np.exp(intercept) * h**4,
#     'k--',
#     label=r'Reference slope $O(h^4)$'
# )
# plt.xlabel(r'Grid spacing h')
# plt.ylabel(r'$\infty$-norm error')
# plt.title('Convergence Plot')
# plt.legend()
# plt.tight_layout()
# plt.savefig("3_1_7.png", dpi=200)
# plt.show()




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
vmin = min(u.data[:].min(), u_exact.data[:].min())
vmax = max(u.data[:].max(), u_exact.data[:].max())
c1.set_clim(vmin, vmax)
c2.set_clim(vmin, vmax)

# Colorbar
cb = fig.colorbar(c2, cax=cax)
cb.set_label('Field u')


for ax in [ax0, ax1]:
    ax.tick_params(axis='x', pad=20)
    ax.tick_params(axis='y', pad=20)

# Layout adjustment
plt.subplots_adjust(left=0.02, right=0.95, top=0.92, bottom=0.12, wspace=0.25)

# Save output
plt.savefig("4thorder_compare.png", dpi=200, bbox_inches='tight', pad_inches=0.2)
plt.show()
