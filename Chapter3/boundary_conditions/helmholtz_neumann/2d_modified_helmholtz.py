import os
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from devito.symbolics import retrieve_functions, INT
from devito import (configuration, Operator, Eq, Grid, Function,
                    SubDomain, switchconfig, norm)
from devito.petsc import petscsolve
from devito.petsc.initialize import PetscInitialize
configuration['compiler'] = 'custom'
os.environ['CC'] = 'mpicc'

# Modified Helmholtz equation
# Ref - https://www.firedrakeproject.org/demos/helmholtz.py.html


PetscInitialize()


Lx = 1.
Ly = Lx

so = 2


class SubTop(SubDomain):
    name = 'subtop'

    def __init__(self, S_O, grid=None):
        self.S_O = S_O
        super().__init__(grid=grid)

    def define(self, dimensions):
        x, y = dimensions
        return {x: ('middle', 1, 1), y: ('right', self.S_O//2)}


class SubBottom(SubDomain):
    name = 'subbottom'

    def __init__(self, S_O, grid=None):
        self.S_O = S_O
        super().__init__(grid=grid)

    def define(self, dimensions):
        x, y = dimensions
        return {x: ('middle', 1, 1), y: ('left', self.S_O//2)}


class SubLeft(SubDomain):
    name = 'subleft'

    def __init__(self, S_O, grid=None):
        self.S_O = S_O
        super().__init__(grid=grid)

    def define(self, dimensions):
        x, y = dimensions
        return {x: ('left', self.S_O//2), y: ('middle', 1, 1)}


class SubRight(SubDomain):
    name = 'subright'

    def __init__(self, S_O, grid=None):
        self.S_O = S_O
        super().__init__(grid=grid)

    def define(self, dimensions):
        x, y = dimensions
        return {x: ('right', self.S_O//2), y: ('middle', 1, 1)}


class SubPointBottomLeft(SubDomain):
    name = 'subpointbottomleft'

    def define(self, dimensions):
        x, y = dimensions
        return {x: ('left', 1), y: ('left', 1)}


class SubPointBottomRight(SubDomain):
    name = 'subpointbottomright'

    def define(self, dimensions):
        x, y = dimensions
        return {x: ('right', 1), y: ('left', 1)}


class SubPointTopLeft(SubDomain):
    name = 'subpointtopleft'

    def define(self, dimensions):
        x, y = dimensions
        return {x: ('left', 1), y: ('right', 1)}


class SubPointTopRight(SubDomain):
    name = 'subpointtopright'

    def define(self, dimensions):
        x, y = dimensions
        return {x: ('right', 1), y: ('right', 1)}


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


def neumann_left(eq, subdomain):
    lhs, rhs = eq.evaluate.args

    # Get horizontal subdimension and its parent
    xfs = subdomain.dimensions[0]
    x = xfs.parent

    # Functions present in stencil
    funcs = retrieve_functions(lhs-rhs)

    mapper = {}
    for f in funcs:
        # Get the x index
        xind = f.indices[-2]
        if (xind - x).as_coeff_Mul()[0] < 0:
            # Symmetric mirror
            # Substitute where index is negative for +ve where
            # index is positive
            mapper.update({f: f.subs({xind: INT(abs(xind))})})

    return Eq(lhs.subs(mapper), rhs.subs(mapper), subdomain=subdomain)


def neumann_right(eq, subdomain):
    lhs, rhs = eq.evaluate.args

    # Get horizontal subdimension and its parent
    xfs = subdomain.dimensions[0]
    x = xfs.parent

    # Functions present in stencil
    funcs = retrieve_functions(lhs-rhs)

    mapper = {}
    for f in funcs:
        # Get the x index
        xind = f.indices[-2]
        if (xind - x).as_coeff_Mul()[0] > 0:
            tmp = x - INT(abs(x.symbolic_max - xind))
            mapper.update({f: f.subs({xind: tmp})})

    return Eq(lhs.subs(mapper), rhs.subs(mapper), subdomain=subdomain)


def analytical_solution(x, y):
    return np.cos(2*np.pi*x)*np.cos(2*np.pi*y)

n_values = [9, 17, 33, 65, 129, 257, 513, 1025, 2049, 4097]
h = np.array([Lx/(n-1) for n in n_values])
infinity_norms = []
discrete_l2_norms = []


for n in n_values:
    grid = Grid(
        shape=(n, n), extent=(Lx, Ly), dtype=np.float64
    )
    top = SubTop(so, grid=grid)
    bottom = SubBottom(so, grid=grid)
    left = SubLeft(so, grid=grid)
    right = SubRight(so, grid=grid)
    bottom_left = SubPointBottomLeft(grid=grid)
    bottom_right = SubPointBottomRight(grid=grid)
    top_left = SubPointTopLeft(grid=grid)
    top_right = SubPointTopRight(grid=grid)

    time = grid.time_dim
    t = grid.stepping_dim
    x, y = grid.dimensions

    u = Function(name='u', grid=grid, space_order=so, dtype=np.float64)
    f = Function(name='f', grid=grid, space_order=so, dtype=np.float64)

    tmpx = np.linspace(0, Lx, n).astype(np.float64)
    tmpy = np.linspace(0, Ly, n).astype(np.float64)
    Y, X = np.meshgrid(tmpx, tmpy)
    f.data[:] = (1.+(8.*(np.pi**2)))*np.cos(2.*np.pi*X)*np.cos(2.*np.pi*Y)

    eqn = Eq(-u.laplace+u, f, subdomain=grid.interior)

    # The reason for the added complexity here is to show how to mantain the symmetry of the 
    # matrix system -> can divide by 0.5 or 0.25 if at a corner
    # check using mat convert if it acc is symmetric
    bc_top = neumann_top(eqn, top)
    bc_top = Eq(0.5*bc_top.lhs, 0.5*bc_top.rhs, subdomain=top)
    bc_bottom = neumann_bottom(eqn, bottom)
    bc_bottom = Eq(0.5*bc_bottom.lhs, 0.5*bc_bottom.rhs, subdomain=bottom)
    bc_left = neumann_left(eqn, left)
    bc_left = Eq(0.5*bc_left.lhs, 0.5*bc_left.rhs, subdomain=left)
    bc_right = neumann_right(eqn, right)
    bc_right = Eq(0.5*bc_right.lhs, 0.5*bc_right.rhs, subdomain=right)
    bc_bottom_left = neumann_left(neumann_bottom(eqn, bottom_left), bottom_left)
    bc_bottom_left = Eq(0.25*bc_bottom_left.lhs, 0.25*bc_bottom_left.rhs, subdomain=bottom_left)
    bc_bottom_right = neumann_right(neumann_bottom(eqn, bottom_right), bottom_right)
    bc_bottom_right = Eq(0.25*bc_bottom_right.lhs, 0.25*bc_bottom_right.rhs, subdomain=bottom_right)
    bc_top_left = neumann_left(neumann_top(eqn, top_left), top_left)
    bc_top_left = Eq(0.25*bc_top_left.lhs, 0.25*bc_top_left.rhs, subdomain=top_left)
    bc_top_right = neumann_right(neumann_top(eqn, top_right), top_right)
    bc_top_right = Eq(0.25*bc_top_right.lhs, 0.25*bc_top_right.rhs, subdomain=top_right)
    bcs = [bc_top, bc_bottom, bc_left, bc_right,
           bc_bottom_left, bc_bottom_right, bc_top_left, bc_top_right]

    solver = petscsolve(
        [eqn]+bcs, target=u,
        solver_parameters={'ksp_rtol': 1e-8, 'ksp_type': 'cg', 'pc_type': 'none'},
        options_prefix='helmholtz_2d'
    )

    op = Operator(solver, language='petsc')
    op.apply()

    analytical = analytical_solution(X, Y)

    diff = Function(name='diff', grid=grid, space_order=2)
    diff.data[:] = analytical[:] - u.data[:]

    # Compute infinity norm using numpy
    infinity_norm = np.linalg.norm(diff.data[:].ravel(), ord=np.inf)
    infinity_norms.append(infinity_norm)

    # Compute discrete L2 norm (RMS error)
    n_interior = np.prod([s - 1 for s in grid.shape])
    discrete_l2_norm = norm(diff) / np.sqrt(n_interior)
    discrete_l2_norms.append(discrete_l2_norm)

    # Keep a coarse solution for the comparison plot
    if n == 33:
        X_plot, Y_plot = X.copy(), Y.copy()
        u_plot, exact_plot = u.data[:].copy(), analytical.copy()

print(infinity_norms)
slope, intercept = np.polyfit(np.log(h), np.log(infinity_norms), 1)
assert slope > 1.9
assert slope < 2.1


plt.figure(figsize=(6, 5))
plt.loglog(h, infinity_norms, 'o-', label=rf'Observed order $\hat{{q}} = {slope:.3f}$', color='orange')
plt.loglog(
    h, np.exp(intercept) * h**2,
    'k--',
    label='Reference line of slope 2'
)
plt.xlabel(r'Grid spacing $h$')
plt.ylabel(r'$\|e\|_\infty$')
plt.legend()
plt.tight_layout()
plt.savefig("helmholtz_neumann.png", dpi=200)
plt.show()





# ######################################## comparison plot at n = 33 ########################################
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
c1 = ax0.contourf(X_plot, Y_plot, u_plot, levels=100, cmap='viridis')
ax0.set_title('Devito Solution')
ax0.set_xlabel('x')
ax0.set_ylabel('y')

# Analytical solution
c2 = ax1.contourf(X_plot, Y_plot, exact_plot, levels=100, cmap='viridis')
ax1.set_title('Analytical Solution')
ax1.set_xlabel('x')
ax1.set_ylabel('y')

# Sync color scales
vmin = min(u_plot.min(), exact_plot.min())
vmax = max(u_plot.max(), exact_plot.max())
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
plt.savefig("helmholtz_comparison.png", dpi=200, bbox_inches='tight', pad_inches=0.2)
plt.show()
