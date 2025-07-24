import os
import numpy as np

from devito import (Grid, Function, Eq, Operator, switchconfig,
                    configuration, SubDomain, norm)
from devito.finite_differences.differentiable import EvalDerivative
from devito.symbolics import retrieve_functions, INT

from devito.petsc import PETScSolve, EssentialBC
from devito.petsc.initialize import PetscInitialize

import matplotlib.pyplot as plt

configuration['compiler'] = 'custom'
os.environ['CC'] = 'mpicc'


# python3 modified_stencil_4th_order.py -ksp_converged_reason -ksp_type gmres -ksp_rtol 1e-12 -pc_type none
# modify the equations near boundaries -> 4th order discretisation


PetscInitialize()


def modified_left(eq, subdomain):
    lhs, rhs = eq.evaluate.args

    # Get horizontal subdimension and its parent
    xfs = subdomain.dimensions[0]
    x = xfs.parent

    yfs = subdomain.dimensions[1]
    y = yfs.parent

    # Functions present in stencil
    funcs = retrieve_functions(lhs-rhs)

    for f in funcs:
        xind = f.indices[-2]
        yind = f.indices[-1]
        h_x = f.grid.spacing_symbols[0]
        if xind == x + h_x:
            u_2_h_x = f
        elif xind == x and yind == y:
            u_h_x = f
        else:
            continue
    mapper = {}
    for f in funcs:
        # Get the x index
        xind = f.indices[-2]
        h_x = f.grid.spacing_symbols[0]
        if xind == x - 2*h_x:
            a3 = (u_2_h_x - 2.*u_h_x) / (6.*h_x**3)
            a1 = (u_h_x - a3*(h_x**3)) / h_x

            new = -1.*a1*h_x - a3*(h_x**3)

            # from IPython import embed; embed()  # Debugging lin
            mapper.update({f: -f.subs({xind: u_h_x.indices[0]})})
            # mapper.update({f: -f})

    return Eq(lhs.subs(mapper), rhs.subs(mapper), subdomain=subdomain)



def modified_right(eq, subdomain):
    lhs, rhs = eq.evaluate.args

    # Get horizontal subdimension and its parent
    xfs = subdomain.dimensions[0]
    x = xfs.parent

    yfs = subdomain.dimensions[1]
    y = yfs.parent

    # Functions present in stencil
    funcs = retrieve_functions(lhs-rhs)

    for f in funcs:
        xind = f.indices[-2]
        yind = f.indices[-1]
        h_x = f.grid.spacing_symbols[0]
        if xind == x and yind == y:
            u_minus_h_x = f
        else:
            continue
    mapper = {}
    for f in funcs:
        # Get the x index
        xind = f.indices[-2]
        h_x = f.grid.spacing_symbols[0]
        if xind == x + 2*h_x:

            new = u_minus_h_x

            mapper.update({f: -f.subs({xind: u_minus_h_x.indices[0]})})
            # mapper.update({f: -f})
    return Eq(lhs.subs(mapper), rhs.subs(mapper), subdomain=subdomain)


def modified_bottom(eq, subdomain):
    lhs, rhs = eq.evaluate.args

    # Get horizontal subdimension and its parent
    xfs = subdomain.dimensions[0]
    x = xfs.parent

    yfs = subdomain.dimensions[1]
    y = yfs.parent

    # Functions present in stencil
    funcs = retrieve_functions(lhs-rhs)

    for f in funcs:
        xind = f.indices[-2]
        yind = f.indices[-1]
        h_y = f.grid.spacing_symbols[1]
        if xind == x and yind == y:
            tmp = f
        else:
            continue
    mapper = {}
    for f in funcs:
        # Get the x index
        yind = f.indices[-1]
        h_y = f.grid.spacing_symbols[1]
        if yind == y - 2*h_y:
            new = -tmp

            mapper.update({f: -f.subs({yind: tmp.indices[-1]})})
            # mapper.update({f: -f})
    return Eq(lhs.subs(mapper), rhs.subs(mapper), subdomain=subdomain)



def modified_top(eq, subdomain):
    lhs, rhs = eq.evaluate.args

    # Get horizontal subdimension and its parent
    xfs = subdomain.dimensions[0]
    x = xfs.parent

    yfs = subdomain.dimensions[1]
    y = yfs.parent

    # Functions present in stencil
    funcs = retrieve_functions(lhs-rhs)

    for f in funcs:
        xind = f.indices[-2]
        yind = f.indices[-1]
        h_x = f.grid.spacing_symbols[0]
        if xind == x and yind == y:
            tmp = f
        else:
            continue
    mapper = {}
    for f in funcs:
        # Get the y index
        yind = f.indices[-1]
        h_y = f.grid.spacing_symbols[1]
        if yind == y + 2*h_y:

            mapper.update({f: -f.subs({yind: tmp.indices[-1]})})
            # mapper.update({f: -f})
    return Eq(lhs.subs(mapper), rhs.subs(mapper), subdomain=subdomain)


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

# n = 9, 17, 33, 65, 129, 257
# for higher n -> round off error starts to dominate
n_values = [2**k + 1 for k in range(3, 9)]
# n_values = [33, 43, 53, 63, 73, 83]
n_values = [33, 35, 37, 39, 41, 43, 45, 47, 49, 51, 53, 55, 57, 59, 61, 63, 65, 67, 69, 71, 73, 75, 77, 79, 81, 83, 85, 87, 89, 91, 93, 95, 97]
# n_values = [6]

h = np.array([Lx/(n-1) for n in n_values])
infinity_norms = []
discrete_l2_norms = []
ksp_iters = []

so = 4

for n in n_values:

    # Subdomains to implement BCs
    class SubTopBC(SubDomain):
        name = 'subtop'

        def define(self, dimensions):
            x, y = dimensions
            return {x: x, y: ('right', 1)}

    class SubBottomBC(SubDomain):
        name = 'subbottom'

        def define(self, dimensions):
            x, y = dimensions
            return {x: x, y: ('left', 1)}

    class SubLeftBC(SubDomain):
        name = 'subleft'

        def define(self, dimensions):
            x, y = dimensions
            return {x: ('left', 1), y: ('middle', 1, 1)}

    class SubRightBC(SubDomain):
        name = 'subright'

        def define(self, dimensions):
            x, y = dimensions
            return {x: ('right', 1), y: ('middle', 1, 1)}
        
    ##################### modified stencil subdomains #####################

    class SubMain(SubDomain):
        name = 'submain'

        def define(self, dimensions):
            x, y = dimensions
            return {x: ('middle', 2, 2), y: ('middle', 2, 2)}
        

    class SubTopModify(SubDomain):
        name = 'subtopmodify'

        def __init__(self, N):
            super().__init__()
            self.N = N

        def define(self, dimensions):
            x, y = dimensions
            return {x: ('middle', 2, 2), y: ('middle', self.N - 2, 1)}
        

    class SubLeftModify(SubDomain):
        name = 'subleftmodify'

        def __init__(self, N):
            super().__init__()
            self.N = N

        def define(self, dimensions):
            x, y = dimensions
            return {x: ('middle', 1, self.N - 2), y: ('middle', 2, 2)}
        

    class SubBottomModify(SubDomain):
        name = 'subbottommodify'

        def __init__(self, N):
            super().__init__()
            self.N = N

        def define(self, dimensions):
            x, y = dimensions
            return {x: ('middle', 2, 2), y: ('middle', 1, self.N - 2)}
        

    class SubRightModify(SubDomain):
        name = 'subrightmodify'

        def __init__(self, N):
            super().__init__()
            self.N = N

        def define(self, dimensions):
            x, y = dimensions
            return {x: ('middle', self.N - 2, 1), y: ('middle', 2, 2)}
        

    class SubTopLeftModify(SubDomain):
        name = 'subtopleftmodify'

        def __init__(self, N):
            super().__init__()
            self.N = N

        def define(self, dimensions):
            x, y = dimensions
            return {x: ('middle', 1, self.N - 2), y: ('middle', self.N - 2, 1)}
        

    class SubBottomLeftModify(SubDomain):
        name = 'subbottomleftmodify'

        def __init__(self, N):
            super().__init__()
            self.N = N

        def define(self, dimensions):
            x, y = dimensions
            return {x: ('middle', 1, self.N - 2), y: ('middle', 1, self.N - 2)}
        
    
    class SubTopRightModify(SubDomain):
        name = 'subtoprightmodify'

        def __init__(self, N):
            super().__init__()
            self.N = N

        def define(self, dimensions):
            x, y = dimensions
            return {x: ('middle', self.N - 2, 1), y: ('middle', self.N - 2, 1)}
        


    class SubBottomRightModify(SubDomain):
        name = 'subbottomrightmodify'

        def __init__(self, N):
            super().__init__()
            self.N = N

        def define(self, dimensions):
            x, y = dimensions
            return {x: ('middle', self.N - 2, 1), y: ('middle', 1, self.N - 2)}


    sub1 = SubTopBC()
    sub2 = SubBottomBC()
    sub3 = SubLeftBC()
    sub4 = SubRightBC()

    sub5 = SubMain()
    sub6 = SubTopModify(n)
    sub7 = SubLeftModify(n)
    sub8 = SubBottomModify(n)
    sub9 = SubRightModify(n)

    sub10 = SubTopLeftModify(n)
    sub11 = SubBottomLeftModify(n)
    sub12 = SubTopRightModify(n)
    sub13 = SubBottomRightModify(n)


    subdomains = (sub1, sub2, sub3, sub4, sub5, sub6, sub7, sub8, sub9, sub10, sub11, sub12, sub13)

    grid = Grid(
        shape=(n, n), extent=(Lx, Ly), subdomains=subdomains, dtype=np.float64
    )

    u = Function(name='u', grid=grid, space_order=so)
    f = Function(name='f', grid=grid, space_order=so)
    bc = Function(name='bc', grid=grid, space_order=so)

    eqn = Eq(u.laplace, f, subdomain=sub5)

    tmpx = np.linspace(0, Lx, n).astype(np.float64)
    tmpy = np.linspace(0, Ly, n).astype(np.float64)

    Y, X = np.meshgrid(tmpx, tmpy)

    k1, k2 = 1., 1.

    f.data[:] = np.float64(np.sin((np.pi*X*k1)/8.0)) * np.float64(np.sin((np.pi*Y*k2)/8.0))

    bc.data[:, 0] = 0.
    bc.data[:, -1] = 0.
    bc.data[0, :] = 0.
    bc.data[-1, :] = 0.

    # Create boundary condition expressions using subdomains
    bcs = [EssentialBC(u, bc, subdomain=sub1)] # top boundary
    bcs += [EssentialBC(u, bc, subdomain=sub2)] # bottom boundary
    bcs += [EssentialBC(u, bc, subdomain=sub3)] # left boundary
    bcs += [EssentialBC(u, bc, subdomain=sub4)] # right boundary

    bcs += [modified_left(eqn, sub7)]  # Left modify
    bcs += [modified_right(eqn, sub9)]  # Right modify
    bcs += [modified_bottom(eqn, sub8)]  # Bottom modify
    bcs += [modified_top(eqn, sub6)]  # Top modify

    bcs += [modified_left(modified_top(eqn, sub10), sub10)]  # Top left modify
    bcs += [modified_right(modified_top(eqn, sub12), sub12)]  # Top right modify
    bcs += [modified_left(modified_bottom(eqn, sub11), sub11)]  # Bottom left modify
    bcs += [modified_right(modified_bottom(eqn, sub13), sub13)]  # Bottom right modify

    # from IPython import embed; embed()

    exprs = [eqn] + bcs

    # exprs = bcs

    # Can play around with initial guess -> if it's zero then cg just converges in 1 iteration because
    # the rhs is an eigenvector of the matrix -> I think?
    # u.data[:] = 0.001

    petsc = PETScSolve(exprs, target=u, solver_parameters={'ksp_rtol': 1e-12})

    with switchconfig(log_level='DEBUG'):
        op = Operator(petsc, language='petsc')
        summary = op.apply()

    iters = summary.petsc[('section0', None)].KSPGetIterationNumber
    ksp_iters.append(iters)

    u_exact = Function(name='u_exact', grid=grid, space_order=so)
    u_exact.data[:] = exact(X, Y)

    diff = Function(name='diff', grid=grid, space_order=so)
    diff.data[:] = u_exact.data[:] - u.data[:]


    # print(u.data[:])


    # Compute infinity norm using numpy
    infinity_norm = np.linalg.norm(diff.data[:].ravel(), ord=np.inf)
    infinity_norms.append(infinity_norm)
    print(infinity_norm)

    # Compute discrete L2 norm (RMS error)
    n_interior = np.prod([s - 1 for s in grid.shape])
    discrete_l2_norm = norm(diff) / np.sqrt(n_interior)
    discrete_l2_norms.append(discrete_l2_norm)
    print(discrete_l2_norm)

    # print(op.ccode)


# print(infinity_norms)
slope, intercept = np.polyfit(np.log(h), np.log(infinity_norms), 1)

print(slope)
# # assert slope > 3.9
# # assert slope < 4.1

# Plot
plt.figure(figsize=(6, 5))
plt.loglog(h, infinity_norms, 'o-', label=f'Observed rate ≈ {slope:.3f}', color='orange')
plt.loglog(
    h, np.exp(intercept) * h**4,
    'k--',
    label=r'Reference slope $O(h^4)$'
)
plt.xlabel(r'Grid spacing h')
plt.ylabel(r'$\infty$-norm error')
plt.title('Convergence Plot')
plt.legend()
plt.tight_layout()
plt.savefig("3_1_8.png", dpi=200)
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
