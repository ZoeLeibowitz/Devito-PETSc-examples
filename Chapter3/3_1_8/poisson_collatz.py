import os
import numpy as np

from devito import (Grid, Function, Eq, Operator, switchconfig,
                    configuration, SubDomain, norm)
from devito.finite_differences.differentiable import EvalDerivative

from devito.petsc import petscsolve, EssentialBC
from devito.petsc.initialize import PetscInitialize

import matplotlib.pyplot as plt

configuration['compiler'] = 'custom'
os.environ['CC'] = 'mpicc'


import devito.finite_differences.finite_difference as fdiff
fdiff._PRECISION = 18

# python3 poisson_collatz.py -ksp_converged_reason -ksp_type cg -ksp_rtol 1e-13 -pc_type none
# ref - https://pdf.sciencedirectassets.com/272570/1-s2.0-S0021999100X02033/1-s2.0-0021999184900226/main.pdf?X-Amz-Security-Token=IQoJb3JpZ2luX2VjELj%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FwEaCXVzLWVhc3QtMSJGMEQCIFQy5wqN8Qzb%2FdGuSbFRGSrbApV%2F6XmfLeQ8FEmkaD7AAiAYSK4zEefc95Igcd%2BpFS3477XMDvXbM%2FLstz7EbJ21uCq8BQjR%2F%2F%2F%2F%2F%2F%2F%2F%2F%2F8BEAUaDDA1OTAwMzU0Njg2NSIMsyWCF9OHFhbLJfSAKpAFVcic7HhOkv4Lq4rQEiA48irK8rI9gKD3fcpuPLe%2F4oL2H709cXc8TVdwm%2FXTIyZFW6pjnqRX9Q3oXGNiFte3A3wMcSusW2A3EpYpPw%2BHi9ZXwB3WyVeYeS1FmBHvqOfe803tZRxMFI%2BSetCtWwfnOLkOUln5l5Me1L4ufFOGFQiqIu258%2F%2FvSpB62K%2FjzqjkiFEP6SxfWyB%2BbxReH46jGE6zyoYAUyfyZ8QveKRjMO38U2Wk5jkuostmvONwmJsfbfLDzrBTWvnY01lh9tcfPF6GhYVHx92TzSclUNY46mI39MvgozABQneceYAeNpMdqECcX5aSRCegGacIsB4OGTsT8lq04yjQ4YNOFvztWSvbzw9dtDEViFzAm3WDQVEYLP6He9hQuKA6xiSZZsBVQGdnSioJG%2BBhmtv1UpDfFgg6qnoMydtxmf112qJtd9fGKLkuR2HU8bfrSy1WxM6lAfq%2Fw2ET1V%2FhlJCQx4%2FTdWLE83XSi18ML4EZ291UXV81J48fw7Q3KvNYT0dlxxlcuU2XeYIFrKXBBP0x%2FDQm4q0e%2BeiqKguW6fcD8K5vcl5g%2B2zg%2FNItHVT2AqacPvEFt1DP9uyaMmHXxBV2g0v3fuGG2vVnESS9mabYZj04qYWnEtmny%2FEx1GbQ4bNAO5VcVxZowWS%2FmFp7iq%2B8KTJ%2B4V1fPa4wM3aA6Zkfs9xiHrWPsUHF0%2BcIvGujEj8SFzDrjKxU%2F6b7UzqNGbRm4ME3kGTFkHR9H8Mnm75H8XQVBNHqWoFhLF8G%2F27ROlABYDYeZbECAAscwrd9VVE0TNVRDapZrBYdTMjLB%2FWb8TyZ2cpM5s4lDJtC7fxKti6JQpZGFjP1XY4XFDA3Y8QmKK7JTbQw3uD3wwY6sgGd2zuWzi7sbhsOT3PxpsuUIoHcMCurHaT83Kq5hMoRef17aN8JnFe0IACENyKX3pD1y4k3xq0U5AywQtI%2FlfxfK3lJpDlBKdPrm28a9gEySfDD3EhHb8V7CiW9q5%2F2Nve10xAfmVYFIFWtm1buOhkAlq3QkPUkCxLbtgV0qan1pzjuRZJUQ3hZUhiXJg4KTsh9GpBznicQY6%2BEeCMkV3035AWG9oRn8Vz55rJid%2BmZpUSi&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Date=20250721T082320Z&X-Amz-SignedHeaders=host&X-Amz-Expires=300&X-Amz-Credential=ASIAQ3PHCVTYZNTJILVC%2F20250721%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Signature=19b1223011648d50d624d6a9aebcace081f1583c24cb86ac6f1d3485d8c5d866&hash=f54ddef1298fe426f37f14dead5eb8cad80473bd7550f05ac4262b82eaffe93a&host=68042c943591013ac2b2430a89b270f6af2c76d8dfd086a07176afe7c76c2c61&pii=0021999184900226&tid=spdf-eb012c0c-d331-46fb-a50f-478e4494ac76&sid=397998d868be274a822ae3254cd5195e41f1gxrqb&type=client&tsoh=d3d3LnNjaWVuY2VkaXJlY3QuY29t&rh=d3d3LnNjaWVuY2VkaXJlY3QuY29t&ua=010c5b52535951050c&rr=96294bb2feadeb19&cc=gb


# instead of using eval derivative, can use custom weights or rotated derivative 45 https://github.com/devitocodes/devito/blob/main/devito/finite_differences/rsfd.py
# or can just use indexed notation (see faq -> but won't work with MPI)

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

# Not acc really used
so = 2


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
    nine_point_stencil_lhs = EvalDerivative(*[w*p/h_x**2 for w, p in zip(weights, points)], base=u)

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
    # TODO: don't use EvalDerivative?
    nine_point_stencil_rhs = EvalDerivative(*[w*p for w, p in zip(weights, points)], base=f)

    eqn = Eq(nine_point_stencil_lhs, nine_point_stencil_rhs, subdomain=grid.interior)

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
    bcs = [EssentialBC(u, bc, subdomain=sub1)]
    bcs += [EssentialBC(u, bc, subdomain=sub2)]
    bcs += [EssentialBC(u, bc, subdomain=sub3)]
    bcs += [EssentialBC(u, bc, subdomain=sub4)]

    exprs = [eqn] + bcs

    # Can play around with initial guess -> if it's zero then cg just converges in 1 iteration because
    # the rhs is an eigenvector of the matrix -> I think?
    u.data[:] = 0.001

    petsc = petscsolve(
        exprs, target=u,
        solver_parameters={'ksp_rtol': 1e-13, 'ksp_type': 'cg', 'pc_type': 'none'},
        options_prefix='poisson_collatz'
    )

    with switchconfig(log_level='DEBUG'):
        op = Operator(petsc, language='petsc')
        summary = op.apply()

    iters = summary.petsc[('section0', 'poisson_collatz')].KSPGetIterationNumber
    ksp_iters.append(iters)

    u_exact = Function(name='u_exact', grid=grid, space_order=so)
    u_exact.data[:] = exact(X, Y)

    diff = Function(name='diff', grid=grid, space_order=so)
    diff.data[:] = u_exact.data[:] - u.data[:]

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
plt.savefig("poisson_collatz_compare.png", dpi=200, bbox_inches='tight', pad_inches=0.2)
plt.show()
