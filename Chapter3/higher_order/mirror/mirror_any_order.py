"""
The antisymmetric mirroring of modified_stencil_4th_order.py, generalised to
any even space order so = 2m.


Same setup as modified_stencil_4th_order.py, but the subdomains are defined
more generally... for example, the "interior" subdomain SubMain is defined as every point at least m points from
every side, where m=so/2. The subdomains for the modified stencisl are also adjusted according to the space order.

"""
import os
import numpy as np
import matplotlib.pyplot as plt

from devito import (Grid, Function, Eq, Operator, switchconfig,
                    configuration, SubDomain, norm, sign)
from devito.symbolics import retrieve_functions, INT

from devito.petsc import petscsolve, EssentialBC
from devito.petsc.initialize import PetscInitialize


configuration['compiler'] = 'custom'
os.environ['CC'] = 'mpicc'

# Increase the precision of finite difference weights to avoid numerical issues
# since we are using high order stencils
import devito.finite_differences.finite_difference as fdiff
fdiff._PRECISION = 18


PetscInitialize()


def modified(eq, subdomain, *sides):
    """Fold every access beyond the given sides ('left', 'right', 'bottom',
    'top') back into the domain, p[i] -> sign(i) p[|i|], counting i inwards
    from the side."""
    lhs, rhs = eq.evaluate.args
    mapper = {}
    for side in sides:
        axis = 0 if side in ('left', 'right') else 1
        dfs = subdomain.dimensions[axis]
        d = dfs.parent
        h = d.spacing
        for f in retrieve_functions(lhs.subs(mapper) - rhs.subs(mapper)):
            ind = f.indices[axis - 2]
            offset = (ind - d).as_coeff_Mul()[0]
            if side in ('left', 'bottom') and offset < 0:
                # Fold about index 0
                s = sign(ind.subs({d: dfs, h: 1}))
                mapper.update({f: s*f.subs({ind: INT(abs(ind))})})
            elif side in ('right', 'top') and offset > 0:
                # Fold about the last index d_M
                d_M = d.symbolic_max
                s = sign(d_M - ind.subs({d: dfs, h: 1}))
                mapper.update({f: s*f.subs({ind: d_M*h - INT(abs(d_M*h - ind))})})
    return Eq(lhs.subs(mapper), rhs.subs(mapper), subdomain=subdomain)


def exact(x, y, k1=1, k2=1):
    tmp1 = np.float64(np.pi/8.0) * np.float64(np.pi/8.0)
    tmp2 = k1**2 + k2**2
    tmp3 = tmp1 * tmp2
    tmp4 = -1.0/tmp3
    tmp5 = np.float64(np.sin((np.pi*x*k1)/8.0))
    tmp6 = np.float64(np.sin((np.pi*y*k2)/8.0))
    return tmp4*tmp5*tmp6


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
# N is the number of grid points and m = so/2. ('middle', a, b) leaves out a
# points at the start of a dimension and b at the end. Next to each side is a
# band of m-1 lines of points, at indices 1, ..., m-1 from the left (or bottom)
# side and N-m, ..., N-2 from the right (or top) side. The four corner blocks
# are where two bands meet, and SubMain is every point at least m points from
# every side. At fourth order (m = 2) each band is one line wide, exactly as in
# modified_stencil_4th_order.py.

class SubMain(SubDomain):
    name = 'submain'

    def __init__(self, m):
        super().__init__()
        self.m = m

    def define(self, dimensions):
        x, y = dimensions
        return {x: ('middle', self.m, self.m), y: ('middle', self.m, self.m)}


class SubTopModify(SubDomain):
    name = 'subtopmodify'

    def __init__(self, N, m):
        super().__init__()
        self.N, self.m = N, m

    def define(self, dimensions):
        x, y = dimensions
        return {x: ('middle', self.m, self.m), y: ('middle', self.N - self.m, 1)}


class SubLeftModify(SubDomain):
    name = 'subleftmodify'

    def __init__(self, N, m):
        super().__init__()
        self.N, self.m = N, m

    def define(self, dimensions):
        x, y = dimensions
        return {x: ('middle', 1, self.N - self.m), y: ('middle', self.m, self.m)}


class SubBottomModify(SubDomain):
    name = 'subbottommodify'

    def __init__(self, N, m):
        super().__init__()
        self.N, self.m = N, m

    def define(self, dimensions):
        x, y = dimensions
        return {x: ('middle', self.m, self.m), y: ('middle', 1, self.N - self.m)}


class SubRightModify(SubDomain):
    name = 'subrightmodify'

    def __init__(self, N, m):
        super().__init__()
        self.N, self.m = N, m

    def define(self, dimensions):
        x, y = dimensions
        return {x: ('middle', self.N - self.m, 1), y: ('middle', self.m, self.m)}


class SubTopLeftModify(SubDomain):
    name = 'subtopleftmodify'

    def __init__(self, N, m):
        super().__init__()
        self.N, self.m = N, m

    def define(self, dimensions):
        x, y = dimensions
        return {x: ('middle', 1, self.N - self.m), y: ('middle', self.N - self.m, 1)}


class SubBottomLeftModify(SubDomain):
    name = 'subbottomleftmodify'

    def __init__(self, N, m):
        super().__init__()
        self.N, self.m = N, m

    def define(self, dimensions):
        x, y = dimensions
        return {x: ('middle', 1, self.N - self.m), y: ('middle', 1, self.N - self.m)}


class SubTopRightModify(SubDomain):
    name = 'subtoprightmodify'

    def __init__(self, N, m):
        super().__init__()
        self.N, self.m = N, m

    def define(self, dimensions):
        x, y = dimensions
        return {x: ('middle', self.N - self.m, 1), y: ('middle', self.N - self.m, 1)}


class SubBottomRightModify(SubDomain):
    name = 'subbottomrightmodify'

    def __init__(self, N, m):
        super().__init__()
        self.N, self.m = N, m

    def define(self, dimensions):
        x, y = dimensions
        return {x: ('middle', self.N - self.m, 1), y: ('middle', 1, self.N - self.m)}


Lx = np.float64(16.)
Ly = np.float64(16.)

# Grids for each space order. At orders 6 and 8 the finer grids are left out,
# since their errors reach the level of rounding error (about 1e-13)
n_values = {2: [17, 33, 65, 129, 257], 4: [17, 33, 65, 129, 257],
            6: [17, 33, 65, 129], 8: [17, 33, 65]}

results = {}

for so, ns in n_values.items():
    m = so // 2
    infinity_norms = []

    for n in ns:
        sub1 = SubTopBC()
        sub2 = SubBottomBC()
        sub3 = SubLeftBC()
        sub4 = SubRightBC()

        sub5 = SubMain(m)
        subdomains = (sub1, sub2, sub3, sub4, sub5)

        # At second order no stencil reaches beyond the boundary
        if m > 1:
            sub6 = SubTopModify(n, m)
            sub7 = SubLeftModify(n, m)
            sub8 = SubBottomModify(n, m)
            sub9 = SubRightModify(n, m)

            sub10 = SubTopLeftModify(n, m)
            sub11 = SubBottomLeftModify(n, m)
            sub12 = SubTopRightModify(n, m)
            sub13 = SubBottomRightModify(n, m)

            subdomains += (sub6, sub7, sub8, sub9, sub10, sub11, sub12, sub13)

        grid = Grid(
            shape=(n, n), extent=(Lx, Ly), subdomains=subdomains, dtype=np.float64
        )

        p = Function(name='p', grid=grid, space_order=so)
        f = Function(name='f', grid=grid, space_order=so)
        bc = Function(name='bc', grid=grid, space_order=so)

        eqn = Eq(-p.laplace, f, subdomain=sub5)

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
        bcs = [EssentialBC(p, bc, subdomain=sub1)]  # top boundary
        bcs += [EssentialBC(p, bc, subdomain=sub2)]  # bottom boundary
        bcs += [EssentialBC(p, bc, subdomain=sub3)]  # left boundary
        bcs += [EssentialBC(p, bc, subdomain=sub4)]  # right boundary

        if m > 1:
            bcs += [modified(eqn, sub7, 'left')]  # Left modify
            bcs += [modified(eqn, sub9, 'right')]  # Right modify
            bcs += [modified(eqn, sub8, 'bottom')]  # Bottom modify
            bcs += [modified(eqn, sub6, 'top')]  # Top modify

            bcs += [modified(eqn, sub10, 'top', 'left')]  # Top left modify
            bcs += [modified(eqn, sub12, 'top', 'right')]  # Top right modify
            bcs += [modified(eqn, sub11, 'bottom', 'left')]  # Bottom left modify
            bcs += [modified(eqn, sub13, 'bottom', 'right')]  # Bottom right modify

        exprs = [eqn] + bcs

        # A nonzero initial guess, so that CG iterates. From zero it converges in
        # one iteration, since f is an eigenvector of the discrete operator.
        p.data[:] = 0.001

        prefix = f'mirror_so{so}'
        petsc = petscsolve(
            exprs, target=p,
            solver_parameters={'ksp_rtol': 1e-13, 'ksp_type': 'cg', 'pc_type': 'none'},
            options_prefix=prefix, get_info=['KSPGetIterationNumber']
        )

        with switchconfig(log_level='ERROR'):
            # TODO: fix in petscsolve. With the default optimisation, Devito
            # (CIRE) hoists the sign() factors, which depend on x or y only, into
            # temporary arrays of size x_size and y_size. The petscsolve
            # callbacks read these sizes from their context struct, which does
            # not contain them, so the generated C fails to compile. Switching
            # CIRE off (a minimum gain no expression can reach) avoids it.
            op = Operator(petsc, language='petsc',
                          opt=('advanced', {'cire-mingain': 10**9}))
            summary = op.apply()

        iters = summary.petsc[prefix].KSPGetIterationNumber

        diff = Function(name='diff', grid=grid, space_order=so)
        diff.data[:] = exact(X, Y) - p.data[:]

        # Compute infinity norm using numpy
        infinity_norm = np.linalg.norm(diff.data[:].ravel(), ord=np.inf)
        infinity_norms.append(infinity_norm)

        # Compute discrete L2 norm (RMS error)
        n_interior = np.prod([s - 1 for s in grid.shape])
        discrete_l2_norm = norm(diff) / np.sqrt(n_interior)

        print(f'space order {so}, n = {n:4d}, its = {iters:5d}, '
              f'inf error {infinity_norm:.6e}, L2 error {discrete_l2_norm:.6e}')

    h = np.array([Lx/(n-1) for n in ns])
    slope, intercept = np.polyfit(np.log(h), np.log(infinity_norms), 1)
    print(f'space order {so}: observed order {slope:.3f}')
    assert abs(slope - so) < 0.1

    results[so] = (h, np.array(infinity_norms), slope)


# Plot
plt.figure(figsize=(6, 5))
for k, (so, (h, e, slope)) in enumerate(results.items()):
    plt.loglog(h, e, 'o-', label=rf'Space order {so}, $\hat{{q}} = {slope:.3f}$')
    plt.loglog(h, e[0]*(h/h[0])**so, 'k--', lw=0.8,
               label=r'Reference lines of slope $q$' if k == len(results) - 1 else None)
plt.xlabel(r'Grid spacing $h$')
plt.ylabel(r'$\|e\|_\infty$')
plt.ylim(top=1e3)
plt.legend(loc='upper left')
plt.tight_layout()
plt.savefig('mirror_any_order.png', dpi=200)
