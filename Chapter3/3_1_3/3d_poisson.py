import os
import numpy as np
import matplotlib.pyplot as plt

from devito import (Grid, Function, Eq, Operator, switchconfig,
                    configuration, SubDomain, norm, mmax)

from devito.petsc import petscsolve, EssentialBC
from devito.petsc.initialize import PetscInitialize

configuration['compiler'] = 'custom'
os.environ['CC'] = 'mpicc'


# 3D test
# Solving -u_xx - u_yy - u_zz = f(x,y,z)
# Dirichlet BCs:
# u(0,y,z) = 0, u(1,y,z)=-e^(y+z)
# u(x,0,z) = -xe^z, u(x,1,z)=-xe^(1+z)
# u(x,y,0) = -xe^y, u(x,y,1)=-xe^(y+1)


# Manufactured solution: u(x,y,z) = -xe^(y+z), with corresponding RHS f(x,y,z) = 2.0*xe^(y+z)
# ref - https://github.com/bueler/p4pdes/blob/master/c/ch6/fish.c

PetscInitialize()

# Subdomains to implement BCs

# Subdomain for z = 1 (top)
class SubTop(SubDomain):
    name = 'subtop'

    def define(self, dimensions):
        x, y, z = dimensions
        return {x: ('middle', 1, 1), y: ('middle', 1, 1), z: ('right', 1)}

# Subdomain for z = 0 (bottom)
class SubBottom(SubDomain):
    name = 'subbottom'

    def define(self, dimensions):
        x, y, z = dimensions
        return {x: ('middle', 1, 1), y: ('middle', 1, 1), z: ('left', 1)}

# Subdomain for y = 1 (back)
class SubBack(SubDomain):
    name = 'subback'

    def define(self, dimensions):
        x, y, z = dimensions
        return {x: ('middle', 1, 1), y: ('right', 1), z: z}

# Subdomain for y = 0 (front)
class SubFront(SubDomain):
    name = 'subfront'

    def define(self, dimensions):
        x, y, z = dimensions
        return {x: ('middle', 1, 1), y: ('left', 1), z: z}

# Subdomain for x = 0 (left)
class SubLeft(SubDomain):
    name = 'subleft'

    def define(self, dimensions):
        x, y, z = dimensions
        return {x: ('left', 1), y: y, z: z}

# Subdomain for x = 1 (right)
class SubRight(SubDomain):
    name = 'subright'

    def define(self, dimensions):
        x, y, z = dimensions
        return {x: ('right', 1), y: y, z: z}


sub1 = SubTop()
sub2 = SubBottom()
sub3 = SubLeft()
sub4 = SubRight()
sub5 = SubBack()
sub6 = SubFront()


subdomains = (sub1, sub2, sub3, sub4, sub5, sub6)

def exact(x, y, z):
    return -x*np.float64(np.exp(y+z))

Lx = np.float64(1.)
Ly = np.float64(1.)
Lz = np.float64(1.)

# n = 9, 17, 33, 65, 129, 257
n_values = [2**k + 1 for k in range(3, 9)]
h = np.array([Lx/(n-1) for n in n_values])
infinity_norms = []
discrete_l2_norms = []
ksp_iters = []

for n in n_values:
    grid = Grid(
        shape=(n, n, n), extent=(Lx, Ly, Lz), subdomains=subdomains, dtype=np.float64
    )

    u = Function(name='u', grid=grid, space_order=2)
    f = Function(name='f', grid=grid, space_order=2)
    bc = Function(name='bc', grid=grid, space_order=2)

    x, y, z = grid.dimensions
    eqn = Eq(-u.laplace, f, subdomain=grid.interior)

    tmpx = np.linspace(0, Lx, n).astype(np.float64)
    tmpy = np.linspace(0, Ly, n).astype(np.float64)
    tmpz = np.linspace(0, Lz, n).astype(np.float64)


    X, Y, Z = np.meshgrid(tmpx, tmpy, tmpz, indexing="ij")

    # RHS
    f.data[:] = 2.0 * X * np.exp(Y + Z)

    # Create the 2D meshes for BCs
    # For faces along y and z (x varies)
    X_Y, Z_Y = np.meshgrid(tmpx, tmpz, indexing="ij")  # For faces where y varies
    X_Z, Y_Z = np.meshgrid(tmpx, tmpy, indexing="ij")  # For faces where z varies
    Y_X, Z_X = np.meshgrid(tmpy, tmpz, indexing="ij")  # For faces where x varies

    # u(0,y,z) = 0
    bc.data[0, :, :] = 0.0

    # u(1,y,z) = -exp(y+z)
    bc.data[-1, :, :] = -np.exp(Y_X + Z_X)

    # u(x,0,z) = -x*exp(z)
    bc.data[:, 0, :] = -X_Y * np.exp(Z_Y)

    # u(x,1,z) = -x*exp(1+z)
    bc.data[:, -1, :] = -X_Y * np.exp(1.0 + Z_Y)

    # u(x,y,0) = -x*exp(y)
    bc.data[:, :, 0] = -X_Z * np.exp(Y_Z)

    # u(x,y,1) = -x*exp(y+1)
    bc.data[:, :, -1] = -X_Z * np.exp(Y_Z + 1.0)

    # # Create boundary condition expressions using subdomains
    bcs = []
    bcs += [EssentialBC(u, bc, subdomain=sub1)]  # top
    bcs += [EssentialBC(u, bc, subdomain=sub2)]  # bottom
    bcs += [EssentialBC(u, bc, subdomain=sub3)]  # subleft
    bcs += [EssentialBC(u, bc, subdomain=sub4)]  # subright
    bcs += [EssentialBC(u, bc, subdomain=sub5)]  # subback
    bcs += [EssentialBC(u, bc, subdomain=sub6)]  # subfront

    exprs = [eqn] + bcs
    petsc = petscsolve(
        exprs, target=u,
        solver_parameters={'ksp_rtol': 1e-12, 'ksp_type': 'cg', 'pc_type': 'none'},
        options_prefix='poisson_3d'
    )

    with switchconfig(log_level='DEBUG'):
        op = Operator(petsc, language='petsc')
        summary = op.apply()

    iters = summary.petsc[('section0', 'poisson_3d')].KSPGetIterationNumber
    ksp_iters.append(iters)

    u_exact = Function(name='u_exact', grid=grid, space_order=2)
    u_exact.data[:] = exact(X, Y, Z)

    diff = Function(name='diff', grid=grid, space_order=2)
    diff.data[:] = u_exact.data[:] - u.data[:]

    # # Compute infinity norm using numpy
    # # TODO: Figure out how to compute the infinity norm using Devito
    infinity_norm = np.linalg.norm(diff.data[:].ravel(), ord=np.inf)
    infinity_norms.append(infinity_norm)

    # Compute discrete L2 norm (RMS error)
    n_interior = np.prod([s - 1 for s in grid.shape])
    discrete_l2_norm = norm(diff) / np.sqrt(n_interior)
    discrete_l2_norms.append(discrete_l2_norm)


slope, intercept = np.polyfit(np.log(h), np.log(infinity_norms), 1)
print(infinity_norms)
assert slope > 1.9
assert slope < 2.1

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
plt.savefig("3_1_3.png", dpi=200)
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


# # Make a table comparing solution against petsc4pdes solution
# # The command used for comparison:
# # export PETSC_ARCH=v3.22.4
# # ./fish -fsh_dim 3 -ksp_type cg -ksp_rtol 1e-12 -pc_type none -ksp_converged_reason -da_refine 2


# # I have changed it on the .fish file to print the infinity norm with 'error |u-uexact|_inf = %.22e'
# # That way, I can easily adjust the precision shown on my table
# petsc4pdes_infinity_norms = [
#     2.3550633478297555711833e-04, # 9
#     6.0292711047349456521260e-05, # 17
#     1.5215275331659228186254e-05, # 33
#     3.8116167768720288222539e-06, # 65
#     9.5329712923586384931696e-07, # 129
#     2.3835349827194818317366e-07, # 257,
#     , # 513
# ]
# formatted_petsc4pdes_infinity_norms = [f"{v:.5e}" for v in petsc4pdes_infinity_norms]
# formatted_devito_infinity_norms = [f"{v:.5e}" for v in infinity_norms]

# petsc4pdes_kspiters = [
#     38,
#     79,
#     159,
#     315,
#     624,
#     1234
# ]

# # I have changed it on the .fish file to print the infinity norm with '|u-uexact|_h = %.22e\n'
# # That way, I can easily adjust the precision shown on my table
# petsc4pdes_l2_norms = [
#     9.4532143978440900142050e-05, # 9
#     2.4201449746629726537039e-05, # 17
#     6.0851860324371993737273e-06, # 33
#     1.5234578767454247633339e-06, # 65
#     3.8099929212974920466535e-07, # 129
#     9.5258245406492944826294e-08, # 257
# ]
# formatted_petsc4pdes_l2_norms = [f"{v:.5e}" for v in petsc4pdes_l2_norms]
# formatted_devito_l2_norms = [f"{v:.5e}" for v in discrete_l2_norms]


# # print infinity norms to screen with a line break
# print("Petsc4pdes Infinity Norms: %s\n" % formatted_petsc4pdes_infinity_norms)
# print("Devito Infinity Norms: %s\n" % formatted_devito_infinity_norms)

# # print l2 discrete norms to screen with a line break
# print("Petsc4pdes L2 Norms: %s\n" % formatted_petsc4pdes_l2_norms)
# print("Devito L2 Norms: %s\n" % formatted_devito_l2_norms)

# assert formatted_petsc4pdes_infinity_norms == formatted_devito_infinity_norms
# assert formatted_petsc4pdes_l2_norms == formatted_devito_l2_norms
# assert ksp_iters == petsc4pdes_kspiters
