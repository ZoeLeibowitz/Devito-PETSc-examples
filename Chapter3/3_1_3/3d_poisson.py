import os
import numpy as np

from devito import (Grid, Function, Eq, Operator, switchconfig,
                    configuration, SubDomain, norm, mmax)

from devito.petsc import PETScSolve, EssentialBC
from devito.petsc.initialize import PetscInitialize

import matplotlib.pyplot as plt

configuration['compiler'] = 'custom'
os.environ['CC'] = 'mpicc'


# python3 3d_poisson.py -ksp_converged_reason -ksp_type cg

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

# n = 9, 17, 33, 65, 129, 257, 513, 1025, 2049
n_values = [2**k + 1 for k in range(3, 12)]
n_values = [65]
h = np.array([Lx/(n-1) for n in n_values])
errors = []

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

    # BCs
    Y2D, Z2D = np.meshgrid(tmpy, tmpz, indexing="ij")
    X2D, Z2D = np.meshgrid(tmpx, tmpz, indexing="ij")
    X2D, Y2D = np.meshgrid(tmpx, tmpy, indexing="ij")

    # u(0,y,z) = 0
    bc.data[0, :, :] = 0.0

    # u(x,0,z) = -x*exp(z)
    bc.data[:, 0, :] = -X2D * np.exp(Z2D)

    # u(x,1,z) = -x*exp(1+z)
    bc.data[:, -1, :] = -X2D * np.exp(1.0 + Z2D)

    # u(x,y,0) = -x*exp(y)
    bc.data[:, :, 0] = -X2D * np.exp(Y2D)

    # u(x,y,1) = -x*exp(y+1)
    bc.data[:, :, -1] = -X2D * np.exp(Y2D + 1.0)


    Y2D, Z2D = np.meshgrid(tmpy, tmpz, indexing="ij")
    # u(1,y,z) = -exp(y+z)
    bc.data[-1, :, :] = -np.exp(Y2D + Z2D)



    print(bc.data[-1, :, :])

    # u.data[:] = 0.0
    # # Create boundary condition expressions using subdomains
    bcs = []
    bcs += [EssentialBC(u, bc, subdomain=sub1)]  # top
    bcs += [EssentialBC(u, bc, subdomain=sub2)]  # bottom
    bcs += [EssentialBC(u, bc, subdomain=sub3)]  # subleft
    bcs += [EssentialBC(u, bc, subdomain=sub4)]  # subright
    bcs += [EssentialBC(u, bc, subdomain=sub5)]  # subback
    bcs += [EssentialBC(u, bc, subdomain=sub6)]  # subfront

    # exprs = bcs
    exprs = [eqn] + bcs
    # TODO: set ksp type to CG
    petsc = PETScSolve(exprs, target=u, solver_parameters={'ksp_rtol': 1e-10})
    op = Operator(petsc, language='petsc')
    op.apply()
    # print(op.ccode)

    # print(u.data[:])
    
    # print(norm(u))
    # print(u_exact.data[:])

    u_exact = Function(name='u_exact', grid=grid, space_order=2)
    u_exact.data[:] = exact(X, Y, Z)

    # print(norm(u_exact))


    # print(np.linalg.norm(u.data[:].ravel(), ord=np.inf))

    # print(u_exact.data[:])

    # print(no/rm(u_exact))
    # print(u_exact.data[:])

    # diff = Function(name='diff', grid=grid, space_order=2)
    # diff.data[:] = u_exact.data[:] - u.data[:]

    diff = u_exact.data[:] - u.data[:]
    diff_norm = np.linalg.norm(diff.ravel(), ord=np.inf)
    print(diff_norm)
    # # Compute infinity norm using numpy
    # # TODO: Figure out how to compute the infinity norm using Devito
    # diff_norm = np.linalg.norm(diff.data[:].ravel(), ord=np.inf)
    # u_error = diff_norm / np.linalg.norm(u_exact.data[:].ravel(), ord=np.inf)
    # errors.append(u_error)

# print(errors)
# slope, intercept = np.polyfit(np.log(h), np.log(errors), 1)

# assert slope > 1.9
# assert slope < 2.1

# # Plot
# plt.figure(figsize=(6, 5))
# plt.loglog(h, errors, 'o-', label=f'Observed rate ≈ {slope:.2f}', color='orange')
# plt.loglog(
#     h, np.exp(intercept) * h**2,
#     'k--',
#     label=r'Reference slope $O(h^2)$'
# )
# plt.xlabel(r'Grid spacing h')
# plt.ylabel(r'Relative $\infty$-norm error')
# plt.title('Convergence Plot')

# plt.legend()
# plt.tight_layout()

# plt.savefig("3_1_3.png", dpi=200)

# plt.show()
