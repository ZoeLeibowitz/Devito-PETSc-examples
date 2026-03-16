import numpy as np
from matplotlib import pyplot, cm
from devito import Grid, TimeFunction, Function, Eq, solve, Operator, configuration, SubDomain, NODE



ny = 41
nx = 41


nt = 2000
nit = 50
dx = 1. / (nx - 1)
dy = 1. / (ny - 1)


x_coord = np.linspace(0, 1, nx)
y_coord = np.linspace(0, 1, ny)


Y, X = np.meshgrid(x_coord, y_coord)

rho = 1
nu = .1
dt = .0005

# p is at cell centres - create subdomain to solve poisson equation only at pressure points
# that fall inside the physical domain (i.e exclude the right and top ghost columns at i=nx-1 and j=ny-1)
class PressureInterior(SubDomain):
    name = 'pressure_interior'
    def define(self, dimensions):
        x, y = dimensions
        return {x: ('middle', 0, 1), y: ('middle', 0, 1)}

grid = Grid(shape=(nx, ny), extent=(1, 1.), subdomains=(PressureInterior(),), dtype=np.float64)
x, y = grid.dimensions
t = grid.stepping_dim

# Staggered MAC grid:
#   u - staggered in y
#   v - staggered in x
#   p - cell centres (staggered in both x and y)
u = TimeFunction(name='u', grid=grid, space_order=2, staggered=y)
v = TimeFunction(name='v', grid=grid, space_order=2, staggered=x)
p = TimeFunction(name='p', grid=grid, space_order=2, staggered=(x, y))


# Plain Functions (no time dimension) for the Poisson source. (this is actually not used in the original notebook,
# which suggests it is slightly wrong since the source is different for each jacobi iteration?)
# The Jacobi loop in oppres advances its own time counter, so any TimeFunction
# in the source would alternate between the two velocity buffers on each
# iteration — the solver would converge to an average of two different sources
# rather than the actual divergence of the current velocity, causing instability.
# So, we use uf anf vf which are time-invariant within the Jacobi loop.
uf = Function(name='uf', grid=grid, space_order=2, staggered=y)
vf = Function(name='vf', grid=grid, space_order=2, staggered=x)


# u-momentum:
# p.dxc is forcing the non staggering so it is not correct - that is why I use p.dx instead
eq_u = Eq(u.dt + u*u.dxc + v*u.dyc, -1./rho * p.dx + nu*(u.dx2 + u.dy2))
stencil_u = solve(eq_u, u.forward)
update_u = Eq(u.forward, stencil_u)


# from IPython import embed; embed()

# v-momentum:
eq_v = Eq(v.dt + u*v.dxc + v*v.dyc, -1./rho * p.dy + nu*(v.dx2 + v.dy2))
stencil_v = solve(eq_v, v.forward)
update_v = Eq(v.forward, stencil_v)

# from IPython import embed; embed()


# manual edit to x0 since using uf.dx and vf.dy alone does not seem to be correct? it seems to be left staggered?
ux_cc = uf.dx(x0=x + x.spacing/2)
vy_cc = vf.dy(x0=y + y.spacing/2)
uy_cc = (uf[x, y+1] + uf[x+1, y+1] - uf[x, y-1] - uf[x+1, y-1]) / (4 * y.spacing)
vx_cc = (vf[x+1, y] + vf[x+1, y+1] - vf[x-1, y] - vf[x-1, y+1]) / (4 * x.spacing)

# eq_p = Eq(p.laplace,
#           rho * (1./dt * (ux_cc + vy_cc) - ux_cc**2 - 2*uy_cc*vx_cc - vy_cc**2),
#           subdomain=grid.subdomains['pressure_interior'])

eq_p = Eq(p.laplace,
          rho * (1./dt * (ux_cc + vy_cc) - ux_cc**2 - 2*uf.dy*vf.dx - vy_cc**2),
          subdomain=grid.subdomains['pressure_interior'])

stencil_p = solve(eq_p, p)
update_p = Eq(p.forward, stencil_p)


# u BCs
# Left/right walls: u is not x-staggered so nodes sit exactly on the walls.
# Bottom wall - average across the wall is zero, so ghost point is negative of first interior point

# Top lid (y=1): value at the wall equals U_lid=1: (u[ny-2] + u[ny-1])/2 = 1
# u[ny-1] = 2 - u[ny-2]
bc_u  = [Eq(u[t+1, 0, y], 0)] # left
bc_u += [Eq(u[t+1, nx-1, y], 0)] # right
bc_u += [Eq(u[t+1, x, -1], -u[t+1, x, 0])] # bottom
bc_u += [Eq(u[t+1, x, ny-1], 2 - u[t+1, x, ny-2])]  # lid: u=1 at y=1

bc_v  = [Eq(v[t+1, -1, y], -v[t+1, 0, y])] # left
bc_v += [Eq(v[t+1, nx-1, y], -v[t+1, nx-2, y])]  # ghost beyond right wall: interp to 0 at x=1
bc_v += [Eq(v[t+1, x, ny-1], 0)] # top
bc_v += [Eq(v[t+1, x, 0], 0)] # bottom


# p is at cell centres
bc_p  = [Eq(p[t+1, -1, y], p[t+1, 0, y])]        # left halo ghost → Neumann at x=0
bc_p += [Eq(p[t+1, nx-1, y], p[t+1, nx-2, y])]   # right ghost → Neumann at x=1
bc_p += [Eq(p[t+1, x, -1], p[t+1, x, 0])] # bottom halo ghost → Neumann at y=0
bc_p += [Eq(p[t+1, x, ny-1], p[t+1, x, ny-2])]   # top ghost → Neumann at y=1
bc_p += [Eq(p[t+1, 0, 0], 0)] # pin pressure at corner 


optime = Operator([update_u, update_v] + bc_u + bc_v)
oppres = Operator([update_p] + bc_p)
# print(oppres.ccode)

configuration['log-level'] = 'ERROR'

for step in range(0, nt):
    if step > 0:
        uf.data[:] = u.data[step % 2]
        vf.data[:] = v.data[step % 2]
        oppres(time_M=nit)
    optime(time_m=step, time_M=step, dt=dt)



# Import u values at x=L/2 (table 6, column 2 rows 12-26) in Marchi et al.
Marchi_Re10_u = np.array([[0.0625, -3.85425800e-2],
                          [0.125,  -6.96238561e-2],
                          [0.1875, -9.6983962e-2],
                          [0.25,   -1.22721979e-1],
                          [0.3125, -1.47636199e-1],
                          [0.375,  -1.71260757e-1],
                          [0.4375, -1.91677043e-1],
                          [0.5,    -2.05164738e-1],
                          [0.5625, -2.05770198e-1],
                          [0.625,  -1.84928116e-1],
                          [0.6875, -1.313892353e-1],
                          [0.75,   -3.1879308e-2],
                          [0.8125,  1.26912095e-1],
                          [0.875,   3.54430364e-1],
                          [0.9375,  6.50529292e-1]])
# Import v values at y=L/2 (table 6, column 2 rows 27-41) in Marchi et al.
Marchi_Re10_v = np.array([[0.0625, 9.2970121e-2],
                          [0.125,  1.52547843e-1],
                          [0.1875, 1.78781456e-1],
                          [0.25,   1.76415100e-1],
                          [0.3125, 1.52055820e-1],
                          [0.375,  1.121477612e-1],
                          [0.4375, 6.21048147e-2],
                          [0.5,    6.3603620e-3],
                          [0.5625,-5.10417285e-2],
                          [0.625, -1.056157259e-1],
                          [0.6875,-1.51622101e-1],
                          [0.75,  -1.81633561e-1],
                          [0.8125,-1.87021651e-1],
                          [0.875, -1.59898186e-1],
                          [0.9375,-9.6409942e-2]])

# run trivial operators to interpolate the staggered fields back onto original grid "nodes" for plotting and analysis

plotfunc_u = Function(name='plotfunc_u', grid=grid, space_order=2, staggered=NODE)
plotfunc_v = Function(name='plotfunc_v', grid=grid, space_order=2, staggered=NODE)
plotfunc_p = Function(name='plotfunc_p', grid=grid, space_order=2, staggered=NODE)

Operator(Eq(plotfunc_u, u))(time_M=0)
Operator(Eq(plotfunc_v, v))(time_M=0)
Operator(Eq(plotfunc_p, p))(time_M=0)


fig = pyplot.figure(figsize=(11, 7), dpi=100)
pyplot.contourf(X, Y, plotfunc_p.data[:], alpha=0.5, cmap=cm.viridis)
pyplot.colorbar()
pyplot.contour(X, Y, plotfunc_p.data[:], cmap=cm.viridis)
pyplot.quiver(X[::2, ::2], Y[::2, ::2], plotfunc_u.data[::2, ::2], plotfunc_v.data[::2, ::2])
pyplot.xlabel('X')
pyplot.ylabel('Y')
pyplot.savefig('02.png', dpi=100, bbox_inches='tight')
pyplot.show()

#NBVAL_IGNORE_OUTPUT
# Again, check results with Marchi et al 2009.
fig = pyplot.figure(figsize=(12, 6))
ax1 = fig.add_subplot(121)
ax1.plot(plotfunc_u.data[int(grid.shape[0]/2),:],y_coord[:])
ax1.plot(Marchi_Re10_u[:,1],Marchi_Re10_u[:,0],'ro')
ax1.set_xlabel('$u$')
ax1.set_ylabel('$y$')
ax1 = fig.add_subplot(122)
ax1.plot(x_coord[:],plotfunc_v.data[:,int(grid.shape[0]/2)])
ax1.plot(Marchi_Re10_v[:,0],Marchi_Re10_v[:,1],'ro')
ax1.set_xlabel('$x$')
ax1.set_ylabel('$v$')
pyplot.savefig('02_comparison.png', dpi=100, bbox_inches='tight')
pyplot.show()
