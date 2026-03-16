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

# p is at cell centres (i+0.5)*dx, (j+0.5)*dy.  The Poisson stencil at i=0
# needs p[-1] (left halo ghost) to enforce Neumann dp/dx=0 at x=0, so we
# solve at i=0..nx-2 (all physical cell columns, skipping only the right ghost
# at i=nx-1).  Same logic applies in y.
class PressureInterior(SubDomain):
    name = 'pressure_interior'
    def define(self, dimensions):
        xi, yi = dimensions
        return {xi: ('middle', 0, 1), yi: ('middle', 0, 1)}

grid = Grid(shape=(nx, ny), extent=(1, 1.), subdomains=(PressureInterior(),), dtype=np.float64)
x, y = grid.dimensions
t = grid.stepping_dim

# Staggered MAC grid:
#   u at (x,        y + hy/2)  -- staggered in y
#   v at (x + hx/2, y       )  -- staggered in x
#   p at (x + hx/2, y + hy/2)  -- cell centres (staggered in both)
u = TimeFunction(name='u', grid=grid, space_order=2, staggered=y)
v = TimeFunction(name='v', grid=grid, space_order=2, staggered=x)
p = TimeFunction(name='p', grid=grid, space_order=2, staggered=(x, y))


# plot_func_p = Function(name='plot_func_p', grid=grid, space_order=2, staggered=NODE)

# interp_order

# Plain Functions (no time dimension) for the Poisson source. (this is actually not used in the original notebook,
# which suggests it is slightly wrong since the source is different for each jacobi iteration?)
# The Jacobi loop in oppres advances its own time counter, so any TimeFunction
# in the source would alternate between the two velocity buffers on each
# iteration — the solver would converge to an average of two different sources
# rather than the actual divergence of the current velocity, causing instability.
# Plain Functions are time-invariant within the Jacobi loop.
uf = Function(name='uf', grid=grid, space_order=2, staggered=y)
vf = Function(name='vf', grid=grid, space_order=2, staggered=x)

# Cross-advection terms mix fields from different staggered locations.
# Devito handles derivative staggering automatically but does NOT interpolate
# field values, so a 4-point average is required for the cross terms only.
#   v lives at (x+hx/2, y); average to u-node (x, y+hy/2):
v_at_u = (v[t, x, y] + v[t, x-1, y] + v[t, x, y+1] + v[t, x-1, y+1]) / 4
#   u lives at (x, y+hy/2); average to v-node (x+hx/2, y):
u_at_v = (u[t, x, y] + u[t, x+1, y] + u[t, x, y-1] + u[t, x+1, y-1]) / 4

# Velocity gradients at cell centres (x+hx/2, y+hy/2) using frozen fields.
# ux and vy use forward half-step derivatives (same as the MAC divergence).
# uy and vx use 4-point averages to reach the cell-centre location.
ux_cc = uf.dx(x0=x + x.spacing/2)
vy_cc = vf.dy(x0=y + y.spacing/2)
uy_cc = (uf[x, y+1] + uf[x+1, y+1] - uf[x, y] - uf[x+1, y]) / (2 * y.spacing)
vx_cc = (vf[x+1, y] + vf[x+1, y+1] - vf[x, y] - vf[x, y+1]) / (2 * x.spacing)

# u-momentum: all terms at u-node (x, y+hy/2)
#   p.dx, u.dxc/dyc, u.dx2/dy2 all evaluate at the u-node automatically
# eq_u = Eq(u.dt + u*u.dxc + v_at_u*u.dyc,
#           -1./rho * p.dxc + nu*(u.dx2 + u.dy2),
#           subdomain=grid.interior)

eq_u = Eq(u.dt + u*u.dxc + v*u.dyc,
          -1./rho * p.dxc + nu*(u.dx2 + u.dy2),
          subdomain=grid.interior)

# v-momentum: all terms at v-node (x+hx/2, y)
#   p.dy, v.dxc/dyc, v.dx2/dy2 all evaluate at the v-node automatically
eq_v = Eq(v.dt + u_at_v*v.dxc + v*v.dyc,
          -1./rho * p.dyc + nu*(v.dx2 + v.dy2),
          subdomain=grid.interior)

# from IPython import embed; embed()


# Eq(u, (u*v).dy).evaluate


# Pressure Poisson at cell centre with full quadratic source (mirrors the
# non-staggered formulation): rho*(1/dt*div - (ux)^2 - 2*uy*vx - (vy)^2).
# The quadratic terms give the correct steady-state pressure even when div->0.
eq_p = Eq(p.laplace,
          rho * (1./dt * (ux_cc + vy_cc) - ux_cc**2 - 2*uy_cc*vx_cc - vy_cc**2),
          subdomain=grid.subdomains['pressure_interior'])





stencil_u = solve(eq_u, u.forward)
stencil_v = solve(eq_v, v.forward)
stencil_p = solve(eq_p, p)

update_u = Eq(u.forward, stencil_u)
update_v = Eq(v.forward, stencil_v)
update_p = Eq(p.forward, stencil_p)


# u BCs
# Left/right walls: u is not x-staggered so nodes sit exactly on the walls.
# Bottom wall (y=0): u is y-staggered so j=0 is a ghost point at y=hy/2;
#   set to 0 (no-slip approximation).
# Top lid (y=1): j=ny-1 is a ghost point above y=1; set so the interpolated
#   value at the wall equals U_lid=1: (u[ny-2] + u[ny-1])/2 = 1
#   => u[ny-1] = 2 - u[ny-2]
bc_u  = [Eq(u[t+1, 0, y], 0)]
bc_u += [Eq(u[t+1, nx-1, y], 0)]
bc_u += [Eq(u[t+1, x, 0], 0)]
bc_u += [Eq(u[t+1, x, ny-1], 2 - u[t+1, x, ny-2])]  # lid: u=1 at y=1

bc_v  = [Eq(v[t+1, 0, y], 0)]
bc_v += [Eq(v[t+1, nx-1, y], -v[t+1, nx-2, y])]  # ghost beyond right wall: interp to 0 at x=1
bc_v += [Eq(v[t+1, x, ny-1], 0)]
bc_v += [Eq(v[t+1, x, 0], 0)]

# p is at cell centres; left/bottom BCs must set halo ghosts (index -1) so
# that the Poisson stencil at i=0/j=0 sees a Neumann (dp/dn=0) condition.
# Right/top ghosts (i=nx-1, j=ny-1) sit outside the domain and are set by
# extrapolation from the last interior cell — same as the non-staggered case.
bc_p  = [Eq(p[t+1, -1, y], p[t+1, 0, y])]        # left halo ghost → Neumann at x=0
bc_p += [Eq(p[t+1, nx-1, y], p[t+1, nx-2, y])]   # right ghost → Neumann at x=1
bc_p += [Eq(p[t+1, x, -1], p[t+1, x, 0])]         # bottom halo ghost → Neumann at y=0
bc_p += [Eq(p[t+1, x, ny-1], p[t+1, x, ny-2])]   # top ghost → Neumann at y=1
bc_p += [Eq(p[t+1, 0, 0], 0)]                      # pin corner to fix gauge


optime = Operator([update_u, update_v] + bc_u + bc_v)
oppres = Operator([update_p] + bc_p)

configuration['log-level'] = 'ERROR'

for step in range(0, nt):
    if step > 0:
        uf.data[:] = u.data[step % 2]
        vf.data[:] = v.data[step % 2]
        oppres(time_M=nit)
    optime(time_m=step, time_M=step, dt=dt)


# Cell-centre coordinates (physical domain, no ghost columns)
nc = nx - 1   # number of physical cell centres in each direction
x_cc = (np.arange(nc) + 0.5) * dx
y_cc = (np.arange(nc) + 0.5) * dy
X_cc, Y_cc = np.meshgrid(x_cc, y_cc, indexing='ij')

# p lives at cell centres — drop the ghost column/row (index nc = nx-1)
p_cc = p.data[0, :nc, :nc]

# u is staggered in y: u[i,j] at (i*dx, (j+0.5)*dy)
#   average in x to reach cell centre (i+0.5)*dx
u_cc = 0.5 * (u.data[0, :nc, :nc] + u.data[0, 1:nc+1, :nc])

# v is staggered in x: v[i,j] at ((i+0.5)*dx, j*dy)
#   average in y to reach cell centre (j+0.5)*dy
v_cc = 0.5 * (v.data[0, :nc, :nc] + v.data[0, :nc, 1:nc+1])

fig = pyplot.figure(figsize=(11, 7), dpi=100)
pyplot.contourf(X_cc, Y_cc, p_cc, alpha=0.5, cmap=cm.viridis)
pyplot.colorbar()
pyplot.contour(X_cc, Y_cc, p_cc, cmap=cm.viridis)
pyplot.quiver(X_cc[::2, ::2], Y_cc[::2, ::2], u_cc[::2, ::2], v_cc[::2, ::2])
pyplot.xlabel('X')
pyplot.ylabel('Y')
pyplot.savefig('stag_cavity_flow_devito.png', dpi=100, bbox_inches='tight')
pyplot.show()




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

#NBVAL_IGNORE_OUTPUT
# Staggered fields share the same array shape as the grid (Devito uses symbolic
# offsets, not extra array points). With positive staggering, physical coords are:
#   u[i, j] lives at y = (j + 0.5)*dy  (y-staggered)
#   v[i, j] lives at x = (i + 0.5)*dx  (x-staggered)
y_u = (np.arange(ny) + 0.5) * dy   # y-coords of u nodes (j=ny-1 is ghost above lid)
x_v = (np.arange(nx) + 0.5) * dx   # x-coords of v nodes (i=nx-1 is ghost beyond right wall)

u_mid = u.data[0, nx // 2, :]      # u at x=L/2
v_mid = v.data[0, :, ny // 2]      # v at y=L/2

# Again, check results with Marchi et al 2009.
fig = pyplot.figure(figsize=(12, 6))
ax1 = fig.add_subplot(121)
ax1.plot(u_mid, y_u)
ax1.plot(Marchi_Re10_u[:, 1], Marchi_Re10_u[:, 0], 'ro')
ax1.set_xlabel('$u$')
ax1.set_ylabel('$y$')
ax1 = fig.add_subplot(122)
ax1.plot(x_v, v_mid)
ax1.plot(Marchi_Re10_v[:, 0], Marchi_Re10_v[:, 1], 'ro')
ax1.set_xlabel('$x$')
ax1.set_ylabel('$v$')
pyplot.savefig('stag_cavity_flow_devito_comparison.png', dpi=100, bbox_inches='tight')
pyplot.show()
