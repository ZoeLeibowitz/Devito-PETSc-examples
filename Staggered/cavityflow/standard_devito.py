import numpy as np
from matplotlib import pyplot, cm
from devito import Grid, TimeFunction, Eq, solve, Operator, configuration, SubDomain



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
dt = .001

grid = Grid(shape=(nx, ny), extent=(1, 1.))
x, y = grid.dimensions
t = grid.stepping_dim

u = TimeFunction(name='u', grid=grid, space_order=2)
v = TimeFunction(name='v', grid=grid, space_order=2)
p = TimeFunction(name='p', grid=grid, space_order=2)

eq_u = Eq(u.dt + u*u.dx + v*u.dy, -1./rho * p.dxc + nu*(u.laplace), subdomain=grid.interior)
eq_v = Eq(v.dt + u*v.dx + v*v.dy, -1./rho * p.dyc + nu*(v.laplace), subdomain=grid.interior)
eq_p = Eq(p.laplace, rho*(1./dt*(u.dxc+v.dyc)-(u.dxc*u.dxc)-2*(u.dyc*v.dxc)-(v.dyc*v.dyc)), subdomain=grid.interior)

stencil_u = solve(eq_u, u.forward)
stencil_v = solve(eq_v, v.forward)
stencil_p = solve(eq_p, p)

update_u = Eq(u.forward, stencil_u)
update_v = Eq(v.forward, stencil_v)
update_p = Eq(p.forward, stencil_p)


print('update_u: %s' % update_u)


bc_u  = [Eq(u[t+1, 0, y], 0)]
bc_u += [Eq(u[t+1, nx-1, y], 0)]
bc_u += [Eq(u[t+1, x, 0], 0)]
bc_u += [Eq(u[t+1, x, ny-1], 1)]  # lid: u=1 at y=1

bc_v  = [Eq(v[t+1, 0, y], 0)]
bc_v += [Eq(v[t+1, nx-1, y], 0)]
bc_v += [Eq(v[t+1, x, ny-1], 0)]
bc_v += [Eq(v[t+1, x, 0], 0)]

bc_p  = [Eq(p[t+1, 0, y], p[t+1, 1, y])]
bc_p += [Eq(p[t+1, nx-1, y], p[t+1, nx-2, y])]
bc_p += [Eq(p[t+1, x, 0], p[t+1, x, 1])]
bc_p += [Eq(p[t+1, x, ny-1], p[t+1, x, ny-2])]
bc_p += [Eq(p[t+1, 0, 0], 0)]


optime = Operator([update_u, update_v] + bc_u + bc_v)
oppres = Operator([update_p] + bc_p)

configuration['log-level'] = 'ERROR'

for step in range(0, nt):
    if step > 0:
        oppres(time_M=nit)
    optime(time_m=step, time_M=step, dt=dt)


fig = pyplot.figure(figsize=(11, 7), dpi=100)
pyplot.contourf(X, Y, p.data[0], alpha=0.5, cmap=cm.viridis)
pyplot.colorbar()
pyplot.contour(X, Y, p.data[0], cmap=cm.viridis)
pyplot.quiver(X[::2, ::2], Y[::2, ::2], u.data[0, ::2, ::2], v.data[0, ::2, ::2])
pyplot.xlabel('X')
pyplot.ylabel('Y')
pyplot.savefig('cavity_flow_devito.png', dpi=100, bbox_inches='tight')
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
# Again, check results with Marchi et al 2009.
fig = pyplot.figure(figsize=(12, 6))
ax1 = fig.add_subplot(121)
ax1.plot(u.data[0,int(grid.shape[0]/2),:],y_coord[:])
ax1.plot(Marchi_Re10_u[:,1],Marchi_Re10_u[:,0],'ro')
ax1.set_xlabel('$u$')
ax1.set_ylabel('$y$')
ax1 = fig.add_subplot(122)
ax1.plot(x_coord[:],v.data[0,:,int(grid.shape[0]/2)])
ax1.plot(Marchi_Re10_v[:,0],Marchi_Re10_v[:,1],'ro')
ax1.set_xlabel('$x$')
ax1.set_ylabel('$v$')
pyplot.savefig('cavity_flow_devito_comparison.png', dpi=100, bbox_inches='tight')
pyplot.show()
