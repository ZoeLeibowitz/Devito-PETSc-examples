# The explicit (FTCS) heat equation in Devito (no petscsolve)
import numpy as np
from devito import Grid, TimeFunction, Eq, Operator, solve

nx, ny = 41, 41
grid = Grid(shape=(nx, ny), extent=(1.0, 1.0))

u = TimeFunction(name='u', grid=grid, space_order=2, time_order=1)

alpha = 0.1
eqn = Eq(u.dt, alpha**2 * (u.dx2 + u.dy2))

stencil = Eq(u.forward, solve(eqn, u.forward), subdomain=grid.interior)

x_coords = np.linspace(0, 1.0, nx)
y_coords = np.linspace(0, 1.0, ny)
X, Y = np.meshgrid(x_coords, y_coords, indexing='ij')
u.data[0] = np.sin(np.pi*X) * np.sin(np.pi*Y)

dt, nt = 0.01, 200
op = Operator(stencil)
op.apply(dt=dt, time_M=nt-1)
print(op.ccode)

# Check against the exact solution at T = nt*dt (the default buffer has two levels)
T = nt*dt
exact = np.exp(-2*np.pi**2*alpha**2*T)*np.sin(np.pi*X)*np.sin(np.pi*Y)
error = np.abs(u.data[nt % 2] - exact).max()
print('T = %g, max |u - u_exact| = %.3e, max |u_exact| = %.3e' % (T, error, exact.max()))
