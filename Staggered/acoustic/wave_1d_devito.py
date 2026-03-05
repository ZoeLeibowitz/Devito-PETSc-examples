from devito import *
from examples.seismic.source import DGaussSource, TimeAxis
from examples.seismic import plot_image
import numpy as np

from sympy import init_printing, latex
init_printing(use_latex='mathjax')


# Initial grid: 1km x 1km, with spacing 100m
extent = (2000., 2000.)
shape = (81, 81)
x = SpaceDimension(name='x', spacing=Constant(name='h_x', value=extent[0]/(shape[0]-1)))
z = SpaceDimension(name='z', spacing=Constant(name='h_z', value=extent[1]/(shape[1]-1)))
grid = Grid(extent=extent, shape=shape, dimensions=(x, z))



# Timestep size from Eq. 7 with V_p=6000. and dx=100
t0, tn = 0., 200.
dt = 1e2*(1. / np.sqrt(2.)) / 60.
time_range = TimeAxis(start=t0, stop=tn, step=dt)

src = DGaussSource(name='src', grid=grid, f0=0.01, time_range=time_range, a=0.004)
src.coordinates.data[:] = [1000., 1000.]



# Now we create the velocity and pressure fields
p = TimeFunction(name='p', grid=grid, staggered=NODE, space_order=2, time_order=1)
v = VectorTimeFunction(name='v', grid=grid, space_order=2, time_order=1)



from devito.finite_differences.operators import div, grad
t = grid.stepping_dim
time = grid.time_dim

# We need some initial conditions
V_p = 4.0
density = 1.

ro = 1/density
l2m = V_p*V_p*density

# The source injection term
src_p = src.inject(field=p.forward, expr=src)

# 2nd order acoustic according to fdelmoc
u_v_2 = Eq(v.forward, solve(v.dt + ro * grad(p), v.forward))
# use leap - frogging (that is why we use v.forward here)
u_p_2 = Eq(p.forward, solve(p.dt + l2m * div(v.forward), p.forward))


op_2 = Operator([u_v_2, u_p_2] + src_p)
print(op_2.ccode)



print(src.time_range.num-1)
# Propagate the source
op_2(time=1, dt=dt)

# print(p.data[:])
# norm_p = norm(p)
# # print(norm_p)
# assert np.isclose(norm_p, .35098, atol=1e-4, rtol=0)