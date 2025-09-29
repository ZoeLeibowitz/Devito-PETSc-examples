import numpy as np
from devito import (Function, TimeFunction, cos, sin, solve,
                    Eq, Operator, configuration, norm)
from examples.seismic import TimeAxis, RickerSource, Receiver, demo_model
from matplotlib import pyplot as plt


# NBVAL_IGNORE_OUTPUT   

shape   = (101,101) # 101x101 grid
spacing = (10.,10.) # spacing of 10 meters
origin  = (0.,0.)  
nbl = 20  # number of pad points

model = demo_model('layers-tti', spacing=spacing, space_order=8,
                   shape=shape, nbl=nbl, nlayers=1)

# initialize Thomsem parameters to those used in Mu et al., (2020)
model.update('vp', np.ones(shape)*3.6) # km/s
model.update('epsilon', np.ones(shape)*0.23)
model.update('delta', np.ones(shape)*0.17)
model.update('theta', np.ones(shape)*(45.*(np.pi/180.))) # radians


# Get symbols from model
theta = model.theta
delta = model.delta
epsilon = model.epsilon
m = model.m

# Use trigonometric functions from Devito
costheta  = cos(theta)
sintheta  = sin(theta)
cos2theta = cos(2*theta)
sin2theta = sin(2*theta)
sin4theta = sin(4*theta)


# NBVAL_IGNORE_OUTPUT

# Values used to compute the time sampling
epsilonmax = np.max(np.abs(epsilon.data[:]))
deltamax = np.max(np.abs(delta.data[:]))
etamax = max(epsilonmax, deltamax)
vmax = model._max_vp
max_cos_sin = np.amax(np.abs(np.cos(theta.data[:]) - np.sin(theta.data[:])))
dvalue = min(spacing)


# Compute the dt and set time range
t0 = 0.   #  Simulation time start
tn = 150. #  Simulation time end (0.15 second = 150 msec)
dt = (dvalue/(np.pi*vmax))*np.sqrt(1/(1+etamax*(max_cos_sin)**2)) # eq. above (cell 3)
time_range = TimeAxis(start=t0,stop=tn,step=dt)
print("time_range; ", time_range)



# NBVAL_IGNORE_OUTPUT

# time stepping 
p = TimeFunction(name="p", grid=model.grid, time_order=2, space_order=2)
q = Function(name="q", grid=model.grid, space_order=8)


# Main equations
term1_p = (1 + 2*delta*(sintheta**2)*(costheta**2) + 2*epsilon*costheta**4)*q.dx4
term2_p = (1 + 2*delta*(sintheta**2)*(costheta**2) + 2*epsilon*sintheta**4)*q.dy4
term3_p = (2-delta*(sin2theta)**2 + 3*epsilon*(sin2theta)**2 + 2*delta*(cos2theta)**2)*((q.dy2).dx2)
term4_p = ( delta*sin4theta - 4*epsilon*sin2theta*costheta**2)*((q.dy).dx3)
term5_p = (-delta*sin4theta - 4*epsilon*sin2theta*sintheta**2)*((q.dy3).dx)

stencil_p = solve(m*p.dt2 - (term1_p + term2_p + term3_p + term4_p + term5_p) + model.damp*p.dt, p.forward)
update_p = Eq(p.forward, stencil_p)

# Poisson eq. (following notebook 6 from CFD examples)
b = Function(name='b', grid=model.grid, space_order=2)
pp = TimeFunction(name='pp', grid=model.grid, space_order=2)

# Create stencil and boundary condition expressions
x, z = model.grid.dimensions
t = model.grid.stepping_dim

update_q = Eq( pp[t+1,x,z],((pp[t,x+1,z] + pp[t,x-1,z])*z.spacing**2 + (pp[t,x,z+1] + pp[t,x,z-1])*x.spacing**2 -
         b[x,z]*x.spacing**2*z.spacing**2) / (2*(x.spacing**2 + z.spacing**2)))

# update_q = Eq( pp[t+1,x,z],damping*(((pp[t,x+1,z] + pp[t,x-1,z])*z.spacing**2 + (pp[t,x,z+1] + pp[t,x,z-1])*x.spacing**2 -
#          b[x,z]*x.spacing**2*z.spacing**2) / (2*(x.spacing**2 + z.spacing**2))))

bc = [Eq(pp[t+1,x, 0], 0.)]
bc += [Eq(pp[t+1,x, shape[1]+2*nbl-1], 0.)]
bc += [Eq(pp[t+1,0, z], 0.)]
bc += [Eq(pp[t+1,shape[0]-1+2*nbl, z], 0.)]


# from IPython import embed; embed()

# set source and receivers
src = RickerSource(name='src',grid=model.grid,f0=0.02,npoint=1,time_range=time_range)
src.coordinates.data[:,0] = model.domain_size[0]* .5
src.coordinates.data[:,1] = model.domain_size[0]* .5
# Define the source injection
src_term = src.inject(field=p.forward,expr=src * dt**2 / m)

rec  = Receiver(name='rec',grid=model.grid,npoint=shape[0],time_range=time_range)
rec.coordinates.data[:, 0] = np.linspace(model.origin[0],model.domain_size[0], num=model.shape[0])
rec.coordinates.data[:, 1] = 2*spacing[1]
# Create interpolation expression for receivers
rec_term = rec.interpolate(expr=p.forward)

# Operators
optime=Operator([update_p] + src_term + rec_term)
oppres=Operator([update_q] + bc)


print(optime.ccode)

# you can print the generated code for both operators by typing print(optime) and print(oppres)


# NBVAL_IGNORE_OUTPUT
psave =np.empty ((time_range.num,model.grid.shape[0],model.grid.shape[1]))
niter_poisson = 8000


# from IPython import embed; embed()
# This is the time loop.
for step in range(0,time_range.num-2):
    q.data[:,:]=pp.data[(niter_poisson+1)%2,:,:]
    optime(time_m=step, time_M=step, dt=dt)
    pp.data[:,:]=0.
    b.data[:,:]=p.data[(step+1)%3,:,:]
    oppres(time_M = niter_poisson)
    psave[step,:,:]=p.data[(step+1)%3,:,:]


# Some useful definitions for plotting if nbl is set to any other value than zero
nxpad,nzpad = shape[0] + 2 * nbl, shape[1] + 2 * nbl
shape_pad   = np.array(shape) + 2 * nbl
origin_pad  = tuple([o - s*nbl for o, s in zip(origin, spacing)])
extent_pad  = tuple([s*(n-1) for s, n in zip(spacing, shape_pad)])



# NBVAL_IGNORE_OUTPUT

# Note: flip sense of second dimension to make the plot positive downwards
plt_extent = [origin_pad[0], origin_pad[0] + extent_pad[0],
              origin_pad[1] + extent_pad[1], origin_pad[1]]

# Plot the wavefields, each normalized to scaled maximum of last time step
kt = (time_range.num - 2) - 1
# amax = 0.05 * np.max(np.abs(psave[kt,:,:]))
amax = np.max(np.abs(psave[kt,:,:]))
# amax = 0.05 * np.max(np.abs(p.data[kt,:,:]))

nsnaps = 10
factor = round(time_range.num/nsnaps)

fig, axes = plt.subplots(2, 5, figsize=(18, 7), sharex=True)
fig.suptitle("Snapshots", size=14)
for count, ax in enumerate(axes.ravel()):
    snapshot = factor*count
    # ax.imshow(np.transpose(psave[snapshot,:,:]), cmap="seismic",
    #            vmin=-amax, vmax=+amax, extent=plt_extent)
    im = ax.imshow(np.transpose(psave[snapshot,nbl:-nbl,nbl:-nbl]), cmap="seismic",
               vmin=-amax, vmax=+amax, extent=plt_extent)
    ax.plot(model.domain_size[0]* .5, model.domain_size[1]* .5, \
         'red', linestyle='None', marker='*', markersize=8, label="Source")
    ax.grid()
    ax.tick_params('both', length=2, width=0.5, which='major',labelsize=10)
    ax.set_title("Wavefield at t=%.2fms" % (factor*count*dt),fontsize=10)

    # add individual colorbar for this subplot
    fig.colorbar(im, ax=ax, orientation='vertical', fraction=0.046, pad=0.04)
for ax in axes[1, :]:
    ax.set_xlabel("X Coordinate (m)",fontsize=10)
for ax in axes[:, 0]:
    ax.set_ylabel("Z Coordinate (m)",fontsize=10)


plt.savefig(f'tti_pure_qp_original_with_damping.png', dpi=300)

# print norm of p
print("norm of p: ", norm(p))