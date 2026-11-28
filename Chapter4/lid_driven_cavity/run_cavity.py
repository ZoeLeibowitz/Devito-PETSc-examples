# Run the lid driven cavity

# python run_cavity.py RE N

# e.g
# python run_cavity.py 100 65
# python run_cavity.py 400 65
# python run_cavity.py 1000 97

import json
import sys

import numpy as np
from scipy.interpolate import RectBivariateSpline
from scipy.optimize import minimize

from solver import make_solver


# final time for each Reynolds number
T = {100: 40., 400: 80., 1000: 200.}
TOL = 1e-7


def run(re_val, n):

    # Run the solver and return the fields and primary vortex
    solve = make_solver(nx=n, ny=n, conservative=True)
    x, y, u, v, omega, psi, rank, u_stag, v_stag, res = solve(re_val, t_end=T[re_val],
                                                              fixed=True)
    if rank != 0:
        return None
    if res > TOL:
        print(f'Re={re_val}, n={n} not at steady state, residual {res} > {TOL}')
    fields = dict(x=x, y=y, u=np.asarray(u), v=np.asarray(v), omega=np.asarray(omega),
                  psi=np.asarray(psi), u_stag=np.asarray(u_stag), v_stag=np.asarray(v_stag))
    info = dict(re=re_val, n=n, T=T[re_val], residual=float(res),
                **primary_vortex(x, y, fields['psi'], fields['omega']))
    return fields, info


def primary_vortex(x, y, psi, omega):

    # centre of the primary vortex
    psi_s = RectBivariateSpline(x, y, psi, kx=3, ky=3)
    omega_s = RectBivariateSpline(x, y, omega, kx=3, ky=3)
    i, j = np.unravel_index(np.argmin(psi), psi.shape)
    h = x[1] - x[0]
    fit = minimize(lambda p: psi_s(p[0], p[1])[0, 0], [x[i], y[j]], method='L-BFGS-B',
                   bounds=[(x[i] - h, x[i] + h), (y[j] - h, y[j] + h)],
                   options={'ftol': 1e-15, 'gtol': 1e-12})
    xc, yc = fit.x
    return dict(psi=float(psi_s(xc, yc)[0, 0]), omega=float(omega_s(xc, yc)[0, 0]),
                x=float(xc), y=float(yc))


if __name__ == '__main__':
    re_val, n = int(sys.argv[1]), int(sys.argv[2])
    out = run(re_val, n)
    if out is not None:
        fields, info = out
        np.savez(f'results/re{re_val}_{n}.npz', **fields)
        with open(f'results/re{re_val}_{n}.json', 'w') as f:
            json.dump(info, f, indent=1)
        print(info)
