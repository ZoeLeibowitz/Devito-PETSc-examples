# Centreline velocities against Ghia et al. (1982) and Marchi et al. (2009)
# python plot_centrelines.py RE N [RE N ...]
# e.g python plot_centrelines.py 100 65

# ... python plot_centrelines.py 100 65 400 65 1000 97

import sys

import matplotlib.pyplot as plt
import numpy as np

from benchmarks import k_marchi, u_ghia, u_marchi, v_ghia, v_marchi, x_ghia, y_ghia

args = sys.argv[1:]
cases = list(zip(map(int, args[::2]), map(int, args[1::2])))

styles = {100: dict(linestyle=(0, (1, 1)), linewidth=2),
          400: dict(linestyle='--'),
          1000: dict(linestyle='-')}

for name in ('u', 'v'):
    plt.figure(figsize=(6, 5))
    for re_val, n in cases:
        d = np.load(f'results/re{re_val}_{n}.npz')
        mid = (n - 1)//2
        if name == 'u':
            plt.plot(d['u'][mid, :], d['y'], color='k', **styles[re_val],
                     label=rf'$\mathrm{{Re}} = {re_val}$, ${n} \times {n}$')
            plt.plot(u_ghia[re_val], y_ghia, 'ko', fillstyle='none')
            if re_val in u_marchi:
                plt.plot(u_marchi[re_val], k_marchi, 'kx')
        else:
            plt.plot(d['x'], d['v'][:, mid], color='k', **styles[re_val],
                     label=rf'$\mathrm{{Re}} = {re_val}$, ${n} \times {n}$')
            plt.plot(x_ghia, v_ghia[re_val], 'ko', fillstyle='none')
            if re_val in v_marchi:
                plt.plot(k_marchi, v_marchi[re_val], 'kx')
    plt.plot([], [], 'ko', fillstyle='none', label=r'Ghia et al. (1982), $129 \times 129$')
    plt.plot([], [], 'kx', label=r'Marchi et al. (2009), $1024 \times 1024$')
    if name == 'u':
        plt.xlabel(r'$u(0.5, y)$')
        plt.ylabel(r'$y$')
    else:
        plt.xlabel(r'$x$')
        plt.ylabel(r'$v(x, 0.5)$')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'figures/centreline_{name}.png', dpi=200)
    plt.close()
