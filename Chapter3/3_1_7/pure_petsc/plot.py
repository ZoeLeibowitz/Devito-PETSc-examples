import numpy as np

import matplotlib.pyplot as plt


# ./fish -ksp_converged_reason -ksp_type cg -ksp_rtol 1e-13 -pc_type none

Lx = np.float64(16.)

n_values = [900, 950, 1000, 1050, 1100, 1150, 1200, 1250, 1300]

h = np.array([Lx/(n-1) for n in n_values])
infinity_norms = [
     7.661e-11,
     5.895e-11,
     4.514e-11,
     3.409e-11,
     2.515e-11,
     1.769e-11,
     1.139e-11,
     5.946e-12,
     1.224e-12
]


slope, intercept = np.polyfit(np.log(h), np.log(infinity_norms), 1)


# infinity_norms = [3.004e-09,
#                   ]
plt.figure(figsize=(6, 5))
plt.loglog(h, infinity_norms, 'o-', label=f'Observed rate ≈ {slope:.3f}', color='orange')
plt.loglog(
    h, np.exp(intercept) * h**4,
    'k--',
    label=r'Reference slope $O(h^4)$'
)
plt.xlabel(r'Grid spacing h')
plt.ylabel(r'$\infty$-norm error')
plt.title('Convergence Plot')
plt.legend()
plt.tight_layout()
plt.savefig("pure_petsc.png", dpi=200)
plt.show()