import os
import numpy as np

from devito import (Grid, Function, Eq, Operator, switchconfig,
                    configuration, SubDomain, norm)
from devito.symbolics import retrieve_functions, INT

from devito.petsc import PETScSolve, EssentialBC
from devito.petsc.initialize import PetscInitialize

import matplotlib.pyplot as plt


# This is with setting the nullspace and removing it from initial guess (initial guess of 0.001)
n_values = [2**k + 1 for k in range(3, 11)]
h = np.array([1.0/(n-1) for n in n_values])

infinity_norms = [6.6029649120150413210695e-02, 1.6455766606729760326289e-02,
                  4.1401939758374250999395e-03, 1.0404542658066784355242e-03,
                  2.6092642011366073973022e-04, 6.5341950621800037879439e-05,
                  1.6349832806517028416238e-05, 4.0892860089236648946098e-06]

slope, intercept = np.polyfit(np.log(h), np.log(infinity_norms), 1)

# Plot
plt.figure(figsize=(6, 5))
plt.loglog(h, infinity_norms, 'o-', label=f'Observed rate ≈ {slope:.3f}', color='orange')
plt.loglog(
    h, np.exp(intercept) * h**2,
    'k--',
    label=r'Reference slope $O(h^2)$'
)
plt.xlabel(r'Grid spacing h')
plt.ylabel(r'$\infty$-norm error')
plt.title('Convergence Plot')
plt.legend()
plt.tight_layout()
plt.savefig("tmp.png", dpi=200)
plt.show()