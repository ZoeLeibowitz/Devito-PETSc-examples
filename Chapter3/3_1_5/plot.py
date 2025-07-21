import os
import numpy as np

from devito import (Grid, Function, Eq, Operator, switchconfig,
                    configuration, SubDomain, norm)
from devito.symbolics import retrieve_functions, INT

from devito.petsc import PETScSolve, EssentialBC
from devito.petsc.initialize import PetscInitialize

import matplotlib.pyplot as plt


# This is with setting the nullspace and removing it from initial guess (initial guess of 0.001)
# using ksp rtol as 1e-10
# 9, 17, 33, 65, 129, 257, 513, 1025, 2049, 4097, 8193
n_values = [2**k + 1 for k in range(3, 14)]
h = np.array([1.0/(n-1) for n in n_values])


# using 1e-10
infinity_norms = [
    6.6029649120150413210695e-02,
    1.6455766606729760326289e-02,
    4.1401939758374250999395e-03,
    1.0404542658066784355242e-03,
    2.6092642011366073973022e-04,
    6.5341950621800037879439e-05,
    1.6349832804074537762062e-05,
    4.0892860095897987093849e-06,
    1.0225522468765291250747e-06,
    2.5566717609670774891129e-07,
    6.3918855763844817374775e-08,
]


iters = [
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    2,
    4,
]


# info without setting a nullspace (initial guess zero -> seems to work)
# infinity_norms = [
#     5.3029287545514947055381e-02,
#     1.2950746721879236034169e-02,
#     3.2189644400790751177510e-03,
#     8.0357767937244695133359e-04,
#     2.0082180970515395301845e-04,
#     5.0200915925113775983846e-05,
#     1.2549945472839496574124e-05,
#     3.1374686646490346220162e-06,
#     7.8436603945242211466393e-07,
#     1.9609143908638770881225e-07,
#     4.9022593096026412240462e-08
# ]  


# iters = [
#     1,
#     1,
#     1,
#     1,
#     1,
#     1,
#     1,
#     1,
#     1,
#     2,
#     4
# ]



# info without setting the nullspace but setting an initial guess of 0.001 (not zero)
# this is clearly wrong

# infinity_norms = [
#     5.4029287545514836921257e-02,
#     1.3950746721879125900045e-02,
#     4.2189644400789649836270e-03,
#     1.8035776793723368172095e-03,
#     1.2008218097050438188944e-03,
#     1.0502009159250036418598e-03,
#     1.0125499454727293624501e-03,
#     1.0031374686645389004980e-03,
#     1.0007843660393422879906e-03,
#     1.0001960914389762535848e-03,
#     1.0000490225929858922882e-03,

# ]

# iters = [
#     1,
#     1,
#     1,
#     1,
#     1,
#     1,
#     1,
#     1,
#     1,
#     2,
#     4
# ]

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




# testing 2d_poisson_symmetric.c -> pure petsc code
# with zero initial guess -> should be 2nd order accuracy
# use rtol 1e-12
# USING KSPCG since we maintin symmetry
n_values = [9, 17, 33, 65, 129, 257, 513, 1025, 2049]
h = np.array([1.0/(n-1) for n in n_values])
norms = [
    6.6029649120150635255300e-02,
    1.6455766606752408875991e-02,
    4.1401939758805017532950e-03,
    1.0404542658237758701034e-03,
    2.6092642011144029368097e-04,
    6.5341950618025279595713e-05,
    1.6349832811846098934438e-05,
    4.0892860380115081397889e-06,
]



# # testing 2d_poisson_symmetric.c -> pure petsc code
# # with zero initial guess -> should be 2nd order accurate
# # use rtol 1e-12
# # USING KSPCG since we maintin symmetry
# # testing the code WITHOUT setting the nullspace - should diverge probably
# n_values = [9, 17, 33, 65, 129, 257, 513, 1025, 2049]
# h = np.array([1.0/(n-1) for n in n_values])
# norms = [
#     6.6029649120150635255300e-02,
#     1.6455766606752408875991e-02,
#     4.1401939758805017532950e-03,
#     1.0404542658237758701034e-03,
#     2.6092642011144029368097e-04,
#     6.5341950618025279595713e-05,
#     1.6349832811846098934438e-05,
#     4.0892860380115081397889e-06,
#     1.0225521651641145126632e-06
# ]



# testing 2d_poisson_symmetric.c -> pure petsc code
# with zero initial guess -> should be 2nd order accurate
# use rtol 1e-12
# USING KSPCG since we maintin symmetry
# testing the code with setting the nullspace
# if i add a constant to the RHS via VecShift, then it should still converge to the exact solution
n_values = [9, 17, 33, 65, 129, 257, 513, 1025, 2049]
h = np.array([1.0/(n-1) for n in n_values])
norms = [
    6.6029649120150635255300e-02,
    1.6455766606752408875991e-02,
    4.1401939758809458425048e-03,
    1.0404542658224436024739e-03,
    2.6092642011366073973022e-04,
    6.5341950615804833546463e-05,
    1.6349832813400411168914e-05,,
    4.0892860184715829063862e-06,

]


slope, intercept = np.polyfit(np.log(h), np.log(norms), 1)

# Plot
plt.figure(figsize=(6, 5))
plt.loglog(h, norms, 'o-', label=f'Observed rate ≈ {slope:.3f}', color='orange')
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
plt.savefig("tmp2.png", dpi=200)
plt.show()