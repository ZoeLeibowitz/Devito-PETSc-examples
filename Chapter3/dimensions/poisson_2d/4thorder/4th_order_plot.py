import numpy as np
import matplotlib.pyplot as plt


# ./fish -ksp_rtol 1e-13

# n_values = [21, 31, 41, 51, 61, 71, 81, 91, 101, 111, 121, 131, 141, 151, 161, 171, 181, 191, 201, 211, 221, 321]
# n_values = [21, 31, 41, 51, 61, 71, 81]
# n_values = [45, 55, 65, 75, 85]
n_values = [15, 25, 35, 45]
dx = np.array([1.0/(n-1) for n in n_values])


infinity_norms = [
    1.3446531887950641e-08,
    1.832979545923763e-09,
    4.875859715980368e-10,
    1.8573098614638184e-10,
    # 8.98812135829985e-11,
    # 5.4303228580465657e-11,
]


# # petsc ones
# infinity_norms = [
#     1.790e-10,
#     8.059e-11,
#     4.147e-11,
# ]


    
# ]

slope, intercept = np.polyfit(np.log(dx), np.log(infinity_norms), 1)
print(infinity_norms)



# Convergence Plot
plt.figure(figsize=(6, 5))
plt.loglog(dx, infinity_norms, 'o-', label=f'Observed rate ≈ {slope:.3f}', color='orange')
plt.loglog(
    dx, np.exp(intercept) * dx**4,
    'k--',
    label=r'Reference slope $O(h^4)$'
)
plt.xlabel(r'Grid spacing h')
plt.ylabel(r'$\infty$-norm error')
plt.title('Convergence Plot')
plt.legend()
plt.tight_layout()
plt.savefig("4thorderconvergence.png", dpi=200)
plt.show()