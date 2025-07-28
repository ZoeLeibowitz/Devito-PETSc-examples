import numpy as np



# These are with mpiexec -n 4
# and with a non zero initial guess

serial_infinity_norms = [
    np.float64(0.0011776102349387307),
    np.float64(0.000295115683632885),
    np.float64(7.382361406488291e-05),
    np.float64(1.8460818343501995e-05),
    np.float64(4.613968733857554e-06),
    np.float64(1.149423846769082e-06)
]


# taken from output:
parallel_infinity_norms = [
    np.float64(0.0011776102342946904),
    np.float64(0.0002951156837122104),
    np.float64(7.382360882207673e-05),
    np.float64(1.8460819333987466e-05),
    np.float64(4.613965593092129e-06),
    np.float64(1.1494330676153908e-06)
]



# Ran the solver with tolernace 1e-11, so absolute tolerance has to be larger than previous examples where the solver
# tolerance was set to 1e-12

# Check all pairs
for i, (serial, mpi) in enumerate(zip(serial_infinity_norms, parallel_infinity_norms)):
    assert np.isclose(
        serial, mpi, rtol=1e-13, atol=1e-11
    ), f"Norm mismatch at index {i}: serial={serial}, mpi={mpi}"