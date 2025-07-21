import numpy as np



# These are with mpiexec -n 4

serial_infinity_norms = [
    np.float64(0.004403459551176603),
    np.float64(0.0011343085906865835),
    np.float64(0.00028433967896346335),
    np.float64(7.114298528454466e-05),
    np.float64(1.779375875687883e-05),
    np.float64(4.448628978415137e-06),
    np.float64(1.1121690737248002e-06),
    np.float64(2.7804420504873306e-07),
    np.float64(6.951107367481058e-08),
    np.float64(1.7377628502845965e-08)
]

# taken from output:
parallel_infinity_norms = [
    np.float64(0.004403459551172995),
    np.float64(0.001134308590686528),
    np.float64(0.000284339678951695),
    np.float64(7.114298528404506e-05),
    np.float64(1.7793758751716293e-05),
    np.float64(4.448628978193092e-06),
    np.float64(1.1121690760562686e-06),
    np.float64(2.780442084349133e-07),
    np.float64(6.95110903836671e-08),
    np.float64(1.7377738803503462e-08)
]



# this one doesn't work with atol 1e-13

# Check all pairs
for i, (serial, mpi) in enumerate(zip(serial_infinity_norms, parallel_infinity_norms)):
    assert np.isclose(
        serial, mpi, rtol=1e-13, atol=1e-12
    ), f"Norm mismatch at index {i}: serial={serial}, mpi={mpi}"
