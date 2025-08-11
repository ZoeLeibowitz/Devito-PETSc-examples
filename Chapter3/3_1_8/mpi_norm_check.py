import numpy as np



# These are with mpiexec -n 4
# and with a non zero initial guess

serial_infinity_norms = [
    np.float64(0.0032844455618405988),
    np.float64(0.00021215401391527777),
    np.float64(1.3355538635462239e-05),
    np.float64(8.361748755625342e-07),
    np.float64(5.2279591766790645e-08),
    np.float64(3.252259794805923e-09)
]


# taken from output:
parallel_infinity_norms = [
    np.float64(0.0032844455618405988),
    np.float64(0.00021215401391483368),
    np.float64(1.3355538635462239e-05),
    np.float64(8.361748773388911e-07),
    np.float64(5.2279590434523016e-08),
    np.float64(3.2522584625382933e-09)
]


# Check all pairs
for i, (serial, mpi) in enumerate(zip(serial_infinity_norms, parallel_infinity_norms)):
    assert np.isclose(
        serial, mpi, rtol=1e-13, atol=1e-13
    ), f"Norm mismatch at index {i}: serial={serial}, mpi={mpi}"