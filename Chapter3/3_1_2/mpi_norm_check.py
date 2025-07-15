import numpy as np

serial_infinity_norms = [
    np.float64(8.59058043676253e-05),
    np.float64(2.183883155537636e-05),
    np.float64(5.477518565166761e-06),
    np.float64(1.371033316877046e-06),
    np.float64(3.428173114272681e-07),
    np.float64(8.571514120703227e-08),
    np.float64(2.1436735497815107e-08),
    np.float64(5.381981305063732e-09)
]

# taken from output:
parallel_infinity_norms = [
    np.float64(8.590580436740325e-05),
    np.float64(2.1838831555598404e-05),
    np.float64(5.477518564056538e-06),
    np.float64(1.371033316877046e-06),
    np.float64(3.428173112052235e-07),
    np.float64(8.571514054089846e-08),
    np.float64(2.1436737274171946e-08),
    np.float64(5.3819804168853125e-09)
]


# abs(a-b) <= atol + rtol * abs(b)

# Check all pairs
for i, (serial, mpi) in enumerate(zip(serial_infinity_norms, parallel_infinity_norms)):
    assert np.isclose(
        serial, mpi, rtol=1e-13, atol=1e-13
    ), f"Norm mismatch at index {i}: serial={serial}, mpi={mpi}"