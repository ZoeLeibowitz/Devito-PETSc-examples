import numpy as np


serial_infinity_norms = [
    np.float64(0.00023550633478208738),
    np.float64(6.0292711047793546e-05),
    np.float64(1.5215275331215139e-05),
    np.float64(3.811616776872029e-06),
    np.float64(9.532971296799531e-07),
    np.float64(2.3835349827194818e-07)
]


# taken from output:
parallel_infinity_norms = [
    np.float64(0.00023550633478253147),
    np.float64(6.0292711047793546e-05),
    np.float64(1.5215275331659228e-05),
    np.float64(3.8116167782042965e-06),
    np.float64(9.532971296799531e-07),
    np.float64(2.3835350182466186e-07)
]


# abs(a-b) <= atol + rtol * abs(b)

# Check all pairs
for i, (serial, mpi) in enumerate(zip(serial_infinity_norms, parallel_infinity_norms)):
    assert np.isclose(
        serial, mpi, rtol=1e-13, atol=1e-13
    ), f"Norm mismatch at index {i}: serial={serial}, mpi={mpi}"