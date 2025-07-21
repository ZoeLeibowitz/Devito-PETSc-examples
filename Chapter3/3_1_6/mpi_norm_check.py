import numpy as np



# These are with mpiexec -n 4

serial_infinity_norms = [
    np.float64(0.05233135619374241),
    np.float64(0.012786704148035177),
    np.float64(0.003178577723158549),
    np.float64(0.0007935197233648328),
    np.float64(0.00019830981110757762),
    np.float64(4.9573167596905776e-05),
    np.float64(1.239310152634232e-05),
    np.float64(3.098334872220221e-06),
    np.float64(7.747051387063664e-07),
    np.float64(1.9374143644945718e-07)
]

# taken from output:
parallel_infinity_norms = [
    np.float64(0.05233135619374241),
    np.float64(0.012786704148035843),
    np.float64(0.0031785777231521095),
    np.float64(0.0007935197233641667),
    np.float64(0.00019830981110979806),
    np.float64(4.9573167587135814e-05),
    np.float64(1.2393101500807191e-05),
    np.float64(3.0983348846547187e-06),
    np.float64(7.747051744555478e-07),
    np.float64(1.9374138493510884e-07)
]



# Check all pairs
for i, (serial, mpi) in enumerate(zip(serial_infinity_norms, parallel_infinity_norms)):
    assert np.isclose(
        serial, mpi, rtol=1e-13, atol=1e-13
    ), f"Norm mismatch at index {i}: serial={serial}, mpi={mpi}"