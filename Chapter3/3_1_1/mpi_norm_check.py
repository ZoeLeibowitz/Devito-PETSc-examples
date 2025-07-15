import numpy as np

serial_infinity_norms = [
    np.float64(0.00027376999356110154),
    np.float64(6.882733511814898e-05),
    np.float64(1.7233852468212518e-05),
    np.float64(4.309849699124513e-06),
    np.float64(1.0775847982813502e-06),
    np.float64(2.693994516356213e-07),
    np.float64(6.735061530704911e-08),
    np.float64(1.6837652383472346e-08),
    np.float64(4.209424364631786e-09),
    np.float64(1.0523915072724321e-09),
    np.float64(2.6304691758127774e-10)
]

# Taken from output:
devito_mpi_norms = [
    np.float64(0.00027376999356154563),
    np.float64(6.882733511570649e-05),
    np.float64(1.723385246643616e-05),
    np.float64(4.309849703121316e-06),
    np.float64(1.0775847980593056e-06),
    np.float64(2.693994520797105e-07),
    np.float64(6.735060997797859e-08),
    np.float64(1.6837660377078123e-08),
    np.float64(4.209419257605873e-09),
    np.float64(1.0523486526636816e-09),
    np.float64(2.6310287282171885e-10)
]



# Check all pairs
for i, (serial, mpi) in enumerate(zip(serial_infinity_norms, devito_mpi_norms)):
    assert np.isclose(
        serial, mpi, rtol=1e-13, atol=1e-13
    ), f"Norm mismatch at index {i}: serial={serial}, mpi={mpi}"