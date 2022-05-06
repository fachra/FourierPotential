from fpm import Circle, Solver
import numpy as np
from numpy.testing import assert_almost_equal


def test_simulation_on_one_circle():
    c = Circle(0, 0, 1, dist_max=0.1)
    parameters = {
        # geometry model
        'geom': (c,),
        # dMRI
        'diffusivity': 2e-3,
        'delta': (1000, 50000),
        'b-values': [1000, 2000],
        'n_direction': 1,
        # simulation
        'dt': 50,  # [us]
        'n_eta': 1,
        'frequency_resolution': 0.25,  # [um^-1]
        'frequency_max': 1,  # [um^-1]
        # data type and device
        'gpu': [0],
        'freq_memory_ratio': 30
    }
    sim = Solver(**parameters)
    sim.run()
    signal_fpm = sim.dMRI_signal[:, -1].real.numpy()

    # reference signals from Matrix Formalism method
    signal_ref = np.array([3.137, 3.133])
    print(signal_ref)
    print(signal_fpm)
    assert_almost_equal(signal_fpm, signal_ref, decimal=1)
