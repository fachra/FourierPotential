#!/usr/bin/env python
from fpm import Circle, Solver, save
import time


if __name__ == "__main__":
    # define a circle with radius 1 um
    c = Circle(0, 0, 1, dist_max=0.01)

    parameters = {
        # geometry model
        'geom': (c,),
        # dMRI
        'diffusivity': 2e-3,
        'delta': (1000, 5000),
        'b-values': [1000, 2000, 3000],
        'n_direction': 2,
        # simulation
        'dt': 2,  # [us]
        'n_eta': 2,
        'frequency_resolution': 0.1,  # [um^-1]
        'frequency_max': 10,  # [um^-1]
        # data type and device
        'gpu': [0, 2, 3],
        'freq_memory_ratio': 30
    }
    sim = Solver(**parameters)
    s = time.time()
    sim.run()
    e = time.time()
    print(f"Elapsed time: {e-s} s.")
    save(sim, 'circle_sim.pt')
