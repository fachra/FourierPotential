#!/usr/bin/env python
import FPM.geom as geo
from FPM.solver import FPM, save
import time


if __name__ == "__main__":
    # stl_path = 'neuron_models/03b_spindle4aACC.stl'
    # stl = geo.stl_3Dto2D(stl_path)

    # c = geo.Circle(0, 0, 1, dist_max=0.01)
    e = geo.Ellipse(10, 10, 2, 10, dist_max=0.1)
    # m = geo.Model2D(stl, refine=True)

    parameters = {
        # geometry model
        'geom': (e,),
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
        'gpu': [0, 1, 2],  # torch.device("cpu")
        'freq_memory_ratio': 30
    }
    circle = FPM(**parameters)
    s = time.time()
    circle.run()
    e = time.time()
    print(e-s)
    save(circle, '/media/chengran/sssd/FP/ellipse_6.pt')
