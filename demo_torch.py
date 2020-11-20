import time
import torch
import numpy as np
import FPM_torch.geom as geo
import FPM_torch.solver as fpm

stl_path = 'neuron_models/03b_spindle4aACC.stl'
stl = geo.load_stl(stl_path)

c = geo.Circle(0, 0, 5)
e = geo.Ellipse(10, 10, 4, 8)
m = geo.Model2D(stl, n_points=500, refine=False)
c.plot()
e.plot()
m.plot()

q = np.array([[0, 0.1, 0.2, 0.3],
              [0, 0,   0,   0]])
parameters = {
    # geometry model
    'models': (c, e, m),
    # dMRI
    'D0': 1e-3,
    'delta': (2500, 5000),
    'q-vectors': q,
    # simulation
    'dt': 5,  # [us]
    'n_eta': 2,
    'frequency_resolution': 0.2,  # [um^-1]
    'frequency_max': 1,  # [um^-1]
    # data type and device
    'ctype': torch.complex64,
    'device': torch.device("cuda:0")  # torch.device("cpu")
}

s = time.time()
time_interval, signal = fpm.solve(**parameters)
e = time.time()
print(e-s)
