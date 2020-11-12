# %%
import matplotlib.pyplot as plt
import scipy.io as io
import time
from tool_func_cupy import *
try:
    import cupy as xp
    import numpy as np
except (ModuleNotFoundError, ImportError):
    import numpy as np
    xp = np
    print('Numpy is used.')
# import numpy as xp
# np = xp

ctype = xp.complex64  # complex float number
rtype = xp.float64
# %% load geometrical model
mat = io.loadmat('geo.mat')
boundary_points, boundary_normals, dl, curvature, region_area = get_geo(mat)
del mat
# %% dMRI settings
D0 = 1e-3  # [um^2/us]
sdelta, bdelta = 2500, 5000  # [us]
b = 3000  # [us/um^2]
q, q_square, q_max = get_q(sdelta, bdelta, b)
# %% simulation setting
dt = 5  # [us]
n_eta = 2
eta = n_eta*dt  # [us]
freq_resolution = 0.01  # [um^-1]
freq_max = 0.5  # [um^-1]

time_val = get_time(sdelta, bdelta, dt)
freq, freq_square = get_freq(freq_max, 0, freq_resolution)
# %% some constants
n_points, n_time = boundary_points.shape[1], time_val.shape[1]
n_freqs, n_q = freq.shape[1], q.shape[1]
# %%init
mu = xp.empty((n_q, n_points, n_time), dtype=ctype)
S_short = xp.empty((n_q, n_points, n_time), dtype=ctype)
K_long = xp.zeros((n_q, n_points, n_time), dtype=ctype)
fhat = xp.zeros((n_q, n_freqs, n_time), dtype=ctype)
# %% neumann data
neu_data = neumann(boundary_points, boundary_normals,
                   q, q_square, time_val, D0=D0)
# %% density mu for [0, eta=n_eta*dt] (us)
mu[:, :, :n_eta+1] = 2 * neu_data[:, :, :n_eta+1] / \
    (1 - curvature.T @ xp.sqrt(D0*time_val[:, :n_eta+1]/xp.pi))
S_short[:, :, :n_eta +
        1] = xp.sqrt(D0*time_val[:, :n_eta+1]/xp.pi)*mu[:, :, :n_eta+1]
# %% density mu for [(n_eta+1)*dt, 2*eta=2*n_eta*dt] (us)
# [n_q x n_freqs]
freq_zero_mask = (xp.abs(q_square.T-freq_square) <= 1e-20)
# [n_freqs x n_points]
fourier_bases = xp.exp(-2*xp.pi*1j*freq.T @ boundary_points)
# [1 x n_freqs]
exp_dt = xp.exp(-4*(xp.pi**2)*D0*dt*freq_square)
# [n_q x n_freqs x n_points]
exponential_term = xp.exp(-2*xp.pi*1j*(freq.T.reshape(1, -1, 2) +
                                       q.T.reshape(-1, 1, 2)) @ boundary_points)
# [n_q x n_points]
normal_derivative_coef = 4*xp.pi*1j*q.T @ boundary_normals
# [n_q x n_freqs x n_points]
coef_temp1 = normal_derivative_coef.reshape(n_q, 1, n_points)*exponential_term
# [n_q x n_freqs]
a = 4*(xp.pi**2)*D0*(q_square.T-freq_square)
# [1 x n_points]
Kshort_appro = 1-xp.sqrt(D0*eta/xp.pi)*curvature
# [n_points x n_freqs]
Klong_fourier_coef = ((2*xp.pi*1j*boundary_normals.T @ freq)
                      * fourier_bases.T.conj())*(D0*freq_resolution**2)
# %%
p_weights = xp.empty((n_q, n_freqs, 2), dtype=rtype)  # [n_q x n_freqs x 2]
p_weights[:, :, 0] = (1+(a*dt-1)*xp.exp(a*dt))/((a**2)*dt)
p_weights[:, :, 1] = (xp.exp(a*dt)-a*dt-1)/((a**2)*dt)

T2 = np.array(range(n_eta+1, 2*n_eta+1))
for it in T2:
    # [n_q x n_freqs x n_points]
    fhat_integrand = coef_temp1 * \
        func_p(it*dt, eta, dt, p_weights, freq_square,
               q_square, freq_zero_mask, curvature, D0)
    # [n_q x n_freqs x n_time]
    fhat[:, :, it] = xp.squeeze(
        fhat_integrand @ dl.T) + fhat[:, :, it-1] * exp_dt

K_long[:, :, T2] = Klong_fourier_coef @ fhat[:, :, T2]
# %% density mu for [(2*n_eta+1)*dt, bdelta-sdelta] (us)
# [1 x n_freqs]
exp_eta = xp.exp(-4*(xp.pi**2)*D0*eta*freq_square)
# [n_q x 1 x n_points]
fhat_h1_coef = (normal_derivative_coef/Kshort_appro).reshape(n_q, 1, n_points)
# [n_freqs x n_points]
fhat_h2_coef = (2*fourier_bases/Kshort_appro).reshape(1, n_freqs, n_points)

# [n_freqs]
h2_a = 4*(xp.pi**2)*D0*freq_square.reshape(-1)
# [n_freqs x 2]
h2_weights = xp.zeros((n_freqs, 2), dtype=rtype)
h2_weights[:, 0] = (1 - xp.exp(-h2_a*dt)*(h2_a*dt + 1))/((h2_a**2)*dt)
h2_weights[:, 1] = (xp.exp(-h2_a*dt) + h2_a*dt - 1)/((h2_a**2)*dt)
# when a == 0
h2_weights[xp.abs(h2_a) <= 1e-20, :] = dt/2

T3 = np.array(range(2*n_eta+1, n_time))
s = time.time()
for it in T3:
    # [n_q x n_freq]
    fhat_it = exp_dt*fhat[:, :, it-1] + \
        xp.squeeze(
            ((func_h1(it*dt, eta, dt, freq_square, q_square, freq_zero_mask, D0) @
              fhat_h1_coef) * exponential_term -
             func_h2(exp_eta, h2_weights, K_long[:, :, it-n_eta-1:it-n_eta+1])*fhat_h2_coef) @ dl.T)
    fhat[:, :, it] = fhat_it
    K_long[:, :, it] = fhat_it @ Klong_fourier_coef.T
xp.cuda.stream.get_current_stream().synchronize()
e = time.time()
print(e-s)
# %%
mu[:, :, n_eta+1:] = 2*(neu_data[:, :, n_eta+1:] -
                        K_long[:, :, n_eta+1:])/Kshort_appro.reshape(1, n_points, 1)

# [n_q x n_points x n_time]
S_short[:, :, n_eta+1:] = xp.sqrt(D0*eta/xp.pi)*mu[:, :, n_eta+1:]
# [n_q x n_points x n_time]
S_long = (fourier_bases.T.conj() @ fhat)*(D0*freq_resolution**2)
# [n_q x n_points x n_time]
S_solution = S_long + S_short

omega_bar = xp.zeros((n_q, n_time), dtype=ctype)
u = xp.zeros((n_q, n_points, n_time), dtype=ctype)
omega_bar_a = 4*(xp.pi**2)*D0*q_square.reshape((n_q, 1, 1))
weight1 = (1 - xp.exp(-omega_bar_a*dt) *
           (omega_bar_a*dt + 1))/(omega_bar_a**2*dt)
weight2 = (xp.exp(-omega_bar_a*dt) +
           omega_bar_a*dt - 1)/(omega_bar_a**2*dt)
# [n_q x n_points x n_time] matrix
u[:, :, 1:] = weight1*S_solution[:, :, 0:-1] + weight2*S_solution[:, :, 1:]

for iq in range(n_q):
    if np.abs(q_square[0, iq]) < 1e-20:
        u[iq, :, 1:] = (S_solution[iq, :, 0:-1] +
                        S_solution[iq, :, 1:])*dt/2

omega_bar_temp = xp.squeeze(dl @ (u*(2*xp.pi*1j*(q.T @ boundary_normals) *
                                     xp.exp(2*xp.pi*1j*q.T@boundary_points)).reshape(n_q, n_points, 1)))
for it in range(1, n_time):
    omega_bar[:, it] = xp.exp(-4*(xp.pi**2)*D0*q_square.reshape(-1)*dt) * \
        omega_bar[:, it-1] - D0*omega_bar_temp[:, it]
omega_bar = omega_bar/region_area

# %%
plt.plot(xp.asnumpy(time_val.reshape(-1)),
         xp.asnumpy(xp.real(omega_bar[-8, :])))
# plt.plot(time_val.reshape(-1),
#          xp.real(omega_bar[-1, :]))
