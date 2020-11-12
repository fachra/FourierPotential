# %%
import matplotlib.pyplot as plt
import torch
import time
import scipy.io as io
import numpy as np
from tool_func_torch import *
rtype = torch.double
ctype = torch.complex64  # complex float number
device = torch.device("cpu")
# device = torch.device("cuda:0")
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
freq_resolution = 0.05  # [um^-1]
freq_max = 1  # [um^-1]

time_val = get_time(sdelta, bdelta, dt)
freq, freq_square = get_freq(freq_max, 0, freq_resolution)
# %% some constants
n_points, n_time = boundary_points.shape[1], time_val.numel()
n_freqs, n_q = freq.shape[1], q.shape[1]
# %%init
mu = torch.empty((n_q, n_points, n_time), device=device, dtype=ctype)
S_short = torch.empty((n_q, n_points, n_time), device=device, dtype=ctype)
K_long = torch.zeros((n_q, n_points, n_time), device=device, dtype=ctype)
fhat = torch.zeros((n_q, n_freqs, n_time), device=device, dtype=ctype)
# %% neumann data
neu_data = neumann(boundary_points, boundary_normals,
                   q, q_square, time_val, D0=D0)
# %% density mu for [0, eta=n_eta*dt] (us)
mu[:, :, :n_eta+1] = 2 * neu_data[:, :, :n_eta+1] / \
    (1 - curvature.t()@torch.sqrt(D0 *
                                  time_val[:, :n_eta+1]/np.pi))
S_short[:, :, :n_eta +
        1] = torch.sqrt(D0*time_val[:, :n_eta+1]/np.pi)*mu[:, :, :n_eta+1]
# %% density mu for [(n_eta+1)*dt, 2*eta=2*n_eta*dt] (us)
# [n_q x n_freqs]
freq_zero_mask = (torch.abs(q_square.t()-freq_square) <= 1e-20)
# [n_freqs x n_points]
fourier_bases = torch.exp(-2*np.pi*1j*freq.t() @ boundary_points)
# [1 x n_freqs]
exp_dt = torch.exp(-4*(np.pi**2)*D0*dt*freq_square)
# [n_q x n_freqs x n_points]
exponential_term = torch.exp(-2*np.pi*1j*(freq.t().reshape(1, -1, 2) +
                                          q.t().reshape(-1, 1, 2)) @ boundary_points)
# [n_q x n_points]
normal_derivative_coef = 4*np.pi*1j*q.t() @ boundary_normals
# [n_q x n_freqs x n_points]
coef_temp1 = normal_derivative_coef.reshape(n_q, 1, n_points)*exponential_term
# [n_q x n_freqs]
a = 4*(np.pi**2)*D0*(q_square.t()-freq_square)
# [1 x n_points]
Kshort_appro = 1-np.sqrt(D0*eta/np.pi)*curvature
# [n_points x n_freqs]
Klong_fourier_coef = ((2*np.pi*1j*boundary_normals.t() @ freq)
                      * fourier_bases.t().conj())*(D0*freq_resolution**2)
# %%
p_weights = torch.empty(n_q, n_freqs, 2)  # [n_q x n_freqs x 2]
p_weights[:, :, 0] = (1+(a*dt-1)*torch.exp(a*dt))/(a.pow(2)*dt)
p_weights[:, :, 1] = (torch.exp(a*dt)-a*dt-1)/(a.pow(2)*dt)

T2 = torch.tensor(range(n_eta+1, 2*n_eta+1))
for it in T2:
    # [n_q x n_freqs x n_points]
    fhat_integrand = coef_temp1 * \
        func_p(it*dt, eta, dt, p_weights, freq_square,
               q_square, freq_zero_mask, curvature, D0)
    # [n_q x n_freqs x n_time]
    fhat[:, :, it] = torch.squeeze(
        fhat_integrand @ dl.t()) + fhat[:, :, it-1] * exp_dt

K_long[:, :, T2] = Klong_fourier_coef @ fhat[:, :, T2]
# %% density mu for [(2*n_eta+1)*dt, bdelta-sdelta] (us)
# [1 x n_freqs]
exp_eta = torch.exp(-4*(np.pi**2)*D0*eta*freq_square)
# [n_q x 1 x n_points]
fhat_h1_coef = (normal_derivative_coef/Kshort_appro).reshape(n_q, 1, n_points)
# [n_freqs x n_points]
fhat_h2_coef = (2*fourier_bases/Kshort_appro).reshape(1, n_freqs, n_points)

# [n_freqs]
h2_a = 4*(np.pi**2)*D0*freq_square.reshape(-1)
# [n_freqs x 2]
h2_weights = torch.zeros(n_freqs, 2, device=device, dtype=ctype)
h2_weights[:, 0] = (1 - torch.exp(-h2_a*dt)*(h2_a*dt + 1))/(h2_a.pow(2)*dt)
h2_weights[:, 1] = (torch.exp(-h2_a*dt) + h2_a*dt - 1)/(h2_a.pow(2)*dt)
# when a == 0
h2_weights[torch.abs(h2_a) <= 1e-20, :] = dt/2

T3 = torch.tensor(range(2*n_eta+1, n_time))
s = time.time()
for it in T3:
    # [n_q x n_freq]
    fhat_it = exp_dt*fhat[:, :, it-1] + \
        torch.squeeze(
            ((func_h1(it*dt, eta, dt, freq_square, q_square, freq_zero_mask, D0) @
              fhat_h1_coef) * exponential_term -
             func_h2(exp_eta, h2_weights.type(ctype), K_long[:, :, it-n_eta-1:it-n_eta+1])*fhat_h2_coef) @ dl.t())
    fhat[:, :, it] = fhat_it
    K_long[:, :, it] = fhat_it @ Klong_fourier_coef.t()
e = time.time()
print(e-s)
# %%
mu[:, :, n_eta+1:] = 2*(neu_data[:, :, n_eta+1:] -
                        K_long[:, :, n_eta+1:])/Kshort_appro.reshape(1, n_points, 1)

# [n_q x n_points x n_time]
S_short[:, :, n_eta+1:] = np.sqrt(D0*eta/np.pi)*mu[:, :, n_eta+1:]
# [n_q x n_points x n_time]
S_long = (fourier_bases.t().conj() @ fhat)*(D0*freq_resolution**2)
# [n_q x n_points x n_time]
S_solution = S_long + S_short

omega_bar = torch.zeros(n_q, n_time, device=device, dtype=ctype)
u = torch.zeros(n_q, n_points, n_time, device=device, dtype=ctype)
omega_bar_a = 4*(np.pi**2)*D0*q_square.reshape(n_q, 1, 1)
weight1 = (1 - torch.exp(-omega_bar_a*dt) *
           (omega_bar_a*dt + 1))/(omega_bar_a**2*dt)
weight2 = (torch.exp(-omega_bar_a*dt) +
           omega_bar_a*dt - 1)/(omega_bar_a**2*dt)
# [n_q x n_points x n_time] matrix
u[:, :, 1:] = weight1*S_solution[:, :, 0:-1] + weight2*S_solution[:, :, 1:]

for iq in range(n_q):
    if np.abs(q_square[0, iq]) < 1e-20:
        u[iq, :, 1:] = (S_solution[iq, :, 0:-1] +
                        S_solution[iq, :, 1:])*dt/2

omega_bar_temp = torch.squeeze(dl @ (u*(2*np.pi*1j*(q.t() @ boundary_normals) *
                                        torch.exp(2*np.pi*1j*q.t()@boundary_points)).reshape(n_q, n_points, 1)))
for it in range(1, n_time):
    omega_bar[:, it] = torch.exp(-4*(np.pi**2)*D0*q_square.reshape(-1)*dt) * \
        omega_bar[:, it-1] - D0*omega_bar_temp[:, it]
omega_bar = omega_bar/region_area

# %%
plt.plot(time_val.reshape(-1), torch.real(omega_bar[-2, :]))
