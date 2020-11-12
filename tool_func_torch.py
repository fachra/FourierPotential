import torch
import numpy as np


def get_geo(mat, ctype=torch.complex64, device=torch.device("cpu")):
    boundary_points = torch.tensor(
        mat['boundary_points'], dtype=ctype, device=device)  # [2 x n_points]
    boundary_normals = torch.tensor(
        mat['boundary_normals'], dtype=ctype, device=device)  # [2 x n_points]
    dl = torch.tensor(mat['dl'], dtype=ctype, device=device)  # [1 x n_points]
    curvature = torch.tensor(
        mat['curvature'], dtype=ctype, device=device)  # [1 x n_points]
    region_area = torch.tensor(
        mat['omega_area'], dtype=ctype, device=device)  # [1 x 1]
    return boundary_points, boundary_normals, dl, curvature, region_area


def get_q(sdelta, bdelta, b, ctype=torch.complex64, device=torch.device("cpu")):
    abs_q = np.sqrt(b/(bdelta-sdelta/3))/(2*np.pi)
    qx = np.linspace(0, abs_q, 10).reshape((1, -1))
    qy = np.zeros(10).reshape((1, -1))
    q = torch.tensor(np.concatenate((qx, qy), axis=0),
                     dtype=ctype, device=device)  # [2 x n_q]
    q_square = q.real.pow(2).sum(axis=0).reshape(
        (1, -1)).type(ctype)  # [1 x n_q]
    return q, q_square, abs_q


def get_time(sdelta, bdelta, dt, ctype=torch.complex64, device=torch.device("cpu")):
    return torch.linspace(0, bdelta-sdelta, round((bdelta-sdelta)/dt) +
                          1, device=device, dtype=ctype).reshape((1, -1))


def get_freq(freq_max, q_max, freq_resolution, ctype=torch.complex64, device=torch.device("cpu")):
    n_temp = int(2*np.ceil((freq_max+q_max)/freq_resolution) + 1)
    temp = torch.linspace(-freq_max-q_max, freq_max+q_max, n_temp,
                          device=device, dtype=torch.double)
    freq_y, freq_x = torch.meshgrid(-temp, temp)
    # freq: [2 x n_freqs]
    freq = torch.cat(
        (freq_x.reshape((1, -1)), freq_y.reshape((1, -1))), axis=0)
    freq_square = freq.pow(2).sum(axis=0).reshape(1, -1)  # [1 x n_freqs]
    return freq.type(ctype), freq_square.type(ctype)


def neumann(points, normals, q, q_square, t_val, D0=1e-3):
    n_q = torch.numel(q_square)

    xq = 2*np.pi*1j*((normals.t() @ q) *
                     torch.exp(-2*np.pi*1j*points.t() @ q)).t().reshape(n_q, -1, 1)
    qt = torch.exp(-4*np.pi**2*D0*q_square.t() @ t_val).reshape(n_q, 1, -1)
    return xq @ qt


def func_p(t, eta, dt, p_weights, freq_square, q_square, zero_mask, curvature, D0):
    # intg: [n_q x n_freqs x n_points]
    D0_curva_coeff = np.sqrt(D0/np.pi)*curvature
    denom1 = 1/(1-D0_curva_coeff*np.sqrt(t-eta-dt))
    denom2 = 1/(1-D0_curva_coeff*np.sqrt(t-eta))
    denom = torch.cat([denom1, denom2], axis=0)  # [2 x n_points]
    # [n_q x n_freqs x n_points]
    intg = (torch.exp(-4*(np.pi**2)*D0*(q_square.t() * (t-eta) +
                                        freq_square*eta)).reshape(q_square.shape[1], -1, 1) *
            p_weights) @ denom
    # when a == 0
    for iq in range(zero_mask.shape[0]):
        mask_temp = zero_mask[iq, :]
        if mask_temp.any():
            intg[iq, mask_temp, :] = -(2*np.pi/D0)*torch.exp(-4*(np.pi**2)*D0*q_square[0, iq]*t) * \
                (curvature*np.sqrt(D0/np.pi)*(np.sqrt(t-eta)-np.sqrt(t-eta-dt)) +
                 np.log((1-curvature*np.sqrt(D0*(t-eta)/np.pi)) /
                        (1-curvature*np.sqrt(D0*(t-eta-dt)/np.pi)))) / \
                curvature.pow(2)
    return intg


def func_h1(t, eta, dt, freq_square, q_square, zero_mask, D0):
    # output: [n_q x n_freqs x 1]
    # [n_q x n_freqs]
    h1 = (torch.exp(-4*(np.pi**2)*D0*(q_square.t()*(t-eta-dt) + freq_square*(eta+dt))) -
          torch.exp(-4*(np.pi**2)*D0*(q_square.t()*(t-eta) + freq_square*eta)))/(4*(np.pi**2)*D0*(q_square.t()-freq_square))
    for iq in range(zero_mask.shape[0]):
        mask_temp = zero_mask[iq, :]
        if mask_temp.any():
            h1[iq, mask_temp] = dt * \
                torch.exp(-4*(np.pi**2)*D0*q_square[0, iq]*t)
    return h1.reshape(q_square.shape[1], -1, 1)


def func_h2(exp_eta, h2_weights, klong):
    # exp_eta: [1 x n_freqs], h2_weights: [n_freqs x 2], klong: [n_q, n_points, 2]
    # output: [n_q x n_freqs x n_points]
    return exp_eta.reshape(1, -1, 1)*(h2_weights @ klong.permute(0, 2, 1))
