import torch
import numpy as np


def solve(**kwargs):
    '''
    '''
    # unpacking kwargs
    ctype, device = kwargs['ctype'], kwargs['device']
    D0 = kwargs['D0']
    sdelta, bdelta = kwargs['delta']
    q_vec = kwargs['q-vectors']
    dt = kwargs['dt']
    n_eta = kwargs['n_eta']
    freq_resolution = kwargs['frequency_resolution']
    freq_max = kwargs['frequency_max']
    models = kwargs['models']
    # prepare models and parameters
    boundary_points, boundary_normals, dl, curvature, region_area = get_geom(
        models, device=device)
    q, q_square, q_max = get_q(q_vec, device=device)
    eta = n_eta*dt
    time_val = get_time(sdelta, bdelta, dt, device=device)
    freq, freq_square = get_freq(
        freq_max, q_max, freq_resolution, device=device)
    # some constants
    n_points, n_time = boundary_points.shape[1], time_val.shape[1]
    n_freqs, n_q = freq.shape[1], q.shape[1]

    # init
    mu = torch.empty((n_q, n_points, n_time), device=device, dtype=ctype)
    S_short = torch.empty((n_q, n_points, n_time), device=device, dtype=ctype)
    K_long = torch.zeros((n_q, n_points, n_time), device=device, dtype=ctype)
    fhat = torch.zeros((n_q, n_freqs, n_time), device=device, dtype=ctype)

    # start computation
    # neumann data
    neu_data = neumann(boundary_points, boundary_normals,
                       q, q_square, time_val, D0=D0)

    # density mu for [0, eta=n_eta*dt] (us)
    mu[:, :, :n_eta+1] = 2 * neu_data[:, :, :n_eta+1] / \
        (1 - curvature.t()@torch.sqrt(D0 *
                                      time_val[:, :n_eta+1]/np.pi))
    S_short[:, :, :n_eta +
            1] = torch.sqrt(D0*time_val[:, :n_eta+1]/np.pi)*mu[:, :, :n_eta+1]

    # density mu for [(n_eta+1)*dt, 2*eta=2*n_eta*dt] (us)
    # reuseable intermediate results
    freq_zero_mask = (torch.abs(q_square.t()-freq_square) <= 1e-20)
    fourier_bases = torch.exp(-2*np.pi*1j*freq.t() @ boundary_points)
    exp_dt = torch.exp(-4*(np.pi**2)*D0*dt*freq_square)
    exponential_term = torch.exp(-2*np.pi*1j*(freq.t().reshape(1, -1, 2) +
                                              q.t().reshape(-1, 1, 2)) @ boundary_points)
    normal_derivative_coef = 4*np.pi*1j*q.t() @ boundary_normals
    coef_temp1 = normal_derivative_coef.reshape(
        n_q, 1, n_points)*exponential_term
    a = 4*(np.pi**2)*D0*(q_square.t()-freq_square)
    Kshort_appro = 1-np.sqrt(D0*eta/np.pi)*curvature
    Klong_fourier_coef = ((2*np.pi*1j*boundary_normals.t() @ freq)
                          * fourier_bases.t().conj())*(D0*freq_resolution**2)
    # weights in function p
    p_weights = torch.empty(n_q, n_freqs, 2, device=device,
                            dtype=ctype)  # [n_q x n_freqs x 2]
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
    # density mu for [(2*n_eta+1)*dt, bdelta-sdelta] (us)
    # [1 x n_freqs]
    exp_eta = torch.exp(-4*(np.pi**2)*D0*eta*freq_square)
    # [n_q x 1 x n_points]
    fhat_h1_coef = (normal_derivative_coef /
                    Kshort_appro).reshape(n_q, 1, n_points)
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
    for it in T3:
        # [n_q x n_freq]
        fhat_it = exp_dt*fhat[:, :, it-1] + \
            torch.squeeze(
                ((func_h1(it*dt, eta, dt, freq_square, q_square, freq_zero_mask, D0) @
                  fhat_h1_coef) * exponential_term -
                 func_h2(exp_eta, h2_weights.type(ctype), K_long[:, :, it-n_eta-1:it-n_eta+1])*fhat_h2_coef) @ dl.t())
        fhat[:, :, it] = fhat_it
        K_long[:, :, it] = fhat_it @ Klong_fourier_coef.t()
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
    # [n_q x n_points x n_time]
    u[:, :, 1:] = weight1*S_solution[:, :, 0:-1] + weight2*S_solution[:, :, 1:]

    for iq in range(n_q):
        if torch.abs(q_square[0, iq]) < 1e-20:
            u[iq, :, 1:] = (S_solution[iq, :, 0:-1] +
                            S_solution[iq, :, 1:])*dt/2
    omega_bar_temp = dl @ (u*(2*np.pi*1j*(q.t() @ boundary_normals) *
                              torch.exp(2*np.pi*1j*q.t()@boundary_points)).reshape(n_q, n_points, 1))
    omega_bar_temp = torch.squeeze(omega_bar_temp)
    for it in range(1, n_time):
        omega_bar[:, it] = torch.exp(-4*(np.pi**2)*D0*q_square.reshape(-1)*dt) * \
            omega_bar[:, it-1] - D0*omega_bar_temp[:, it]
    omega_bar = omega_bar/region_area
    return time_val.to('cpu'), omega_bar.to('cpu')


def get_geom(models, ctype=torch.complex64, device=torch.device("cpu")):
    boundary_points = torch.empty((2, 0), dtype=ctype)
    boundary_normals = torch.empty((2, 0), dtype=ctype)
    dl = torch.empty((1, 0), dtype=ctype)
    curvature = torch.empty((1, 0), dtype=ctype)
    region_area = torch.zeros(1, dtype=ctype)

    for m in models:
        boundary_points = torch.cat(
            (boundary_points, torch.from_numpy(m.points)), axis=1)
        boundary_normals = torch.cat(
            (boundary_normals, torch.from_numpy(m.normals)), axis=1)
        dl = torch.cat((dl, torch.from_numpy(m.dl.reshape((1, -1)))), axis=1)
        curvature = torch.cat((curvature, torch.from_numpy(
            m.curvature.reshape((1, -1)))), axis=1)
        region_area = region_area + m.area
    return boundary_points.to(device).type(ctype), boundary_normals.to(device).type(ctype), \
        dl.to(device).type(ctype), curvature.to(device).type(
            ctype), region_area.to(device).type(ctype)


def get_q(q_vec, ctype=torch.complex64, device=torch.device("cpu")):
    q = torch.tensor(q_vec, device=device)
    q_square = q.pow(2).sum(axis=0).reshape(
        (1, -1))  # [1 x n_q]
    return q.type(ctype), q_square.type(ctype), torch.max(torch.sqrt(q_square.cpu())).item()


def get_time(sdelta, bdelta, dt, ctype=torch.complex64, device=torch.device("cpu")):
    return torch.linspace(0, bdelta-sdelta, round((bdelta-sdelta)/dt) +
                          1, device=device, dtype=ctype).reshape((1, -1))


def get_freq(freq_max, q_max, freq_resolution,
             ctype=torch.complex64, device=torch.device("cpu")):
    n_temp = int(2*np.ceil((freq_max+q_max)/freq_resolution) + 1)
    temp = torch.linspace(-freq_max-q_max, freq_max+q_max, n_temp,
                          device=device, dtype=torch.double)
    freq_y, freq_x = torch.meshgrid(-temp, temp)
    freq = torch.cat(
        (freq_x.reshape((1, -1)), freq_y.reshape((1, -1))), axis=0)
    freq_square = freq.pow(2).sum(axis=0).reshape(1, -1)
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
                 torch.log((1-curvature*np.sqrt(D0*(t-eta)/np.pi)) /
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
