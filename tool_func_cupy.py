try:
    import cupy as xp
    import numpy as np
except (ModuleNotFoundError, ImportError):
    import numpy as np
    xp = np
# import numpy as xp
# np = xp


def get_geo(mat, dtype=xp.float64):
    boundary_points = xp.array(
        mat['boundary_points'], dtype=dtype)  # [2 x n_points]
    boundary_normals = xp.array(
        mat['boundary_normals'], dtype=dtype)  # [2 x n_points]
    dl = xp.array(mat['dl'], dtype=dtype)  # [1 x n_points]
    curvature = xp.array(
        mat['curvature'], dtype=dtype)  # [1 x n_points]
    region_area = xp.array(
        mat['omega_area'], dtype=dtype)  # [1 x 1]
    return boundary_points, boundary_normals, dl, curvature, region_area


def get_q(sdelta, bdelta, b, dtype=xp.float64):
    abs_q = np.sqrt(b/(bdelta-sdelta/3))/(2*np.pi)
    qx = xp.linspace(0, abs_q, 10).reshape((1, -1))
    qy = xp.zeros(10).reshape((1, -1))
    q = xp.concatenate((qx, qy), axis=0)  # [2 x n_q]
    q_square = (q**2).sum(axis=0).reshape((1, -1))  # [1 x n_q]
    return q, q_square, abs_q


def get_time(sdelta, bdelta, dt, dtype=xp.float64):
    return xp.linspace(0, bdelta-sdelta, round((bdelta-sdelta)/dt) +
                       1, dtype=dtype).reshape((1, -1))


def get_freq(freq_max, q_max, freq_resolution, dtype=xp.float64):
    n_temp = int(2*xp.ceil((freq_max+q_max)/freq_resolution) + 1)
    temp = xp.linspace(-freq_max-q_max, freq_max+q_max,
                       n_temp, dtype=dtype)
    freq_y, freq_x = xp.meshgrid(-temp, temp)
    # freq: [2 x n_freqs]
    freq = xp.concatenate(
        (freq_x.reshape((1, -1)), freq_y.reshape((1, -1))), axis=0)
    freq_square = (freq**2).sum(axis=0).reshape(1, -1)  # [1 x n_freqs]
    return freq, freq_square


def neumann(points, normals, q, q_square, t_val, D0=1e-3):
    n_q = q_square.shape[1]
    xq = 2*xp.pi*1j*((normals.T @ q) *
                     xp.exp(-2*xp.pi*1j*points.T @ q)).T.reshape(n_q, -1, 1)
    qt = xp.exp(-4*xp.pi**2*D0*q_square.T @ t_val).reshape(n_q, 1, -1)
    return xq @ qt


def func_p(t, eta, dt, p_weights, freq_square, q_square, zero_mask, curvature, D0):
    # intg: [n_q x n_freqs x n_points]
    D0_curva_coeff = xp.sqrt(D0/xp.pi)*curvature
    denom1 = 1/(1-D0_curva_coeff*xp.sqrt(t-eta-dt))
    denom2 = 1/(1-D0_curva_coeff*xp.sqrt(t-eta))
    denom = xp.concatenate([denom1, denom2], axis=0)  # [2 x n_points]
    # [n_q x n_freqs x n_points]
    intg = (xp.exp(-4*(xp.pi**2)*D0*(q_square.T * (t-eta) +
                                     freq_square*eta)).reshape(q_square.shape[1], -1, 1) *
            p_weights) @ denom
    # when a == 0
    for iq in range(zero_mask.shape[0]):
        mask_temp = zero_mask[iq, :]
        if mask_temp.any():
            zero_inds = mask_temp.nonzero()
            for ind in zero_inds:
                intg[iq, ind, :] = \
                    -(2*xp.pi/D0)*xp.exp(-4*(xp.pi**2)*D0*q_square[0, iq]*t) * \
                    (curvature*xp.sqrt(D0/xp.pi)*(xp.sqrt(t-eta)-xp.sqrt(t-eta-dt)) +
                     xp.log((1-curvature*xp.sqrt(D0*(t-eta)/xp.pi)) /
                            (1-curvature*xp.sqrt(D0*(t-eta-dt)/xp.pi)))) / \
                    curvature**2
    return intg


def func_h1(t, eta, dt, freq_square, q_square, zero_mask, D0):
    # output: [n_q x n_freqs x 1]
    # [n_q x n_freqs]
    h1 = (xp.exp(-4*(xp.pi**2)*D0*(q_square.T*(t-eta-dt) + freq_square*(eta+dt))) -
          xp.exp(-4*(xp.pi**2)*D0*(q_square.T*(t-eta) + freq_square*eta))) / \
        (4*(xp.pi**2)*D0*(q_square.T-freq_square))
    for iq in range(zero_mask.shape[0]):
        mask_temp = zero_mask[iq, :]
        if mask_temp.any():
            zero_inds = mask_temp.nonzero()
            for ind in zero_inds:
                h1[iq, ind] = dt * \
                    xp.exp(-4*(xp.pi**2)*D0*q_square[0, iq]*t)
    return h1.reshape(q_square.shape[1], -1, 1)


def func_h2(exp_eta, h2_weights, klong):
    # exp_eta: [1 x n_freqs], h2_weights: [n_freqs x 2], klong: [n_q, n_points, 2]
    # output: [n_q x n_freqs x n_points]
    return exp_eta.reshape(1, -1, 1)*(h2_weights @ klong.transpose(0, 2, 1))
