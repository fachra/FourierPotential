import torch
import torch.multiprocessing as mp
import matplotlib.pyplot as plt
import numpy as np
from functools import partial
from itertools import product
import psutil
import time
mp.set_start_method('spawn', force=True)


def q2b_PGSE(q_norm, sdelta, bdelta):
    return (2*np.pi*q_norm)**2*(bdelta-sdelta/3)


def b2q_PGSE(b, sdelta, bdelta, direction):
    q_norm = np.sqrt(b/(bdelta-sdelta/3))/(2*np.pi)
    q = np.zeros((2, len(q_norm)*len(direction.T)))
    i = 0
    for _q, _dir in product(q_norm, direction.T):
        q[:, i] = _q*_dir
        i = i + 1
    return q


def is_direction(arr):
    if (len(arr.shape) != 2) or (arr.shape[0] != 2):
        return False
    else:
        return True


def dir_semicircle(n):
    if n == 0:
        raise ValueError('n_direction should be a positive integer (>0).')
    try:
        angle = np.linspace(0, np.pi, n, endpoint=False)
    except (ValueError, RuntimeError, TypeError, NameError):
        raise ValueError('n_direction is illegal.')
    c, s = np.cos(angle), np.sin(angle)
    return np.vstack((c, s))


def get_dtype_size(dtype):
    return torch.ones(1, dtype=dtype).element_size()


def res_stack(res_list):
    result = [None] * 8
    for i in range(8):
        result[i] = torch.stack([res[i] for res in res_list])
    return result


save = torch.save
load = torch.load


class FPM:
    """Class conducting a fourier potential simulation."""
    # define alias
    _2pij = 2*np.pi*1j
    _4pis = 4*np.pi**2

    def __str__(self):
        s = 'Solving narrow-pulsed Bloch-Torrey PDE '
        s += 'using layer potentials and Fourier transform.\n'
        if self.has_run():
            s += '[THE SIMULATION IS COMPLETED] '
            s += f'Elapsed time is {self.elapsed_time} seconds.\n'
        s += 'Diffusion MRI parameters:\n'
        s += f'    initial density = {self.rho}\n'
        s += f'    diffusivity = {self.D0} um^2/us (mm^2/s)\n'
        s += f'    delta = {self.sdelta} us, Delta = {self.bdelta} us\n'
        s += f'    {self.n_q} q-vector um^-1'
        s += f' / {len(self.b)} b-value us/um^2 (s/mm^2):\n'
        for iq, qq in enumerate(self.q.real.numpy().T):
            s += f'        q = {qq},'
            bb = q2b_PGSE(self.q_norm[iq].item(), self.sdelta, self.bdelta)
            s += f' b = {round(bb, 4)}\n'
        s += 'Simulation settings:\n'
        s += f'    eta = {self.eta} us\n'
        s += f'    time step = {self.dt} us\n'
        s += f'    frequency_max = {self.f_max} um^-1\n'
        s += f'    frequency step = {self.df} um^-1\n'
        s += f'    number of time steps = {self.n_time}\n'
        s += f'    number of fourier modes = {self.n_freq}\n'
        s += 'Geometrical models:\n'
        s += f'    number of points = {self.n_points}\n'
        s += f'    total area = {self.region_area:.4f} um^2\n'
        s += '    '
        s += f'curvature max = {self.curvature.real.max().item():.4f} um^-1,'
        s += f' curvature min = {self.curvature.real.min().item():.4f} um^-1\n'
        s += f'    spatial step max = {self.dl.real.max().item():.4f} um,'
        s += f' spatial step min = {self.dl.real.min().item():.4f} um\n'
        s += 'Other settings:\n'
        temp = 'CPU' if self.n_gpu <= 0 else f'GPU: {self.devices}'
        s += f'    device = {temp}\n'
        s += f'    datatype = {self.dtype}\n'
        s += f'    freq_memory_ratio = {self.freq_memory_ratio}\n'
        s += '\n'
        s += 'Individual info of models. Make sure there is no overlapping.\n'
        for m in self.geom:
            s += f'{m}\n'
        return s

    def __init__(self,
                 geom,
                 delta,
                 dt,
                 n_eta,
                 frequency_max,
                 frequency_resolution,
                 diffusivity=1e-3,
                 initial_density=1,
                 dtype=torch.complex128,
                 gpu=False,
                 freq_memory_ratio=20,
                 **kwargs) -> None:
        """

        """
        # set and check parameters
        self.geom = geom
        if (not isinstance(self.geom, (tuple, list))) or \
                (len(self.geom) < 1):
            raise ValueError(
                "'geom' should be a non-empty tuple/list.")
        self.sdelta, self.bdelta = delta
        if (self.sdelta <= 0) or (self.sdelta > self.bdelta):
            raise ValueError(
                "'sdelta' should be less than or equal to " +
                "'bdelta' and greater than zero.")
        self.dt = dt
        if self.dt <= 0:
            raise ValueError(
                "'dt' should be greater than zero.")
        elif self.dt > self.bdelta-self.sdelta:
            raise ValueError("'dt' is too large.")
        self.n_eta = n_eta
        if self.n_eta <= 0:
            raise ValueError(
                "'n_eta' should be an integer greater than zero.")
        elif self.n_eta*self.dt > self.bdelta+self.sdelta:
            raise ValueError(
                "'n_eta' is too large.")
        self.f_max = frequency_max
        if self.f_max <= 0:
            raise ValueError(
                "'frequency_max' should be positive.")
        self.df = frequency_resolution
        if self.df <= 0:
            raise ValueError(
                "'frequency_resolution' should be positive.")
        elif self.df > self.f_max:
            raise ValueError(
                "'frequency_resolution' should be less than 'frequency_max'.")
        self.D0 = diffusivity
        if self.D0 <= 0:
            raise ValueError(
                "'diffusivity' should be positive.")
        self.rho = initial_density
        if self.rho <= 0:
            raise ValueError(
                "'initial_density' should be positive.")
        self.dtype = dtype
        if not isinstance(self.dtype, torch.dtype):
            raise ValueError(
                "'dtype' should be one of the torch.dtype instances.")
        self.freq_memory_ratio = freq_memory_ratio
        if (self.freq_memory_ratio > 100) or (self.freq_memory_ratio < 0):
            raise ValueError(
                "'freq_memory_ratio' should be between 0 and 100.")
        self._run = False

        # set device
        if isinstance(gpu, bool):
            if gpu:
                if torch.cuda.is_available():
                    self.n_gpu = 1
                    self.devices = (torch.device("cuda:0"),)
                else:
                    print(f'{type(self).__name__} warning: ' +
                          'GPU is not available, CPU is used.')
                    self.n_gpu = 0
                    self.devices = (torch.device("cpu"),)
            else:
                self.n_gpu = 0
                self.devices = (torch.device("cpu"),)
        elif isinstance(gpu, int):
            if gpu >= 0:
                if torch.cuda.is_available():
                    _gpu = np.min([gpu, torch.cuda.device_count()-1])
                    self.devices = (torch.device(f"cuda:{_gpu}"),)
                    self.n_gpu = 1
                else:
                    print(f'{type(self).__name__} warning: ' +
                          'GPU is not available, CPU is used.')
                    self.n_gpu = 0
                    self.devices = (torch.device("cpu"),)
            else:
                self.n_gpu = 0
                self.devices = (torch.device("cpu"),)
        elif isinstance(gpu, list):
            # multiple GPUs support
            if min(gpu) < 0:
                print(f"{type(self).__name__} warning: " +
                      "illegal 'gpu' entry, CPU is used.")
                self.n_gpu = 0
                self.devices = (torch.device("cpu"),)
            elif not torch.cuda.is_available():
                print(f'{type(self).__name__} warning: ' +
                      'GPU is not available, CPU is used.')
                self.n_gpu = 0
                self.devices = (torch.device("cpu"),)
            else:
                gpu = np.array(gpu).astype(int)
                gpu = np.clip(gpu, 0, torch.cuda.device_count()-1)
                gpu = np.unique(gpu)
                self.n_gpu = len(gpu)
                self.devices = tuple(torch.device(f"cuda:{i}") for i in gpu)
        else:
            print(f"{type(self).__name__} warning: " +
                  "unrecognized 'gpu' entry, CPU is used.")
            self.n_gpu = 0
            self.devices = (torch.device("cpu"),)

        # get q-vectors and b-values
        if 'q-vectors' in kwargs:
            self.q = np.array(kwargs['q-vectors'])
            if not is_direction(self.q):
                raise ValueError("The 'q-vectors' entry should be a 2D array" +
                                 " with shape being (2, -1)")
            self.q_norm = np.linalg.norm(self.q, axis=0)
            self.direction = self.q / self.q_norm
            self.b = q2b_PGSE(self.q_norm, self.sdelta, self.bdelta)
        elif 'b-values' in kwargs:
            self.b = np.array(kwargs['b-values']).reshape((-1,))
            if 'direction' in kwargs:
                self.direction = np.array(kwargs['direction'])
                if not is_direction(self.direction):
                    raise ValueError("The 'direction' entry should be a" +
                                     " 2D array with shape being (2,-1)")
                dir_norm = np.linalg.norm(self.direction, axis=0)
                if np.any(dir_norm < 1e-12):
                    raise ValueError(
                        "The 'direction' entry contains a zero vector.")
                self.direction = self.direction / dir_norm
                self.q = b2q_PGSE(self.b, self.sdelta,
                                  self.bdelta, self.direction)
                self.q_norm = np.linalg.norm(self.q, axis=0)
            elif 'n_direction' in kwargs:
                self.direction = dir_semicircle(kwargs['n_direction'])
                self.q = b2q_PGSE(self.b, self.sdelta,
                                  self.bdelta, self.direction)
                self.q_norm = np.linalg.norm(self.q, axis=0)
            else:
                raise ValueError("The entry 'direction' or 'n_direction' \
                        is needed when b-values are given.")
        else:
            raise ValueError("The 'q-vectors' or 'b-values' \
                    should be provided in the input.")

        # prepare suitable data structure for simulation
        # convert numpy.array to torch.tensor
        self._geom_init()
        self._q_init()
        self._time_init()
        self._freq_init()

    def _geom_init(self):
        self.n_points = 0
        for m in self.geom:
            self.n_points += m.n_points

        self.points = torch.empty((2, self.n_points), dtype=self.dtype)
        self.normals = torch.empty((2, self.n_points), dtype=self.dtype)
        self.dl = torch.empty((1, self.n_points), dtype=self.dtype)
        self.curvature = torch.empty((1, self.n_points), dtype=self.dtype)
        self.region_area = 0

        start = 0
        end = 0
        for m in self.geom:
            end += m.n_points
            self.points[:, start:end] = torch.from_numpy(m.points)
            self.normals[:, start:end] = torch.from_numpy(m.normals)
            self.dl[:, start:end] = torch.from_numpy(m.dl.reshape((1, -1)))
            self.curvature[:, start:end] = torch.from_numpy(
                m.curvature.reshape((1, -1)))
            self.region_area = self.region_area + m.area
            start = end

        self.n_points = self.points.shape[1]
        return self.n_points

    def _q_init(self):
        self.q = torch.from_numpy(self.q).type(self.dtype)
        self.q_norm = torch.from_numpy(self.q_norm)
        self.q_square = (self.q_norm.pow(2)).type(self.dtype).reshape((1, -1))
        self.n_q = self.q.shape[1]
        return self.n_q

    def _time_init(self):
        self.T = torch.linspace(0, self.bdelta-self.sdelta,
                                round((self.bdelta-self.sdelta)/self.dt) + 1)
        if len(self.T) > 1:
            self.dt = (self.T[1] - self.T[0]).item()
            self.eta = self.n_eta * self.dt
        else:
            raise ValueError("'dt' is too large.")
        self.T = self.T.reshape((1, -1)).type(self.dtype)
        self.n_time = self.T.shape[1]
        return self.n_time

    def _freq_init(self):
        n_temp = int(2*np.ceil(self.f_max/self.df) + 1)
        temp = torch.linspace(-self.f_max, self.f_max,
                              n_temp, dtype=torch.double)
        if len(temp) > 1:
            self.df = (temp[1] - temp[0]).item()
        else:
            raise ValueError(
                "'frequency_resolution' should be less than 'frequency_max'.")
        freq_y, freq_x = torch.meshgrid(-temp, temp)
        self.freq = torch.cat(
            (freq_x.reshape((1, -1)), freq_y.reshape((1, -1))), axis=0)
        self.freq_square = self.freq.pow(2).sum(axis=0)
        self.freq_square = self.freq_square.reshape((1, -1)).type(self.dtype)
        self.freq = self.freq.type(self.dtype)
        self.n_freq = self.freq.shape[1]
        return self.n_freq

    def check(self, plot=False) -> None:
        s = self.__str__()
        print(s)
        if plot:
            _, axs = plt.subplots(1)
            for m in self.geom:
                axs.plot(m.points[0, :], m.points[1, :], 'b--')
            axs.axis('equal')

    def has_run(self):
        return self._run

    def save_fft(self):
        pass

    def run(self):
        # print simulation info
        self.check()
        if self.has_run():
            print("The simulation is completed. " +
                  "Use rerun() to relaunch the simulation.")
            return None

        # check if devices have enough memory
        for idev in self.devices:
            chk_mem = self._is_mem_enough(idev)
            if not chk_mem[0]:
                s = f"{type(self).__name__} {idev}: "
                s += f"at least {chk_mem[1] / 1e6 : .1f} MB memory is needed, "
                s += f"only {chk_mem[2] / 1e6 : .1f} MB is available."
                raise MemoryError(s)

        s = time.time()
        # launch simulations
        if len(self.devices) == 1:
            q_ind = torch.arange(self.n_q)
            result = [self._launch(q_ind, self.devices[0])]
        elif self.n_gpu > 1:
            q_ind = torch.tensor_split(torch.arange(
                self.n_q), min(self.n_gpu, self.n_q))
            var_list = [None] * self.n_gpu
            for ind, dev in enumerate(self.devices):
                var_list[ind] = [q_ind[ind], dev]
            with mp.Pool(processes=self.n_gpu) as p:
                result = p.starmap(self._launch, var_list)
        else:
            raise RuntimeError("Unknown device.")
        e = time.time()
        # store results
        self._store(result)
        self.elapsed_time = e-s
        print(f'Elapsed time is {self.elapsed_time} seconds.')
        self._run = True
        return True

    def rerun(self):
        self._run = False
        return self.run()

    def _store(self, result) -> None:
        self.mu = torch.cat([res[0] for res in result])
        self.S_short = torch.cat([res[1] for res in result])
        self.S_long = torch.cat([res[2] for res in result])
        self.K_short = torch.cat([res[3] for res in result])
        self.K_long = torch.cat([res[4] for res in result])
        self.omega = torch.cat([res[5] for res in result])
        self.omega_bar = torch.cat([res[6] for res in result])
        self.dMRI_signal = torch.cat([res[7] for res in result])

    def _is_mem_enough(self, device):
        # get the available memory size
        if device == torch.device("cpu"):
            mem_available = psutil.virtual_memory().available
        else:
            mem_available = torch.cuda.get_device_properties(
                device).total_memory
        # estimate memory usage
        least_mem = 4*self.n_time * \
            self.n_points*get_dtype_size(self.dtype)
        return (least_mem < mem_available, least_mem, mem_available)

    def _launch(self, ind, device):
        n_q = len(ind)
        # move parameters to device
        variables = (self.points.to(device),
                     self.normals.to(device),
                     self.dl.to(device),
                     self.curvature.to(device),
                     self.T.to(device),
                     self.freq.to(device),
                     self.freq_square.to(device))
        q = self.q[:, ind].to(device)
        q_square = self.q_square[:, ind].to(device)

        # split into simulation units based on q
        res_list = [None] * n_q
        for iq in range(n_q):
            print(f"Running FPM simulation: {ind[iq]+1} / {self.n_q}.")
            res_list[iq] = self.simulation_unit(*variables,
                                                q[:, [iq]], q_square[:, iq],
                                                device, save_fft=False)
        return res_stack(res_list)

    def simulation_unit(self, points, normals,
                        dl, curvature, T, freq, freq_square,
                        q, q_square, device, save_fft=False):
        """This is a simulation unit for one q-vector.

        Args:
            points (torch.tensor) [2 x n_points]:
                The coordinates of spatial samplings on the boundary.
            normals (torch.tensor) [2 x n_points]:
                The out-pointing normal vectors on the boundary.
            dl (torch.tensor) [1 x n_points]:
                The length of the boundary segments.
            curvature (torch.tensor) [2 x n_points]:
                The boundary curvature.
            T (torch.tensor) [1 x n_time]:
                The equispaced temporal samplings of
                the time interval [0, bdelta-sdelta].
            freq (torch.tensor) [2 x n_freq]:
                The coordinates of the samplings on the frequency domain.
            freq_square (torch.tensor) [1 x n_freq]:
                The frequency norm square. |f|^2
            q (torch.tensor) [2 x 1]:
                The q-vector.
            q_square (torch.tensor) [1]:
                The q square. |q|^2.
            device (torch.device):
                The device to run this unit.
            save_fft (bool, optional):
                If true, save the spectrum. Default is false.

        Returns:
            mu (torch.tensor) [n_points x n_time]:
                The density funciton mu defined on the boundary.
            S_short (torch.tensor) [n_points x n_time]:
                The local part of the single layer potential S[mu].
            S_long (torch.tensor) [n_points x n_time]:
                The history part of the single layer potential S[mu].
            K_short (torch.tensor) [n_points x n_time]:
                The local part of K[mu].
            K_long (torch.tensor) [n_points x n_time]:
                The history part of K[mu].
            omega (torch.tensor) [n_points x n_time]:
                The values of the transformed magnetization omega
                on the boundary.
            omega_bar (torch.tensor) [n_time]:
                The integration of the transformed magnetization omega.
            dMRI_signal (torch.tensor) [n_time]:
                The final dMRI signal.
        """
        # initialize
        S_long = torch.zeros((self.n_points, self.n_time),
                             device=device, dtype=self.dtype)
        K_long = torch.zeros((self.n_points, self.n_time),
                             device=device, dtype=self.dtype)
        fhat_last = torch.zeros(
            (1, self.n_freq), device=device, dtype=self.dtype)

        # start computation for T = [0, bdelta-sdelta] (us)
        # define neumann data
        neumann = partial(self._neumann, points, normals, q, q_square)

        if self.n_time == self.n_eta+1:
            # first time interval T1 = [0, n_eta]*dt (us)
            t1 = torch.arange(0, self.n_time)
            # compute mu
            mu = torch.empty((self.n_points, self.n_time),
                            device=device, dtype=self.dtype)
            mu[:, t1] = self.fun_t1(neumann, curvature, T[:, t1])

            # compute K_short and move it to cpu for saving memory
            K_short = - (torch.sqrt(self.D0*T[:, t1]/np.pi)
                                * curvature.t()/2) * mu[:, t1]
            K_short = K_short.cpu()
            # zero quantities
            K_long = torch.tensor([0])
            S_long = torch.tensor([0], device=device)
        elif (self.n_time > self.n_eta+1) and (self.n_time <= 2*self.n_eta+1):
            # first time interval T1 = [0, n_eta]*dt (us)
            t1 = torch.arange(0, self.n_eta+1)
            # second time interval T2 = [n_eta+1, n_time-1]*dt (us)
            t2 = torch.arange(self.n_eta+1, self.n_time)
            freq_schedule = self._set_freq_schedule(
                device=device, r=self.freq_memory_ratio)
            if len(freq_schedule) > 1:
                for it in t2:
                    for ifreq in freq_schedule:
                        fhat_temp, K_long_temp, S_long_temp = \
                            self.fun_t2(it, fhat_last[:, ifreq], freq[:, ifreq],
                                        freq_square[:, ifreq], q, q_square,
                                        points, normals, curvature, dl)
                        fhat_last[:, ifreq] = fhat_temp
                        K_long[:, it].add_(K_long_temp)
                        S_long[:, it].add_(S_long_temp)
                    if save_fft:
                        # to do
                        torch.cuda.synchronize(device)
            else:
                fourier_bases = torch.exp(-self._2pij*freq.t() @ points)
                for it in t2:
                    fhat_last, K_long[:, it], S_long[:, it] = \
                        self.fun_t2(it, fhat_last, freq, freq_square, q,
                                    q_square, points, normals, curvature, dl,
                                    fourier_bases=fourier_bases)
                    if save_fft:
                        # to do
                        torch.cuda.synchronize(device)

            # compute mu
            mu = torch.empty((self.n_points, self.n_time),
                            device=device, dtype=self.dtype)
            D0_curva = np.sqrt(self.D0*self.eta/np.pi)*curvature.reshape((-1, 1))

            mu[:, t1] = self.fun_t1(neumann, curvature, T[:, t1])
            mu[:, t2] = (neumann(T[:, t2]) - K_long[:, t2])/((1 - D0_curva)/2)

            # K_long is no longer used, move it to cpu for saving memory
            K_long = K_long.cpu()
            
            # compute K_short and move it to cpu for saving memory
            K_short = - (D0_curva/2) * mu
            K_short[:, t1] = - (torch.sqrt(self.D0*T[:, t1]/np.pi)
                                * curvature.t()/2) * mu[:, t1]
            K_short = K_short.cpu()
        elif self.n_time > 2*self.n_eta:
            # first time interval T1 = [0, n_eta]*dt (us)
            # the computation of mu is moved to the end for saving memory
            t1 = torch.arange(0, self.n_eta+1)

            # second time interval T2 = [n_eta+1, 2*n_eta]*dt (us)
            t2 = torch.arange(self.n_eta+1, 2*self.n_eta+1)
            freq_schedule = self._set_freq_schedule(
                device=device, r=self.freq_memory_ratio)
            if len(freq_schedule) > 1:
                for it in t2:
                    for ifreq in freq_schedule:
                        fhat_temp, K_long_temp, S_long_temp = \
                            self.fun_t2(it, fhat_last[:, ifreq], freq[:, ifreq],
                                        freq_square[:, ifreq], q, q_square,
                                        points, normals, curvature, dl)
                        fhat_last[:, ifreq] = fhat_temp
                        K_long[:, it].add_(K_long_temp)
                        S_long[:, it].add_(S_long_temp)
                    if save_fft:
                        # to do
                        torch.cuda.synchronize(device)
            else:
                fourier_bases = torch.exp(-self._2pij*freq.t() @ points)
                for it in t2:
                    fhat_last, K_long[:, it], S_long[:, it] = \
                        self.fun_t2(it, fhat_last, freq, freq_square, q,
                                    q_square, points, normals, curvature, dl,
                                    fourier_bases=fourier_bases)
                    if save_fft:
                        # to do
                        torch.cuda.synchronize(device)

            # third time interval T3 = [2*n_eta+1, n_time-1]*dt (us)
            t3 = torch.arange(self.n_eta*2+1, self.n_time)
            freq_schedule = self._set_freq_schedule(
                device=device, r=self.freq_memory_ratio)
            if len(freq_schedule) > 1:
                for it in t3:
                    for ifreq in freq_schedule:
                        fhat_temp, K_long_temp, S_long_temp = \
                            self.fun_t3(
                                K_long[:, it-self.n_eta-1:it-self.n_eta+1].t(),
                                it, fhat_last[:, ifreq], freq[:, ifreq],
                                freq_square[:, ifreq], q, q_square,
                                points, normals, curvature, dl)
                        fhat_last[:, ifreq] = fhat_temp
                        K_long[:, it].add_(K_long_temp)
                        S_long[:, it].add_(S_long_temp)
                    if save_fft:
                        # to do
                        torch.cuda.synchronize(device)
            else:
                fourier_bases = torch.exp(-self._2pij*freq.t() @ points)
                for it in t3:
                    fhat_last, K_long[:, it], S_long[:, it] = \
                        self.fun_t3(
                            K_long[:, it-self.n_eta-1:it-self.n_eta+1].t(),
                            it, fhat_last, freq, freq_square, q,
                            q_square, points, normals, curvature, dl,
                            fourier_bases=fourier_bases)
                    if save_fft:
                        # to do
                        torch.cuda.synchronize(device)

            # compute mu
            mu = torch.empty((self.n_points, self.n_time),
                            device=device, dtype=self.dtype)
            t23 = torch.cat([t2, t3])
            D0_curva = np.sqrt(self.D0*self.eta/np.pi)*curvature.reshape((-1, 1))

            mu[:, t1] = self.fun_t1(neumann, curvature, T[:, t1])
            mu[:, t23] = (neumann(T[:, t23]) - K_long[:, t23])/((1 - D0_curva)/2)

            # K_long is no longer used, move it to cpu for saving memory
            K_long = K_long.cpu()
            
            # compute K_short and move it to cpu for saving memory
            K_short = - (D0_curva/2) * mu
            K_short[:, t1] = - (torch.sqrt(self.D0*T[:, t1]/np.pi)
                                * curvature.t()/2) * mu[:, t1]
            K_short = K_short.cpu()
        else:
            raise ValueError("Error with time parameters.")

        # compute S_short
        S_short = np.sqrt(self.D0*self.eta/np.pi) * mu
        S_short[:, t1] = torch.sqrt(self.D0*T[:, t1]/np.pi) * mu[:, t1]

        # mu is no longer used, move it to cpu for saving memory
        mu = mu.cpu()

        # compute the transformed magnetization
        omega = S_short + S_long
        # S_short and S_long are no longer used
        # move them to cpu for saving memory
        S_short = S_short.cpu()
        S_long = S_long.cpu()
        # compute the dMRI signal
        omega_bar, dMRI_signal = self.fun_dMRI_signal(omega, q, q_square,
                                                      points, normals, dl, T)

        return (mu,
                S_short,
                S_long,
                K_short,
                K_long,
                omega.cpu(),
                omega_bar.cpu(),
                dMRI_signal.cpu())

    def _set_freq_schedule(self, device=torch.device("cpu"), r=25):
        """Make a computation schedule for the spectrum.
        The computation of the whole spectrum needs enormous memory.
        Luckily, the computation at one frequency point is independent
        from others. Therefore, we can split the enitre spectrum to
        several blocks according to the memory limit.

        Args:
            device (torch.device):
                The device to run the simulation.
            r (float):
                The percentage of the available memory
                that can be used for saving one spectrum block.
                Default is 25%. Suggested to be less than 30%.

        Returns:
            schedule (list):
                A list contaning the index of the frequency points
                in each spectrum block.
        """
        # get available memory in the device
        if device == torch.device("cpu"):
            mem_available = psutil.virtual_memory().available
        else:
            mem_available = torch.cuda.get_device_properties(
                device).total_memory - torch.cuda.memory_allocated(device)
        if mem_available < 4e9:
            print(f'''{type(self).__name__} warning: {mem_available / 1e6} MB
            memory is available. If you encounter out of memory error,
            please set your_FPM_instance.freq_memory_ratio to a smaller number
            (current value: {r}) and rerun the simulation.''')
        mem_per_freq = self.n_points*get_dtype_size(self.dtype)
        # number of frequency per loop
        npl = max(1, int(r * mem_available / mem_per_freq / 100))
        # number of loops
        nl = np.ceil(self.n_freq / npl).astype('int')
        schedule = [torch.arange(ii*npl, min((ii+1)*npl, self.n_freq)) for
                    ii in range(nl)]
        return schedule

    def _neumann(self, points, normals, q, q_square, t_val):
        """The Neumann boundary condition corresponding to Algorithm 1 in the paper.

        Args:
            points (torch.tensor) [2 x n_points]:
                The coordinates of spatial samplings on the boundary.
            normals (torch.tensor) [2 x n_points]:
                The out-pointing normal vectors on the boundary.
            q (torch.tensor) [2 x 1]:
                The q-vector.
            q_square (torch.tensor) [1]:
                The q square. |q|^2.
            t_val (torch.tensor) [1 x n_t]
                The temporal sampling points.

        Returns:
            neumann (np.array, complex)[n_points x n_t]:
                The density funciton mu defined on the boundary.
        """
        # xq: [n_points x 1]
        xq = self._2pij*self.rho*((normals.t() @ q) *
                                  torch.exp(-self._2pij*points.t()@q))
        # qt: [1 x n_time]
        qt = torch.exp(-self._4pis*self.D0*q_square*t_val)
        return xq @ qt

    def fun_t1(self, neumann, curvature, T):
        """Computation of the density function mu
        for the first time interval [0, eta].
        This function corresponds to the Algorithm 2 in the paper.

        Args:
            neumann (function handle):
                A function computing the neumann data.
                Pointing to self._neumann.
            curvature (torch.tensor) [2 x n_points]:
                The boundary curvature.
            T (torch.tensor) [1 x n_t1]
                The temporal sampling points.

        Returns:
            mu (np.array, complex)[n_points x n_t]:
                The density funciton mu for the time interval [0, eta].
        """
        return 2 * neumann(T) / (1 - curvature.t() @
                                 torch.sqrt(self.D0 * T/np.pi))

    def fun_t2(self, it, fhat_last, freq, freq_square, q, q_square,
               points, normals, curvature, dl, fourier_bases=None):
        """Computation of the density function mu, the Fourier coefficients fhat,
        and the long time parts K_long, S_long for the timepoint it*dt
        in the second time interval T2 = [n_eta+1, 2*n_eta]*dt (us).
        This function mainly corresponds to the Algorithm 3 in the paper.

        Args:
            it (int):
                The it-th time step.
            fhat_last (torch.tensor, complex) [1 x n_freq]:
                The Fourier coefficients fhat of the last time step.
            freq (torch.tensor) [2 x n_freq]:
                The coordinates of the samplings on the frequency domain.
            freq_square (torch.tensor) [1 x n_freq]:
                The frequency norm square. |f|^2
            q (torch.tensor) [2 x 1]:
                The q-vector.
            q_square (torch.tensor) [1]:
                The q square. |q|^2.
            points (torch.tensor) [2 x n_points]:
                The coordinates of spatial samplings on the boundary.
            normals (torch.tensor) [2 x n_points]:
                The out-pointing normal vectors on the boundary.
            curvature (torch.tensor) [2 x n_points]:
                The boundary curvature.
            dl (torch.tensor) [1 x n_points]:
                The length of the boundary segments.
            fourier_bases (None or torch.tensor) [n_freq x n_points]:
                Matrix of Fourier bases exp(-2*pi*j*freq*x).
                If not None (memory is enough), use the inputted fourier_bases
                for saving time.

        Returns:
            fhat_it (torch.tensor, complex) [1 x n_freq]:
                The Fourier coefficients fhat at the current (it-th) time step.
            K_long_temp (torch.tensor, complex) [n_points]:
                The long time part K_long at at the current (it-th) time step.
            S_long_temp (torch.tensor, complex) [n_points]:
                The long time part S_long at at the current (it-th) time step.
        """
        # matrix of Fourier bases exp(-2*pi*j*freq*x)  [n_freq x n_points]
        if fourier_bases is None:
            fourier_bases = torch.exp(-self._2pij*freq.t() @ points)

        fhat_temp1 = \
            self._fhat_temp1(it*self.dt, fourier_bases, freq_square,
                             q, q_square, points, normals, curvature, dl)

        # compute the Fourier coefficients
        # for the it-th time step  [1 x n_freq]
        exp_dt = torch.exp((-self._4pis*self.D0*self.dt)*freq_square)
        fhat_it = fhat_temp1 + exp_dt * fhat_last

        # coefficients related to the K_long [n_freq x n_points]
        # compute the coef's conjugation to avoid conjugating fourier_bases
        Klong_fourier_coef = -self._2pij*freq.t() @ normals
        Klong_fourier_coef.mul_(fourier_bases)
        # get K_long and S_long by inverse Fourier transform
        K_long_temp = torch.squeeze(
            fhat_it.conj()@Klong_fourier_coef).conj()*(self.D0*self.df**2)
        S_long_temp = torch.squeeze(
            fhat_it.conj()@fourier_bases).conj()*(self.D0*self.df**2)
        return fhat_it, K_long_temp, S_long_temp

    def _fhat_temp1(self, t, fourier_bases, freq_square, q, q_square,
                    points, normals, curvature, dl):
        """The updating Fourier coefficients fhat_temp1 at timepoint t.
        This function corresponds to the Function fhat_temp1 in the paper.

        Args:
            t (float):
                The time point t.
            fourier_bases (torch.tensor, complex) [n_freq x n_points]:
                Matrix of Fourier bases exp(-2*pi*j*freq*x).
            freq_square (torch.tensor) [1 x n_freq]:
                The frequency norm square. |f|^2
            q (torch.tensor) [2 x 1]:
                The q-vector.
            q_square (torch.tensor) [1]:
                The q square. |q|^2.
            points (torch.tensor) [2 x n_points]:
                The coordinates of spatial samplings on the boundary.
            normals (torch.tensor) [2 x n_points]:
                The out-pointing normal vectors on the boundary.
            curvature (torch.tensor) [2 x n_points]:
                The boundary curvature.
            dl (torch.tensor) [1 x n_points]:
                The length of the boundary segments.

        Returns:
            fhat_temp1 (torch.tensor, complex) [1 x n_freq]
        """
        # coefficients related to normal vectors on the boundary [1 x n_points]
        normal_coef = (2*self.rho*self._2pij*q.t() @ normals) * \
            torch.exp(-self._2pij*q.t() @ points)
        # integrand of fhat_temp1  [n_freq x n_points]
        # in-place multiplication for less memory usage
        integrand = self._p(t, curvature, freq_square, q_square)
        integrand.mul_(fourier_bases)
        integrand.mul_(normal_coef)
        return (integrand@dl.t()).reshape((1, -1))

    def _p(self, t, curvature, freq_square, q_square):
        """The time integration within the integral fhat_temp1.
        This function corresponds to the Function p in the paper.

        Args:
            t (float):
                The time point t.
            curvature (torch.tensor) [2 x n_points]:
                The boundary curvature.
            freq_square (torch.tensor) [1 x n_freq]:
                The frequency norm square. |f|^2
            q_square (torch.tensor) [1]:
                The q square. |q|^2.

        Returns:
            p (torch.tensor, complex) [n_freq x n_points]
        """
        # weights [n_freq x 1]
        a = self.dt*self._4pis*self.D0*(q_square-freq_square.t())
        p_weights1 = (1+(a-1)*torch.exp(a))/a.pow(2)
        p_weights2 = (torch.exp(a)-a-1)/a.pow(2)
        # prevent numerical rounding error using Taylor expansion
        mask = torch.abs(a) < 1e-5
        if mask.any():
            a_temp = a[mask]
            p_weights1[mask] = 1/2 + a_temp/3 + a_temp.pow(2)/8 + \
                a_temp.pow(3)/30 + a_temp.pow(4)/144 + a_temp.pow(5)/840
            p_weights2[mask] = 1/2 + a_temp/6 + a_temp.pow(2)/24 + \
                a_temp.pow(3)/120 + a_temp.pow(4)/720 + a_temp.pow(5)/5040
        p_weights = torch.cat([p_weights1, p_weights2], axis=1)
        exp_term = torch.exp(-self._4pis*self.D0 *
                             (q_square*(t-self.eta)+freq_square.t()*self.eta))
        p_weights.mul_(exp_term)

        # denominator
        D0_curva = np.sqrt(self.D0/np.pi)*curvature
        denom1 = 1-D0_curva*torch.sqrt(torch.abs(t-self.eta-self.dt))
        denom2 = 1-D0_curva*torch.sqrt(torch.abs(t-self.eta))
        denom = 1/torch.cat([denom1, denom2], axis=0)  # [2 x n_points]

        # intg: [n_freq x n_points]
        intg = (p_weights*self.dt) @ denom

        # when a == 0
        mask = torch.abs(a).reshape(-1) < 1e-12
        if mask.any():
            ln = torch.log(denom2 / denom1)
            add = D0_curva * (torch.sqrt(torch.abs(t-self.eta)) -
                              torch.sqrt(torch.abs(t-self.eta-self.dt)))
            intg[mask, :] = -2*torch.exp(-self._4pis*self.D0*q_square*t) * \
                (add + ln) / D0_curva.pow(2)
        return intg

    def fun_t3(self, K_long, it, fhat_last, freq, freq_square, q, q_square,
               points, normals, curvature, dl, fourier_bases=None):
        """Computation of the density function mu, the Fourier coefficients fhat,
        and the long time parts K_long, S_long for the timepoint it*dt
        in the third time interval T3 = [(2*n_eta+1)*dt, bdelta-sdelta] (us)
        This mainly corresponds to the Algorithm 4 in the paper.

        Args:
            it (int):
                The it-th time step.
            fhat_last (torch.tensor, complex) [1 x n_freq]:
                The Fourier coefficients fhat of the last time step.
            freq (torch.tensor) [2 x n_freq]:
                The coordinates of the samplings on the frequency domain.
            freq_square (torch.tensor) [1 x n_freq]:
                The frequency norm square. |f|^2
            q (torch.tensor) [2 x 1]:
                The q-vector.
            q_square (torch.tensor) [1]:
                The q square. |q|^2.
            points (torch.tensor) [2 x n_points]:
                The coordinates of spatial samplings on the boundary.
            normals (torch.tensor) [2 x n_points]:
                The out-pointing normal vectors on the boundary.
            curvature (torch.tensor) [2 x n_points]:
                The boundary curvature.
            dl (torch.tensor) [1 x n_points]:
                The length of the boundary segments.
            fourier_bases (None or torch.tensor) [n_freq x n_points]:
                If not None (memory is enough), use the inputted fourier_bases
                for saving time.

        Returns:
            fhat_it (torch.tensor, complex) [1 x n_freq]:
                The Fourier coefficients fhat at the current (it-th) time step.
            K_long_temp (torch.tensor, complex) [n_points]:
                The long time part K_long at at the current (it-th) time step.
            S_long_temp (torch.tensor, complex) [n_points]:
                The long time part S_long at at the current (it-th) time step.
        """
        # matrix of Fourier bases exp(-2*pi*j*freq*x)  [n_freq x n_points]
        if fourier_bases is None:
            fourier_bases = torch.exp(-self._2pij*freq.t() @ points)

        fhat_temp2 = \
            self._fhat_temp2(it*self.dt, K_long, fourier_bases, freq_square,
                             q, q_square, points, normals, curvature, dl)

        # compute the Fourier coefficients for the it-th time step
        # fhat_it size: [1 x n_freq]
        exp_dt = torch.exp((-self._4pis*self.D0*self.dt)*freq_square)
        fhat_it = fhat_temp2 + fhat_last * exp_dt

        # coefficients related to the K_long  [n_freq x n_points]
        # compute the coef's conjugation to avoid conjugating fourier_bases
        Klong_fourier_coef = -self._2pij*freq.t() @ normals
        Klong_fourier_coef.mul_(fourier_bases)
        # get K_long and S_long by inverse Fourier transform
        K_long_temp = torch.squeeze(
            fhat_it.conj()@Klong_fourier_coef).conj()*(self.D0*self.df**2)
        S_long_temp = torch.squeeze(
            fhat_it.conj()@fourier_bases).conj()*(self.D0*self.df**2)
        return fhat_it, K_long_temp, S_long_temp

    def _fhat_temp2(self, t, K_long, fourier_bases, freq_square, q, q_square,
                    points, normals, curvature, dl):
        """The updating Fourier coefficients fhat_temp2 at timepoint t.
        This function corresponds to the Function fhat_temp2 in the paper.

        Args:
            t (float):
                The time point t.
            K_long (torch.tensor, complex) [n_points x 2]:
                The long time part K_long at timepoint t-eta-dt and t-eta.
            fourier_bases (torch.tensor, complex) [n_freq x n_points]:
                Matrix of Fourier bases exp(-2*pi*j*freq*x).
            freq_square (torch.tensor) [1 x n_freq]:
                The frequency norm square. |f|^2
            q (torch.tensor) [2 x 1]:
                The q-vector.
            q_square (torch.tensor) [1]:
                The q square. |q|^2.
            points (torch.tensor) [2 x n_points]:
                The coordinates of spatial samplings on the boundary.
            normals (torch.tensor) [2 x n_points]:
                The out-pointing normal vectors on the boundary.
            curvature (torch.tensor) [2 x n_points]:
                The boundary curvature.
            dl (torch.tensor) [1 x n_points]:
                The length of the boundary segments.

        Returns:
            fhat_temp2 (torch.tensor, complex) [1 x n_freq]
        """
        # coefficients related to normal vectors on the boundary [1 x n_points]
        normal_coef = (self._2pij*self.rho*q.t() @ normals) * \
            torch.exp(-self._2pij*q.t() @ points)
        # denominator  [1 x n_points]
        denom = (1 - np.sqrt(self.D0*self.eta/np.pi)*curvature) / 2

        # integrand of fhat_temp2  [n_freq x n_points]
        # in-place multiplication for less memory usage
        integrand = self._h1(t, freq_square,
                             q_square).t() @ normal_coef
        integrand.sub_(self._h2(freq_square, K_long))
        integrand.mul_(fourier_bases)
        integrand.divide_(denom)
        return (integrand@dl.t()).reshape((1, -1))

    def _h1(self, t, freq_square, q_square):
        """The first time integration within the integral fhat_temp2.
        This function corresponds to the Function h1 in the paper.

        Args:
            t (float):
                The time point t.
            freq_square (torch.tensor) [1 x n_freq]:
                The frequency norm square. |f|^2
            q_square (torch.tensor) [1]:
                The q square. |q|^2.

        Returns:
            h1 (torch.tensor, complex) [1 x n_freq]
        """
        a = self._4pis*self.D0*(q_square-freq_square)*self.dt
        mask = torch.abs(a) < 1e-5
        h1 = (torch.exp(a) - 1)/a
        # prevent numerical rounding error using Taylor expansion
        if mask.any():
            a_temp = a[mask]
            h1[mask] = 1 + a_temp/2 + a_temp.pow(2)/6 + a_temp.pow(3)/24 + \
                a_temp.pow(4)/120 + a_temp.pow(5)/720
        exp_term = torch.exp(-self._4pis*self.D0*(q_square*(t-self.eta) +
                             freq_square*self.eta))
        h1.mul_(exp_term)
        return h1*self.dt

    def _h2(self, freq_square, K_long):
        """The second time integration within the integral fhat_temp2.
        This function corresponds to the Function h2 in the paper.

        Args:
            freq_square (torch.tensor) [1 x n_freq]:
                The frequency norm square. |f|^2
            K_long (torch.tensor, complex) [n_points x 2]:
                The long time part K_long at timepoint t-eta-dt and t-eta.

        Returns:
            h2 (torch.tensor, complex) [n_freq x n_points]
        """
        # h2_weights  [2 x n_freq]
        a = self._4pis*self.D0*freq_square.reshape(-1)*self.dt
        h2_weights1 = (1 - torch.exp(-a)*(a + 1))/a.pow(2)
        h2_weights2 = (torch.exp(-a) + a - 1)/a.pow(2)
        # prevent numerical rounding error using Taylor expansion
        mask = torch.abs(a) < 1e-5
        if mask.any():
            a_temp = -a[mask]
            h2_weights1[mask] = 1/2 + a_temp/3 + a_temp.pow(2)/8 + \
                a_temp.pow(3)/30 + a_temp.pow(4)/144 + a_temp.pow(5)/840
            h2_weights2[mask] = 1/2 + a_temp/6 + a_temp.pow(2)/24 + \
                a_temp.pow(3)/120 + a_temp.pow(4)/720 + a_temp.pow(5)/5040

        h2_weights = torch.vstack([h2_weights1, h2_weights2])*self.dt
        h2_weights.mul_(torch.exp(-self._4pis*self.D0*self.eta*freq_square))
        return h2_weights.t() @ K_long

    def fun_dMRI_signal(self, omega, q, q_square, points, normals, dl, T):
        """Computation of the omega_bar and diffusion MRI signal.
        This function corresponds to the Algorithm 6 in the paper.

        Args:
            omega (torch.tensor, complex) [n_points x n_time]:
                The transformed magnetization. (= S_short + S_long)
            q (torch.tensor) [2 x 1]:
                The q-vector.
            q_square (torch.tensor) [1]:
                The q square. |q|^2.
            points (torch.tensor) [2 x n_points]:
                The coordinates of spatial samplings on the boundary.
            normals (torch.tensor) [2 x n_points]:
                The out-pointing normal vectors on the boundary.
            dl (torch.tensor) [1 x n_points]:
                The length of the boundary segments.
            T (torch.tensor) [1 x n_time]
                The temporal samplings in the interval [0, bdelta-sdelta].

        Returns:
            omega_bar (torch.tensor, complex) [n_time]
            dMRI_signal (torch.tensor, complex) [n_time]
        """
        # initilize omega_bar
        omega_bar = torch.zeros(
            self.n_time, device=dl.device, dtype=self.dtype)
        omega_bar_temp = \
            self._omega_bar_temp(omega, q, q_square, points, normals, dl)
        exp_term = torch.exp(-self._4pis*self.D0*q_square*self.dt)
        for it in range(1, self.n_time):
            omega_bar[it] = \
                exp_term*omega_bar[it-1] - self.D0*omega_bar_temp[it]
        dMRI_signal = self.region_area*self.rho * \
            torch.exp(-self._4pis*self.D0*q_square*T[0, :]) + omega_bar
        return omega_bar, dMRI_signal

    def _omega_bar_temp(self, omega, q, q_square, points, normals, dl):
        """Computation of the updating term omega_bar_temp.
        This function corresponds to the Function omega_bar_temp in the paper.

        Args:
            omega (torch.tensor, complex) [n_points x n_time]:
                The transformed magnetization. (= S_short + S_long)
            q (torch.tensor) [2 x 1]:
                The q-vector.
            q_square (torch.tensor) [1]:
                The q square. |q|^2.
            points (torch.tensor) [2 x n_points]:
                The coordinates of spatial samplings on the boundary.
            normals (torch.tensor) [2 x n_points]:
                The out-pointing normal vectors on the boundary.
            dl (torch.tensor) [1 x n_points]:
                The length of the boundary segments.

        Returns:
            omega_bar_temp (torch.tensor, complex) [n_time]
        """
        # coefficients related to normal vectors on the boundary
        normal_coef = ((self._2pij*q.t()) @ normals) * \
            torch.exp((self._2pij*q.t()) @ points)

        integrand = self._u(omega, q_square)
        integrand.mul_(normal_coef.t())
        return (dl@integrand).reshape(-1)

    def _u(self, omega, q_square):
        """Computation of the time integration u.
        This function corresponds to the Function u in the paper.

        Args:
            omega (torch.tensor, complex) [n_points x n_time]:
                The transformed magnetization. (= S_short + S_long)
            q_square (torch.tensor) [1]:
                The q square. |q|^2.

        Returns:
            u (torch.tensor, complex) [n_time]
        """
        # init
        u = torch.zeros_like(omega)
        # compute u
        a = -self._4pis*self.D0*q_square*self.dt
        if torch.abs(a) < 1e-5:
            weight1 = 1/2 + a/3 + a.pow(2)/8 + \
                a.pow(3)/30 + a.pow(4)/144 + a.pow(5)/840
            weight2 = 1/2 + a/6 + a.pow(2)/24 + \
                a.pow(3)/120 + a.pow(4)/720 + a.pow(5)/5040
        else:
            weight1 = (1 + torch.exp(a)*(a-1)) / a.pow(2)
            weight2 = (torch.exp(a) - a - 1) / a.pow(2)
        u[:, 1:] = weight1*omega[:, 0:-1] + weight2*omega[:, 1:]
        return u*self.dt
