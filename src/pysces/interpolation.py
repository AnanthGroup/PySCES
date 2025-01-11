import numpy as np
from collections import deque
from scipy.integrate import simps
from scipy.interpolate import interp1d
import scipy.linalg as la
from typing import Literal
class _GradEstimator():
    def __init__(self, order: Literal[1, 2, 3], history_size=None, interval=1.0, name='UNKNOWN'):

        if order not in [1, 2, 3]:
            raise ValueError("order must be 1, 2, or 3")

        if history_size is None:
            history_size = order + 1
        elif history_size < order + 1:
            raise ValueError("history_size must be at least order + 1")
        
        self.order = order
        self.history_size = history_size
        self.interval = interval
        self.history_grads = deque(maxlen=history_size)
        self.history_t = deque(maxlen=history_size)
        self._interp_func = None
        self.name = name
        self._interp_kind = {1: 'linear', 2: 'quadratic', 3: 'cubic'}[order]


    def update_history(self, new_t, new_grads):

        self.history_grads.append(np.array(new_grads))
        self.history_t.append(new_t)

        if len(self.history_grads) > self.order:
            self._interpolate()

    def reset_history(self):
        self.history_grads.clear()
        self.history_t.clear()
        self._interp_func = None

    def _interpolate(self):
        stacked_history = np.vstack([np.ravel(matrix) for matrix in self.history_grads])
        self._interp_func = interp1d(self.history_t, stacked_history, 
                                     kind=self._interp_kind, 
                                     axis=0, 
                                     fill_value='extrapolate')
        
    def _evaluate(self, time):
        if not self._interp_func:
            return None
        return self._interp_func(time)
    
    def __call__(self, t) -> np.ndarray:
        pass

    def check_if_ready(self):
        is_ready = len(self.history_grads) > self.order
        is_ready *= (self._interp_func is not None)
        return is_ready

    def check_run_deriv(self, new_time):
        if len(self.history_grads) <= self.order:
            return True, 'History too short'
        elif new_time - self.history_t[-1] >= self.interval:
            return True, 'Interval reached'
        else:
            return False, 'OK'
    
    def guess_f(self, f0, times, velocities):
        if not self._interp_func:
            raise ValueError("Not enough history to estimate yet")            

        grads = self._evaluate(times)
        prod = grads * velocities
        prod = np.sum(prod, axis=prod.ndim-1)
        delta_f = simps(prod, times, axis=0)
        extrap_f = f0 + delta_f

        return extrap_f

class GradientInterpolation():
    def __init__(self, grads, masses) -> None:
        self._grads = grads.copy()
        AMU_2_AU = 1.822888486*10**3      # atomic mass unit to a.u. of mass
        self._masses = np.array(masses) * AMU_2_AU
        self._grad_estimator = {g: _GradEstimator(order=2, interval=8.0, name=f'grad_{g}') for g in self._grads}
        self._comparisons = {}

    def update_history(self, grad_id, time, gradient):

        if self._grad_estimator[grad_id].check_if_ready():
            guess = self._grad_estimator[grad_id]._evaluate(time)
            self._comparisons[grad_id] = (guess, gradient, time)

        self._grad_estimator[grad_id].update_history(time, gradient)
    
    def print_update_messages(self):
        if len(self._comparisons) == 0:
            return
        print('\nComparisons of Gradients at interpolation update')
        out_str = ''
        for g in list(self._comparisons):
            guess, gradient, time = self._comparisons.pop(g)
            error = np.abs(guess - gradient.ravel())
            max_error = np.max(error)
            rmsd_error = np.sqrt(np.mean(error**2))
            rrmsd_error = rmsd_error / np.sqrt(np.mean(gradient**2))
            out_str += f' Energy   {g:2d}  {time:8.1f}  {max_error:16.10f}  {rmsd_error:16.10f}  {rrmsd_error:6.4f}\n'
            self._comparisons
        print('  Gradient    Time (a.u.)     Max Error            RMSD      RRMSD (%)')
        print(' -------------------------------------------------------------------')
        print(out_str[:-1])
        # print(' -------------------------------------------------------------------')


    def get_guess(self, time: float, gradient: int):
        return self._grad_estimator[gradient]._evaluate(time)
    
    def get_guesses(self, time: float):
        return tuple((g, self._grad_estimator[g]._evaluate(time)) for g in self._grads)

    def get_gradient_states(self, current_energies, previous_energies, time_hist, nuc_p_hist):
        guess_energy = {g: None for g in self._grads}
        grads_to_run = {}
        curr_time = time_hist[-1]

        for g, grad_est in self._grad_estimator.items():
            if not grad_est.check_if_ready():
                grads_to_run[g] = (True, 'History too short')
            else:
                grads_to_run[g] = grad_est.check_run_deriv(curr_time)
 
                #   setup the velocities for the gradient estimator
                #   current time is -1, previous time is -2 for the energy history
                prev_time = time_hist[-2]
                vel_history = nuc_p_hist/(self._masses)
                interp_vel = interp1d(time_hist, vel_history, axis=0, kind='quadratic') # TODO: change to the same order as the grad interpolator
                time_pts = np.linspace(prev_time, curr_time, 100)
                vel_pts = interp_vel(time_pts)
                guess_energy[g] = grad_est.guess_f(previous_energies[g], time_pts, vel_pts)
                energy_diff = guess_energy[g] - current_energies[g]

                if abs(energy_diff) > 0.0001:
                    grads_to_run[g] = (True, 'guess energy error')

        print('\nInteprolation Gradient Time: ', curr_time)
        print('          State   Estimate (a.u.)     Actual (a.u.)  Error (a.u.)      Estimate?')
        print('---------------------------------------------------------------------------------------------')
        for g in self._grads:
            reason = grads_to_run[g][1]
            guess_str, error_str = f'{"---":^16s}', f'{"---":^16s}'
            if guess_energy[g] is not None:
                guess_str = f'{guess_energy[g]:16.10f}'
                error_str = f'{guess_energy[g] - current_energies[g]:16.10f}'
            print(f'  Energy    {g:2d}   {guess_str:>16s}  {current_energies[g]:16.10f}  {error_str:>16s}  {reason}')
        # print('---------------------------------------------------------------------------------------------')

        return grads_to_run

def _norm_dot_prod(x, y):
    a = np.array(x).flatten()
    b = np.array(y).flatten()
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

class NACnterpolation():
    def __init__(self, nacs, masses) -> None:
        self._NACs = tuple(tuple(x) for x in nacs)
        AMU_2_AU = 1.822888486*10**3      # atomic mass unit to a.u. of mass
        self._masses = np.array(masses) * AMU_2_AU
        self._estimator = {}
        self._comparisons = {}
        for x in self._NACs:
            self._estimator[x] = _GradEstimator(order=2, interval=8.0, name=f'nac_{x[0]}_{x[1]}')
            # self._estimator[tuple(reversed(x))] = self._estimator[x]


    def update_history(self, nac_pair, time, gradient):
        estimator: _GradEstimator = self._estimator[nac_pair]
        if estimator.check_if_ready():
            guess = estimator._evaluate(time)
            self._comparisons[nac_pair] = (guess, gradient, time)

            dp = _norm_dot_prod(gradient, guess)
            if dp < 0.80:
                print(f'NAC dot product with current extrapolation is too large: {dp:.4f}')
                estimator.reset_history()

        self._estimator[tuple(nac_pair)].update_history(time, gradient)


    def print_update_messages(self):
        if len(self._comparisons) == 0:
            return
        print('\nComparisons of NACs at interpolation update')
        out_str = ''
        # for (x1, x2), (guess, gradient, time) in self._comparisons.items():
        for x1, x2 in list(self._comparisons):
            guess, gradient, time = self._comparisons.pop((x1, x2))
            error = np.abs(guess - gradient.ravel())
            max_error = np.max(error)
            rmsd_error = np.sqrt(np.mean(error**2))
            rrmsd_error = rmsd_error / np.sqrt(np.mean(gradient**2))
            out_str += f' TDC   {x1} - {x2}  {time:8.1f}  {max_error:16.10f}  {rmsd_error:16.10f}  {rrmsd_error:6.4f}\n'
        print('  Gradient    Time (a.u.)     Max Error            RMSD      RRMSD (%)')
        print(' -------------------------------------------------------------------')
        print(out_str[:-1])
        # print(' -------------------------------------------------------------------')

    
    def get_guess(self, time: float, gradient: int):
        return self._estimator[gradient]._evaluate(time)
    
    def get_guesses(self, time: float):
        return tuple((g, self._estimator[g]._evaluate(time)) for g in self._NACs)

    def get_nac_states_OLD(self, overlaps, time_hist, nuc_p_hist):
        overlaps = np.transpose(overlaps)
        print_str = ''

        for x in self._NACs:
            estimator: _GradEstimator = self._estimator[x]
            if not estimator.check_if_ready():
                nacs_to_run[x] = (True, 'History too short')

        if overlaps is None:
            nacs_to_run = {x: (True, 'Overlaps not available') for x in self._NACs}
        elif len(time_hist) < 3:
            nacs_to_run = {x: (True, 'History too short') for x in self._NACs}
        else:


            nacs_to_run = {}
            #   setup the velocities for the gradient estimator
            curr_time = time_hist[-1]
            prev_time = time_hist[-2]
            vel_history = nuc_p_hist/(self._masses)
            interp_vel = interp1d(time_hist, vel_history, axis=0, kind='quadratic') # TODO: change to the same order as the grad interpolator
            time_pts = np.linspace(prev_time, curr_time, 100)
            vel_pts = interp_vel(time_pts)
            dt = curr_time - prev_time

            U = overlaps @ np.linalg.inv(la.sqrtm(overlaps.T @ overlaps)) # unitary matrix
            log_U = la.logm(U)
            avg_T = log_U / dt

            guess_avg_T = np.zeros_like(U)

            #   compute estimates for the gradients and if they are good enough            
            for x in self._NACs:
                estimator: _GradEstimator = self._estimator[x]
                required_run, reason = estimator.check_run_deriv(curr_time)
                if not required_run:
                    #   check difference with time derivative coupling
                    guess_avg_T[x] = estimator.guess_f(0.0, time_pts, vel_pts)/dt
                    diff = guess_avg_T[x]  - avg_T[x]
                    print_str += (f'  {x}  {guess_avg_T[x]:16.10f}  {avg_T[x]:16.10f}  {diff:16.10f}\n')
                    if abs(diff) > 0.0001:
                        required_run = True
                        reason = 'TDC guess error'

                    #   check angle between histories
                    dp12 = _norm_dot_prod(estimator.history_grads[-1], estimator.history_grads[-2])
                    dp13 = _norm_dot_prod(estimator.history_grads[-1], estimator.history_grads[-3])
                    if dp12 < 0.80 or dp13 < 0.80:
                        required_run = True
                        reason = f'History dot-product: {np.max([dp12, dp13]):.4f}'
                        estimator.reset_history()

                nacs_to_run[x] = (required_run, reason)
            guess_avg_T += guess_avg_T.T
            

        if print_str != '':
            print('Inteprolation NAC Time: ', curr_time)
            print('  States      Estimate            Actual      Error (a.u.)')
            print(' ------------------------------------------------------------')
            print(print_str[:-1])
            print(' -------------------------------------------------------------')
        print('')
        print('NACs to Run')
        print('     NAC       Run?    Reason')
        print('   -------------------------------------------')
        for (x1, x2), (required_run, reason) in nacs_to_run.items():
            print(f'     {x1}, {x2}        {required_run}    {reason}')
        print('   -------------------------------------------')

        return nacs_to_run


    def get_nac_states(self, overlaps, time_hist, nuc_p_hist):
        overlaps = np.transpose(overlaps)
        nacs_to_run = {}
        guess_avg_T = {x: None for x in self._NACs}
        curr_time = time_hist[-1]

        if overlaps.ndim == 0:
            dim = np.max(self._NACs).astype(int) + 1
            current_avg_T = np.zeros((dim, dim))
        else:
            dt = curr_time - time_hist[-2]
            U = overlaps @ np.linalg.inv(la.sqrtm(overlaps.T @ overlaps)) # unitary matrix
            log_U = la.logm(U)
            current_avg_T = log_U / dt

        for x in self._NACs:
            estimator: _GradEstimator = self._estimator[x]
            if not estimator.check_if_ready():
                nacs_to_run[x] = (True, 'History too short')

            else:
                nacs_to_run[x] = estimator.check_run_deriv(time_hist[-1])
            
                #   setup the velocities for the gradient estimator
                prev_time = time_hist[-2]
                vel_history = nuc_p_hist/(self._masses)
                interp_vel = interp1d(time_hist, vel_history, axis=0, kind='quadratic') # TODO: change to the same order as the grad interpolator
                time_pts = np.linspace(prev_time, curr_time, 100)
                vel_pts = interp_vel(time_pts)

                guess_avg_T[x] = estimator.guess_f(0.0, time_pts, vel_pts)/dt
                diff = guess_avg_T[x]  - current_avg_T[x]
                if abs(diff) > 0.0001:
                    nacs_to_run[x] = (True, 'TDC guess error')

                #   check angle between histories
                dp12 = _norm_dot_prod(estimator.history_grads[-1], estimator.history_grads[-2])
                dp13 = _norm_dot_prod(estimator.history_grads[-1], estimator.history_grads[-3])
                if dp12 < 0.80 or dp13 < 0.80:
                    nacs_to_run[x] = (True, f'History dot-product: {np.max([dp12, dp13]):.4f}')
                    estimator.reset_history()

        print('\nInteprolation NAC Time: ', curr_time)
        print('          State   Estimate (a.u.)     Actual (a.u.)  Error (a.u.)      Estimate?')
        print('---------------------------------------------------------------------------------------------')
        for x in self._NACs:
            reason = nacs_to_run[x][1]
            guess_str, error_str = f'{"---":^16s}', f'{"---":^16s}'
            if guess_avg_T[x] is not None:
                guess_str = f'{guess_avg_T[x]:16.10f}'
                error_str = f'{guess_avg_T[x] - current_avg_T[x]:16.10f}'
            state_str = str(x[0])+' - '+str(x[1])
            print(f' TDC    {state_str:^5s}  {guess_str:>16s}  {current_avg_T[x]:16.10f}  {error_str:>16s}  {reason}')
        # print('---------------------------------------------------------------------------------------------')

        return nacs_to_run

class SignFlipper():
    _debug = False
    _ref_nac = None
    def __init__(self, n_states: int, hist_length: int, n_nuc: int, name: str='UNK') -> None:
        ''' Checks which sign for the NAC is expected. 
            Artificial sign flips will be corrected and history is logged

            Parameters
            ----------
            n_states: int
                number of states that will be passed in to the NAC arrays
            hist_length: int
                number of points to keep track of in the history.
                if hist_length=2, we use a linear extrapolation, hist_length=3 is quadratic, and so on
            n_nuc: int
                number of nuclei coordinates
            name: str
                name of the sign flipper to use when printing sign flip messages
                
            Notes
            -----
            Predict d(t) from d(t-1),d(t-2) with a linear extrapolation 
            The line through the points (-2,k),(-1,l)
            is p(x) = (l-k)*x + 2*l - k
            Extrapolation to the next time step yields
            p(0) = 2*l - k
            Do that for all NACs and all vector components
            If available, countercheck if transition dipole moment has also flipped sign
            One can also do a higher degree polynomial or choose more points, which uses numpy polyfit then.
            Right now, the degree is hard coded to 1 with 2 history points

            If there is not history, do not correct artificial sign flips
            if len(nac_hist) == 0:
            return nac, []
            if tdm is not None:
                use_tdm = True
            else:
                use_tdm = False
        '''
        self.hist_length = hist_length
        self.n_states = n_states
        self.n_nuc = n_nuc
        self.nac_hist = np.zeros((n_states, n_states, n_nuc, hist_length))
        self.tdm_hist = np.zeros((n_states, n_states, 3, hist_length))
        self.name = name

    def serialize(self) -> dict:
        return {
            'hist_length': self.hist_length,
            'n_states': self.n_states,
            'n_nuc': self.n_nuc,
            'nac_hist': self.nac_hist,
            'tdm_hist': self.tdm_hist,
            'name': self.name
        }
    
    @staticmethod
    def deserialize(data: dict) -> 'SignFlipper':
        sf = SignFlipper(data['n_states'], data['hist_length'], data['n_nuc'], data['name'])
        sf.nac_hist = data['nac_hist']
        sf.tdm_hist = data['tdm_hist']
        return sf

   
    def set_history(self, nac, nac_hist_in: np.ndarray, trans_dips: np.ndarray = None, tdm_hist_in: np.ndarray=None):
        # If nac_hist and tdm_hist array does not exist yet, create it as zeros array
        if nac_hist_in.size == 0:
            self.nac_hist = np.zeros((self.n_states, self.n_states, self.n_nuc, self.hist_length))
            # fill array with current nac
            for it in range(0,self.hist_length):
                self.nac_hist[:,:,:,it] = nac
        else:
            if nac_hist_in.shape[3] != self.hist_length:
                raise ValueError("nac_hist_in does not have the correct history length")
            if nac_hist_in.shape[0] != self.n_states:
                raise ValueError("nac_hist_in does not have the correct number of states")
            self.nac_hist = nac_hist_in
        # exit()
        if tdm_hist_in.size == 0:
            self.tdm_hist = np.zeros((self.n_states, self.n_states, 3, self.hist_length))
            # fill array with current tdm (if available)
            if trans_dips is not None:
                for it in range(0,self.hist_length):
                    self.tdm_hist[:,:,:,it] = trans_dips
        else:
            if tdm_hist_in.shape[3] != self.hist_length:
                raise ValueError("tdm_hist_in does not have the correct history length")
            if tdm_hist_in.shape[0] != self.n_states:
                raise ValueError("tdm_hist_in does not have the correct number of states")
            self.tdm_hist = tdm_hist_in

    def correct_nac_sign(self, nac: np.ndarray, tdm: np.ndarray=None):
        '''Check which sign for the nac is expected and correct artificial sign flips

        Parameters
        ----------
        nac: np.ndarray, shape (n, n, N)
            nonadiabatic coupling vectors for `n` states and `N` nuclei.
        tdm: np.ndarray, shape (n, n, 3)
            transition dipole moment from the ground state to each of the `n` states with `N` nuclei.
            For now, the diagonal elements (i, i, :) should be a 3-vector of zeros
        '''

        if tdm is None:
            use_tdm = False
        elif len(tdm) == 0:
            use_tdm = False
        else:
            use_tdm = True
        if self._debug: print("TDM HIST: ", self.tdm_hist, self.hist_length)

        polynom_degree = 1 # hardcoded. 1 is usually sufficient. 2 is in principle better but could lead to artificial oscillations

        # Allocate array for extrapolated vector
        nac_expol = np.empty_like(nac)
        if (polynom_degree == 1):
            # default
            # uses only the last 2 points
            nac_expol = 2.0*self.nac_hist[:,:,:,-1] - 1.0*self.nac_hist[:,:,:,-2]
        else:
            # for scientific purposes only
            # uses the whole history
            timesteps = np.arange(self.hist_length)
            for i in range(0, self.n_states):
                for j in range(0, self.n_states):
                    for ix in range(0,nac.shape[2]):
                        coefficients = np.polyfit(timesteps, self.nac_hist[i,j,ix,:], polynom_degree)
                        nac_expol[i,j,ix] = np.polyval(coefficients,self.hist_length)

        # Do similar with transition dipole moment
        if use_tdm:
            tdm_expol = np.empty_like(tdm)
            if (polynom_degree == 1):
                tdm_expol = 2.0*self.tdm_hist[:,:,:,-1] - 1.0*self.tdm_hist[:,:,:,-2]
            else:
                timesteps = np.arange(self.hist_length)
                for i in range(0, self.n_states):
                    for j in range(0, self.n_states):
                        for ix in range(0,3):
                            coefficients = np.polyfit(timesteps,self.tdm_hist[i,j,ix,:], polynom_degree)
                            tdm_expol[i,j,ix] = np.polyval(coefficients,self.hist_length)


        # check whether the TC/GAMESS vector goes in the same or opposite direction
        # (means an angle with more than 90 degree) as the estimation
        # if the angle is < 90 degree -> np.sign(dot_product)== 1 -> no flip
        # if the angle is > 90 degree -> np.sign(dot_product)==-1 -> flip
        message = ''
        for i in range(0, self.n_states):
            for j in range(i+1, self.n_states):
                flip_detected = False
                nac_dot_product = np.dot(nac[i,j,:],nac_expol[i,j,:])
                # if tdm is available: check if it also flips sign. if not, no correction
                # if tdm is not available rely only on nac
                if use_tdm:
                    tdm_dot_product = np.dot(tdm[i,j,:],tdm_expol[i,j,:])
                    sign_tdm = np.sign(tdm_dot_product)
                    sign_nac = np.sign(nac_dot_product)
                    if sign_tdm == sign_nac:
                        if sign_nac < 0:
                            flip_detected = True
                            print('FLIP 1: ', i, j, nac_dot_product, sign_tdm)
                            for k in range(0,nac.shape[2]):
                                print(f'{nac[i,j,k]:12.6f} {nac_expol[i,j,k]:12.6f}')
                        nac[i,j,:] = sign_nac*nac[i,j,:]
                        nac[j,i,:] = sign_nac*nac[j,i,:]
                        
                        tdm[i,j,:] = sign_tdm*tdm[i,j,:]
                        tdm[j,i,:] = sign_tdm*tdm[j,i,:]
                else:
                    sign = np.sign(nac_dot_product)
                    if sign < 0:
                        flip_detected = True
                        # print('FLIP 1: ', sign)

                    nac[i,j,:] = sign*nac[i,j,:]
                    nac[j,i,:] = sign*nac[j,i,:]

                if flip_detected:
                    message += f'{self.name} NAC sign-flip detected between states {i} and {j}\n'
                    mag = np.linalg.norm(nac[i,j,:])
                    message += f'  NAC magnitude: {mag}\n'
        
        if message != '':
            print(f'\n{message}\n')

        if self._debug:
            print("nac_hist vor roll: ")
            print("nh[:,:,0]")
            print(self.nac_hist[:,:,:,0])
            print("nh[:,:,1]")
            print(self.nac_hist[:,:,:,1])
            # print("nh[:,:,2]")
            # print(self.nac_hist[:,:,:,2])

        # roll array and update newest entry
        self.nac_hist = np.roll(self.nac_hist,-1,axis=3)
        self.nac_hist[:,:,:,self.hist_length-1] = nac
        if use_tdm:
            self.tdm_hist = np.roll(self.tdm_hist,-1,axis=3)
            self.tdm_hist[:,:,:,self.hist_length-1] = tdm

        if self._debug and self._ref_nac is not None:
            print("DEBUG CORRECTION")
            # Check if the NACs are the same as the reference NACsuse signs of the reference NACs
            #   this is done only for the first frame
            for i in range(self.n_states):
                for j in range(i+1, self.n_states):
                    sign = np.sign(np.dot(nac[i,j], self._ref_nac[i,j]))
                    print(f"Sign of NAC between {i} and {j} is {sign}")
                    nac[i, j] = sign * nac[i, j]
                    nac[j, i] = sign * nac[j, i]
            self._ref_nac = None

        if self._debug:
            print("nac_hist nach roll: ")
            print("nh[:,:,0]")
            print(self.nac_hist[:,:,:,0])
            print("nh[:,:,1]")
            print(self.nac_hist[:,:,:,1])
            print("nh[:,:,2]")
            # print("nac_dot[0,1] history post roll:",nac_dot_hist[0,1,0],nac_dot_hist[0,1,1],nac_dot_hist[0,1,2])
            # input()
    
        # if self._debug: print("nac_dot[0,1] history: post upda",nac_dot_hist[0,1,0],nac_dot_hist[0,1,1],nac_dot_hist[0,1,2])

        return nac