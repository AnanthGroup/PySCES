import numpy as np
from collections import deque
from scipy.integrate import simpson
from scipy.interpolate import interp1d
class GradEstimator():


    def __init__(self, order, history_size, interval=1, name='UNKNOWN'):
        self.order = order
        self.history_size = history_size
        self.interval = interval
        self.history_grads = deque(maxlen=history_size)
        self.history_t = deque(maxlen=history_size)
        self.history_f = deque(maxlen=history_size)
        self.polynomial_coeffs = None
        self.name = name


    def update_history(self, new_t, new_grads, new_f):

        #   before we update, let's get one more extrapolation to check how close we are
        if self.polynomial_coeffs is not None:
            guess = np.polyval(self.polynomial_coeffs, new_t)
            guess = guess.reshape(self.history_grads[0].shape)
            error = np.abs(guess - new_grads)
            max_error = np.max(error)
            rms_error = np.sqrt(np.mean(error**2))
            print(f'Before updating extrapolation history in {self.name} at time {new_t}: ')
            print(f'    {max_error=:.5e}, {rms_error=:.5e}')

        self.history_grads.append(np.array(new_grads))
        self.history_t.append(new_t)
        self.history_f.append(new_f)
        print(f'Updating {self.name} history at time {new_t}')
        if len(self.history_grads) == self.history_size:
            self._fit_polynomial()

    def _fit_polynomial(self):
        # Flatten the matrices and stack them into a 2D array
        stacked_history = np.vstack([np.ravel(matrix) for matrix in self.history_grads])
        # Fit a polynomial to each column of the 2D array
        self.polynomial_coeffs = np.polyfit(self.history_t, stacked_history, self.order)

    def _evaluate(self, time):
        return np.polyval(self.polynomial_coeffs, time)
    
    def __call__(self, t) -> np.ndarray:
        pass

    def check_if_ready(self):
        return len(self.history_grads) == self.history_size
    
    def check_run_deriv(self, new_time):
        if len(self.history_grads) < self.history_size:
            return True, 'History not long enough'
        elif (1+new_time - len(self.history_grads)) % self.interval == 0:
            return True, 'Interval reached'
        else:
            return False, 'OK'
    
    def guess_f(self, f0, times, velocities):
        if self.polynomial_coeffs is None:
            raise ValueError("Not enough history to fit a polynomial yet")            

        grads = self._evaluate(times)
        prod = grads * velocities[:, None]
        axis = len(grads.shape) - 2
        delta_f = simpson(prod, times, axis=axis)
        extrap_f = f0 + delta_f

        return extrap_f

    


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