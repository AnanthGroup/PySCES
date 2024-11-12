import numpy as np

class SignFlipper():
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
        # exit()
        if tdm_hist_in.size == 0:
            self.tdm_hist = np.zeros((self.n_states, self.n_states, 3, self.hist_length))
            # fill array with current tdm (if available)
            if trans_dips is not None:
                for it in range(0,self.hist_length):
                    self.tdm_hist[:,:,:,it] = trans_dips

    def correct_nac_sign(self, nac: np.ndarray, tdm: np.ndarray=None, debug=False):
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
        if debug: print("TDM HIST: ", self.tdm_hist, self.hist_length)


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
                        nac[i,j,:] = sign_nac*nac[i,j,:]
                        nac[j,i,:] = sign_nac*nac[j,i,:]
                        
                        tdm[i,j,:] = sign_tdm*tdm[i,j,:]
                        tdm[j,i,:] = sign_tdm*tdm[j,i,:]
                else:
                    sign = np.sign(nac_dot_product)
                    if sign < 0:
                        flip_detected = True

                    nac[i,j,:] = sign*nac[i,j,:]
                    nac[j,i,:] = sign*nac[j,i,:]

                if flip_detected:
                    message += f'{self.name} NAC sign-flip detected between states {i} and {j}\n'
                    mag = np.linalg.norm(nac[i,j,:])
                    message += f'  NAC magnitude: {mag}\n'
        
        if message != '':
            print(f'\n{message}\n')

        if debug:
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

        if debug:
            print("nac_hist nach roll: ")
            print("nh[:,:,0]")
            print(self.nac_hist[:,:,:,0])
            print("nh[:,:,1]")
            print(self.nac_hist[:,:,:,1])
            print("nh[:,:,2]")
            # print("nac_dot[0,1] history post roll:",nac_dot_hist[0,1,0],nac_dot_hist[0,1,1],nac_dot_hist[0,1,2])
            input()
    
        # if debug: print("nac_dot[0,1] history: post upda",nac_dot_hist[0,1,0],nac_dot_hist[0,1,1],nac_dot_hist[0,1,2])

        return nac