import numpy as np
from scipy.interpolate import interp1d
from collections import deque
from pysces.input_simulation import * 
from typing import Optional
from abc import abstractmethod


class PhaseVars:
    def __init__(self, time=0.0, elec_q0=None, elec_p0=None, nuc_q0=None, nuc_p0=None):
        self.elec_q = elec_q0
        self.elec_p = elec_p0
        self.nuc_q = nuc_q0
        self.nuc_p = nuc_p0
        self.time = time

    def get_concatenated(self):
        return np.concatenate((self.elec_q, self.elec_p, self.nuc_q, self.nuc_p))
    
    @staticmethod
    def from_concatenated(concatenated):
        out_vars = PhaseVars()
        out_vars.elec_q = concatenated[:nel]
        out_vars.elec_p = concatenated[nel:2*nel]
        out_vars.nuc_q = concatenated[2*nel:2*nel+nnuc]
        out_vars.nuc_p = concatenated[2*nel+nnuc:]
        return out_vars
    
    @property
    def coordinates(self):
        return np.concatenate((self.elec_q, self.nuc_q))
    
    @property
    def momenta(self):
        return np.concatenate((self.elec_p, self.nuc_p))

class PhaseVarHistory:
    def __init__(self, initial_vars: PhaseVars = None, max_history=None) -> None:
        if initial_vars is None:
            initial_vars = PhaseVars()

        self.var_history = deque([initial_vars.get_concatenated()], maxlen=max_history)
        self.time_history = deque([initial_vars.time], maxlen=max_history)

        self._interp_func = None
        self._need_to_recalculate_interp = True

    def append(self, phase_vars: PhaseVars) -> None:
        if phase_vars.time < self.time_history[-1]:
            raise ValueError("Time must be increasing")

        self.var_history.append(phase_vars.get_concatenated())
        self.time_history.append(phase_vars.time)


    def evaluate(self, time) -> PhaseVars:
        #   redo the interpolation function
        if len(self.var_history) > 2:
            self._interp_func = interp1d(self.time_history, np.array(self.var_history).T, kind='quadratic', axis=0, fill_value='extrapolate')
        elif len(self.var_history) == 2:
            self._interp_func = interp1d(self.time_history, np.array(self.var_history).T, kind='linear', axis=0, fill_value='extrapolate')
        else:
            self._interp_func = lambda t: self.var_history[0] 

        return PhaseVars.from_concatenated(self._interp_func(time))
    
class QCRunner:
    def __init__(self):
        pass

    @abstractmethod
    def run_new_geom(self, phase_vars: PhaseVars, geom=None):
        pass

    def report(self):
        pass

    def cleanup(self):
        pass
    
class ESVars:
    def __init__(self, 
                all_energies: Optional[np.array] = None, 
                elecE: Optional[np.array] = None, 
                grads: Optional[np.array] = None,
                nacs: Optional[np.array] = None,
                trans_dips: Optional[np.array] = None,
    ) -> None:

        '''
        Parameters
        ----------

        all_energies : np.ndarray
            All energies of the system, including those of the states being propogated on
        elecE : np.ndarray, shape=(nstates,)
            Electronic energies being propogated on. Corresponds to the gradients and NACs.
            This is a subset of all_energies.
        grads : np.ndarray, shape=(nstates, ndof)
            Gradients of the electronic energies being propogated on.
        nacs : np.ndarray, shape=(nstates, nstates, ndof)
            Non-adiabatic couplings between the electronic energies being propogated.
        trans_dips : np.ndarray, shape=(nstates, nstates, 3)
            Transition dipoles between the electronic energies being propogated.
        '''

        self.all_energies = all_energies
        self.elecE = elecE
        self.grads = grads
        self.nacs = nacs
        self.trans_dips = trans_dips

    
