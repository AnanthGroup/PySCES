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
                time: float,
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

        self.time = time
        self.all_energies = all_energies
        self.elecE = elecE
        self.grads = grads
        self.nacs = nacs
        self.trans_dips = trans_dips

class _HistoryInterpolation:
    def __init__(self):
        self._data_history = None
        self._time_history = None

        self._interpolation_function = None

    def append(self, vals: np.array, time: float) -> None:
        if vals is None:
            self._interpolation_function = None
            return
        
        if self._data_history is None:
            self._data_history = deque([vals], maxlen=100)
            self._time_history = deque([time], maxlen=100)
        else:
            if time < self._time_history[-1]:
                raise ValueError("Time must be increasing")
            self._data_history.append(vals)
            self._time_history.append(time)

        self._interpolation_function = None  # Invalidate the interpolation function

    def __call__(self, time: float) -> np.array:
        if self._data_history is None or len(self._data_history) == 0:
            raise ValueError("No data to interpolate")
        
        valid_history = [x for x in self._data_history if x is not None]
        time_history = [t for t, x in zip(self._time_history, self._data_history) if x is not None]

        if self._interpolation_function is None:
            if len(valid_history) > 2:
                self._interpolation_function = interp1d(time_history, valid_history, kind='quadratic', axis=0, fill_value='extrapolate')
            elif len(valid_history) == 2:
                self._interpolation_function = interp1d(time_history, valid_history, kind='linear', axis=0, fill_value='extrapolate')
            else:
                self._interpolation_function = lambda t: self._data_history[-1]

        interpolated_vals = self._interpolation_function(time)
        return interpolated_vals

    def __getitem__(self, idx: int) -> np.array:
        if self._data_history is None or len(self._data_history) == 0:
            return np.array([])
        if idx >= len(self._data_history):
            raise IndexError(f'index {idx} out of range for history of length {len(self._data_history)}')
        return self._data_history[idx]

class ESVarsHistory:
    def __init__(self, initial_vars: ESVars = None, max_history=None) -> None:

        self._time_history = deque([0.0], maxlen=max_history)
        self._all_energies_history = _HistoryInterpolation()
        self._ElecE_history = _HistoryInterpolation()
        self._grads_history = _HistoryInterpolation()
        self._nacs_history = _HistoryInterpolation()
        self._trans_dips_history = _HistoryInterpolation()

        if initial_vars is not None:
            self.append(initial_vars)

    def append(self, es_vars: ESVars) -> None:
        if es_vars.time < self._time_history[-1]:
            raise ValueError("Time must be increasing")

        self._time_history.append(es_vars.time)
        self._all_energies_history.append(es_vars.all_energies, es_vars.time)
        self._ElecE_history.append(es_vars.elecE, es_vars.time)
        self._grads_history.append(es_vars.grads, es_vars.time)
        self._nacs_history.append(es_vars.nacs, es_vars.time)
        self._trans_dips_history.append(es_vars.trans_dips, es_vars.time)


    @property
    def all_energies(self):
        return self._all_energies_history
    
    @property
    def elecE(self):
        return self._ElecE_history

    @property
    def grads(self):
        return self._grads_history

    @property
    def nacs(self):
        return self._nacs_history

    @property
    def trans_dips(self):
        return self._trans_dips_history
