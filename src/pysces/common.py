import numpy as np
from scipy.interpolate import interp1d
from collections import deque
# from pysces.input_simulation import * 
import pysces.input_simulation as opts
from typing import Optional
from abc import abstractmethod


class PhaseVars:
    def __init__(self, time, elec_q0=None, elec_p0=None, nuc_q0=None, nuc_p0=None):
        self.time = time
        self.elec_q = elec_q0
        self.elec_p = elec_p0
        self.nuc_q = nuc_q0
        self.nuc_p = nuc_p0
        

    def get_vec(self):
        # return np.concatenate((self.elec_q, self.elec_p, self.nuc_q, self.nuc_p))
        return np.concatenate((self.elec_q, self.nuc_q, self.elec_p, self.nuc_p))
    
    @staticmethod
    def from_vec(time, vec):
        nel = opts.nel
        nnuc = opts.natom*3
        ndof = nel + nnuc
        out_vars = PhaseVars(time)
        out_vars.elec_q = vec[:nel]
        out_vars.nuc_q = vec[nel:ndof]
        out_vars.elec_p = vec[ndof:ndof+nel]
        out_vars.nuc_p = vec[ndof+nel:]
        return out_vars
    
    @property
    def elec_nuc_q(self):
        if self.elec_q is None or self.nuc_q is None:
            return None
        return np.concatenate((self.elec_q, self.nuc_q))
    
    @property
    def elec_nuc_p(self):
        if self.elec_p is None or self.nuc_p is None:
            return None
        return np.concatenate((self.elec_p, self.nuc_p))

class PhaseVarHistory(deque):

    def __init__(self) -> None:
        super().__init__()
        self._cache = {}

    def append(self, obj):
        if not isinstance(obj, PhaseVars):
            raise TypeError("Only PhaseVars objects can be added to history.")
        super().append(obj)
        self._cache.clear()

    def __getattribute__(self, __name: str):
        if __name in ['time', 'elec_q', 'elec_p', 'nuc_q', 'nuc_p', 'elec_nuc_q', 'elec_nuc_p', 'get_vec']:
            if __name in self._cache:
                return self._cache[__name]
            else:
                # self._cache[__name] = [getattr(obj, __name) for obj in self]
                self._cache[__name] = super().__getattribute__(__name)
                return self._cache[__name]

        return super().__getattribute__(__name)

    @property
    def time(self):
        return np.array([obj.time for obj in self])
    
    @property
    def elec_q(self):
        return np.array([obj.elec_q for obj in self])
    
    @property
    def elec_p(self):
        return np.array([obj.elec_p for obj in self])
    
    @property
    def nuc_q(self):
        return np.array([obj.nuc_q for obj in self])
    
    @property
    def nuc_p(self):
        return np.array([obj.nuc_p for obj in self])
    
    @property
    def elec_nuc_q(self):
        return np.array([obj.elec_nuc_q for obj in self])
    
    @property
    def elec_nuc_p(self):
        return np.array([obj.elec_nuc_p for obj in self])
    
    @property
    def get_vec(self):
        return np.array([obj.get_vec() for obj in self])

class ESResults:
    def __init__(self,
                time: Optional[float] = None,
                all_energies: Optional[np.array] = None, 
                elecE: Optional[np.array] = None, 
                grads: Optional[np.array] = None,
                nacs: Optional[np.array] = None,
                trans_dips: Optional[np.array] = None,
                timings: Optional[dict[str, float]] = None
    ) -> None:

        '''
        Parameters
        ----------
        time : float
            Simulation time at which the results are obtained
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
        self.timings = timings

        if all_energies is not None and self.elecE is None:
            self.elecE = all_energies

class ESResultsHistory(deque):
    def __init__(self):
        super().__init__()
        self._cache = {}

    def append(self, obj):
        if not isinstance(obj, ESResults):
            raise TypeError("Only ESResults objects can be added to history.")
        super().append(obj)
        self._cache.clear()

    def __getattribute__(self, __name: str):
        if __name in ['time', 'all_energies', 'elecE', 'grads', 'nacs', 'trans_dips', 'timings']:
            if __name in self._cache:
                return self._cache[__name]
            else:
                self._cache[__name] = np.array([getattr(obj, __name) for obj in self])
                return self._cache[__name]

        return super().__getattribute__(__name)

    @property
    def time(self):
        return np.array([obj.time for obj in self])

    @property
    def all_energies(self):
        return np.array([obj.all_energies for obj in self])

    @property
    def elecE(self):
        return np.array([obj.elecE for obj in self])

    @property
    def grads(self):
        return np.array([obj.grads for obj in self])

    @property
    def nacs(self):
        return np.array([obj.nacs for obj in self])

    @property
    def trans_dips(self):
        return np.array([obj.trans_dips for obj in self])

    @property
    def timings(self):
        return np.array([obj.timings for obj in self])


class QCRunner:
    def __init__(self):
        pass

    @abstractmethod
    def run_new_geom(self, phase_vars: PhaseVars, geom=None) -> ESResults:
        pass

    def report(self):
        pass

    def cleanup(self):
        pass