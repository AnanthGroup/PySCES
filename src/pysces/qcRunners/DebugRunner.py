import numpy as np
import h5py
from pysces.common import PhaseVars, ESVars

class DebugRunner:
    def __init__(self, atoms: list[str], model_params: dict):
        self.k1 = model_params.get("k1", 0.1)          # force constant for state 1
        self.k2 = model_params.get("k2", 0.1)          # force constant for state 2
        self.R1 = model_params.get("R1", 1.0)           # equilibrium bond length for state 1
        self.R2 = model_params.get("R2", 1.5)           # equilibrium bond length for state 2
        self.V12 = model_params.get("V12", 0.005)       # constant electronic coupling
        self.atoms = atoms

        self._prev_evecs = None

    def set_logger_file(self, logger_file: h5py.File):
        """Set the logger file for storing results."""
        self.logger_file = logger_file

    def save_restart(self):
        out_data = {
            'k1': self.k1,
            'k2': self.k2,
            'R1': self.R1,
            'R2': self.R2,
            'V12': self.V12,
        }
        return out_data
    
    def load_restart(self, data: dict):
        self.k1 =  data.get("k1", 0.1)
        self.k2 =  data.get("k2", 0.1)
        self.R1 =  data.get("R1", 1.0)
        self.R2 =  data.get("R2", 1.5)
        self.V12 = data.get("V12", 0.005)

    @staticmethod
    def get_molecule_props(model_params):
        data = {
            'atoms': ['H', 'H'],
            'xyz': np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
            'hessian_vecs': np.zeros((2, 2)),
            'freq':np.zeros(2),
            'reduced_mass': np.zeros(2)
        }
        return data

    def run_new_geom(self, phase_vars: PhaseVars) -> ESVars:
        ''' run a new calculation '''
        nuc_coords = phase_vars.nuc_q.reshape(-1, 3)

        if len(nuc_coords) != 2:
            raise ValueError("DebugRunner requires exactly two nuclear coordinates.")

        energy, U = self._diagonalize(nuc_coords)
        dH = self._dHdx(nuc_coords)
        grads = self._gradients(nuc_coords, dH, U)
        nacs = self._nac(nuc_coords, dH, energy, U)

        return ESVars(
            all_energies=energy,
            elecE=energy,
            grads=grads,
            nacs=nacs,
        )


    def _H(self, coords: np.ndarray) -> np.ndarray:
        
        R = np.linalg.norm(coords[0] - coords[1])
        print(f' ######     {R:8.4f}     ######')
        V11 = 0.5 * self.k1 * (R - self.R1)**2
        V22 = 0.5 * self.k2 * (R - self.R2)**2
        return np.array([[V11, self.V12], [self.V12, V22]])

    def _dHdx(self, coords: np.ndarray) -> np.ndarray:
        R = np.linalg.norm(coords[0] - coords[1])
        dV11dx = self.k1 * (R - self.R1)
        dV22dx = self.k2 * (R - self.R2)
        return np.array([[dV11dx, 0.0], [0.0, dV22dx]])

    
    def _diagonalize(self, coords: np.ndarray):
        H = self._H(coords)
        evals, evecs = np.linalg.eigh(H)
        order = np.argsort(evals)
        evals = evals[order]
        evecs = evecs[:, order]

        if self._prev_evecs is not None:
            sign_flips = np.sign(np.sum(self._prev_evecs * evecs, axis=0))
            evecs *= sign_flips
        self._prev_evecs = evecs.copy()

        return evals, evecs

    def _gradients(self, coords: np.ndarray, dHx: np.ndarray, U: np.ndarray) -> tuple[float, float]:
        """Return energy gradients dE1/dx and dE2/dx using Hellmann–Feynman theorem."""

        dX = coords[0] - coords[1]
        dX_norm = dX / np.linalg.norm(dX)

        grad_E1 = U[:, 0] @ dHx @ U[:, 0]
        grad_E2 = U[:, 1] @ dHx @ U[:, 1]

        grads = np.zeros((2, 6))
        grads[0, 0:3] =  grad_E1 * dX_norm
        grads[0, 3:6] = -grad_E1 * dX_norm
        grads[1, 0:3] =  grad_E2 * dX_norm
        grads[1, 3:6] = -grad_E2 * dX_norm

        return grads

    def _nac(self, coords: np.ndarray, dHx: np.ndarray, E: np.ndarray, U: np.ndarray) -> float:

        dX = coords[0] - coords[1]
        dX_norm = dX / np.linalg.norm(dX)

        phi1 = U[:, 0]
        phi2 = U[:, 1]
        delta_E = E[1] - E[0]
        if abs(delta_E) < 1e-8:
            return 0.0  # Avoid division by zero
        d12 = phi1 @ dHx @ phi2 / delta_E

        nacs = np.zeros((2, 2, 6)) 
        nacs[0, 1, 0:3] =  d12 * dX_norm
        nacs[0, 1, 3:6] = -d12 * dX_norm
        nacs[1, 0] = -nacs[0, 1] 

        return nacs
