import numpy as np
import h5py
from pysces.common import PhaseVars, ESVars

class DebugRunner:
    def __init__(self, atoms: list[str], model_params: dict):
        self.A = model_params.get("A", 0.01)
        self.B = model_params.get("B", 1.6)
        self.C = model_params.get("C", 0.005)
        self.D = model_params.get("D", 1.0)
        self.atoms = atoms

    def set_logger_file(self, logger_file: h5py.File):
        """Set the logger file for storing results."""
        self.logger_file = logger_file

    def save_restart(self):
        pass

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
        print("DebugRunner: Running new geometry with nuclear coordinates:", nuc_coords)
        if len(nuc_coords) != 2:
            raise ValueError("DebugRunner requires exactly two nuclear coordinates.")
        
        R = np.linalg.norm(nuc_coords[0] - nuc_coords[1])

        energy, U = self._diagonalize(R)
        dH = self._dHdx(R)
        grads = self._gradients(R, dH, U)
        nacs = self._nac(R, dH, energy, U)

        return ESVars(
            all_energies=energy,
            elecE=energy,
            grads=grads,
            nacs=nacs,
        )


    def _H(self, x: float) -> np.ndarray:
        """Return 2x2 diabatic Hamiltonian at position x."""
        V = self.A * np.tanh(self.B * x)
        coupling = self.C * np.exp(-self.D * x**2)
        return np.array([[ V, coupling],
                         [coupling, -V]])

    def _dHdx(self, x: float) -> np.ndarray:
        """Return derivative of diabatic Hamiltonian with respect to x."""
        dVdx = self.A * self.B / np.cosh(self.B * x)**2
        dcoupling_dx = -2 * self.D * x * self.C * np.exp(-self.D * x**2)
        return np.array([[ dVdx, dcoupling_dx],
                         [dcoupling_dx, -dVdx]])

    
    def _diagonalize(self, x: float):
        H = self._H(x)
        evals, evecs = np.linalg.eigh(H)
        order = np.argsort(evals)
        evals = evals[order]
        evecs = evecs[:, order]

        return evals, evecs

    def _gradients(self, x: float, dHx: np.ndarray, U: np.ndarray) -> tuple[float, float]:
        """Return energy gradients dE1/dx and dE2/dx using Hellmann–Feynman theorem."""
        grad_E1 = U[:, 0] @ dHx @ U[:, 0]
        grad_E2 = U[:, 1] @ dHx @ U[:, 1]
        grads = np.zeros((2, 6))
        grads[0, 0] = grad_E1
        grads[1, 0] = grad_E2
        grads[0, 3] = -grad_E1
        grads[1, 3] = -grad_E2
        return grads

    def _nac(self, x: float, dHx: np.ndarray, E: np.ndarray, U: np.ndarray) -> float:

        phi1 = U[:, 0]
        phi2 = U[:, 1]
        delta_E = E[1] - E[0]
        if abs(delta_E) < 1e-8:
            return 0.0  # Avoid division by zero
        d12 = phi1 @ dHx @ phi2 / delta_E

        nac_vec = np.zeros(6)
        nac_vec[0] = d12
        nac_vec[3] = -d12  # Assuming symmetry in the coupling

        return np.array([
            [np.zeros(6), nac_vec],
            [-nac_vec, np.zeros(6)]
        ])
