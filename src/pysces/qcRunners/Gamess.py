import pysces.subroutines as subs
from pysces.common import PhaseVars, ESVars, QCRunner
import qcelemental as qcel
import time

class GamessRunner(QCRunner):

    def __init__(self, atoms, AN_mat) -> None:
        super().__init__()
        self.atoms = atoms
        self.AN_mat = AN_mat
        self.input_name = 'cas'

    def __eq__(self, o: object) -> bool:
        if isinstance(o, str):
            if o == 'GamessRunner' or o.lower() == 'gamess':
                return True
        return super().__eq__(o)

    def run_new_geom(self, phase_vars: PhaseVars, geom=None):

        if phase_vars is not None:
            geom = phase_vars.nuc_q
        elif geom is not None:
            pass
        else:
            raise ValueError('Either phase_vars or geom must be provided')
        
        start_time = time.time()
        elecE, grad, nac = subs.run_gamess_at_geom(self.input_name, self.AN_mat, geom, self.atoms)
        timings = {'Total': time.time() - start_time}

        out_data = ESVars(elecE, elecE, grad, nac, None)
        return out_data, timings