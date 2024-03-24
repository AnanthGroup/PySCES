import os
import json
import numpy as np

def read_restart(file_loc: str, ndof: int, integrator: str='RK4') -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    '''
        Reads in a restart file and extracts it's data
        Parameters
        ----------
        file_loc: str
            File path to read in. The file extension must be either .out or .json.
        integrator: str
            Integrator being used to restart the simulation
        ndof: int
            Number of DoF. Needed for .out files.
        
        Returns
        -------
            q: list[np.ndarray]
                array of initial coordinates with the first N being the electronic
                DOF and the remaining elements belonging to the nuclei
            p: list[np.ndarray]
                same as p, but with momenta as it's elements
            nac_hist: list[np.ndarray]
                history of nonadiabatic coupling vectors
            energy: float
                Total energy of the system
            time: float
                Total elapsed time
    '''
    if integrator.lower() == 'rk4':
        extension = os.path.splitext(file_loc)[-1]
        if extension == '.out':
            #   original output file data
            q, p = np.zeros(ndof), np.zeros(ndof)
            with open(file_loc, 'r') as ff:
                ff.readline() 
                for i in range(ndof):
                    x = ff.readline().split()
                    q[i], p[i] = float(x[0]), float(x[1]) # Mapping variables already in Cartesian coordinate
                [ff.readline() for i in range(2)]

                init_energy = float(ff.readline()) # Total energy
                [ff.readline() for i in range(2)]

                initial_time = float(ff.readline()) # Total simulation time at the beginning of restart run
                t = initial_time  
            return q, p, init_energy, t

        elif extension == '.json':
            #  json data format
            with open(file_loc) as file:
                data = json.load(file)
            rst_integrator = data['integrator'].lower()
            if rst_integrator != integrator.lower():
                exit(f'ERROR: Restart file integrator {rst_integrator} does not match request integrator "{integrator}"')
            nucl_q = data['nucl_q']
            nucl_p = data['nucl_p']
            elec_q = data['elec_q']
            elec_p = data['elec_p']
            energy = data['energy']
            time = data['time']
            nac_hist = np.array(data['nac_hist'])

            combo_q = np.array(elec_q + nucl_q)
            combo_p = np.array(elec_p + nucl_p)
            return combo_q, combo_p, nac_hist, energy, time

        else:
            exit(f'ERROR: File extension "{extension}" is not a valid restart file')
    else:
        exit(f'ERROR: only RK4 is implimented fileIO')

def write_restart(file_loc: str, coord: list | np.ndarray, nac_hist: np.ndarray, energy: float, time: float, n_states: int, integrator='rk4'):
    '''
        Writes a restart file for restarting a simulation from the previous conditions

        Parameters
        ----------
        file_loc: str
            File path to store the file. The file extension must be either .out or .json.
        coord: list or ndarray
            must be an array of size (2xN) where N is the total number DoF. The first row are
            the coordinates and the second are the momenta. For each row, the first M values are
            the electronic DoF and the remaining are the nuclear DoF.
        nac_hist: ndarray
            Contains the nonadiabatic coupling vectors of previous time steps
        energy: float
            The last total energy of the the system in a.u.
        time: float
            The last time stamp of the simulation in a.u.
        n_states: int
            number of electronic states
        integrator: str
            The integrator used to run the simulation
    '''
    if integrator.upper() == 'RK4':
        extension = os.path.splitext(file_loc)[-1]
        
        if extension == '.out':
            #   original output file data
            with open(file_loc, 'w') as gg:
                ndof = len(coord)
                # Write the coordinates
                gg.write('Coordinates (a.u.) at the last update: \n')
                for i in range(ndof):
                    gg.write('{:>16.10f}{:>16.10f} \n'.format(coord[0,i,-1], coord[1,i,-1]))
                gg.write('\n')

                # Record the energy and the total time
                gg.write('Energy at the last time step \n')
                gg.write('{:>16.10f} \n'.format(energy[-1]))
                gg.write('\n')

                # Record the total time
                gg.write('Total time in a.u. \n')
                gg.write('{:>16.10f} \n'.format(time))
                gg.write('\n')

        elif extension == '.json':
            coord = np.array(coord).tolist()
            data = {'time': time, 'energy': energy, 'integrator': 'rk4'}
            data['elec_q'] = coord[0][0:n_states]
            data['elec_p'] = coord[1][0:n_states]
            data['nucl_q'] = coord[0][n_states:]
            data['nucl_p'] = coord[1][n_states:]
            data['nac_hist'] = np.array(nac_hist).tolist()
            with open(file_loc, 'w') as file:
                json.dump(data, file, indent=2)
    else:
        exit(f'ERROR: only RK4 is implimented fileIO')

class SimulationLogger():
    def __init__(self, n_states, save_energy=False, save_grad=True, save_nac=True, save_corr=True, dir=None) -> None:
        if dir is None:
            dir = os.path.abspath(os.path.curdir)
        self._energy_logger = None
        self._grad_logger = None
        self._nac_logger = None
        self._corr_logger = None
        if save_energy:
            self._energy_logger = EnergyLogger(os.path.join(dir, 'energy.txt'), n_states)
        if save_grad:
            self._grad_logger = GradientLogger(os.path.join(dir, 'grad.txt'), n_states)
        if save_nac:
            self._nac_logger = NACLogger(os.path.join(dir, 'nac.txt'), n_states)
        if save_corr:
            self._corr_logger = CorrelationLogger(os.path.join(dir, 'corr.out'), n_states)

    def write(self, time, total_E=None, elec_E=None, grads=None, NACs=None, pops=None):
        if self._energy_logger is not None:
            self._energy_logger.write(time, total_E, elec_E)
        if self._grad_logger is not None:
            self._grad_logger.write(time, grads)
        if self._nac_logger is not None:
            self._nac_logger.write(time, NACs)
        if self._corr_logger is not None:
            self._corr_logger.write(time, pops)
        
class CorrelationLogger():
    def __init__(self, file_loc: str, n_states: int) -> None:
        self._total_format = '{:>12.4f}'
        for i in range(n_states + 1):
            self._total_format += '{:>16.10f}'
        self._total_format += '\n'

        self._file = open(file_loc, 'w')
        
        #   write file header
        self._file.write('%12s' % 'Time')
        self._file.write(' %16s' % 'Total')
        for i in range(n_states):
            self._file.write(' %16s' % f'S{i}')
        self._file.write('\n')
        self._write_header = False

    def __del__(self):
        self._file.close()

    def write(self, time: float, pops):
        total = np.sum(pops)
        out_str = f'{time:12.6f} {total:16.10f}'
        for i in range(len(pops)):
            out_str += f' {pops[i]:16.10f}'
        self._file.write(f'{out_str}\n')
        self._file.flush()

class EnergyLogger():
    def __init__(self, file_loc: str, n_states: int) -> None:
        self._file = open(file_loc, 'w')
        # self._write_header = True
        self._total_writes = 0

        #   write file header
        self._file.write('%12s' % 'Time')
        self._file.write(' %16s' % 'Total')
        for i in range(n_states):
            self._file.write(' %16s' % f'S{i}')
        self._file.write('\n')
        self._write_header = False

    def __del__(self):
        self._file.close()

    def write(self, time: float, total: float, elec: list | np.ndarray):
        out_str = f'{time:12.6f} {total:16.10f}'
        for i in range(len(elec)):
            out_str += f' {elec[i]:16.10f}'
        self._file.write(f'{out_str}\n')

        self._file.flush()
    

class GradientLogger():
    def __init__(self, file_loc: str, n_states: int) -> None:
        self._file = open(file_loc, 'w')
        self._n_states = n_states
        self._total_writes = 0

        # self._file.write('%16s' % 'Time')
        for i in range(n_states):
            self._file.write('%16s' % f'S{i}')
        self._file.write('\n')

    def __del__(self):
        self._file.close()

    def write(self, time, grads):

        np.savetxt(self._file, np.transpose(grads), fmt='%16.10f', 
            header=f'time_step {self._total_writes}\ntime {time}')
        self._file.flush()
        self._total_writes += 1


class NACLogger():
    def __init__(self, file_loc: str, n_NACs: int, labels: list[str] = None) -> None:
        self._file = open(file_loc, 'w')
        self._n_states = n_NACs
        self._total_writes = 0

        if labels is None:
            labels = []
            for i in range(n_NACs):
                for j in range(i+1, n_NACs):
                    labels.append(f'S{i}_S{j}')

        # self._file.write('%16s' % 'Time')
        for label in labels:
            self._file.write('%16s' % label)
        self._file.write('\n')

    def __del__(self):
        self._file.close()

    def write(self, time: float, NACs: np.ndarray):
        out_data = []
        print("N STATES: ", self._n_states)
        for i in range(self._n_states):
            for j in range(i+1, self._n_states):
                out_data.append(NACs[i, j])
        np.savetxt(self._file, np.transpose(out_data), fmt='%16.10f', 
            header=f'time_step {self._total_writes}\ntime {time}')
        self._file.flush()
        self._total_writes += 1
