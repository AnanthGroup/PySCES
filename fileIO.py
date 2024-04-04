import os
import json
import numpy as np

def read_restart(file_loc: str='restart.out', ndof: int=0, integrator: str='RK4') -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
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
            if ndof <= 0:
                raise ValueError('`ndof` must be supplied when using .out restart files')
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
            return q, p, np.array([]), init_energy, t

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

class LoggerData():
    def __init__(self, time, atoms=None, total_E=None, elec_E=None, grads=None, NACs=None, timings=None, elec_p=None, elec_q=None, nuc_p=None, nuc_q=None, state_labels=None) -> None:
        self.time = time
        self.atoms = atoms
        self.total_E = total_E
        self.elec_E = elec_E
        self.grads = grads
        self.NACs = NACs
        self.timings = timings
        self.elec_p = elec_p
        self.elec_q = elec_q
        self.nuc_p = nuc_p
        self.nuc_q = nuc_q
        self.state_labels = state_labels

class SimulationLogger():
    def __init__(self, n_states, save_energy=True, save_grad=True, save_nac=True, save_corr=True, save_timigs=True, dir=None, save_geo=True, save_elec=True, save_p=True, atoms=None) -> None:
        if dir is None:
            dir = os.path.abspath(os.path.curdir)
        self.atoms = atoms


        self._loggers = []
        if save_energy:
            self._loggers.append(EnergyLogger(os.path.join(dir, 'energy.txt')))
        if save_grad:
            self._loggers.append(GradientLogger(os.path.join(dir, 'grad.txt')))
        if save_nac:
            self._loggers.append(NACLogger(os.path.join(dir, 'nac.txt')))
        if save_corr:
            self._loggers.append(CorrelationLogger(os.path.join(dir, 'corr.txt')))
        if save_timigs:
            self._loggers.append(TimingsLogger(os.path.join(dir, 'timings.txt')))
        if save_elec:
            self._loggers.append(ElectricPQLogger(os.path.join(dir, 'electric_pq.txt')))
        if save_p:
            self._loggers.append(NuclearPLogger(os.path.join(dir, 'nuclear_P.txt')))

        self._nuc_geo_logger = None
        if save_geo:
        #     self._loggers.append(NucGeoLogger(os.path.join(dir, 'nuc_geo.xyz')))
            self._nuc_geo_logger = NucGeoLogger(os.path.join(dir, 'nuc_geo.xyz'))

        self.state_labels = None


    def write(self, time, total_E=None, elec_E=None, grads=None, NACs=None, timings=None, elec_p=None, elec_q=None, nuc_p=None, nuc_q=None):
        data = LoggerData(time, self.atoms, total_E, elec_E, grads, NACs, timings, elec_p, elec_q, nuc_p, None, self.state_labels)
        for logger in self._loggers:
            logger.write(data)

        #TODO: add nuc_geo logging here

class NucGeoLogger():
    def __init__(self, file_loc: str) -> None:
        self._file = open(file_loc, 'w')

    def __del__(self):
        self._file.close()

    def write(self, total_time: float, atoms, qCart_ang, com_ang=None):
        if com_ang is None:
            com_ang = np.zeros(3)

        natom = len(atoms)
        self._file.write('%d \n' %natom)
        self._file.write('%f \n' %total_time)
        for i in range(natom):
            self._file.write('{:<5s}{:>12.6f}{:>12.6f}{:>12.6f} \n'.format(
                atoms[i],
                qCart_ang[3*i+0] + com_ang[0],
                qCart_ang[3*i+1] + com_ang[1],
                qCart_ang[3*i+2] + com_ang[2]))
        self._file.flush()
        
class ElectricPQLogger():
    def __init__(self, file_loc: str) -> None:
        self._file = open(file_loc, 'w')
        self._write_header = True

    def __del__(self):
        self._file.close()

    def _write_header_to_file(self, n_states):
        #   write file header
        self._file.write('%12s ' % 'Time')
        for i in range(n_states):
            self._file.write(' %16s' % f'p{i}')
            self._file.write(' %16s' % f'q{i}')
        self._file.write('\n')
        self._write_header = False

    def write(self, data: LoggerData):
        time = data.time
        if self._write_header:
            self._write_header_to_file(len(data.elec_p))
        out_str = f'{time:12.6f} '
        for i in range(len(data.elec_p)):
            out_str += f' {data.elec_p[i]:16.10f}'
            out_str += f' {data.elec_q[i]:16.10f}'
        self._file.write(f'{out_str}\n')
        self._file.flush()

class NuclearPLogger():
    def __init__(self, file_loc: str) -> None:
        self._file = open(file_loc, 'w')
        self._write_header = True

    def __del__(self):
        self._file.close()

    def write(self, data: LoggerData):
        atoms = data.atoms
        natom = len(atoms)
        self._file.write('%d \n' %natom)
        self._file.write('%f \n' %data.time)
        for i in range(natom):
            self._file.write('{:<5s}{:>12.6f}{:>12.6f}{:>12.6f} \n'.format(
                atoms[i],
                data.nuc_p[3*i+0],
                data.nuc_p[3*i+1],
                data.nuc_p[3*i+2]))
        self._file.flush()

class TimingsLogger():
    def __init__(self, file_loc: str) -> None:
        self._file = open(file_loc, 'w')
        self._write_header = True

    def __del__(self):
        self._file.close()

    def write(self, data: LoggerData):
        times = data.timings
        if self._write_header:
            #   write file header
            self._file.write(f'{"Total":>12s}')
            for key, value in times.items():
                self._file.write(f'{key:>12s}')
            self._file.write('\n')
            self._write_header = False

        #   compute total
        total = 0.0
        for key, value in times.items():
            total += value

        # Write timings
        self._file.write(f'{total:12.3f}')
        for key, value in times.items():
            self._file.write(f'{value:12.3f}')
        self._file.write('\n')
        self._file.flush()

class CorrelationLogger():
    def __init__(self, file_loc: str) -> None:
        self._file = open(file_loc, 'w')
        self._write_header = True
    
    def _write_header_to_file(self, n_states, labels=None):
        #   write file header
        self._file.write('%12s' % 'Time')
        self._file.write(' %16s' % 'Total')
        if labels is None:
            labels = [f'S{i}' for i in range(self._n_states)]
        for i in range(n_states):
            self._file.write(' %16s' % labels[i])
        self._file.write('\n')
        self._write_header = False

    def __del__(self):
        self._file.close()

    def write(self, data: LoggerData):
        p, q = data.elec_p, data.elec_q
        time = data.time
        if self._write_header:
            self._write_header_to_file(len(p), data.state_labels)
        ### Compute the estimator of electronic state population ###
        nel = len(q)
        pops = np.zeros(nel)
        common_TCF = 2**(nel+1) * np.exp(-np.dot(q, q) - np.dot(p, p))
        for i in range(nel):
                final_state_TCF = q[i]**2 + p[i]**2 - 0.5
                pops[i] = common_TCF * final_state_TCF
    
        total = np.sum(pops)
        out_str = f'{time:12.6f} {total:16.10f}'
        for i in range(len(pops)):
            out_str += f' {pops[i]:16.10f}'
        self._file.write(f'{out_str}\n')
        self._file.flush()

class EnergyLogger():
    def __init__(self, file_loc: str) -> None:
        self._file = open(file_loc, 'w')
        self._write_header = True
        self._total_writes = 0

    def __del__(self):
        self._file.close()

    def _write_header_to_file(self, labels=None):
        self._file.write('%12s' % 'Time')
        self._file.write(' %16s' % 'Total')
        if labels is None:
            labels = [f'S{i}' for i in range(self._n_states)]
        for i in range(self._n_states):
            self._file.write(' %16s' % labels[i])
        self._file.write('\n')
        self._write_header = False

    def write(self, data: LoggerData):
        self._n_states = len(data.elec_E)
        if self._write_header:
            self._write_header_to_file(data.state_labels)
        out_str = f'{data.time:12.6f} {data.total_E:16.10f}'
        for i in range(len(data.elec_E)):
            out_str += f' {data.elec_E[i]:16.10f}'
        self._file.write(f'{out_str}\n')
        self._file.flush()

class GradientLogger():
    def __init__(self, file_loc: str) -> None:
        self._file = open(file_loc, 'w')
        self._total_writes = 0
        self._write_header = True
        
    def __del__(self):
        self._file.close()

    def _write_header_to_file(self, labels=None):
        if labels is None:
            labels = [f'S{i}' for i in range(self._n_states)]
        for i in range(self._n_states):
            self._file.write('%16s' % labels[i])
        self._file.write('\n')

    def write(self, data: LoggerData):
        grads = data.grads
        time = data.time
        self._n_states = len(grads)
        if self._write_header:
            self._write_header_to_file(data.state_labels)
        np.savetxt(self._file, np.transpose(grads), fmt='%16.10f', 
            header=f'time_step {self._total_writes}\ntime {time}')
        self._file.flush()
        self._total_writes += 1

class NACLogger():
    def __init__(self, file_loc: str, labels: list[str] = None) -> None:
        self._file = open(file_loc, 'w')
        self._write_header = True
        self._total_writes = 0
        self._labels = labels

    def __del__(self):
        self._file.close()

    def _write_header_to_file(self, state_labels=None):
        n_NACs = self._n_states
        if state_labels is None:
            state_labels = [f'S{i}' for i in range(self._n_states)]
        if self._labels is None:
            labels = []
            for i in range(n_NACs):
                for j in range(i+1, n_NACs):
                    # labels.append(f'S{i}_S{j} ')
                    print(i, j)
                    labels.append(f'{state_labels[i]}_{state_labels[j]} ')

        # self._file.write('%16s' % 'Time')
        for label in labels:
            self._file.write('%16s' % label)
        self._file.write('\n')
        self._write_header = False

    def write(self, data: LoggerData):
        NACs = data.NACs
        time = data.time
        self._n_states = len(NACs)
        if self._write_header:
            self._write_header_to_file(data.state_labels)
        out_data = []
        print("N STATES: ", self._n_states)
        for i in range(self._n_states):
            for j in range(i+1, self._n_states):
                out_data.append(NACs[i, j])
        np.savetxt(self._file, np.transpose(out_data), fmt='%15.10f', 
            header=f'time_step {self._total_writes}\ntime {time}')
        self._file.flush()
        self._total_writes += 1
