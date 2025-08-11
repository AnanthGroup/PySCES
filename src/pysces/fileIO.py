from __future__ import annotations
from typing import TYPE_CHECKING

import os
import json
import shutil
import numpy as np
import argparse
import sys
from typing import Optional

from .h5file import H5File, H5Group, H5Dataset
from . import input_simulation as opts
from .serialization import serialize, deserialize, TCRunner_Deserialize
from .common import PhaseVars, ESVars
from .interpolation import SignFlipper
from .subroutines import compute_CF_single_LSC, compute_CF_single_SQC

ANG_2_BOHR = 1.8897259886

def _to_symmetric_matrix(data: np.ndarray) -> np.ndarray:
    ''' Convert an array of upper triangular matrix elements to a full symmetric matrix '''
    a = data.shape[0]
    b = data.shape[1:]
    n = int((1 + np.sqrt(1 + 8 * a)) / 2)
    mat = np.zeros((n, n) + b)
    mat[np.triu_indices(n, 1)] = data
    mat += mat.swapaxes(0, 1)
    return mat


def run_restart_module():
    print('Running restart module')
    arg_parser = argparse.ArgumentParser(description='Run the restart module')
    arg_parser.add_argument('--file',   '-f', type=str,     help='HDF5 file to read', required=True)
    arg_parser.add_argument('--time',   '-t', type=float,   help='Time to create restart from', default=0.0)
    arg_parser.add_argument('--output', '-o', type=str,     help='Output json file name', default='restart.json')
    args = arg_parser.parse_args(sys.argv[2:])

    print(args)

    if args.file.endswith('.h5'):
        traj_file = H5File(args.file, 'r', swmr=True)
        times = np.array(traj_file['electronic/time'])
        dt = times[1] - times[0]
        print('\nReading in data from HDF5 file\n')
        print(f'Number of frames in trajectory: ', len(times))
        print(f'Start time: {times[0]} a.u.')
        print(f'End time: {times[-1]} a.u.')
        print(f'Trajectory time length: {times[-1] - times[0]} a.u.', )
        print(f'Trajectory time step: {dt} a.u.')

        #   find the closest time to the requested time
        print('\nRequest a restart file at time: ', args.time)
        time_idx = np.argmin(np.abs(times - args.time))
        print(f'Closest time step: {times[time_idx]} a.u.')

        print('\nExtracting data from the closest time step')
        elec_pq = np.array(traj_file['electronic/electric_pq'])
        elec_p = elec_pq[time_idx, 0:elec_pq.shape[1]//2]
        elec_q = elec_pq[time_idx, elec_pq.shape[1]//2:]
        nuc_P = np.array(traj_file['electronic/nuclear_P'])[time_idx].flatten()
        nuc_Q = np.array(traj_file['electronic/nuclear_Q'])[time_idx].flatten()*ANG_2_BOHR
        grads = np.array(traj_file['electronic/grad'])[time_idx]

        #   test correlation function
        q, p = elec_q, elec_p
        nel = len(q)
        pops = np.zeros(nel)
        common_TCF = 2**(nel+1) * np.exp(-np.dot(q, q) - np.dot(p, p))
        for i in range(nel):
                final_state_TCF = q[i]**2 + p[i]**2 - 0.5
                pops[i] = common_TCF * final_state_TCF
        print('Restart populations:')
        for i in range(nel):
            print(f'  State {i+1}: {pops[i]:12.8f}')
        print(f'Total population: {np.sum(pops):8f}')

        # energies
        energies = np.array(traj_file['electronic/energy'])[time_idx]
        total_e = energies[-1]
        elec_E = energies[0:-1]
        if elec_E.shape[0] != elec_p.shape[0]:
            print(f'Warning: Length of electronic energies ({len(elec_E)}) does not match electronic coordinates ({len(elec_p)})')
            print(f'         Assuming the last {len(elec_p)} energies are used in dynamics')
            elec_E = elec_E[-len(elec_p):]
        
        #   form the NAC matrix
        nac_data = traj_file['electronic/nac']
        nac = np.array(nac_data)[time_idx]
        a, b = nac.shape
        nac_mat = _to_symmetric_matrix(nac)


        #   NAC history
        if time_idx == 0:
            print('Warning: Nac history can only be formed when selected frame is not the first')
            print('         Assuming NAC history is zero')
            nac_hist = np.array([])
        else:
            print('Forming NAC history with 2 previous timesteps')
            nac_hist = np.zeros(nac_mat.shape + (2,))
            nac_hist[:,:,:,0] = _to_symmetric_matrix(np.array(nac_data)[time_idx-1])
            nac_hist[:,:,:,1] = nac_mat

        #   TDM History
        #   TODO: Add TDM history

        #   Center of Mass
        if 'com' in traj_file['electronic']:
            com = np.array(traj_file['electronic/com'])[time_idx]
        else:
            print('Warning: No COM data found\n         Assuming COM=[0, 0, 0]')
            com = np.zeros(3)

        #   Integrator type
        if 'integrator' in traj_file['electronic']:
            integrator = traj_file['electronic/integrator'][0]
        else:
            print('Warning: No integrator data found\n         Assuming RK4')
            integrator = 'RK4'


        coord_out = np.concatenate((elec_q, nuc_Q, elec_p, nuc_P))
        sign_flipper = SignFlipper(len(elec_p), 2, len(nuc_Q))
        sign_flipper.nac_hist = nac_hist


        write_restart(args.output, coord_out, sign_flipper, total_e, times[time_idx], integrator, elec_E, grads, nac_mat, com)

    exit()

def check_json_format(data: dict, root: str='/'):
    ''''
        Recursively checks if the data provided is json serializable.
    '''
    if isinstance(data, dict):
        for key, value in data.items():
            if not isinstance(key, str):
                raise ValueError(f'Key {root}/{key} is not a string')
            check_json_format(value, f'{root}/{key}')
    elif isinstance(data, (list, tuple)):
        for i, item in enumerate(data):
            check_json_format(item, f'{root}[{i}]')
    elif not isinstance(data, (str, int, float, bool, None.__class__)):
        raise ValueError(f'Value {root}/{type(data)} is not json serializable')

def read_restart(file_loc: str='restart.out', 
                 ndof: int=0, 
                 integrator: str='RK4',
                 tc_runner=None, 
                 qc_runner=None) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float, ESVars]:
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
            tdm_hist: list[np.ndarray]
                history of transition dipole moments
            energy: float
                Total energy of the system
            time: float
                Total elapsed time
    '''
    ndof = opts.ndof
    if integrator.lower() in ['rk4', 'rk4-uprop', 'verlet-uprop']:
    

        def _read_array_data(ff):
            shape = [int(x) for x in next(ff).split()]
            lines = []
            n_spaces = int(np.prod(shape[0:-2]))
            n_lines =  int(np.prod(shape[0:-1])) + n_spaces
            for i in range(n_lines):
                vals = next(ff).split()
                if len(vals) == 0:
                    continue
                lines.append([float(x) for x in vals])
            data = np.reshape(lines, shape)
            return data


        extension = os.path.splitext(file_loc)[-1]

        #   Chris: We should start to phase this out and use .json instead, as new
        #   entries are trivial to add and parse
        if extension == '.out':
            nac_hist = np.array([])
            nac_mat = np.array([])
            elecE = np.array([])
            grads = np.array([])
            nac_mat = np.array([])
            com = None
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

                for line in ff:
                    if 'COM' in line:
                        opts.com_ang = [float(x) for x in ff.readline().split()]
                    if 'NAC History' in line:
                        nac_hist = _read_array_data(ff)
                    if 'NAC Matrix' in line:
                        nac_mat = _read_array_data(ff)
                    if 'Gradients' in line:
                        grads = _read_array_data(ff)
                    if 'Electronic Energies' in line:
                        elecE = _read_array_data(ff)    

            es_vars = ESVars(elecE=elecE, grads=grads, nacs=nac_mat)
            return q, p, nac_hist, np.array([]), init_energy, t, es_vars

        elif extension == '.json':
            #  json data format
            with open(file_loc) as file:
                data = json.load(file)
            rst_integrator = data.pop('integrator').lower()
            if rst_integrator != integrator.lower():
                raise ValueError(f'ERROR: Restart file integrator {rst_integrator} does not match request integrator "{integrator}"')
            if any([x not in data for x in ['nucl_q', 'nucl_p', 'elec_q', 'elec_p', 'energy', 'time']]):
                raise ValueError(f'ERROR: Restart file requires at least the following keys: nucl_q, nucl_p, elec_q, elec_p, energy, time')


            data: dict
            nucl_q = data.pop('nucl_q')
            nucl_p = data.pop('nucl_p')
            elec_q = data.pop('elec_q')
            elec_p = data.pop('elec_p')
            energy = data.pop('energy')
            time = data.pop('time')
            nac_hist = np.array(data.pop('nac_hist', np.empty(0)))
            tdm_hist = np.array(data.pop('tdm_hist', np.empty(0)))

            combo_q = np.array(elec_q + nucl_q)
            combo_p = np.array(elec_p + nucl_p)

            if 'com' in data: 
                opts.com_ang = np.array(data.pop('com'))
   
            elecE = np.array(data.pop('elec_E')) if 'elec_E' in data else None
            grads = np.array(data.pop('grads')) if 'grads' in data else None
            nac_mat = np.array(data.pop('nac_mat')) if 'nac_mat' in data else None

            if 'TCJobBatch__batch_counter' in data:
                from .qcRunners.TeraChem import TCJobBatch
                TCJobBatch.set_ID_counter(data.pop('TCJobBatch__batch_counter'))

            if 'TCJob__job_counter' in data:
                from .qcRunners.TeraChem import TCJob
                TCJob.set_ID_counter(data.pop('TCJob__job_counter'))

            if 'TCRunner' in data:
                TCRunner_Deserialize(data.pop('TCRunner'), tc_runner)

            if 'qc_runner' in data and qc_runner is not None:
                qc_runner.load_restart(data.pop('qc_runner'))


            es_vars = ESVars(elecE=elecE, grads=grads, nacs=nac_mat)
            return combo_q, combo_p, nac_hist, tdm_hist, energy, time, es_vars

        else:
            exit(f'ERROR: File extension "{extension}" is not a valid restart file')
    else:
        exit(f'ERROR: only RK4 is implimented fileIO')

def write_restart(  coord: np.ndarray | list, 
                    sign_flipper: SignFlipper,
                    energy: float, 
                    time: float,
                    es_vars: ESVars,
                    tc_runner=None,
                    qc_runner=None,
                    file_loc: Optional[str] = None,
                    integrator: Optional[str] = None, 
                    com=None,
                    ):
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
        tdm_hist: ndarray
            Contains the transition dipole moments of previous time steps
        energy: float
            The last total energy of the the system in a.u.
        time: float
            The last time stamp of the simulation in a.u.
        integrator: str
            The integrator used to run the simulation
    '''

    #   use default values if not provided
    if integrator is None:
        integrator = opts.integrator
    if com is None:
        com = opts.com_ang
    if file_loc is None:
        file_loc = opts.restart_file_out
    

    #   electronic variables
    elecE = es_vars.elecE
    grads = es_vars.grads
    nac_mat = es_vars.nacs

    #   sign flipper data
    nac_hist = sign_flipper.nac_hist
    tdm_hist = sign_flipper.tdm_hist
    n_states = len(elecE)

    #   positions and momenta
    coord = np.reshape(coord, (2, -1))
    q = np.array(coord[0])
    p = np.array(coord[1])

    if com is None:
        com = opts.com_ang

    if integrator.lower() in ['rk4', 'rk4-uprop', 'verlet-uprop']:
        extension = os.path.splitext(file_loc)[-1]
        
        if extension == '.out':
            #   original output file data
            with open(file_loc, 'w') as gg:
                ndof = len(coord[0])
                # Write the coordinates
                gg.write('Coordinates (a.u.) at the last update: \n')
                for i in range(ndof):
                    gg.write('{:>16.10f}{:>16.10f} \n'.format(coord[0][i], coord[1][i]))
                gg.write('\n')

                # Record the energy and the total time
                gg.write('Energy at the last time step \n')
                gg.write('{:>16.10f} \n'.format(energy))
                gg.write('\n')

                # Record the total time
                gg.write('Total time in a.u. \n')
                gg.write('{:>16.10f} \n'.format(time))
                gg.write('\n')

                def write_array_data(data_name, data):
                    if data.size > 0:
                        gg.write(f'{data_name}:\n')
                        gg.write(' '.join(map(str, data.shape)) + '\n')
                        if data.ndim == 1:
                            gg.write(' '.join(map(str, data)) + '\n\n')
                        else:
                            for outer_col in data.reshape((-1,) + data.shape[-2:]):
                                for row in outer_col:
                                    gg.write(' '.join(map(str, row)) + '\n')
                                gg.write('\n')
                        # gg.write(np.array2string(data, separator=' ').replace('[', '').replace(']', '') + '\n')

                gg.write('COM: \n')
                gg.write(' '.join(map(str, com)) + '\n\n')
                write_array_data('NAC History', nac_hist)
                write_array_data('NAC Matrix', nac_mat)
                write_array_data('Gradients', grads)
                write_array_data('Electronic Energies', elecE)


        elif extension == '.json':
            coord = np.array(coord).tolist()
            data = {'time': float(time), 'energy': float(energy), 'integrator': 'rk4'}
            data['elec_q'] = coord[0][0:n_states]
            data['elec_p'] = coord[1][0:n_states]
            data['nucl_q'] = coord[0][n_states:]
            data['nucl_p'] = coord[1][n_states:]
            data['nac_hist'] = np.array(nac_hist).tolist()
            data['tdm_hist'] = np.array(tdm_hist).tolist()
            data['elec_E'] = np.array(elecE).tolist()
            data['grads'] = np.array(grads).tolist()
            data['nac_mat'] = np.array(nac_mat).tolist()
            data['com'] = np.array(com).tolist()

            from .qcRunners.TeraChem import TCJobBatch, TCJob

            if TCJobBatch.get_ID_counter() > 0:
                data['TCJobBatch__batch_counter'] = TCJobBatch.get_ID_counter()

            if TCJob.get_ID_counter() > 0:
                data['TCJob__job_counter'] = TCJob.get_ID_counter()

            if tc_runner:
                data[tc_runner.__class__.__name__] = serialize(tc_runner)

            elif qc_runner is not None and not isinstance(qc_runner, str):
                serialized_runner = qc_runner.save_restart()
                check_json_format(serialized_runner, root='qc_runner')
                data['qc_runner'] = serialized_runner

            with open(file_loc, 'w') as file:
                json.dump(data, file, indent=2)
            
    else:
        exit(f'ERROR: only RK4 is implimented fileIO')

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)
    
#   TODO: Convert to a @dataclass
class LoggerData():
    def __init__(self, time, atoms=None, total_E=None, elec_E=None, grads=None, NACs=None, timings=None, elec_p=None, elec_q=None, nuc_p=None, nuc_q=None, state_labels=None, all_energies = None) -> None:
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
        self.all_energies = all_energies


class SimulationLogger():
    def __init__(self, save_energy=True, save_grad=True, save_nac=True, save_corr=True, save_timigs=False, dir=None, save_geo=True, save_elec=True, save_p=True, save_jobs=True, atoms=None, hdf5=False, hdf5_name='') -> None:
        if dir is None:
            dir = os.path.abspath(os.path.curdir)
        self.atoms = atoms

        self._h5_file = None
        self._h5_group = None
        if hdf5:
            if 'r' in opts.logging_mode and os.path.exists('logs.h5'):
                raise ValueError('logging mode must be either "w", "a", or "x/w-"')
            if opts.logging_mode == 'w' and os.path.exists('logs.h5'):
                for n in range(100):
                    if not os.path.exists(f'logs_{n}.h5'):
                        shutil.move('logs.h5', f'logs_{n}.h5')
                        break
            self._h5_file = H5File('logs.h5', opts.logging_mode, libver='latest')
            self._h5_file.swmr_mode = True

            if hdf5_name == '':
                hdf5_name = 'electronic'
            self._h5_group = self._h5_file.create_group(hdf5_name)
            self._h5_group.create_dataset('time', shape=(0,), maxshape=(None,), chunks=True)

        self.loggers = {}
        if save_energy:
            self.loggers[EnergyLogger.name] =      EnergyLogger(os.path.join(dir, 'energy.txt'), self._h5_group)
            self.loggers[ExEnergyLogger.name] =    ExEnergyLogger(os.path.join(dir, 'ex_energy.txt'), self._h5_group)
        if save_grad:
            self.loggers[GradientLogger.name] =    GradientLogger(os.path.join(dir, 'grad.txt'), self._h5_group)
        if save_nac:
            self.loggers[NACLogger.name] =         NACLogger(os.path.join(dir, 'nac.txt'), self._h5_group)
        if save_corr:
            self.loggers[CorrelationLogger.name] = CorrelationLogger(os.path.join(dir, 'corr.txt'), self._h5_group)
        if save_timigs:
            self.loggers[TimingsLogger.name] =     TimingsLogger(os.path.join(dir, 'timings.txt'), self._h5_group)
        if save_elec:
            self.loggers[ElectricPQLogger.name] =  ElectricPQLogger(os.path.join(dir, 'electric_pq.txt'), self._h5_group)
        if save_p:
            self.loggers[NuclearPLogger.name] =    NuclearPLogger(os.path.join(dir, 'nuclear_P.txt'), self._h5_group)
        # if save_jobs:
        #     # self.loggers[TCJobsLogger.name] =      TCJobsLogger(None, self._h5_file)
        #     # self.loggers[TCJobsLogger.name] =      None
        #     self.loggers[TCJobsLoggerSequential.name] =      TCJobsLoggerSequential(None, self._h5_file)


        self._nuc_geo_logger = None
        if save_geo:
            self._nuc_geo_logger = NucGeoLogger(os.path.join(dir, 'nuc_geo.xyz'), self._h5_group)

        #   append additionally specified loggers
        for extra_logger in opts.extra_loggers:
            extra_logger.setup(dir, self._h5_file)
            self.loggers[extra_logger.name] = extra_logger

        self.state_labels = opts.state_labels


    def write(self, time, total_E=None, es_vars: ESVars = None, coord=None):
        all_energies = es_vars.all_energies if es_vars is not None else None
        elecE = es_vars.elecE if es_vars is not None else None
        grads = es_vars.grads if es_vars is not None else None
        NACs = es_vars.nacs if es_vars is not None else None
        timings = es_vars.timings if es_vars is not None else None

        nel = opts.nel
        ndof = opts.ndof
        elec_q = coord[0:ndof][0:nel] if coord is not None else None
        nuc_q = coord[0:ndof][nel:] if coord is not None else None
        elec_p = coord[ndof:][0:nel] if coord is not None else None
        nuc_p = coord[ndof:][nel:] if coord is not None else None


        data = LoggerData(time, self.atoms, total_E, elecE, grads, NACs, timings, elec_p, elec_q, nuc_p, None, self.state_labels, all_energies)
        for logger in self.loggers.values():
            logger.write(data)

        if self._h5_group:
            H5File.append_dataset(self._h5_group['time'], data.time)
            if 'atoms' not in self._h5_group:
                out_data = [np.void(str.encode(x)) for x in data.atoms]
                out_data = np.array(data.atoms, dtype='S10')
                # g = self._h5_group.create_dataset('atoms', data=out_data)

        if self._h5_file:
            self._h5_file.flush()
class BaseLogger():
    name = 'Unnamed Logger'
    def __init__(self, file_loc: str = None, h5_group: H5Group = None) -> None:
        self._file = None
        if file_loc:
            self._file = open(file_loc, 'w')
        self._h5_dataset = None
        self._h5_group = h5_group
        self._initialized = False
        self._next_dataset: H5Dataset = None

    def __del__(self):
        if self._file:
            self._file.close()

    def set_next_dataset(self, dataset: H5Dataset):
        self._next_dataset = dataset

    def write(self, data: LoggerData) -> None:
        if not self._initialized:
            self._initialize(data)
            self._initialized = True
 
    
class NucGeoLogger():
    def __init__(self, file_loc: str = None, h5_group: H5Group = None) -> None:
        self._file = None
        if file_loc:
            self._file = open(file_loc, 'w')
        self._h5_dataset = None
        self._h5_group = h5_group
        self._initialized = False

    def __del__(self):
        if self._file:
            self._file.close()

    def _initialize(self, qCart_ang):
        self._initialized = True
        if self._h5_group:
            tmp_data = np.array(qCart_ang).reshape((-1, 3))
            self._h5_dataset = self._h5_group.create_dataset('nuclear_Q', shape=(0,)+tmp_data.shape, maxshape=(None,)+tmp_data.shape)

    def write(self, total_time: float, atoms, qCart_ang, com=None):
        if not self._initialized:
            self._initialize(qCart_ang)

        if com is None:
            com = opts.com_ang

        if self._file:
            natom = len(atoms)
            self._file.write('%d \n' %natom)
            self._file.write('%f \n' %total_time)
            for i in range(natom):
                self._file.write('{:<5s}{:>12.6f}{:>12.6f}{:>12.6f} \n'.format(
                    atoms[i],
                    qCart_ang[3*i+0] + com[0],
                    qCart_ang[3*i+1] + com[1],
                    qCart_ang[3*i+2] + com[2]))
            self._file.flush()
        if self._h5_dataset:
            shifted = np.array(qCart_ang).reshape((-1, 3)) + com
            H5File.append_dataset(self._h5_dataset, shifted)
        
class ElectricPQLogger(BaseLogger):
    name = 'electric_pq'

    def __init__(self, file_loc: str = None, h5_group: H5Group = None) -> None:
        super().__init__(file_loc, h5_group)

    def _initialize(self, data: LoggerData):
        n_states = len(data.elec_p)
        p_labels = [f'p{i}' for i in range(n_states)]
        q_labels = [f'q{i}' for i in range(n_states)]
        if self._file:
            #   write file header
            self._file.write('%12s ' % 'Time')
            for i in range(n_states):
                self._file.write(' %16s' % p_labels[i])
                self._file.write(' %16s' % q_labels[i])
            self._file.write('\n')
        if self._h5_group:
            n_elms = len(data.elec_p) + len(data.elec_q)
            self._h5_dataset = self._h5_group.create_dataset(self.name, shape=(0, n_elms), maxshape=(None, n_elms))
            self._h5_dataset.attrs.create('labels', p_labels + q_labels)
        # self._initialized = True

    def write(self, data: LoggerData):
        super().write(data)
        time = data.time
        if self._file:
            out_str = f'{time:12.6f} '
            for i in range(len(data.elec_p)):
                out_str += f' {data.elec_p[i]:16.10f}'
                out_str += f' {data.elec_q[i]:16.10f}'
            self._file.write(f'{out_str}\n')
            self._file.flush()
        if self._h5_dataset:
            H5File.append_dataset(self._h5_dataset, list(data.elec_p) + list(data.elec_q))

class NuclearPLogger(BaseLogger):
    name = 'nuclear_P'

    def __init__(self, file_loc: str=None, h5_group: H5Group = None) -> None:
        super().__init__(file_loc, h5_group)

    def _initialize(self, data: LoggerData):
        if self._h5_group:
            tmp_data = np.array(data.nuc_p).reshape((-1, 3))
            self._h5_dataset = self._h5_group.create_dataset(self.name, shape=(0,) + tmp_data.shape, maxshape=(None,) + tmp_data.shape)

    def write(self, data: LoggerData):
        super().write(data)
        if self._file:
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
        if self._h5_dataset:
            H5File.append_dataset(self._h5_dataset, np.array(data.nuc_p).reshape((-1, 3)))

class TimingsLogger(BaseLogger):
    name = 'timings'

    def __init__(self, file_loc: str = None, h5_group: H5Group = None) -> None:
        super().__init__(file_loc, h5_group)
        labels = ['gradient_0', 'gradient_n', 'nac_0_n', 'nac_n_m', 'total']
        self._descriptions = ['Ground state gradient', 'Excited state gradients', 
                              'Ground-excited NACs', 'Excited-excited NACs', 'Total']
        self._label_to_desc = dict(zip(labels, self._descriptions))
        self._totals = {l: 0.0 for l in labels}
        self._n_steps = 0

    def __del__(self):
        super().__del__()
        self._print_final_sumamry()

    def _print_final_sumamry(self):
        #   print final summary
        n_steps = self._n_steps
        if self._n_steps == 0:
            n_steps = 1
        total = self._totals['total']
        if total == 0.0:
            return
        print("Electronic Structure Average Timings")
        for l, v in self._totals.items():
            description = self._label_to_desc[l] + ':'
            print(f'    {description:25s} {v/n_steps:8.2f} s  {100*v/total:5.1f} %')
        print()

    def _initialize(self, data: LoggerData):
        if self._file:
            self._file.write(f'{"Total_QC":>12s}')
            for key, value in data.timings.items():
                self._file.write(f'{key:>12s}')
            self._file.write('\n')

        if self._h5_group:
            n_elems = len(data.timings) + len(self._label_to_desc)
            self._h5_dataset = self._h5_group.create_dataset(self.name, shape=(0, n_elems), maxshape=(None, n_elems))
            all_labels = list(data.timings.keys()) + list(self._label_to_desc.keys())
            self._h5_dataset.attrs.create('labels', all_labels)

    def write(self, data: LoggerData):
        super().write(data)

        times = data.timings
        #   compute total
        total = 0.0
        for key, value in times.items():
            total += value
            # if 'gradient_' in key or 'nac_' in key or 'energy_' in key:
                # total += value

        if self._file:
            # Write timings
            self._file.write(f'{total:12.3f}')
            for key, value in times.items():
                self._file.write(f'{value:12.3f}')
            self._file.write('\n')
            self._file.flush()

        #   print a sumamry for this timestep
        print("Electronic Structure Timings:")
        g_0, g_n, d_0n, d_nm, other = 0.0, 0.0, 0.0, 0.0, 0.0
        for key, value in times.items():
            if 'gradient_0' == key:
                g_0 = value
                self._totals['gradient_0'] += value
            elif 'gradient_' in key:
                g_n += value
                self._totals['gradient_n'] += value
            elif 'nac_0_' in key:
                d_0n += value
                self._totals['nac_0_n'] += value
            elif 'nac_' in key:
                d_nm += value
                self._totals['nac_n_m'] += value
            else:
                other += value
        self._totals['total'] += total
        
        print(f'    Ground state gradient:   {g_0:.2f} s')
        print(f'    Excited state gradients: {g_n:.2f} s')
        print(f'    Ground-Excited NACs:     {d_0n:.2f} s')
        print(f'    Excited-Excited NACs:    {d_nm:.2f} s')
        print(f'    Other:                   {other:.2f} s')
        print(f'    Total:                   {total:.2f} s')
        print("")
        self._n_steps += 1

        if self._h5_dataset:
            all_times = list(times.values()) + [g_0, g_n, d_0n, d_nm, total]
            H5File.append_dataset(self._h5_dataset, all_times)

class CorrelationLogger(BaseLogger):
    name = 'corr'

    def __init__(self, file_loc: str = None, h5_group: H5Group = None) -> None:
        super().__init__(file_loc, h5_group)
    
    def _initialize(self, data: LoggerData):
        # if data.all_energies is not None:
        #     labels = [f'S{i}' for i in range(len(data.all_energies))]
        # elif labels is None:

        if data.state_labels is None:
            labels = [f'S{i}' for i in range(len(data.elec_q))]
        else:
            labels = data.state_labels

        if opts.debug_eff_wigner_nel > 0:
            labels += [f'Sx{i}' for i in range(len(data.elec_q), opts.debug_eff_wigner_nel)]
            
        if self._file:
            #   write file header
            self._file.write('%12s' % 'Time')
            self._file.write(' %16s' % 'Total')
            # if labels is None:
            #     labels = [f'S{i}' for i in range(n_states)]
            for l in labels:
                self._file.write(' %16s' % l)
            self._file.write('\n')
        if self._h5_group:
            self._h5_dataset = self._h5_group.create_dataset(self.name, shape=(0, len(labels)+1), maxshape=(None, len(labels)+1))
            self._h5_dataset.attrs.create('labels', labels + ['Total'])

    def write(self, data: LoggerData):
        super().write(data)
        p, q = data.elec_p, data.elec_q
        time = data.time
        
        ### Compute the estimator of electronic state population ###
        if opts.debug_eff_wigner_nel > len(q):
            print(f'DEBUG: Using {opts.debug_eff_wigner_nel} effective Wigner states')
            nel = opts.debug_eff_wigner_nel
            n_diff = nel - len(q)
            p = np.concatenate((p, np.zeros(n_diff)))
            q = np.concatenate((q, np.ones(n_diff)*np.sqrt(0.5)))
        else:
            nel = len(q)
        
        if not opts.sqc:
            pops = compute_CF_single_LSC(q, p)
        else:
            pops = compute_CF_single_SQC(q, p)

        total = np.sum(pops)

        if self._file:
            out_str = f'{time:12.6f} {total:16.10f}'
            for i in range(len(pops)):
                out_str += f' {pops[i]:16.10f}'
            self._file.write(f'{out_str}\n')
            self._file.flush()
        if self._h5_dataset:
            H5File.append_dataset(self._h5_dataset, pops.tolist() + [total])

class EnergyLogger(BaseLogger):
    name = 'energy'
    def __init__(self, file_loc: str=None, h5_group: H5Group = None) -> None:
        super().__init__(file_loc, h5_group)


    def _initialize(self, data: LoggerData):
        if data.all_energies is not None:
            labels = [f'S{i}' for i in range(len(data.all_energies))]
        elif data.state_labels is None:
            labels = [f'S{i}' for i in range(len(data.elec_E))]
        else:
            labels = data.state_labels

        if self._file:
            self._file.write('%12s' % 'Time')
            self._file.write(' %16s' % 'Total')
            for l in labels:
                self._file.write(' %16s' % l)
            self._file.write('\n')
        
        if self._h5_group:
            self._h5_dataset = self._h5_group.create_dataset(self.name, shape=(0, len(labels)+1), maxshape=(None, len(labels)+1))
            self._h5_dataset.attrs.create('labels', list(labels) + ['total'])

    def write(self, data: LoggerData):
        super().write(data)
        if self._file:
            out_str = f'{data.time:12.6f} {data.total_E:16.10f}'
            if data.all_energies is not None:
                energies = data.all_energies
            else:
                energies = data.elec_E
            for e in energies:
                out_str += f' {e:16.10f}'
            self._file.write(f'{out_str}\n')
            self._file.flush()
        
        if self._h5_dataset:
            if data.all_energies is not None:
                H5File.append_dataset(self._h5_dataset, list(data.all_energies) + [data.total_E,])
            else:
                H5File.append_dataset(self._h5_dataset, list(data.elec_E) + [data.total_E, ])

class ExEnergyLogger(BaseLogger):
    name = 'ex_energy'
    def __init__(self, file_loc: str=None, h5_group: H5Group = None) -> None:
        super().__init__(file_loc, h5_group)

    def _initialize(self, data: LoggerData):
        if data.all_energies is None:
            print('WARNING: Excited state energies requires that all energies be provided')
            print('Excited state energies will not be logged')
            self._file.close()
            self._file = None
            return
        
        labels = [f'S{i}' for i in range(1, len(data.all_energies))]

        if self._file:
            self._file.write('%12s' % 'Time')
            for l in labels:
                self._file.write(' %16s' % l)
            self._file.write('\n')
        
        if self._h5_group:
            self._h5_dataset = self._h5_group.create_dataset(self.name, shape=(0, len(labels)), maxshape=(None, len(labels)))
            self._h5_dataset.attrs.create('labels', labels)

    def write(self, data: LoggerData):
        super().write(data)
        if self._file:
            out_str = f'{data.time:12.6f}'
            ex_energies = (np.array(data.all_energies[1:]) - data.all_energies[0])*27.2114079527 # Convert Hartree to eV

            for e in ex_energies:
                out_str += f' {e:16.10f}'
            self._file.write(f'{out_str}\n')
            self._file.flush()
        
        if self._h5_dataset:
            H5File.append_dataset(self._h5_dataset, ex_energies)


class GradientLogger(BaseLogger):
    name = 'grad'

    def __init__(self, file_loc: str = None, h5_group: H5Group = None) -> None:
        super().__init__(file_loc, h5_group)
        self._total_writes = 0

    def _initialize(self, data: LoggerData):
        labels = data.state_labels
        n_states = len(data.grads)
        if labels is None:
            labels = [f'S{i}' for i in range(n_states)]

        if self._file:
            for i in range(n_states):
                self._file.write('%16s' % labels[i])
            self._file.write('\n')

        if self._h5_group:
            self._h5_dataset = self._h5_group.create_dataset(self.name, shape=(0,) + data.grads.shape, maxshape=(None, ) + data.grads.shape)
            self._h5_dataset.attrs.create('labels', labels)


    def write(self, data: LoggerData):
        super().write(data)

        grads = data.grads
        time = data.time

        if self._file:
            np.savetxt(self._file, np.transpose(grads), fmt='%16.10f', 
                header=f'time_step {self._total_writes}\ntime {time}')
            self._file.flush()

        if self._h5_dataset:
            H5File.append_dataset(self._h5_dataset, data.grads)

        self._total_writes += 1

class NACLogger(BaseLogger):
    name = 'nac'

    def __init__(self, file_loc: str, h5_group: H5Group = None) -> None:
        super().__init__(file_loc, h5_group)
        self._total_writes = 0


    def _initialize(self, data: LoggerData):
        n_NACs = len(data.NACs)

        state_labels = data.state_labels
        if data.state_labels is None:
            state_labels = [f'S{i}' for i in range(n_NACs)]

        labels = []
        for i in range(n_NACs):
            for j in range(i+1, n_NACs):
                labels.append(f'{state_labels[i]}_{state_labels[j]}')

        if self._file:
            for label in labels:
                self._file.write('%16s' % label)
            self._file.write('\n')
            self._initialized = True
        if self._h5_group:
            indices = np.triu_indices(data.NACs.shape[0], 1)
            tmp_data = data.NACs[indices]
            shape = (0, ) + tmp_data.shape
            self._h5_dataset = self._h5_group.create_dataset(self.name, shape=shape, maxshape=(None, ) + tmp_data.shape)
            self._h5_dataset.attrs.create('labels', labels)

    def write(self, data: LoggerData):
        super().write(data)
        NACs = data.NACs
        time = data.time
        _n_states = len(NACs)

        out_data = []
        for i in range(_n_states):
            for j in range(i+1, _n_states):
                out_data.append(NACs[i, j])
        if self._file:
            np.savetxt(self._file, np.transpose(out_data), fmt='%15.10f', 
                header=f'time_step {self._total_writes}\ntime {time}')
            self._file.flush()
            self._total_writes += 1

        if self._h5_dataset:
            H5File.append_dataset(self._h5_dataset, out_data)

def print_ascii_art():
    #   Useful for debugging
    if os.environ.get('NO_SPLASH', None):
        return
    art = '''                                                                                     
                                        @@@@@@@                                 
                                 @@@@@@@@@   @@@@@@@@@                          
                                 @@@                  @@@@@                     
                                   @@@                    @@@@                  
                                     @@                      @@@                
                              @@@@@@@@@   @@@@@@@@             @@@              
                          @@@@@                  @@@@@@@         @@@            
                       @@@                             @@@@@       @@@          
                     @@@                                   @@       @@          
                   @@                                              @@@          
                 @@@                                              @@            
                @@@                              @@@@@@     @@     @@@          
               @@@   @@@@@@@@@@@@@@@            @@    @@  @@@        @@         
              @@  @@@@              @@@@        @@     @@@@           @@        
          @@@@  @@@                    @@@@     @@@@@@  @@             @@       
       @@@@      @@                 @@@@@@@@@       @@@@@@        @@@  @@@      
    @@@          @@               @@@       @@         @@         @@@   @@      
   @@     @@@     @@                @@@@@                                @@     
  @@   @@@@ @@   @@                     @@@@@@      @@@                  @@@    
 @@@@@@     @@  @@                         @@@    @@@  @@@                @@    
 @@@       @@ @@@                         @@@@@@@@@       @@@@            @@    
           @@@@                            @@@                @@@@@       @@    
          @@@@                                                    @@@@@@@@@     
          @@                                                                                                                                           
                                                                                
       @@@@@@@@              @@@@@@@@     @@@@@@@@    @@@@@@@@   @@@@@@@@       
       @@@    @@@           @@@    @@@  @@@@    @@@   @@        @@@    @@@      
       @@@    @@@ @@@   @@@ @@@@@       @@            @@         @@@@           
       @@@@@@@@@   @@  @@@    @@@@@@@  @@@            @@@@@@@@     @@@@@@@      
       @@@         @@@ @@          @@@  @@@     @@@   @@               @@@      
       @@@          @@@@@   @@@   @@@@   @@@   @@@@   @@        @@@@   @@@      
       @@@           @@@      @@@@@@       @@@@@@     @@@@@@@@    @@@@@@        
                  @@@@@                                                         
                  @@@                                                                                                                          
'''
    print(art)