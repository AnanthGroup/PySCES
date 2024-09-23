import os
import json
import shutil
import numpy as np
import h5py
from copy import deepcopy
import yaml
from .qcRunners import TeraChem as TC
from .h5file import H5File, H5Group, H5Dataset
# from .input_simulation import extra_loggers, logging_mode
from . import input_simulation as opts


def read_restart(file_loc: str='restart.out', ndof: int=0, integrator: str='RK4') -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float]:
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
    if integrator.lower() == 'rk4':

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
                        com = [float(x) for x in ff.readline().split()]
                    if 'NAC History' in line:
                        nac_hist = _read_array_data(ff)
                    if 'NAC Matrix' in line:
                        nac_mat = _read_array_data(ff)
                    if 'Gradients' in line:
                        grads = _read_array_data(ff)
                    if 'Electronic Energies' in line:
                        elecE = _read_array_data(ff)    

            return q, p, nac_hist, np.array([]), init_energy, t, elecE, grads, nac_mat, com

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
            nac_hist = np.array(data.get('nac_hist', np.array([])))
            tdm_hist = np.array(data.get('tdm_hist', np.array([])))

            combo_q = np.array(elec_q + nucl_q)
            combo_p = np.array(elec_p + nucl_p)

            com = np.array(data.get('com', np.array([])))
            elecE = np.array(data.get('elec_E', np.array([])))
            grads = np.array(data.get('grads', np.array([])))
            nac_mat = np.array(data.get('nac_mat', np.array([])))

            return combo_q, combo_p, nac_hist, tdm_hist, energy, time, elecE, grads, nac_mat, com

        else:
            exit(f'ERROR: File extension "{extension}" is not a valid restart file')
    else:
        exit(f'ERROR: only RK4 is implimented fileIO')

def write_restart(file_loc: str, coord: list | np.ndarray, nac_hist: np.ndarray, tdm_hist: np.ndarray, energy: float, time: float, n_states: int, integrator='rk4', elecE: float=np.empty(0), grads: np.ndarray = np.empty(0), nac_mat: np.ndarray=np.empty(0), com=None):
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
            data = {'time': time, 'energy': energy, 'integrator': 'rk4'}
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
            with open(file_loc, 'w') as file:
                json.dump(data, file, indent=2)
    else:
        exit(f'ERROR: only RK4 is implimented fileIO')


    
#   TODO: Convert to a @dataclass
class LoggerData():
    def __init__(self, time, atoms=None, total_E=None, elec_E=None, grads=None, NACs=None, timings=None, elec_p=None, elec_q=None, nuc_p=None, nuc_q=None, state_labels=None, jobs_data=None, all_energies = None) -> None:
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

        #   place holder for all other types of data
        self.jobs_data: dict | TC.TCJobBatch = jobs_data

class SimulationLogger():
    def __init__(self, save_energy=True, save_grad=True, save_nac=True, save_corr=True, save_timigs=True, dir=None, save_geo=True, save_elec=True, save_p=True, save_jobs=True, atoms=None, hdf5=False, hdf5_name='') -> None:
        if dir is None:
            dir = os.path.abspath(os.path.curdir)
        self.atoms = atoms

        self._h5_file = None
        self._h5_group = None
        if hdf5:
            # self._h5_file = H5File(os.path.join(dir, 'logs.h5'), 'w')
            if 'r' in opts.logging_mode and os.path.exists('logs.h5'):
                raise ValueError('logging mode must be either "w", "a", or "x/w-"')
            if opts.logging_mode == 'w' and os.path.exists('logs.h5'):
                for n in range(100):
                    if not os.path.exists(f'logs_{n}.h5'):
                        shutil.move('logs.h5', f'logs_{n}.h5')
                        break
            self._h5_file = H5File('logs.h5', opts.logging_mode)
            if hdf5_name == '':
                hdf5_name = 'electronic'
            self._h5_group = self._h5_file.create_group(hdf5_name)
            self._h5_group.create_dataset('time', shape=(0,), maxshape=(None,), chunks=True)

        self._loggers = []
        if save_energy:
            self._loggers.append(EnergyLogger(os.path.join(dir, 'energy.txt'), self._h5_group))
            self._loggers.append(ExEnergyLogger(os.path.join(dir, 'ex_energy.txt'), self._h5_group))
        if save_grad:
            self._loggers.append(GradientLogger(os.path.join(dir, 'grad.txt'), self._h5_group))
        if save_nac:
            self._loggers.append(NACLogger(os.path.join(dir, 'nac.txt'), self._h5_group))
        if save_corr:
            self._loggers.append(CorrelationLogger(os.path.join(dir, 'corr.txt'), self._h5_group))
        if save_timigs:
            self._loggers.append(TimingsLogger(os.path.join(dir, 'timings.txt'), self._h5_group))
        if save_elec:
            self._loggers.append(ElectricPQLogger(os.path.join(dir, 'electric_pq.txt'), self._h5_group))
        if save_p:
            self._loggers.append(NuclearPLogger(os.path.join(dir, 'nuclear_P.txt'), self._h5_group))
        if save_jobs:
            self._loggers.append(TCJobsLogger(self._h5_file))

        self._nuc_geo_logger = None
        if save_geo:
        #     self._loggers.append(NucGeoLogger(os.path.join(dir, 'nuc_geo.xyz')))
            self._nuc_geo_logger = NucGeoLogger(os.path.join(dir, 'nuc_geo.xyz'), self._h5_group)

        #   append additionally specified loggers
        for ex_logger in opts.extra_loggers:
            ex_logger.setup(dir, self._h5_file)
            self._loggers.append(ex_logger)

        self.state_labels = opts.state_labels

    def __del__(self):
        pass
        # if self._h5_file:
        #     self._h5_file.to_file_and_dir()


    def write(self, time, total_E=None, elec_E=None, grads=None, NACs=None, timings=None, elec_p=None, elec_q=None, nuc_p=None, nuc_q=None, jobs_data=None, all_energies=None):
        data = LoggerData(time, self.atoms, total_E, elec_E, grads, NACs, timings, elec_p, elec_q, nuc_p, None, self.state_labels, jobs_data, all_energies)
        for logger in self._loggers:
            logger.write(data)

        if self._h5_group:
            H5File.append_dataset(self._h5_group['time'], data.time)
            if 'atoms' not in self._h5_group:
                out_data = [np.void(str.encode(x)) for x in data.atoms]
                out_data = np.array(data.atoms, dtype='S10')
                # g = self._h5_group.create_dataset('atoms', data=out_data)

        if self._h5_file:
            self._h5_file.flush()

        #TODO: add nuc_geo logging here

class TCJobsLogger_OLD():
    def __init__(self, file_loc: str) -> None:
        self._file_loc = file_loc
        self._file = None

        if not _TC_AVAIL:
            raise ImportError('Could not import TeraChem Runner: TCJobsLogger not available for use')
        
    def __del__(self):
        if self._file is not None:
            self._file.close()

    def write(self, data: LoggerData):
        if data.jobs_data is None:
            return
        if self._file is None:
            self._file = open(self._file_loc, 'w')

        if isinstance(data.jobs_data, TC.TCJobBatch):
            results = data.jobs_data.results_list
        else:
            results = deepcopy(data.jobs_data)
        cleaned = TC.TCRunner.cleanup_multiple_jobs(results, 'orb_energies', 'bond_order', 'orb_occupations', 'spins')

        out_data = {'time': data.time, 'jobs_data': cleaned}
        yaml.dump(out_data, self._file, allow_unicode=True, explicit_start=True, default_flow_style=False, sort_keys=False)
        self._file.flush()

class TCJobsLogger():
    # def __init__(self, file_loc: str, file: h5py.File =None) -> None:
    def __init__(self, file: str | h5py.File) -> None:

        if isinstance(file, str):
            self._file_loc = file
            self._file = None
        elif isinstance(file, h5py.File):
            self._file_loc = None
            self._file = file
        else:
            raise ValueError('file must be a string or h5py.File object')

        self._data_fields = [
            'energy', 'gradient', 'dipole_moment', 'dipole_vector', 'nacme', 'cis_', 'cas_'
        ]
        self._units_from_field = {'energy': 'a.u.', 'dipole_moment': 'Debye'}
        self._job_datasets = {}
        self._group_name = 'tc_job_data'

        if self._file is None:
            self._file = H5File(self._file_loc, 'w')
        
    def __del__(self):
        if self._file is not None:
            self._file.close()

    def _initialize(self, cleaned_batch: TC.TCJobBatch):
        self._file.create_group(self._group_name)
        
        str_dt = h5py.string_dtype(encoding='utf-8')

        for job in cleaned_batch.jobs:
            group = self._file[self._group_name].create_group(name=job.name)
            group.create_dataset(name='timestep', shape=(0,1), maxshape=(None, 1))
            for key, value in job.results.items():
                for k in self._data_fields:
                    if k in key:
                        if isinstance(value, list):
                            shape = (0,) + np.shape(value)
                        else:   #   assume it is a single value
                            shape = (0,1)
                        ds = group.create_dataset(name=key, shape=shape, maxshape=(None,) + shape[1:])
                        if k in self._units_from_field:
                            ds.attrs['units'] = self._units_from_field[k]

            #   couldn't figure out how to initialize with an empty shape when using strings,
            #   so I just resized afterwards
            ds = group.create_dataset(name='tc.out', shape=(1,1), maxshape=(None, 1), data='', dtype=str_dt)
            ds.resize((0, 1))
            ds = group.create_dataset(name='other', shape=(1,1), maxshape=(None, 1), data='', dtype=str_dt)
            ds.resize((0, 1))


        self._file.create_dataset(name = f'{self._group_name}/atoms', 
                                  data = cleaned_batch.results_list[0]['atoms'], dtype=str_dt)
        geom = np.array(cleaned_batch.results_list[0]['geom'])
        geom_ds = self._file.create_dataset(name = f'{self._group_name}/geom', 
                                            shape=(0,) + geom.shape,
                                            maxshape=(None,) + geom.shape)
        geom_ds.attrs.create('units', 'angstroms')
        

    def write(self, data: LoggerData):

        if data.jobs_data is None:
            return
        
        results: list[dict] = deepcopy(data.jobs_data.results_list)
        cleaned_results = TC.TCRunner.cleanup_multiple_jobs(results, 'orb_energies', 'bond_order', 'orb_occupations', 'spins')
        # cleaned_batch = deepcopy(data.jobs_data)
        cleaned_batch = data.jobs_data
        for job, res in zip(cleaned_batch.jobs, cleaned_results):
            job.results = res

        #   the first job is used to establish dataset sizes
        if self._group_name not in self._file:
            self._initialize(cleaned_batch)

        group = self._file[self._group_name]
        H5File.append_dataset(group['geom'], cleaned_batch.results_list[0]['geom'])
        for job in cleaned_batch.jobs:
            results = job.results.copy()
            results.pop('geom')
            results.pop('atoms')
            for key in group[job.name]:
                if key in ['other', 'timestep', 'tc.out']:
                    continue
                H5File.append_dataset(group[job.name][key], results[key])
                results.pop(key)
            H5File.append_dataset(group[job.name]['tc.out'], json.dumps(results['tc.out']))
            results.pop('tc.out')
            H5File.append_dataset(group[job.name]['timestep'], data.time)

            #   everything else goes into 'other'
            other_data = json.dumps(results)
            H5File.append_dataset(group[job.name]['other'], other_data)
            
        self._file.flush()     
    

class BaseLogger():
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

    def write(self, total_time: float, atoms, qCart_ang, com_ang=None):
        if not self._initialized:
            self._initialize(qCart_ang)
        if com_ang is None:
            com_ang = np.zeros(3)

        if self._file:
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
        if self._h5_dataset:
            shifted = np.array(qCart_ang).reshape((-1, 3)) + com_ang
            H5File.append_dataset(self._h5_dataset, shifted)
        
class ElectricPQLogger(BaseLogger):
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
            self._h5_dataset = self._h5_group.create_dataset('electric_pq', shape=(0, n_elms), maxshape=(None, n_elms))
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
    def __init__(self, file_loc: str=None, h5_group: H5Group = None) -> None:
        super().__init__(file_loc, h5_group)

    def _initialize(self, data: LoggerData):
        if self._h5_group:
            tmp_data = np.array(data.nuc_p).reshape((-1, 3))
            self._h5_dataset = self._h5_group.create_dataset('nuclear_P', shape=(0,) + tmp_data.shape, maxshape=(None,) + tmp_data.shape)

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
            self._h5_dataset = self._h5_group.create_dataset('timings', shape=(0, n_elems), maxshape=(None, n_elems))
            all_labels = list(data.timings.keys()) + list(self._label_to_desc.keys())
            self._h5_dataset.attrs.create('labels', all_labels)

    def write(self, data: LoggerData):
        super().write(data)

        times = data.timings
        #   compute total
        total = 0.0
        for key, value in times.items():
            if 'gradient_' in key or 'nac_' in key or 'energy_' in key:
                total += value

        if self._file:
            # Write timings
            self._file.write(f'{total:12.3f}')
            for key, value in times.items():
                self._file.write(f'{value:12.3f}')
            self._file.write('\n')
            self._file.flush()

        #   print a sumamry for this timestep
        print("Electronic Structure Timings:")
        g_0, g_n, d_0n, d_nm = 0.0, 0.0, 0.0, 0.0
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
        self._totals['total'] += total
        
        print(f'    Ground state gradient:  { g_0:.2f} s')
        print(f'    Excited state gradients: {g_n:.2f} s')
        print(f'    Ground-Excited NACs:     {d_0n:.2f} s')
        print(f'    Excited-Excited NACs:    {d_nm:.2f} s')
        print(f'    Total:                   {total:.2f} s')
        print("")
        self._n_steps += 1

        if self._h5_dataset:
            all_times = list(times.values()) + [g_0, g_n, d_0n, d_nm, total]
            H5File.append_dataset(self._h5_dataset, all_times)

class CorrelationLogger(BaseLogger):
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
            self._h5_dataset = self._h5_group.create_dataset('corr', shape=(0, len(labels)+1), maxshape=(None, len(labels)+1))
            self._h5_dataset.attrs.create('labels', labels + ['Total'])

    def write(self, data: LoggerData):
        super().write(data)
        p, q = data.elec_p, data.elec_q
        time = data.time
        
        ### Compute the estimator of electronic state population ###
        nel = len(q)
        pops = np.zeros(nel)
        common_TCF = 2**(nel+1) * np.exp(-np.dot(q, q) - np.dot(p, p))
        for i in range(nel):
                final_state_TCF = q[i]**2 + p[i]**2 - 0.5
                pops[i] = common_TCF * final_state_TCF
    
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
            self._h5_dataset = self._h5_group.create_dataset('energy', shape=(0, len(labels)+1), maxshape=(None, len(labels)+1))
            self._h5_dataset.attrs.create('labels', labels + ['total'])

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
    def __init__(self, file_loc: str=None, h5_group: H5Group = None) -> None:
        super().__init__(file_loc, h5_group)

    def _initialize(self, data: LoggerData):

        if data.all_energies is None:
            print('WARNING: Excited state energies requires that all energies be provided')
            print('Excited state energies will not be logged')
            return
        
        labels = [f'S{i}' for i in range(1, len(data.all_energies))]

        if self._file:
            self._file.write('%12s' % 'Time')
            for l in labels:
                self._file.write(' %16s' % l)
            self._file.write('\n')
        
        if self._h5_group:
            self._h5_dataset = self._h5_group.create_dataset('ex_energy', shape=(0, len(labels)), maxshape=(None, len(labels)))
            self._h5_dataset.attrs.create('labels', labels)

    def write(self, data: LoggerData):
        super().write(data)
        if self._file:
            out_str = f'{data.time:12.6f}'
            ex_energies = (data.all_energies[1:] - data.all_energies[0])*27.2114079527 # Convert Hartree to eV

            for e in ex_energies:
                out_str += f' {e:16.10f}'
            self._file.write(f'{out_str}\n')
            self._file.flush()
        
        if self._h5_dataset:
            H5File.append_dataset(self._h5_dataset, ex_energies)


class GradientLogger(BaseLogger):
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
            self._h5_dataset = self._h5_group.create_dataset('grad', shape=(0,) + data.grads.shape, maxshape=(None, ) + data.grads.shape)
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
            self._h5_dataset = self._h5_group.create_dataset('nac', shape=(0,) + tmp_data.shape, maxshape=(None, ) + tmp_data.shape)
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