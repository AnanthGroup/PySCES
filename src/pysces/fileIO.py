import os
import json
import numpy as np
import h5py
from copy import deepcopy
import yaml
from .qcRunners import TeraChem as TC
from .input_simulation import extra_loggers


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

            return q, p, np.array([]), np.array([]), init_energy, t

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
            return combo_q, combo_p, nac_hist, tdm_hist, energy, time

        else:
            exit(f'ERROR: File extension "{extension}" is not a valid restart file')
    else:
        exit(f'ERROR: only RK4 is implimented fileIO')

def write_restart(file_loc: str, coord: list | np.ndarray, nac_hist: np.ndarray, tdm_hist: np.ndarray, energy: float, time: float, n_states: int, integrator='rk4'):
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
            data['tdm_hist'] = np.array(tdm_hist).tolist()
            with open(file_loc, 'w') as file:
                json.dump(data, file, indent=2)
    else:
        exit(f'ERROR: only RK4 is implimented fileIO')

class H5File(h5py.File):
    def __init__(self, name, mode='r', driver=None, libver=None, userblock_size=None, swmr=False, rdcc_nslots=None, rdcc_nbytes=None, rdcc_w0=None, track_order=None, fs_strategy=None, fs_persist=False, fs_threshold=1, fs_page_size=None, page_buf_size=None, min_meta_keep=0, min_raw_keep=0, locking=None, alignment_threshold=1, alignment_interval=1, meta_block_size=None, **kwds):
        super().__init__(name, mode, driver, libver, userblock_size, swmr, rdcc_nslots, rdcc_nbytes, rdcc_w0, track_order, fs_strategy, fs_persist, fs_threshold, fs_page_size, page_buf_size, min_meta_keep, min_raw_keep, locking, alignment_threshold, alignment_interval, meta_block_size, **kwds)

    @staticmethod
    def append_dataset(ds: h5py.Dataset, value):
        ds.resize(ds.shape[0]+1, axis=0)
        ds[-1] = value

    def print_data_structure(self, data: h5py.File | h5py.Dataset | h5py.Group = None, string=''):
        if data is None:
            data = self
        if isinstance(data, h5py.Group) or isinstance(data, h5py.File):
            for key, val in data.items():
                # print(f'{string}/{key}: attrs = ', dict(val.attrs))
                print(f'{string}/{key}: ')
                print(' '*len(f'{string}/{key}'), ' attrs = ', dict(data.attrs))
                self.print_data_structure(val, string + '/' + key)
        elif isinstance(data, h5py.Dataset):
            print(' '*len(string), ' shape = ', data.shape)


    def print_data(self, data: h5py.File |  h5py.Dataset | h5py.Group = None):
        if data is None:
            data = self
        out_data = {}
        if isinstance(data, h5py.Group) or isinstance(data, h5py.File):
            for key, val in data.items():
                out_data[key] = self.print_data(val)
        elif isinstance(data, h5py.Dataset):
            return {'type': str(data.dtype), 'shape': str(data.shape)}

        return out_data
    
    def to_file_and_dir(self):
        self._to_file_and_dir_details(self, '')

    @staticmethod
    def _to_file_and_dir_details(data: h5py.File |  h5py.Dataset | h5py.Group, parent_dir=''):

        if isinstance(data, h5py.Group) or isinstance(data, h5py.File):
            dir_path = data.name
            #   paths can not be at root or blank
            if dir_path != '' and dir_path != '/':
                if dir_path[0] == '/':
                    dir_path = dir_path[1:]
                os.makedirs(dir_path, exist_ok=True)

            attribs = dict(data.attrs)
            for key, val in data.items():
                val: h5py.Dataset | h5py.Group
                H5File._to_file_and_dir_details(val, dir_path)

                #   update attributes if present
                if len(val.attrs) == 0:
                    continue
                val_key = os.path.basename(val.name)
                attribs[val_key] = {}
                for attr_k, attr_v in val.attrs.items():

                    if isinstance(attr_v, np.ndarray):
                        attribs[val_key][attr_k] = attr_v.tolist()
                    else:
                        attribs[val_key][attr_k] = attr_v
                # attribs.update(dict(val.attrs))

            #   write atribues to a json file
            if len(attribs) > 0:
                json_file_path = os.path.join(dir_path, 'attr.json')
                with open(json_file_path, 'w') as file:
                    json.dump(attribs, file, indent=4)

        elif isinstance(data, h5py.Dataset):
            file_name = data.name
            #   paths can not be at root
            if file_name[0] == '/':
                file_name = file_name[1:]
            if data.dtype=='object':
                np.save(file_name + '.npy', data[:])
            elif data.ndim <= 2:
                np.savetxt(file_name + '.txt', data[:])
            else:
                np.save(file_name, data[:])


    def to_json(self, data: h5py.File |  h5py.Dataset | h5py.Group = None):
        if data is None:
            data = self
        out_data = {}
        if isinstance(data, h5py.Group) or isinstance(data, h5py.File):
            for key, val in data.items():
                print(key)
                out_data[key] = self._to_json(val)
        elif isinstance(data, h5py.Dataset):
            if data.dtype == 'object':
                conv_data = np.array(data[:]).astype(str).tolist()
            else:
                conv_data = data[:].tolist()
            return conv_data
        else:
            print("Could not convert ", data)
        return out_data

    
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
            self._h5_file = H5File(os.path.join(dir, 'logs.h5'), 'w')
            if hdf5_name == '':
                hdf5_name = 'electronic'
            self._h5_group = self._h5_file.create_group(hdf5_name)
            self._h5_group.create_dataset('time', shape=(0,), maxshape=(None,), chunks=True)

        self._loggers = []
        if save_energy:
            self._loggers.append(EnergyLogger(os.path.join(dir, 'energy.txt'), self._h5_group))
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
            self._loggers.append(TCJobsLogger(os.path.join(dir, 'jobs_data.yaml')))

        self._nuc_geo_logger = None
        if save_geo:
        #     self._loggers.append(NucGeoLogger(os.path.join(dir, 'nuc_geo.xyz')))
            self._nuc_geo_logger = NucGeoLogger(os.path.join(dir, 'nuc_geo.xyz'), self._h5_group)

        #   append additionally specified loggers
        for logger in extra_loggers:
            self._loggers.append(logger(self._h5_file))

        self.state_labels = None

    def __del__(self):
        if self._h5_file:
            self._h5_file.to_file_and_dir()


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
    def __init__(self, file_loc: str, file: h5py.File =None) -> None:
        self._file_loc = file_loc
        self._file: h5py.File = file

        self._data_fields = [
            'energy', 'gradient', 'dipole_moment', 'dipole_vector', 'nacme', 'cis_', 'cas_'
        ]
        self._units_from_field = {'energy': 'a.u.', 'dipole_moment': 'Debye'}
        self._job_datasets = {}
        self._group_name = 'tc_job_data'

        if self._file is None:
            self._file = H5File(self._file_loc, 'w')
        
    def __del__(self):
        # print(self._file.print_data_structure(self._file))
        # self._file.to_file_and_dir()

        if self._file is not None:
            self._file.close()

    def _initialize(self, cleaned_batch: TC.TCJobBatch):
        self._file.create_group(self._group_name)
        
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
            ds = group.create_dataset(name='other', shape=(1,1), maxshape=(None, 1), data='')
            ds.resize((0, 1))


        self._file.create_dataset(name = f'{self._group_name}/atoms', 
                                  data = cleaned_batch.results_list[0]['atoms'])
        geom = np.array(cleaned_batch.results_list[0]['geom'])
        geom_ds = self._file.create_dataset(name = f'{self._group_name}/geom', 
                                  data = [geom], 
                                  maxshape=(None,) + geom.shape)
        geom_ds.attrs.create('units', 'angstroms')
        

    def write(self, data: LoggerData):

        if data.jobs_data is None:
            return
        
        results: list[dict] = deepcopy(data.jobs_data.results_list)
        cleaned_results = TC.TCRunner.cleanup_multiple_jobs(results, 'orb_energies', 'bond_order', 'orb_occupations', 'spins')
        cleaned_batch = deepcopy(data.jobs_data)
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
                if key in ['other', 'timestep']:
                    continue
                H5File.append_dataset(group[job.name][key], results[key])
                results.pop(key)
            other_data = json.dumps(results)
            H5File.append_dataset(group[job.name]['other'], other_data)
            H5File.append_dataset(group[job.name]['timestep'], data.time)
            

        self._file.flush()     
    

class BaseLogger():
    def __init__(self, file_loc: str = None, h5_group: h5py.Group = None) -> None:
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
    def __init__(self, file_loc: str = None, h5_group: h5py.Group = None) -> None:
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
    def __init__(self, file_loc: str = None, h5_group: h5py.Group = None) -> None:
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
    def __init__(self, file_loc: str=None, h5_group: h5py.Group = None) -> None:
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
    def __init__(self, file_loc: str = None, h5_group: h5py.Group = None) -> None:
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
    def __init__(self, file_loc: str = None, h5_group: h5py.Group = None) -> None:
        super().__init__(file_loc, h5_group)
    
    def _initialize(self, data: LoggerData):
        # if data.all_energies is not None:
        #     labels = [f'S{i}' for i in range(len(data.all_energies))]
        # elif labels is None:
        if data.state_labels is None:
            labels = [f'S{i}' for i in range(len(data.elec_E))]
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
    def __init__(self, file_loc: str=None, h5_group: h5py.Group = None) -> None:
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
                H5File.append_dataset(self._h5_dataset, list(data.elec_E) + [data.total_E])

class GradientLogger(BaseLogger):
    def __init__(self, file_loc: str = None, h5_group: h5py.Group = None) -> None:
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
    def __init__(self, file_loc: str, h5_group: h5py.Group = None) -> None:
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