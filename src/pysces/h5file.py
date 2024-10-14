import pickle
import h5py
import json
import numpy as np
import os
import sys
import argparse
import gzip
import fnmatch

class H5Group(h5py.Group):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def create_group(self, name, track_order=None, exist_ok=True):
        if exist_ok and name in self:
            return self[name]
        grp = super().create_group(name, track_order)
        return H5Group(grp.id)
    
    def create_dataset(self, name, shape=None, dtype='f8', data=None, exist_ok=True, **kwargs):
        if name in self and exist_ok:
            return self[name]
        dset = super().create_dataset(name, shape, dtype, data, **kwargs)
        return H5Dataset(dset.id)

class H5Dataset(h5py.Dataset):
    def __init__(self, bind, *, readonly=False):
        super().__init__(bind, readonly=readonly)

    def append(self, value):
        self.resize(self.shape[0]+1, axis=0)
        self[-1] = value

class H5File(h5py.File):
    def __init__(self, name, mode='r', driver=None, libver=None, userblock_size=None, swmr=False, rdcc_nslots=None, rdcc_nbytes=None, rdcc_w0=None, track_order=None, fs_strategy=None, fs_persist=False, fs_threshold=1, fs_page_size=None, page_buf_size=None, min_meta_keep=0, min_raw_keep=0, locking=None, alignment_threshold=1, alignment_interval=1, meta_block_size=None, **kwds):
        super().__init__(name, mode, driver, libver, userblock_size, swmr,
                 rdcc_nslots, rdcc_nbytes, rdcc_w0, track_order,
                 fs_strategy, fs_persist, fs_threshold, fs_page_size,
                 page_buf_size, min_meta_keep, min_raw_keep, locking,
                 alignment_threshold, alignment_interval, meta_block_size, **kwds)
        
        self._dictionaried_data = None

    def create_group(self, name, track_order=None, exist_ok=True):
        if exist_ok and name in self:
            return self[name]
        
        grp = super().create_group(name, track_order)
        return H5Group(grp.id)
    
    def create_dataset(self, name, shape=None, dtype='f8', data=None, exist_ok=True, **kwargs):
        if name in self and exist_ok:
            return self[name]
        dset = super().create_dataset(name, shape, dtype, data, **kwargs)
        return H5Dataset(dset.id)
    
    # Override __getitem__ to return either H5Group or H5Dataset
    def __getitem__(self, name):
        obj = super().__getitem__(name)
        if isinstance(obj, h5py.Group):
            return H5Group(obj.id)
        elif isinstance(obj, h5py.Dataset):
            return H5Dataset(obj.id)
        return obj

    @staticmethod
    def append_dataset(ds: h5py.Dataset, value):
        ds.resize(ds.shape[0]+1, axis=0)
        ds[-1] = value

    def get_data_structure(self, data: h5py.File | h5py.Dataset | h5py.Group = None):
        if data is None:
            data = self
        
        shapes = {}
        def _print_dataset_info(name, obj):
            if isinstance(obj, h5py.Dataset):
                shapes[name] = obj.shape
        data.visititems(_print_dataset_info)
        return shapes

    def print_data_structure(self, data: h5py.File | h5py.Dataset | h5py.Group = None, string=''):
        if data is None:
            data = self

        shapes = {}
        def _print_dataset_info(name, obj):
            if isinstance(obj, h5py.Dataset):
                print(f"Dataset: {name}, Shape: {obj.shape}")
                shapes[name] = obj.shape

        data.visititems(_print_dataset_info)
        print()
        self.print_trajectory_summary(data)

    def print_trajectory_summary(self, data: h5py.File | h5py.Dataset | h5py.Group = None):
        if data is None:
            data = self
        print('Trajectory Summary')
        print('------------------')
        times = np.array(data['electronic/time'])
        dt = times[1] - times[0]
        print(f'Number of frames in trajectory: ', len(times))
        print(f'Start time: {times[0]} a.u.')
        print(f'End time: {times[-1]} a.u.')
        print(f'Trajectory time length: {times[-1] - times[0]} a.u.', )
        print(f'Trajectory time step: {dt} a.u.')

    @staticmethod
    def combine_files(new_file_loc: str, *file_locs):
        structures = {}
        open_files = []
        for f in file_locs:
            file = H5File(f, 'r')
            open_files.append(file)
            structures[f] = file.get_data_structure()

            print()
            print(f"File: {f}")
            file.print_trajectory_summary()
            print()
        
        #   make sure all files have the same structure, except for hte number of frames
        for file in file_locs[1:]:
            for ds_path in structures[file]:
                if ds_path not in structures[file_locs[0]]:
                    print(f"Dataset {ds_path} in {file_locs[0]} not found in {file}")
                    return
                if structures[file][ds_path][1:] != structures[file_locs[0]][ds_path][1:]:
                    print(f"Dataset {ds_path} in {file_locs[0]} has shape {structures[file_locs[0]][ds_path]} while {file} has shape {structures[file][ds_path]}")

        #   TODO: Fix these later
        exceptions = ['tc_job_data/atoms', 'tc_job_data/gradient_1/cis_dipole_deriv', 'tc_job_data/gradient_2/cis_transition_dipole_deriv', 'tc_job_data/gradient_1/other']
        str_dt = h5py.string_dtype(encoding='utf-8')

        #   TODO: re-order based on simulation time
        #   TODO: check for gaps or replicants in time

        with H5File(new_file_loc, 'w') as new_file:
            for ds in structures[file_locs[0]]:
                print('Combining dataset:', ds)

                try:

                    #   first copy over all data as numpy arrays
                    new_data = np.concatenate([open_files[0][ds][:]] + [file[ds][:] for file in open_files[1:]])

                    #   create the new dataset
                    new_ds = new_file.create_dataset(ds, data=new_data)
                    new_ds.attrs.update(open_files[0][ds].attrs)
                
                except Exception as e:
                    print(f"Failed to combine dataset {ds}")
                    print(e)

        print('New dataset stucture:')
        with H5File(new_file_loc, 'r') as new_file:
            new_file.print_data_structure()

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
    
    def to_file_and_dir(self, data_path: str = None, frame=None):
        if data_path is None:
            self._to_file_and_dir_details(self)
        else:

            if isinstance(self[data_path], (h5py.File, h5py.Group)):
                os.makedirs(data_path[1:], exist_ok=True)
            else:
                dir_path = os.path.dirname(data_path[1:])
                if dir_path != '':
                    os.makedirs(dir_path, exist_ok=True)


            data = self[data_path]

            if data_path.endswith('tc.out'):
                frames = np.arange(len(data))
                if frame is not None:
                    data = [data[frame]]
                    frames = [frame]
                for file_lines, frame in zip(data, frames):
                    file_name = dir_path.replace('tc.out', f'tc_{frame}.out')
                    with open(file_name, 'w') as file:
                        if len(file_lines) == 1 and type(file_lines[0]) == bytes:
                            file_lines = eval(file_lines[0].decode('utf-8'))
                        for line in file_lines:
                            file.write(line + '\n')
                return

            if frame is not None:
                data = data[frame]

            if isinstance(data, np.ndarray):
                #   NumPy array can't be save directly to a file
                #   create a temporary file with a dataset to save
                temp_file = H5File('_temp.h5', 'w')
                temp_file.create_dataset(data_path, data=data)
                H5File._to_file_and_dir_details(temp_file)
                os.remove('_temp.h5')
                return
            self._to_file_and_dir_details(data)

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

    def to_json(self, file_loc, data_path: str = None, frame=None):
        print('Saving to json:')
        if data_path is None:
            data = self
        else:
            data = self[data_path]

        out_data = self.to_dict(data, print_message=True, frame=frame)
        if file_loc.endswith('.gz'):
            print('    Writing data to a compressed .json file')
            with gzip.open(file_loc, 'wt') as file:
                json.dump(out_data, file, indent=4)
        else:
            print('    Writing data to a .json file')
            with open(file_loc, 'w') as file:
                json.dump(out_data, file, indent=4)
        print('    Done!')

    def to_dict(self, data: h5py.File |  h5py.Dataset | h5py.Group = None, print_message=False, frame=None):
        if self._dictionaried_data is not None:
            return self._dictionaried_data
        if print_message:
            print('    Converting data to a dictionary')
        if data is None:
            data = self
        return self._to_dict_recursive(data, frame)
        
    
    def _to_dict_recursive(self, data: h5py.File |  h5py.Dataset | h5py.Group, frame=None):
        if data is None:
            data = self
        out_data = {}
        if isinstance(data, h5py.Group) or isinstance(data, h5py.File):
            for key, val in data.items():
                out_data[key] = self._to_dict_recursive(val, frame)
        elif isinstance(data, h5py.Dataset):
            if data.dtype == 'object':
                conv_data = np.array(data[:]).astype(str).tolist()
            else:
                conv_data = data[:].tolist()

            if frame is not None:
                print('Frame:', frame)
                return conv_data[frame]
            return conv_data
        else:
            print("Could not convert ", data)
        self._dictionaried_data = out_data
        return out_data
    
    def to_pickle(self, file_loc):
        print('Saving data to a pickled dictionary:')
        data = self.to_dict(print_message=True)

        if file_loc.endswith('.gz'):
            print('    Writing data to a compressed pickle file')
            with gzip.open(file_loc, 'wb') as file:
                pickle.dump(data, file)
        else:
            print('    Writing data to a pickle file')
            with open(file_loc, 'wb') as file:
                pickle.dump(data, file)

        print('    Done!')
    
    def remove_tc_jobs(self, new_file_loc):
        with H5File(new_file_loc, 'w') as new_file:
            self._remove_tc_jobs(self, new_file)

    # def extract_path(self, new_file_loc: str, paths: list[str], frame: int):
    #     data = self[paths[0]]
    #     for p in paths[1:]:

        
    #     if frame is not None:
    #         data = data[frame]
    #     new_file = H5File(new_file_loc, 'w')
    #     if isinstance(data, h5py.Dataset):
    #         new_file.create_dataset(path, data=data)
    #     else:
    #         self.copy(data, new_file)
    #     new_file.close()
            
    @staticmethod
    def copy_with_frame(source, dest_file, frame=None):
        
        if isinstance(source, (h5py.File, h5py.Group)):
            for key in source:
                H5File.copy_with_frame(source[key], dest_file, frame)
        elif isinstance(source, h5py.Dataset):
            if frame is None:
                print('   Copying dataset:', source.name)
                dest_file.create_dataset(source.name, data=source[:])
                for k, v in source.attrs.items():
                    dest_file[source.name].attrs[k] = v
            else:
                # Check if the dataset is large enough to grab the n-th element
                if len(source) > frame:
                    n_th_element = source[frame]
                    print('   Copying dataset:', source.name)
                    dest_file.create_dataset(source.name, data=n_th_element)
                    for k, v in source.attrs.items():
                        dest_file[source.name].attrs[k] = v
                else:
                    print(f"Dataset: {source.name} is too small for frame {frame}")

    @staticmethod
    def find_paths_with_wildcard(h5file, pattern):
        """
        Recursively find paths of datasets and groups that match a wildcard pattern.
        
        :param h5file: The HDF5 file object.
        :param pattern: The wildcard pattern (e.g., '*/dataset*').
        :return: A list of paths that match the pattern.
        """
        matching_paths = []
        
        def visitor_func(name, obj):
            # Use fnmatch to compare the name with the pattern
            if fnmatch.fnmatch(name, pattern):
                matching_paths.append(name)
        
        # Traverse the file and apply the visitor_func
        h5file.visititems(visitor_func)
        
        return matching_paths


    def write_new_file(self, new_file_loc: str, paths: list[str], frame: int):
        all_paths = []
        for p in paths:
            all_paths += H5File.find_paths_with_wildcard(self, p)

        with h5py.File(new_file_loc, 'w') as new_file:
            # Iterate through the list of paths to copy and apply the function
            for path in all_paths:
                if path in self:
                    obj = self[path] 
                    H5File.copy_with_frame(obj, new_file, frame=frame)

def run_h5_module():
    parser = argparse.ArgumentParser()
    parser.add_argument('--file',       '-f', type=str, help='HDF5 file to read', required=True)
    parser.add_argument('--json',       '-j', type=str, help='Convert to JSON')
    parser.add_argument('--rm_tc_files','-r', type=str, help='Remove tc.out file data from tc_job_data and save to this file')
    parser.add_argument('--pickle',     '-P', type=str, help='Convert to a pickled HDF5 file')
    parser.add_argument('--concat',     '-c', type=str, help='Concatenate HDF5 files', nargs='+')
    parser.add_argument('--extract',    '-x', help='Extract a dataset from the HDF5 file', action='store_true')
    parser.add_argument('--path',       '-p', type=str, help='Dataset or Group path to extract', nargs='+')
    parser.add_argument('--frame',      '-F', type=int, help='Frame to extract')
    parser.add_argument('--new',        '-n', help='Create a new H5 file')
    args = parser.parse_args(sys.argv[2:])


    if args.concat is not None:
        print('Combining files')
        files = [args.file] + args.concat
        H5File.combine_files(args.new, *files)
        exit()

    with H5File(args.file, 'r') as file:

        if args.json is not None:
            file.to_json(args.json, args.path, args.frame)
        elif args.pickle is not None:
            pass
        elif args.rm_tc_files is not None:
            file.remove_tc_jobs(args.rm_tc_files)
        elif args.pickle is not None:
            file.to_pickle(args.pickle)
        elif args.extract:
            print('Extracting data to file and directory')
            file.to_file_and_dir(args.path, args.frame)
        elif args.new:
            print('Creating new h5 file')
            file.write_new_file(args.new, args.path, args.frame)

        else:
            print('Printing data structure:')
            file.print_data_structure()

    exit()
if __name__ == '__main__':
    run_h5_module()
