import pickle
import h5py
import json
import numpy as np
import os
import sys
import argparse
import gzip
import gzip

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

    def print_data_structure(self, data: h5py.File | h5py.Dataset | h5py.Group = None, string=''):
        if data is None:
            data = self
        if isinstance(data, h5py.Group) or isinstance(data, h5py.File):
            for key, val in data.items():
                self.print_data_structure(val, string + '/' + key)
        elif isinstance(data, h5py.Dataset):
            # print(string, ' shape = ', data.shape, ', attrs = ', dict(data.attrs))
            print(string, ' shape = ', data.shape)



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
            if data_path[0] == '/':
                dir_path = data_path[1:]
                os.makedirs(os.path.dirname(dir_path), exist_ok=True)
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

def run_h5_module():
    parser = argparse.ArgumentParser()
    parser.add_argument('--file',       '-f', type=str, help='HDF5 file to read', required=True)
    parser.add_argument('--json',       '-j', type=str, help='Convert to JSON')
    parser.add_argument('--rm_tc_files', '-r', type=str, help='Remove tc.out file data from tc_job_data and save to this file')
    parser.add_argument('--pickle',     '-P', type=str, help='Convert to a pickled HDF5 file')
    parser.add_argument('--extract', '-x', help='Extract a dataset from the HDF5 file', action='store_true')
    parser.add_argument('--path', '-p', type=str, help='Dataset or Group path to extract')
    parser.add_argument('--frame', '-F', type=int, help='Frame to extract')
    args = parser.parse_args(sys.argv[2:])

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
        else:
            file.print_data_structure()

    exit()
if __name__ == '__main__':
    run_h5_module()
