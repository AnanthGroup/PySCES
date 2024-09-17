import h5py
import json
import numpy as np
import os
import sys
import argparse

class H5Group(h5py.Group):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def create_group(self, name, track_order=None, exist_ok=True):
        if exist_ok and name in self:
            return self[name]
        grp = super().create_group(name, track_order)
        return H5Group(grp.id)
    
    def create_dataset(self, name, shape=None, dtype=None, data=None, exist_ok=True, **kwargs):
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
        super().__init__(name, mode, driver, libver, userblock_size, swmr, rdcc_nslots, rdcc_nbytes, rdcc_w0, track_order, fs_strategy, fs_persist, fs_threshold, fs_page_size, page_buf_size, min_meta_keep, min_raw_keep, locking, alignment_threshold, alignment_interval, meta_block_size, **kwds)

    def create_group(self, name, track_order=None, exist_ok=True):
        if exist_ok and name in self:
            return self[name]
        
        grp = super().create_group(name, track_order)
        return H5Group(grp.id)
    
    def create_dataset(self, name, shape=None, dtype=None, data=None, exist_ok=True, **kwargs):
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

    def to_json(self, file_loc):
        out_data = self.to_dict()
        with open(file_loc, 'w') as file:
            json.dump(out_data, file, indent=4)

    def to_dict(self, data: h5py.File |  h5py.Dataset | h5py.Group = None):
        if data is None:
            data = self
        out_data = {}
        if isinstance(data, h5py.Group) or isinstance(data, h5py.File):
            for key, val in data.items():
                out_data[key] = self.to_dict(val)
        elif isinstance(data, h5py.Dataset):
            if data.dtype == 'object':
                conv_data = np.array(data[:]).astype(str).tolist()
            else:
                conv_data = data[:].tolist()
            return conv_data
        else:
            print("Could not convert ", data)
        return out_data
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', '-f', type=str, help='HDF5 file to read')
    parser.add_argument('--json', '-j', type=str, help='Convert to JSON')
    args = parser.parse_args()

    if args.file is None:
        print('Please provide a file to read')
        sys.exit(1)


    with H5File(args.file, 'r') as file:
        if args.json is not None:
            file.to_json(args.json)
        else:
            file.print_data_structure()
