import numpy as np
import io
import gc
import unittest
import os
import shutil
import inspect
import json
import pandas

def parse_xyz_data(file_loc):
    n_atoms = None
    all_frames = []
    with open(file_loc) as file:
        for line in file:
            if n_atoms is None:
                n_atoms = int(line)
                next(file)
                frame_data = {'atoms': [], 'positions': []}
                for i in range(n_atoms):
                    sp = next(file).split()
                    coords = [float(x) for x in sp[1:4]]
                    frame_data['atoms'].append(sp[0])
                    frame_data['positions'].append(coords)
                frame_data['atoms'] = np.array(frame_data['atoms'])
                frame_data['positions'] = np.array(frame_data['positions'])
                n_atoms = None
                all_frames.append(frame_data)

    return all_frames

def assert_dictionary(testcase: unittest.TestCase, dict_ref, dict_tst, atol=1e-6, rtol=1e-6,):
    for key, value in dict_ref.items():
        if key not in dict_tst:
            raise ValueError(f'key "{key}" not in test dictionary')
        msg = f'value for key "{key}" not equal between both dicts'
        if isinstance(value, dict):
            assert_dictionary(testcase, dict_ref[key], dict_tst[key], atol, rtol)
        elif isinstance(value, str) or isinstance(value, int):
            testcase.assertEqual(value, dict_tst[key], msg=msg)
        elif isinstance(value, float):
            testcase.assertAlmostEqual(value, dict_tst[key], places=8, msg=msg)
        elif isinstance(value, np.ndarray):
            np.testing.assert_allclose(value, dict_tst[key], atol=atol, rtol=rtol, err_msg=msg)
        elif isinstance(value, list):
            array_ref = np.array(value)
            array_tst = np.array(dict_tst[key])
            np.testing.assert_allclose(array_ref, array_tst, atol=atol, rtol=rtol, err_msg=msg)

def check_for_open_files():
    # Trigger garbage collection
    gc.collect()

    # Find all file objects that are still open
    open_files = [obj for obj in gc.get_objects() if isinstance(obj, io.IOBase) and not obj.closed]

    if open_files:
        print("Unclosed file objects detected:")
        for f in open_files:
            print("File: ", f)
            # print(f"File: {f.name}, Mode: {f.mode}")
    else:
        print("No unclosed file objects found.")

def cleanup(*files_and_dirs):
    
    #   clean up
    for file in ['progress.out', 'corr.out', 'restart.json', 'restart.out', 'cas.dat', 'cas.inp', 'cas_old.inp', 'cas.out']:
        if os.path.isfile(file):
            os.remove(file)
    if os.path.isdir('logs'):
        try:
            shutil.rmtree('logs')

        except OSError:
            print('Could not delete logs directory, check open files')
            check_for_open_files()

    #   remove directories logs.*
    for file in os.listdir():
        if os.path.isdir(file) and file.startswith('logs.'):
            shutil.rmtree(file)

    #   remove others
    for x in files_and_dirs:
        if os.path.isfile(x):
            os.remove(x)
        elif os.path.isdir(x):
            shutil.rmtree(x)

def reset_directory():
    #   directory of test_tools.py, could be called from anywhere
    this_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    os.chdir(this_dir)

def assert_logs_dir(tst_dir_name, ref_dir_name):
    #   check simple panda readable data
    for file in ['corr.txt', 'electric_pq.txt', 'energy.txt', 'grad.txt', 'nac.txt']:
        data_ref = pandas.read_csv(f'{ref_dir_name}/{file}', sep='\s+', comment='#')
        data_tst = pandas.read_csv(f'{tst_dir_name}/{file}', sep='\s+', comment='#')
        for key in data_ref:
            np.testing.assert_allclose(data_tst[key], data_ref[key], 
                                        rtol=1e-5, verbose=True,
                                        err_msg=f'file: {file}')
    
    #   check data in xyz formats
    for file in ['nuc_geo.xyz', 'nuclear_P.txt']:
        data_ref = parse_xyz_data(f'{ref_dir_name}/{file}')
        data_tst = parse_xyz_data(f'{tst_dir_name}/{file}')
        for frame, (frame_tst, frame_ref) in enumerate(zip(data_tst, data_ref)):
            np.testing.assert_equal(frame_tst['atoms'], frame_ref['atoms'])
            np.testing.assert_allclose(frame_tst['positions'], frame_ref['positions'],
                                        atol=1e-16, verbose=True,
                                        err_msg=f'file: {file}; frame {frame}')
            
def assert_reset_files(testcase: unittest.TestCase, tst_file, ref_file):
    with open(ref_file) as file:
        restart_ref: dict = json.load(file)
        if 'TCRunner' in restart_ref:
            restart_ref.pop('TCRunner')
    with open(tst_file) as file:
        restart_tst: dict = json.load(file)
        if 'TCRunner' in restart_tst:
            restart_tst.pop('TCRunner')
    assert_dictionary(testcase, restart_ref, restart_tst)

class Tester(unittest.TestCase):
    '''
        Wrapper for unittest.TestCase with explicit logging.
        FOR DEBUGGING ONLY
    '''
    def __init__(self, methodName: str = "runTest") -> None:
        super().__init__(methodName)
        self.logger = io.StringIO()
        self.logger.write(f"{'Label':30s} {'Max Diff':12s} {'Max Rel Diff':12s}\n")
        self.logger.write('-------------------------------------------------------------\n')

        #   All functions start this verable set to True and exit seeting it false
        #   If an error is raised, logs will be print
        self.print_log = False

    def assert_allclose(self, label, actual, desired, rtol=1e-7, atol=0, equal_nan=True):
        self.print_log = True
        diff = actual - desired
        max_diff = np.max(np.abs(diff))
        rel_diff = np.abs(diff)/max_diff
        self.log(label, max_diff, rel_diff)
        np.testing.assert_allclose(actual, desired, rtol, atol, equal_nan, verbose=True, strict=True)
        self.print_log = False

    def log(self, file, key, diff, rel_diff):
        self.logger.write(f'{file:20s} {key:10s} {diff:12.5e} {rel_diff:12.5e}\n')


    def __dell__(self):
        if self.print_log:
            print(self.logger.getvalue())

class ArrayCompareMessage:
    def __init__(self, tst, ref):
        self.tst = np.array(tst)
        self.ref = np.array(ref)

    def __str__(self):
        out_str = ''
        #   get the index of the array. It can be of arbitray shape
        for idx, ref_val in np.ndenumerate(self.ref):
            tst_val = self.tst[idx]
            idx_str = ''
            for i in idx:
                idx_str += f'{i:3d},'
            abs_diff = np.abs(tst_val - ref_val)
            if ref_val == 0:
                rel_diff = abs_diff
            else:
                rel_diff = abs_diff / np.abs(ref_val)
            
            out_str += f'[{idx_str}] tst: {tst_val:12.5e} ref: {ref_val:12.5e} abs_diff: {abs_diff:12.5e} rel_diff: {rel_diff:12.5e}\n'
        return out_str