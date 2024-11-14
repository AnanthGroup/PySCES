import numpy as np
import io
import unittest
import os
import shutil
import inspect

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
            assert_dictionary(dict_ref[key], dict_tst[key])
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

def cleanup(*files_and_dirs):
    #   clean up
    for file in ['progress.out', 'corr.out', 'restart.json', 'restart.out', 'cas.dat', 'cas.inp', 'cas_old.inp', 'cas.out']:
        if os.path.isfile(file):
            os.remove(file)
    if os.path.isdir('logs'):
        for file in os.listdir('logs'):
            if os.path.isfile(os.path.join('logs', file)):
                os.remove(os.path.join('logs', file))
        os.removedirs('logs')

    #   remove directories logs.*
    for file in os.listdir():
        if os.path.isdir(file) and file.startswith('logs.'):
            for file2 in os.listdir(file):
                os.remove(os.path.join(file, file2))
            os.removedirs(file)

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