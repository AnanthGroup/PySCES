import unittest
import pandas
import io
import numpy as np
import os
from tools import parse_xyz_data, assert_dictionary
import json
# from pysces import main

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


class Test_TC_CIS(unittest.TestCase):
    def __init__(self, methodName: str = "runTest") -> None:
        super().__init__(methodName)

    def test_jobs(self):
        this_dir = os.path.abspath(os.path.curdir)
        os.chdir('test_precomp_traj')
        os.environ['DEBUG_TRAJ'] = 'debug_traj.pkl'

        import pysces
        pysces.main()

        #   check simple panda readable data
        for file in ['corr.txt', 'electric_pq.txt', 'energy.txt', 'grad.txt', 'nac.txt']:
            data_ref = pandas.read_csv(f'logs_ref/{file}', sep='\s+', comment='#')
            data_tst = pandas.read_csv(f'logs/{file}', sep='\s+', comment='#')
            for key in data_ref:
                np.testing.assert_allclose(data_tst[key], data_ref[key], 
                                           rtol=1e-5, verbose=True, strict=True,
                                           err_msg=f'file: {file}')
        
        #   check data in xyz formats
        for file in ['nuc_geo.xyz', 'nuclear_P.txt']:
            data_ref = parse_xyz_data(f'logs_ref/{file}')
            data_tst = parse_xyz_data(f'logs/{file}')
            for frame, (frame_tst, frame_ref) in enumerate(zip(data_tst, data_ref)):
                np.testing.assert_equal(frame_tst['atoms'], frame_ref['atoms'])
                np.testing.assert_allclose(frame_tst['positions'], frame_ref['positions'],
                                           atol=1e-16, verbose=True, strict=True,
                                           err_msg=f'file: {file}; frame {frame}')
        
        #   check checkpoint files
        with open('restart_end.json') as file:
            restart_ref = json.load(file)
        with open('restart.json') as file:
            restart_tst = json.load(file)
        assert_dictionary(self, restart_ref, restart_tst)

        self.cleanup()
        os.chdir(this_dir)
    

    def cleanup(self):
        #   clean up
        for file in ['progress.out', 'corr.out', 'restart.json', 'restart.out']:
            if os.path.isfile(file):
                os.remove(file)
        for file in os.listdir('logs'):
            if os.path.isfile(os.path.join('logs', file)):
                os.remove(os.path.join('logs', file))
        if os.path.isdir('logs'):
            os.removedirs('logs')


                
if __name__ == '__main__':
    test = Test_TC_CIS()
    test.test_jobs()