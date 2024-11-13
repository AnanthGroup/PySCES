import unittest
import pandas
import io
import numpy as np
import os
from tools import parse_xyz_data, assert_dictionary
import json
# from pysces import main

class Test_TC_CIS(unittest.TestCase):
    def __init__(self, methodName: str = "runTest") -> None:
        super().__init__(methodName)

    def test_jobs(self):
        this_dir = os.path.abspath(os.path.curdir)
        os.chdir('test_gamess_wigner')

        with open('geo_gamess', 'w') as file:
            file.write('6\n')
            file.write('C    6.0   0.6655781462   0.0000000000   0.0000000000\n')
            file.write('C    6.0  -0.6655781462   0.0000000000   0.0000000000\n')
            file.write('H    1.0  -1.2398447212   0.9238158831   0.0000000000\n')
            file.write('H    1.0   1.2398447212   0.9238158831   0.0000000000\n')
            file.write('H    1.0  -1.2398447212  -0.9238158831   0.0000000000\n')
            file.write('H    1.0   1.2398447212  -0.9238158831   0.0000000000\n')

        import pysces
        pysces.run_simulation()

        #   check simple panda readable data
        for file in ['corr.txt', 'electric_pq.txt', 'energy.txt', 'grad.txt', 'nac.txt']:
            data_ref = pandas.read_csv(f'logs_ref/{file}', sep='\s+', comment='#')
            data_tst = pandas.read_csv(f'logs/{file}', sep='\s+', comment='#')
            for key in data_ref:
                np.testing.assert_allclose(data_tst[key], data_ref[key], 
                                           rtol=1e-5, verbose=True,
                                           err_msg=f'file: {file}')
        
        #   check data in xyz formats
        for file in ['nuc_geo.xyz', 'nuclear_P.txt']:
            data_ref = parse_xyz_data(f'logs_ref/{file}')
            data_tst = parse_xyz_data(f'logs/{file}')
            for frame, (frame_tst, frame_ref) in enumerate(zip(data_tst, data_ref)):
                np.testing.assert_equal(frame_tst['atoms'], frame_ref['atoms'])
                np.testing.assert_allclose(frame_tst['positions'], frame_ref['positions'],
                                           atol=1e-16, verbose=True,
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
        for file in ['progress.out', 'corr.out', 'restart.json', 'restart.out', 'cas.dat', 'cas.inp', 'cas_old.inp', 'cas.out']:
            if os.path.isfile(file):
                os.remove(file)
        for file in os.listdir('logs'):
            if os.path.isfile(os.path.join('logs', file)):
                os.remove(os.path.join('logs', file))
        if os.path.isdir('logs'):
            os.removedirs('logs')

        #   remove directories logs.*
        print(os.listdir())
        for file in os.listdir():
            if os.path.isdir(file) and file.startswith('logs.'):
                for file2 in os.listdir(file):
                    os.remove(os.path.join(file, file2))
                os.removedirs(file)


                
if __name__ == '__main__':
    test = Test_TC_CIS()
    test.test_jobs()