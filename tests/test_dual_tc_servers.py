import unittest
import pandas
import io
import numpy as np
import os
from tools import parse_xyz_data, assert_dictionary
import json

class Test_TC_CIS(unittest.TestCase):
    def __init__(self, methodName: str = "runTest") -> None:
        super().__init__(methodName)

    def test_jobs(self):
        this_dir = os.path.abspath(os.path.curdir)
        os.chdir('test_dual_tc_servers')

        import pysces
        pysces.main()

        for file in ['corr.txt', 'electric_pq.txt', 'energy.txt', 'grad.txt', 'nac.txt']:
            data_ref = pandas.read_csv(f'logs_ref/{file}', sep='\s+', comment='#')
            data_tst = pandas.read_csv(f'logs/{file}', sep='\s+', comment='#')
            for key in data_ref:
                np.testing.assert_allclose(data_tst[key], data_ref[key], 
                                           rtol=1e-5, verbose=True, strict=True,
                                           err_msg=f'file: {file}')
                
        for file in ['nuc_geo.xyz', 'nuclear_P.txt']:
            data_ref = parse_xyz_data(f'logs_ref/{file}')
            data_tst = parse_xyz_data(f'logs/{file}')
            for frame, (frame_tst, frame_ref) in enumerate(zip(data_tst, data_ref)):
                np.testing.assert_equal(frame_tst['atoms'], frame_ref['atoms'])
                np.testing.assert_allclose(frame_tst['positions'], frame_ref['positions'],
                                           rtol=1e-5, verbose=True, strict=True,
                                           err_msg=f'file: {file}; frame {frame}')
                
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