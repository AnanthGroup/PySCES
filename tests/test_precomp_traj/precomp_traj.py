import unittest
import pandas
import io
import numpy as np
import os
from tools import parse_xyz_data, assert_dictionary, cleanup, reset_directory
import json
import pysces

class Test_Precompute(unittest.TestCase):
    def __init__(self, methodName: str = "runTest") -> None:
        super().__init__(methodName)

    def test_jobs(self):
        reset_directory()
        os.chdir('test_precomp_traj')

        pysces.qcRunners.TeraChem._DEBUG_TRAJ = 'debug_traj.pkl'
        pysces.reset_settings()
        pysces.run_simulation()

        #   check simple panda readable data
        for file in ['corr.txt', 'electric_pq.txt', 'energy.txt', 'grad.txt', 'nac.txt']:
            data_ref = pandas.read_csv(f'logs_ref/{file}', sep=r'\s+', comment='#')
            data_tst = pandas.read_csv(f'logs/{file}', sep=r'\s+', comment='#')
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

        cleanup()

                
if __name__ == '__main__':
    test = Test_Precompute()
    test.test_jobs()