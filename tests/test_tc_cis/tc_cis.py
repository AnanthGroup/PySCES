import unittest
import pandas
import numpy as np
import os
import json
import pysces
import sys
sys.path.insert(1, os.path.join(os.path.dirname(pysces.__file__), '../../tests'))
from tools import parse_xyz_data, assert_dictionary, cleanup, reset_directory
from pysces.qcRunners.TeraChem import TCJobBatch, TCJob

class Test_TC_CIS(unittest.TestCase):
    def __init__(self, methodName: str = "runTest") -> None:
        super().__init__(methodName)

    def setUp(self) -> None:
        #   reset class variables
        TCJobBatch._TCJobBatch__batch_counter = 0
        TCJob._TCJob__job_counter = 0

        reset_directory()
        os.chdir('test_tc_cis')

        #   Load the reference nacs and use their values as sign references
        #   This will be later fixed with overlaps
        ref_nacs = np.loadtxt('logs_ref/nac.txt', skiprows=3, max_rows=18)
        pysces.SignFlipper._debug = False
        pysces.SignFlipper._ref_nacs = ref_nacs

    def test_jobs(self):

        pysces.reset_settings()
        pysces.run_simulation()

        #   check simple panda readable data
        for file in ['corr.txt', 'electric_pq.txt', 'energy.txt', 'grad.txt', 'nac.txt']:
            data_ref = pandas.read_csv(f'logs_ref/{file}', sep='\s+', comment='#')
            data_tst = pandas.read_csv(f'logs/{file}', sep='\s+', comment='#')
            for key in data_ref:
                np.testing.assert_allclose(data_tst[key], data_ref[key], 
                                           atol=1e-4, verbose=True,
                                           err_msg=f'file: {file} key: {key}')
        
        #   check data in xyz formats
        for file in ['nuc_geo.xyz', 'nuclear_P.txt']:
            data_ref = parse_xyz_data(f'logs_ref/{file}')
            data_tst = parse_xyz_data(f'logs/{file}')
            for frame, (frame_tst, frame_ref) in enumerate(zip(data_tst, data_ref)):
                np.testing.assert_equal(frame_tst['atoms'], frame_ref['atoms'])
                np.testing.assert_allclose(frame_tst['positions'], frame_ref['positions'],
                                           atol=1e-5, verbose=True,
                                           err_msg=f'file: {file}; frame {frame}')
        
        #   check checkpoint files
        with open('restart_end.json') as file:
            restart_ref = json.load(file)
        with open('restart.json') as file:
            restart_tst = json.load(file)
        assert_dictionary(self, restart_ref, restart_tst, atol=1e-3)

        cleanup()

                
if __name__ == '__main__':
    test = Test_TC_CIS()
    test.test_jobs()