import unittest
import pandas
import numpy as np
import os
from tools import parse_xyz_data, assert_dictionary, cleanup, reset_directory
import json
import pysces
from pysces import options as opts
from pysces.input_simulation import TCRunnerOptions as tcr_opts

class Test_Dual_TC_Servers(unittest.TestCase):
    def __init__(self, methodName: str = "runTest") -> None:
        super().__init__(methodName)

    def test_jobs(self):
        reset_directory()
        os.chdir('test_dual_tc_servers')

        pysces.reset_settings()
        ref_nacs = np.loadtxt('logs_ref/nac.txt', skiprows=3, max_rows=18)
        pysces.SignFlipper._debug = True
        pysces.SignFlipper._ref_nac = ref_nacs
        
        if os.path.isfile('../host_ports.txt'):
            from numpy import loadtxt
            host_ports = loadtxt('../host_ports.txt', dtype=str)
            tcr_opts.host = host_ports[:, 0]
            tcr_opts.port = host_ports[:, 1].astype(int)
            tcr_opts.server_root = host_ports[:, 2]
        else:
            tcr_opts.host = ['localhost', 'localhost']
            tcr_opts.port = [1234, 1235]
            tcr_opts.server_root = ['', '']


        pysces.run_simulation()

        for file in ['corr.txt', 'electric_pq.txt', 'energy.txt', 'grad.txt', 'nac.txt']:
            data_ref = pandas.read_csv(f'logs_ref/{file}', sep='\s+', comment='#')
            data_tst = pandas.read_csv(f'logs/{file}', sep='\s+', comment='#')
            for key in data_ref:
                np.testing.assert_allclose(data_tst[key], data_ref[key], 
                                           rtol=1e-5, verbose=True,
                                           err_msg=f'file: {file}')
                
        for file in ['nuc_geo.xyz', 'nuclear_P.txt']:
            data_ref = parse_xyz_data(f'logs_ref/{file}')
            data_tst = parse_xyz_data(f'logs/{file}')
            for frame, (frame_tst, frame_ref) in enumerate(zip(data_tst, data_ref)):
                np.testing.assert_equal(frame_tst['atoms'], frame_ref['atoms'])
                np.testing.assert_allclose(frame_tst['positions'], frame_ref['positions'],
                                           rtol=1e-5, verbose=True,
                                           err_msg=f'file: {file}; frame {frame}')
                
        with open('restart_end.json') as file:
            restart_ref = json.load(file)
        with open('restart.json') as file:
            restart_tst = json.load(file)
        assert_dictionary(self, restart_ref, restart_tst)

        cleanup()

                
if __name__ == '__main__':
    test = Test_Dual_TC_Servers()
    test.test_jobs()