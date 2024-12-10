import unittest
import numpy as np
import os
import sys
import pysces
from pysces import options as opts
from pysces.input_simulation import TCRunnerOptions as tcr_opts
sys.path.insert(1, os.path.join(os.path.dirname(pysces.__file__), '../../tests'))
from tools import parse_xyz_data, assert_reset_files, cleanup, reset_directory, assert_logs_dir

class Test_Dual_TC_Servers(unittest.TestCase):
    def __init__(self, methodName: str = "runTest") -> None:
        super().__init__(methodName)

    def test_jobs(self):
        reset_directory()
        os.chdir('test_dual_tc_servers')

        pysces.reset_settings()
        pysces.run_simulation()

        assert_logs_dir('logs', 'logs_ref')
        assert_reset_files(self, 'restart.json', 'restart_end.json')

        cleanup()

                
if __name__ == '__main__':
    test = Test_Dual_TC_Servers()
    test.test_jobs()