import unittest
import pandas
import numpy as np
import os
import json
import pysces
import sys
import shutil
sys.path.insert(1, os.path.join(os.path.dirname(pysces.__file__), '../../tests'))
from tools import assert_reset_files, cleanup, reset_directory, assert_logs_dir
from pysces.qcRunners.TeraChem import TCJobBatch, TCJob

class Test_TC_Restart(unittest.TestCase):
    def __init__(self, methodName: str = "runTest") -> None:
        super().__init__(methodName)

    def test_jobs(self):
        reset_directory()
        os.chdir('test_tc_restart')
        cleanup('logs_1', 'logs_2', 'restart_1.json', 'restart_2.json') # in case of previous failed tests

        shutil.copy2('input_simulation_local_1.py', 'input_simulation_local.py')
        pysces.reset_settings()
        pysces.run_simulation()
        shutil.move('logs', 'logs_1')
        shutil.move('restart.json', 'restart_1.json')

        shutil.copy2('input_simulation_local_2.py', 'input_simulation_local.py')
        pysces.reset_settings()
        pysces.run_simulation()
        shutil.move('logs', 'logs_2')
        shutil.move('restart.json', 'restart_2.json')

        assert_logs_dir('logs_1', 'logs_1_ref')
        assert_logs_dir('logs_2', 'logs_2_ref')

        assert_reset_files(self, 'restart_1.json', 'restart_1_ref.json')
        assert_reset_files(self, 'restart_2.json', 'restart_2_ref.json')

        cleanup('logs_1', 'logs_2', 'restart_1.json', 'restart_2.json')
    
if __name__ == '__main__':
    unittest.main()