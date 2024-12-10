import unittest
import numpy as np
import os
from tools import cleanup, reset_directory, assert_reset_files, assert_logs_dir
import pysces

class Test_GAMESS_RK4(unittest.TestCase):
    def __init__(self, methodName: str = "runTest") -> None:
        super().__init__(methodName)

    def test_jobs(self):
        reset_directory()
        os.chdir('test_gamess_rk4')

        with open('geo_gamess', 'w') as file:
            file.write('6\n')
            file.write('C    6.0   0.6655781462   0.0000000000   3.0000000000\n')
            file.write('C    6.0  -0.6655781462   0.0000000000   3.0000000000\n')
            file.write('H    1.0  -1.2398447212   0.9238158831   3.0000000000\n')
            file.write('H    1.0   1.2398447212   0.9238158831   3.0000000000\n')
            file.write('H    1.0  -1.2398447212  -0.9238158831   3.0000000000\n')
            file.write('H    1.0   1.2398447212  -0.9238158831   3.0000000000\n')

        pysces.reset_settings()
        pysces.run_simulation()

        assert_logs_dir('logs', 'logs_ref')
        assert_reset_files(self, 'restart.json', 'restart_end.json')

        cleanup('logs', 'logs_1', 'logs_2', 'logs_combo')
    