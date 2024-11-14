import unittest
import pandas
import io
import numpy as np
import os
from tools import parse_xyz_data, assert_dictionary, cleanup, reset_directory
import json

import pysces

import re

def get_restart_data(file_loc):

    data = {
        "coordinates": [],
        "forces": {},
        "energy": None,
        "total_time": None,
    }
    
    with open(file_loc, "r") as file:
        file_content = file.read()

    lines = file_content.splitlines()
    section = None

    for line in lines:
        line = line.strip()
        
        if "Coordinates (a.u.) at the last update" in line:
            section = "coordinates"
            continue
        elif "Forces (a.u.) at the last 4 time steps" in line:
            section = "forces"
            continue
        elif "Energy at the last time step" in line:
            section = "energy"
            continue
        elif "Total time in a.u." in line:
            section = "total_time"
            continue

        if section == "coordinates":
            if re.match(r"^-?\d+\.\d+", line):
                coords = list(map(float, line.split()))
                data["coordinates"].append(coords)
        
        elif section == "forces":
            if line.startswith("t = "):
                time_step = int(re.search(r"-?\d+", line).group())
                data["forces"][time_step] = []
            elif re.match(r"^-?\d+\.\d+", line):
                forces = list(map(float, line.split()))
                data["forces"][time_step].append(forces)
        
        elif section == "energy":
            if re.match(r"^-?\d+\.\d+", line):
                data["energy"] = float(line)
        
        elif section == "total_time":
            if re.match(r"^-?\d+\.\d+", line):
                data["total_time"] = float(line)
    
    return data

class Test_GAMESS_ABM(unittest.TestCase):
    def __init__(self, methodName: str = "runTest") -> None:
        super().__init__(methodName)

    def test_jobs(self):
        reset_directory()
        os.chdir('test_gamess_wigner_abm')

        with open('geo_gamess', 'w') as file:
            file.write('6\n')
            file.write('C    6.0   0.6655781462   0.0000000000   0.0000000000\n')
            file.write('C    6.0  -0.6655781462   0.0000000000   0.0000000000\n')
            file.write('H    1.0  -1.2398447212   0.9238158831   0.0000000000\n')
            file.write('H    1.0   1.2398447212   0.9238158831   0.0000000000\n')
            file.write('H    1.0  -1.2398447212  -0.9238158831   0.0000000000\n')
            file.write('H    1.0   1.2398447212  -0.9238158831   0.0000000000\n')

        pysces.reset_settings()
        pysces.run_simulation()

        coor_ref = np.loadtxt('logs_ref/corr.out')
        corr_tst = np.loadtxt('corr.out')
        np.testing.assert_allclose(corr_tst, coor_ref, rtol=1e-5, verbose=True)

        energy_ref = np.loadtxt('logs_ref/energy.out')
        energy_tst = np.loadtxt('energy.out')
        np.testing.assert_allclose(energy_tst, energy_ref, rtol=1e-5, verbose=True)
        
        #   check data in xyz formats
        data_ref = parse_xyz_data(f'logs_ref/nuc_geo.xyz')
        data_tst = parse_xyz_data(f'nuc_geo.xyz')
        for frame, (frame_tst, frame_ref) in enumerate(zip(data_tst, data_ref)):
            np.testing.assert_equal(frame_tst['atoms'], frame_ref['atoms'])
            np.testing.assert_allclose(frame_tst['positions'], frame_ref['positions'],
                                        atol=1e-16, verbose=True,
                                        err_msg=f'file: nuc_geo.xyz; frame {frame}')
            

        restart_ref = get_restart_data('logs_ref/restart.out')
        restart_tst = get_restart_data('restart.out')
        for key in restart_ref:
            if key == "forces":
                for time_step in restart_ref[key]:
                    np.testing.assert_allclose(restart_tst[key][time_step], restart_ref[key][time_step], rtol=1e-5, verbose=True)
            else:
                np.testing.assert_allclose(restart_tst[key], restart_ref[key], rtol=1e-5, verbose=True)

        cleanup('logs', 'nuc_geo.xyz', 'corr.out', 'energy.out', 'restart.out')
    