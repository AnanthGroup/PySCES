#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 11 15:28:22 2023

@author: user
"""

'''
Main code to run LSC-IVR dynamics
'''

import os
import numpy as np
import argparse
from pysces.input_gamess import * 
from pysces.subroutines import *
from pysces.fileIO import print_ascii_art, run_restart_module
from pysces.h5file import run_h5_module
__location__ = ''

from pysces.input_simulation import * 


def main():
    print_ascii_art()
    run_modules()
    run_simulation_module()

def run_simulation_module():
    description = 'Main code to run LSC-IVR dynamics\n'
    description += 'When run, PySCES will look for a file named "local_settings.py" in \n'
    description += 'the current directory and use it to set the simulation parameters.\n'
    parser = argparse.ArgumentParser(description=description)
    args = parser.parse_args(sys.argv[2:])

    make_logging_dir()
    input_local_settings()
    print_settings()
    run_simulation()

def run_simulation():
    # TODO: Temporary fix for the global variables
    for k, v in opts.__dict__.items():
        globals()[k] = v
    set_subroutine_globals()

    ###################################
    ### Propagation of a trajectory ###
    ###################################
    ndof = 3*natom + nel
    initq, initp = np.zeros(ndof-6), np.zeros(ndof-6)

    # Read geo_gamess and hess_gamess
    amu_mat, xyz_ang, frq, redmas, L, U, AN_mat = get_geo_hess()

    if restart == 0: # If this is not a restart run
        # Rotate Cartesian coordinate into normal coordinate (normal_geo is in A.U.)
        normal_geo = get_normal_geo(U, xyz_ang, amu_mat)

        # Sample initial phase space configuration
        if sampling == 'wigner':
            if nel == 1:
                print('WARNING: Wigner population estimator with nel=1 will result in\n')
                print('an unphysical radius of sampling. Use "sc" option instead.\n')
                exit()
            coord = sample_wignerLSC(normal_geo, frq)
        elif sampling == 'sc':
            coord = sample_scLSC(normal_geo, frq)
        elif sampling == 'spin':
            if nel != 3:
                print('WARNING: Spin mapping population estimator with nel being other than 3\n')
                print('is not implemented. Use "wigner" or "sc" option instead.\n')
                exit()
            coord = sample_spinLSC(normal_geo, frq)
        
        initq = coord[0,:] # A.U.
        initp = coord[1,:] # A.U.

    # Start the propagation routine
    if integrator == 'ABM':
        time_array, coord, flag_energy, flag_grad, flag_nac, flag_orb, initial_time = ME_ABM(restart, initq, initp, amu_mat, U, AN_mat)
        if flag_energy == 0: # If energy is conserved,
            if all([el == 0 for el in flag_grad]) and flag_nac == 0 and flag_orb == 0: # If no error is raised by the CAS calculations
                compute_CF(time_array, coord)

    elif integrator == 'BSH':
        time_array, coord, initial_time = BulStoer(initq,initp,tmax_bsh,Hbsh,tol,restart,amu_mat,U, AN_mat)
        compute_CF(time_array, coord)

    elif integrator.lower() in ['rk4', 'rk4-uprop', 'verlet-uprop']:
        time_array, coord, initial_time = rk4(initq, initp, tmax_rk4, Hrk4, restart, amu_mat, U, AN_mat)
        compute_CF(time_array, coord)


    print("\n\nSimulation completed successfully")

    '''End of the program'''


modules_map = {'h5': run_h5_module, 
               'run': run_simulation_module, 
               'genrst': run_restart_module
               }
def run_modules():
    if len(sys.argv) > 1:
        if sys.argv[1] in modules_map:
            modules_map[sys.argv[1]]()
        else:
            print("Invalid module specified, must be one of:")
            for k in modules_map.keys():
                print(f'  {k}')
            print('For details, run `pysces <module> -h`')
            exit()

if __name__ == '__main__':
    run_modules()