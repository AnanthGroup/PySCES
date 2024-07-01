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
import pandas
from pysces.input_gamess import * 
from pysces.subroutines import *
# __location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
__location__ = ''

from pysces.input_simulation import * 

def main():
    print_ascii_art()
    print_settings()


    ###################################
    ### Propagation of a trajectory ###
    ###################################
    ndof = 3*natom + nel
    initq, initp = np.zeros(ndof-6), np.zeros(ndof-6)

    # Read geo_gamess and hess_gamess
    amu_mat, xyz_ang, frq, redmas, L, U, com_ang, AN_mat = get_geo_hess()

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
        time_array, coord, flag_energy, flag_grad, flag_nac, flag_orb, initial_time = ME_ABM(restart, initq, initp, amu_mat, U, com_ang, AN_mat)
        if flag_energy == 0: # If energy is conserved,
            if all([el == 0 for el in flag_grad]) and flag_nac == 0 and flag_orb == 0: # If no error is raised by the CAS calculations
                compute_CF(time_array, coord)

    elif integrator == 'BSH':
        time_array, coord, initial_time = BulStoer(initq,initp,tmax_bsh,Hbsh,tol,restart,amu_mat,U, com_ang, AN_mat)
        compute_CF(time_array, coord)

    elif integrator == 'RK4':
        time_array, coord, initial_time = rk4(initq, initp, tmax_rk4, Hrk4, restart, amu_mat, U, com_ang, AN_mat)
        compute_CF(time_array, coord)


    print("\n\nSimulation completed successfully")

    '''End of the program'''

if __name__ == '__main__':
    main()