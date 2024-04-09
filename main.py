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
from input_gamess import * 
from subroutines import *
# __location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
__location__ = ''

from input_simulation import * 

# Print git commit
command_git_tag="git -C "+str(os.path.dirname(os.path.realpath(__file__)))+" describe --tags"
print("git tag: "+str(os.popen(command_git_tag).readline()))

###################################
### Propagation of a trajectory ###
###################################
initq, initp = np.zeros(ndof-6), np.zeros(ndof-6)

# Read geo_gamess and hess_gamess
amu_mat, xyz_ang, frq, redmas, L, U, com_ang = get_geo_hess()

if restart == 0: # If this is not a restart run
    # Rotate Cartesian coordinate into normal coordinate (normal_geo is in A.U.)
    normal_geo = get_normal_geo(U, xyz_ang, amu_mat)

    # Sample initial phase space configuration
    if sampling == 'conventional':
        coord = sample_conventionalLSC(normal_geo, frq)
    elif sampling == 'modified':
        coord = sample_modifiedLSC(normal_geo, frq)
    elif sampling == 'spin':
        coord = sample_spinLSC(normal_geo, frq)
    
    initq = coord[0,:] # A.U.
    initp = coord[1,:] # A.U.

# Start the propagation routine
if integrator == 'ABM':
    time_array,coord,flag_energy,flag_grad,flag_nac,flag_orb,initial_time = ME_ABM(restart, initq, initp, amu_mat, U, com_ang)
    if flag_energy == 0: # If energy is conserved,
        if all([el == 0 for el in flag_grad]) and flag_nac == 0 and flag_orb == 0: # If no error is raised by the CAS calculations
            compute_CF(time_array, coord)

elif integrator == 'BSH':
    time_array, coord, initial_time = BulStoer(initq,initp,tmax_bsh,Hbsh,tol,restart,amu_mat,U, com_ang)
    compute_CF(time_array, coord)

elif integrator == 'RK4':
    time_array, coord, initial_time = rk4(initq,initp,tmax_rk4,Hrk4,restart,amu_mat,U, com_ang)
    compute_CF(time_array, coord)


'''End of the program'''
