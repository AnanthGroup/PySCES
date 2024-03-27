#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  3 19:05:28 2023

@author: user
"""

"""
Repository of simulation variables
"""
import sys, os
import shutil
import input_simulation as opts


########## DEFAULT SETTINGS ##########

nel, natom = 3, 6 # number of electronic states, number of atoms in the molecule
nnuc = 3*natom # number of nuclear DOFs
ndof = nel + nnuc 
temp = 300 # simulation temperature in Kelvin

# Initial sampling function ('conventioanl', 'modified', or 'spin' LSC-IVR)
sampling = 'conventional' 

# Centers of initial electronic coherent states (positions q0 & momenta p0)
# From left, electronic state 1, 2, 3, ...
q0 = [0.0, 0.0, 0.0] 
p0 = [0.0, 0.0, 0.0]

# Center of initial momentum of nuclear modes (same value for all nuc DOFs)
# NOTE: the centers of initial position are determined by normal coordinates
pN0 = 0.0

# ELectronic coherent state width parameters
# From left, electronic state 1, 2, 3, ...
width = [1.0, 1.0, 1.0]

# Specify an integrator (Choose from 'ABM', 'BSH', and 'RK4')
integrator = 'RK4'
# Size of time step (a.u.), number of steps (Only relevant for ABM)
timestep, nstep = 1.0, 16700
# Maximum propagation time (a.u.), BSH step to be tried (a.u.), error tolerance (ratio) (Only relevant for BSH)
tmax_bsh, Hbsh, tol = 10, 3.0, 0.01 
# Maximum propagation time (a.u.), one Runge-Kutta step (a.u.) (Only relevant for RK4)
tmax_rk4, Hrk4 = 20, 5.0 

# Scaling factor of normal mode frequencies
frq_scale = 0.967

# Number of CPUs and nodes in each internal NACME calculation
ncpu, nnode = 1, 1

# Partition of Pool cluster (astra or common)
partition = 'astra'

# Indices of electronic states for which NACME to be read from GAMESS output file
elab = [1, 2, 3]

# Index of initially occupied electronic state
init_state = 2

# Restart request: 0 = no restart, 1 = restart
restart = 0

#   type of QC runner, either 'gamess' or 'terachem'
QC_RUNNER = 'gamess'
#QC_RUNNER = 'terachem'

#   TeraChem runner options
tcr_host = '10.1.1.154'
tcr_port = 9876
tcr_server_root = '.'
tcr_job_options = {}
tcr_state_options = {}

# Terachem files
fname_tc_xyz      = "tmp/tc_hf/hf.spherical.freq/Geometry.xyz"
fname_tc_geo_freq = "tmp/tc_hf/hf.spherical.freq/Geometry.frequencies.dat"
fname_tc_redmas   = "tmp/tc_hf/hf.spherical.freq/Reduced.mass.dat"
fname_tc_freq     = "tmp/tc_hf/hf.spherical.freq/Frequencies.dat"

#   GAMESS submission script name
sub_script = None

#   type of QC interface to use for inputs (masses, hessian, etc.)
#   either 'gamess' or 'terachem'
mol_input_format = ''

#   logging directory
logging_dir = 'logs'


########## END DEFAULT SETTINGS ##########




#   TODO: check that there are no conflicting settings
def _check_settings():
    #   set input format to the same type of QC runner
    if opts.mol_input_format == '':
        opts.mol_input_format = opts.QC_RUNNER

    #   logging directory
    opts.logging_dir = os.path.abspath(opts.logging_dir)
    if os.path.isdir(opts.logging_dir):
        #   logging dir alreayd exists (from a previous job)
        #   so we'll copy it to a new directory
        coppied = False
        count = 1
        while not coppied and count < 100:
            new_dir = f'{opts.logging_dir}.{count}'
            if os.path.isdir(new_dir):
                count += 1
            else:
                shutil.move(opts.logging_dir, new_dir, shutil.copytree)
                coppied = True
        if count == 100:
            raise RecursionError('logging dir already eists, cou not copy to new numbered dir')
    os.makedirs(opts.logging_dir)
    

def _set_seed():
    '''
        set the random number generator seed
    '''
    if 'input_seed' in opts.__dict__:
        import numpy as np
        import random
        input_seed = opts.__dict__['input_seed']
        random.seed(input_seed)
        np.random.seed(input_seed)

#   load in local settings, which will overwrite the default ones above
try:
    sys.path.append(os.path.abspath(os.path.curdir))
    print("Importing local settings")
    from input_simulation_local import * 
except Exception as e:
    print("Using default settings: ", e)
    from input_simulation import * 

_check_settings()
_set_seed()
