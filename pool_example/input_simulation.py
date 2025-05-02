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

########## DEFAULT SETTINGS ##########

nel, natom = 1, 6 # number of electronic states, number of atoms in the molecule
temp = 300 # simulation temperature in Kelvin

# Initial sampling function ('wigner', 'sc', or 'spin' LSC-IVR)
sampling = 'sc'

# Center of initial momentum of nuclear modes (same value for all nuc DOFs)
# NOTE: the centers of initial position are determined by normal coordinates
pN0 = 0.0

# Specify an integrator (Choose from 'ABM', 'BSH', and 'RK4')
integrator = 'RK4'
# Size of time step (a.u.), number of steps (Only relevant for ABM)
timestep, nstep = 1.0, 16700
# Maximum propagation time (a.u.), BSH step to be tried (a.u.), error tolerance (ratio) (Only relevant for BSH)
tmax_bsh, Hbsh, tol = 10, 3.0, 0.01 
# Maximum propagation time (a.u.), one Runge-Kutta step (a.u.) (Only relevant for RK4)
tmax_rk4, Hrk4 = 60, 5.0 

# Scaling factor of normal mode frequencies
frq_scale = 1.0

# Number of CPUs and nodes in each internal NACME calculation
ncpu, nnode = 1, 1

# Partition of Pool cluster (astra or common)
partition = 'astra'

# Indices of electronic states for which NACME to be read from GAMESS output file
elab = [1]

# Index of initially occupied electronic state
init_state = 1

# Restart request: 0 = no restart, 1 = restart
restart = 0
restart_file_in = 'restart.out'

# type of QC runner, either 'gamess' or 'terachem'
QC_RUNNER = 'gamess'

# log TC job results (False when using GAMESS)
tcr_log_jobs = False

# GAMESS submission script name
sub_script = None

# type of QC interface to use for inputs (masses, hessian, etc.)
# either 'gamess' or 'terachem'
mol_input_format = 'gamess'

# logging directory (In a restart run, recommended to give a name different from the original run)
logging_dir = 'logs'

########## END DEFAULT SETTINGS ##########

