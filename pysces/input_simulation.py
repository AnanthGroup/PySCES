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
from subprocess import Popen
from fileIO import print_ascii_art


########## DEFAULT SETTINGS ##########

nel, natom = 3, 6 # number of electronic states, number of atoms in the molecule
temp = 300 # simulation temperature in Kelvin

# Initial sampling function ('wigner', 'sc', or 'spin' LSC-IVR)
sampling = 'wigner'

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
tmax_rk4, Hrk4 = 20671, 1.0 

# Scaling factor of normal mode frequencies
frq_scale = 1.0

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
restart_file_in = 'restart.out'

#   type of QC runner, either 'gamess' or 'terachem'
QC_RUNNER = 'gamess'
#QC_RUNNER = 'terachem'

#   TeraChem runner options
tcr_host = '10.1.1.154'
tcr_port = 9876
tcr_server_root = '.'
tcr_job_options = {}
tcr_state_options = {
    'max_state': nel-1, 'grads': 'all'
}
#   TeraChem runner job specific options
tcr_spec_job_opts = {}
#    TeraChem runner job options for first frame only
tcr_initial_frame_opts = {
    'n_frames': 0
}
#   log TC job results
tcr_log_jobs = True

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



def _check_settings():
    opts.nnuc = 3*opts.natom # number of nuclear DOFs
    opts.ndof = opts.nel + opts.nnuc 

    if 'q0' not in local:
        opts.q0 = [0.0]*nel
    if 'p0' not in local:
        opts.p0 = [0.0]*nel


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

    #   TeraChem settings
    if opts.QC_RUNNER == 'terachem':
        max_state = opts.tcr_state_options.get('max_state', False)
        grads = opts.tcr_state_options.get('grads', False)
        
        if max_state and not grads:
            grads = list(range(max_state + 1))
            opts.tcr_state_options['grads'] = grads
        elif grads and not max_state:
            max_state = max(grads)
            opts.tcr_state_options['max_state'] = max_state
        elif grads and max_state:
            if max_state != max(grads):
                raise ValueError('"max_state" and highest "grads" index in "tcr_run_options" do not match')
        
        if 'nacs' not in opts.tcr_state_options:
            opts.tcr_state_options['nacs'] = 'all'

        opts.nel = len(grads)
        if nel != len(opts.q0) or nel != len(opts.p0):
            print(f"WARNING: Number of initial electronic coherent states (q0 and p0)")
            print(f"         does not match the number of TeraChem states ({opts.nel}): ")
            print(f"         Resetting q0 and p0 to all zeros")
            opts.q0 = [0.0]*opts.nel
            opts.p0 = [0.0]*opts.nel
     
    opts.ndof = opts.nel + opts.nnuc 
       
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
    import input_simulation_local as local_opts
    local = local_opts.__dict__
except Exception as e:
    print("Using default settings: ", e)
    from input_simulation import * 
    local = {}

def _print_settings():
    print
    print(f'Number of atoms:                    {natom}')
    print(f'Number of electronic states:        {nel}')
    print(f'Total degress of freedom:           {3*natom + nel}')
    print(f'Sampling method:                    {sampling}')
    print(f'Type fo integrator:                 {integrator}')
    if integrator == 'RK4:':
        print(f'Maximum simulation time:            {tmax_rk4:.2f} a.u.')
        print(f'Integrator time step:               {Hrk4} a.u.')

    print(f'Normal mode frequency scaling:      {frq_scale}')
    print(f'Electronic structure runner:        {QC_RUNNER}')
    print(f'Restart file will be written to     {restart_file_in}')
    print(f'current working directory:          {os.path.abspath(os.path.curdir)}')
    print(f'Logs will be written to:            {logging_dir}')

    # Print git commit
    try:
        command_git_tag="git -C "+str(os.path.dirname(os.path.realpath(__file__)))+" describe --tags"
        print("git tag: "+str(os.popen(command_git_tag).readline()))
    except:
        #   older versions of git don't have the -C option
        print("Cound not obtain git tag: If ithis is not desired, check your git version")


    # print(fmt_string.format("Number of atoms")'natom')
    # print(fmt_string.format("Number of electronic states", nel))
    # print(fmt_string.format("Total degress of freedom", 3*natom+nel))
    # print(fmt_string.format("Sampling method", sampling))
    # print(fmt_string.format("Type fo integrator", integrator))
    # print(fmt_string.format("Maximum simulation time", integrator))

_check_settings()
_set_seed()
print_ascii_art()
_print_settings()