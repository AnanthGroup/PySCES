#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  3 19:05:28 2023

@author: user
"""

"""
Repository of simulation variables
"""
import os
import sys
import shutil
from pysces import input_simulation as opts
from subprocess import Popen
from typing import Callable
import numpy as np


class TCRunnerOptions:
    #   TeraChem runner options
    host:str = '10.1.1.154'
    port:int = 9876
    server_root: str = '.'
    job_options: dict = {}
    state_options: dict = {
        'max_state': 1, 'grads': 'all'
    }

    #   TeraChem runner job specific options
    spec_job_opts: dict[str, dict] = {}

    #    TeraChem runner job options for first frame only
    initial_frame_opts: dict = {
        'n_frames': 0
    }
    #   TeraChem runner client assignments based on job names
    client_assignments: list[list[str]] = []

    #   log TC job results
    log_jobs: bool = True

    #   pysces should start it's own TeraChem servers
    server_gpus: list = []

    # Terachem frequency files
    fname_tc_xyz: str = "tmp/tc_hf/hf.spherical.freq/Geometry.xyz"
    fname_tc_geo_freq: str = "tmp/tc_hf/hf.spherical.freq/Geometry.frequencies.dat"
    fname_tc_redmas: str = "tmp/tc_hf/hf.spherical.freq/Reduced.mass.dat"
    fname_tc_freq: str = "tmp/tc_hf/hf.spherical.freq/Frequencies.dat"

    #   overlap data
    fname_exciton_overlap_data: str = None

    #   sometimes nacs can have different signs,
    #   this is a reference for the first frame
    _initial_ref_nacs = None


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
restart_file_out = 'restart.json'

#   type of QC runner, either 'gamess' or 'terachem'
qc_runner: str | Callable[[list], int] = 'gamess'
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
#   TeraChem runner client assignments based on job names
tcr_client_assignments = []
#   log TC job results
tcr_log_jobs = True
#   pysces should start it's own TeraChem servers
tcr_server_gpus = []

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

#   HDF5 File logging
hdf5_logging = True

logging_mode = 'a'

#   additional loggers
extra_loggers = []

#   for logging only
state_labels = None

#   DEBUG ONLY
tcr_ref_job = None

########## END DEFAULT SETTINGS ##########



########## GLOBAL SETTINGS, SHOULD NOT BE SET BY USER ##########
tc_runner_opts = TCRunnerOptions()
_set_defaults = False
defaults = {}
if not _set_defaults:
    for k, v in globals().copy().items():
        defaults[k] = v
    _set_defaults = True

def reset_settings():
    for k, v in defaults.items():
        globals()[k] = v

nnuc = 3*natom
ndof = nnuc + nel
com_ang = np.array([0.0, 0.0, 0.0])
##### END GLOBAL SETTINGS #####



def input_local_settings():
    '''
        Load in settings from the local input file
    '''
    if os.path.isfile('input_simulation_local.py'):
        print("Loading local settings")
        local_lines = []
        with open('input_simulation_local.py', 'r') as f:
            local_lines = f.read()
        locals = {}

        try:
            exec(local_lines, globals())

        except Exception as e:
            print("Error loading local settings: ", e)
            return


    _check_settings(locals)
    _set_seed()

def make_logging_dir():
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

def _check_settings(local: dict):
    opts.nnuc = 3*opts.natom # number of nuclear DOFs
    opts.ndof = opts.nel + opts.nnuc 

    #   legacy settings
    if 'q0' not in local:
        opts.q0 = [0.0]*nel
    if 'p0' not in local:
        opts.p0 = [0.0]*nel
    if 'QC_RUNNER' in globals():
        opts.qc_runner = globals()['QC_RUNNER']
        print('IN QC_RUNNER')

    #   set input format to the same type of QC runner
    if opts.mol_input_format == '':
        opts.mol_input_format = opts.qc_runner

    #   logging directory


    #   TeraChem settings
    for k, v in globals().items():
        if k.startswith('tcr_') or k.startswith('_tcr_'):
            tc_runner_opts.__dict__[k[4:]] = v
        elif k.startswith('fname_'):
            tc_runner_opts.__dict__[k] = v

    if opts.qc_runner == 'terachem':
        max_state = tc_runner_opts.state_options.get('max_state', False)
        grads = tc_runner_opts.state_options.get('grads', False)
        
        if max_state and not grads:
            grads = list(range(max_state + 1))
            tc_runner_opts.state_options['grads'] = grads
        elif grads and not max_state:
            max_state = max(grads)
            tc_runner_opts.state_options['max_state'] = max_state
        elif grads and max_state:
            if max_state != max(grads):
                raise ValueError('"max_state" and highest "grads" index in "tcr_run_options" do not match')
        
        if 'nacs' not in tc_runner_opts.state_options:
            tc_runner_opts.state_options['nacs'] = 'all'

        opts.nel = len(grads)
        if nel != len(opts.q0) or nel != len(opts.p0):
            print(f"WARNING: Number of initial electronic coherent states (q0 and p0)")
            print(f"         does not match the number of TeraChem states ({opts.nel}): ")
            print(f"         Resetting q0 and p0 to all zeros")
            opts.q0 = [0.0]*opts.nel
            opts.p0 = [0.0]*opts.nel
     
    opts.ndof = opts.nel + opts.nnuc

    #   check restart file
    if opts.restart == 1 and (not os.path.isfile(restart_file_in)):
        print('WARNING: Restart file not found: defaulting to initial condition generation')
        opts.restart = 0

    for k, v in opts.__dict__.items():
        globals()[k] = v
       
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


def print_settings():
    print()
    print(f'Number of atoms:                    {natom}')
    print(f'Number of electronic states:        {nel}')
    print(f'Total degress of freedom:           {3*natom + nel}')
    print(f'Sampling method:                    {sampling}')
    print(f'Type fo integrator:                 {integrator}')
    if integrator == 'RK4':
        print(f'Maximum simulation time:            {tmax_rk4:.2f} a.u.')
        print(f'Integrator time step:               {Hrk4} a.u.')

    print(f'Normal mode frequency scaling:      {frq_scale}')
    print(f'Electronic structure runner:        {qc_runner}')
    print(f'Molecule input format:              {mol_input_format}')
    print(f'Restart file will be written to     {restart_file_in}')
    print(f'current working directory:          {os.path.abspath(os.path.curdir)}')
    print(f'Logs will be written to:            {logging_dir}')

    # Print git commit
    try:
        command_git_tag="git -C "+str(os.path.dirname(os.path.realpath(__file__)))+" describe --tags"
        print("git tag: "+str(Popen(command_git_tag).readline()))
    except:
        #   older versions of git don't have the -C option
        print("Cound not obtain git tag: If ithis is not desired, check your git version")


    # print(fmt_string.format("Number of atoms")'natom')
    # print(fmt_string.format("Number of electronic states", nel))
    # print(fmt_string.format("Total degress of freedom", 3*natom+nel))
    # print(fmt_string.format("Sampling method", sampling))
    # print(fmt_string.format("Type fo integrator", integrator))
    # print(fmt_string.format("Maximum simulation time", integrator))

# _check_settings()
# _set_seed()

