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
from input_gamess import option as gms 
from subroutines import *
from fileIO import print_ascii_art
# __location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
__location__ = ''

from input_simulation import * 
import input_simulation as opts

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


def main():
    ###################################
    ### Propagation of a trajectory ###
    ###################################
    if opts.QC_RUNNER.casefold() == 'gamess':
        if opts.nel > 1 and gms['contrl']['runtyp'].casefold() != 'nacme':
            print('SETUP ERROR: A nonadiabatic run (nel > 1) using GAMESS is requested without calling NACME.')
            print('Check the GAMESS input and make sure to call NACME via runtyp=nacme.')
            exit()
        if opts.nel == 1 and len(opts.elab) > 1:
            print("WARNING: 'elab' in input_simulation specifies more than one electronic state while nel = 1.")
            print('Only the first entry is considered in this simulation.')
        
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
            compute_CF = compute_CF_wigner
        elif sampling == 'sc':
            coord = sample_scLSC(normal_geo, frq)
            compute_CF = compute_CF_sc
        elif sampling == 'spin':
            if nel != 3:
                print('WARNING: Spin mapping population estimator with nel being other than 3\n')
                print('is not implemented. Use "wigner" or "sc" option instead.\n')
                exit()
            coord = sample_spinLSC(normal_geo, frq)
            compute_CF = compute_CF_spin
        
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
