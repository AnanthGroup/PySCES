#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  3 18:55:55 2023

@author: kenmiyazaki
@coauthors: Chris Myers, Tom Trepl
"""

"""
The repository of functions to perform ab initio LSC-IVR nonadiabatic 
dynamics of polyatomic molecules
"""
import numpy as np
import scipy.integrate as it
import os
import sys
import subprocess as sp
import random
import pandas
import time
from input_simulation import * 
from input_gamess import option as opt 
from fileIO import SimulationLogger, write_restart, read_restart
# __location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
__location__ = ''

from input_simulation import *
nnuc = 3*natom
ndof = nnuc + nel
    
# Physical constants and unit conversion factors
pi       = np.pi
hplanck  = 6.62607015*10**-34   # Planck's constant in SI
hbar     = hplanck/(2*pi)       # reduced Planck's constant in SI
clight   = 2.99792458*10**8     # speed of light in SI
kb       = 1.380649*10**-23     # Boltzmann constant
eh2j     = 4.359744650*10**-18  # Hartree to Joule
amu2au   = 1.822888486*10**3    # atomic mass unit to atomic unit 
autime2s = hbar/eh2j            # atomic unit time to second
au2fs    = autime2s*10**15      # atomic unit time to second
ang2bohr = 1.8897259886         # angstroms to bohr
k2autmp  = kb/eh2j              # Kelvin to atomic unit temperature
beta     = 1.0/(temp * k2autmp) # inverse temperature in atomic unit

#####################################################
### Read geometry & hessian file, returns 
### 1. AMU matrix
### 2. molecular xyz geometry
### 3. Normal mode frequencies
### 4. Normal mode reduced masses
### 5. L matrix (GAMESS hessian output)
### 6. U matrix (Hessian unitary matrix)
### 7. center of mass vector
####################################################
def get_geo_hess():
    if mol_input_format == "terachem":
        amu_mat, xyz_ang, frq, redmas, L, U, com_ang, atom_number_mat = get_geo_hess_terachem()
    elif mol_input_format == "gamess":
        amu_mat, xyz_ang, frq, redmas, L, U, com_ang, atom_number_mat = get_geo_hess_gamess()
    else:
        print("Error: get_geo_hess ran in undefined 'mol_input_format' case")        
        exit()
    return(amu_mat, xyz_ang, frq, redmas, L, U, com_ang, atom_number_mat)

def get_geo_hess_terachem():
    ##--------------------------------------------------
    ## 1 & 2 Read Cartesian coordinate of initial geometry
    ##--------------------------------------------------
    
    # initialize arrays
    amu = []
    xyz_ang = np.zeros(nnuc)
    amu_mat = np.zeros((nnuc,nnuc))
    
    n_vib_modes = nnuc - 6
    
    # Open TeraChem Geometry.frequencies.dat file from scratch dir
    # This file contains the geometry in bohr after removing the center of mass
    # Note: if a different file is used for coords, check whether COM is removed
    # Note: The xyz file is not used for this part of the code. It is only used for the labels
    with open(os.path.join(__location__,fname_tc_geo_freq)) as f:
        f_lines = f.readlines()
    
    # Coordinates start in second line
    for ia in range(0,natom):
        current_line = f_lines[ia+1].split()
        amu.append(float(current_line[0]))
        for ja in range(0,3):
            # mass entries in TC geometry frequencies file are in amu
            amu_mat[3*ia+ja,3*ia+ja] = float(current_line[0])
            # xyz entries in TC geometry file are in a.u.
            xyz_ang[3*ia+ja] = 1.0/ang2bohr * float(current_line[ja+1])

    ##--------------------------------------------------
    ## 4. Read in reduced mass
    ##--------------------------------------------------
    # Allocate reduced mass array
    #TODO tom: can be calulated - reading unnecessary
    redmas = np.zeros(nnuc) # in a.u.
   
    # Open Reduced.mass.dat
    with open(os.path.join(__location__,fname_tc_redmas)) as f:
        f_lines = f.readlines()
    
    # Read reduced masses   
    for ivm in range(0,n_vib_modes):
        current_line = f_lines[ivm+1].split()
        # keep first 6 entries empty for 3 trans. and 3 rot. modes
        # and convert from amu to a.u.
        redmas[ivm+6] = amu2au*float(current_line[2])

    ##--------------------------------------------------
    ## 3. Read in frequencies
    ## 5. Read in eigenvectors
    ##--------------------------------------------------
    # Definition of L:
    # Columns of L (e.g. L[:,0]) contain the eigenvectors
    # Rows of L (e.g. L[0,:]) contain (dx1/dq1, dx1/dq2, dx1/dq3, ...)
    # Note: L is not unitless! L has units 1/sqrt(amu)

    # Initialize structures
    # allocate frequencies array
    frq = np.zeros(nnuc)
    # L contains eigenvectors (EV). E.g. L[:,0] is the first EV
    L = np.zeros((nnuc,nnuc))
    with open(os.path.join(__location__,fname_tc_freq)) as f:
        f_lines = f.readlines()

    for ivm in range(0,n_vib_modes):
        # line begin (which contains the first data) of mode ivm
        # add 6 for the 6 header lines; 4 resembles number of columns
        lbegin = 6+int((ivm-ivm%4)/4)*((3*natom)+4)
        # column of mode ivm
        lcolumn = ivm%4
        # get frequency in 1/cm (skip 6 indices for trans+rot modes)
        # and convert to a.u. and scale by frq_scale
        frq[ivm+6] = f_lines[lbegin-2].split()[lcolumn]
        frq[ivm+6] = frq[ivm+6] * 2.0*pi*clight*100*autime2s*frq_scale
        # Write a warning if the frequency is negative
        if(frq[ivm+6] < 0.0): print(f"Warning: Vibrational normal mode {ivm} has a negative frequency - its initial momentum is set to 0.")
        # Get eigenvectors
        # For the x direction we need one more column, because this line also contains the atom number
        # (skip 6 indices in L for the trans+rot modes)
        for ia in range(0,natom):
            L[3*ia+0,ivm+6] = float(f_lines[lbegin+ia*3+0].split()[lcolumn+1])
            L[3*ia+1,ivm+6] = float(f_lines[lbegin+ia*3+1].split()[lcolumn+0])
            L[3*ia+2,ivm+6] = float(f_lines[lbegin+ia*3+2].split()[lcolumn+0])

    #TODO tom: remove the following begin debugging test
    #test_vec = L[:,6] #first EV
    #print(test_vec)
    #for i in range(0,nnuc):
    #    test_vec = L[i,:]
    #    #print("testvec",test_vec)
    #    print("L2-norm testvec",np.dot(test_vec,test_vec))
    #    print("tv*m(au)*tv",np.dot(test_vec,amu2au*np.matmul(amu_mat,test_vec)))
    #exit()
    # end TODO debugging test
    
    # -------------------------------------------------
    # 6. U matrix (mass-weighted eigenmodes)
    # -------------------------------------------------
    # U contains sqrt(mass)-weighted EV as rows. It has no units
    # U is defined in a transposed way compared to L, i.e.,
    # U[0,:] contains the first sqrt(mass)-weighted eigenvector.
    U = np.zeros((nnuc,nnuc))
    U = np.matmul(L.T,amu_mat**0.5) 

    # U is a unitary matrix and normalization is not necessary. 
    # If one still wants to do it, outcomment the following lines
    # imo, you would need to rescale the rows.
    #for i in range(7,nnuc):
    #    norm = sum(U[i,:]**2)
    #    U[i,:] = U[i,:]/np.sqrt(norm)

    # COM is already substracted in Geometry.Frequencies.dat but better save than sorry
    # compute center of mass and remove from geometry
    amu = np.array(amu)
    xyz_shaped = xyz_ang.reshape((-1, 3))
    com = np.average(xyz_shaped, axis=0, weights=amu)
    xyz_ang = (xyz_shaped - com).flatten()

    atom_number_mat = [] # Returns an empty array. Not necessary for terachem option
    return(amu_mat, xyz_ang, frq, redmas, L, U, com, atom_number_mat)


def get_geo_hess_gamess():
    # Read Cartesian coordinate of initial geometry
    atom_number = []
    xyz_ang = np.zeros(nnuc)
    atom_number_mat = np.zeros((nnuc, nnuc))
    with open(os.path.join(__location__, 'geo_gamess'), 'r') as f:
        f.readline()
        for i in range(natom):
            x = f.readline().split()
            atom_number.append(x[1])
            for j in range(3):
                xyz_ang[3*i+j] = float(x[2+j])

    # Form a diagonal matrix for atomic masses in amu as well as atomic numbers
    amu = []
    with open(os.path.join(__location__, 'mass_gamess'), 'r') as f:
        for i in range(natom):
            x = f.readline().split()
            amu.append(float(x[2]))
    amu_mat = np.zeros((nnuc, nnuc))
    for i in range(natom):
        for j in range(3):
            amu_mat[3*i+j,3*i+j] = amu[i]
            atom_number_mat[3*i+j,3*i+j] = atom_number[i]

    # Read hessian from hess_gamess
    frq, redmas = np.zeros(nnuc), np.zeros(nnuc)
    L, U = np.zeros((nnuc, nnuc)), np.zeros((nnuc, nnuc))
    if nnuc%5 == 0:
        nchunk = int(nnuc/5)
    else:
        nchunk = int(nnuc/5) + 1
    nline = 6 + nnuc + 11
    f = open(os.path.join(__location__,'hess_gamess'))
    for ichunk in range(nchunk):
        if ichunk == nchunk-1:
            if nnuc%5 == 0:
                ncolumn = 5
            else:
                ncolumn = int(nnuc%5)
        else:
            ncolumn = 5

        for iline in range(nline):
            x = f.readline()
            if iline == 1:
                x = x.split()
                for icolumn in range(ncolumn):
                    frq[ichunk*5+icolumn] = float(x[icolumn])
            elif iline == 3:
                x = x.split()
                for icolumn in range(ncolumn):
                    redmas[ichunk*5+icolumn] = float(x[icolumn])
            elif iline >= 6:
                if iline < nnuc+6:
                    x = x.split()
                    for icolumn in range(ncolumn):
                        L[ichunk*5+icolumn, iline-6] = float(x[icolumn])
    f.close()

    # Convert frq & red. mass into atomic unit
    for i in range(nnuc):
        frq[i] *= 2.0*pi * clight*100 * autime2s * frq_scale
        redmas[i] *= amu2au

    # Redefine L so that the row of L, for example L(1,:),
    # is (dx1/dq1, dx1/dq2, ..., dx1/dq3N)
    L = L.T

    # Define a unitary matrix U based on L
    U = np.matmul(L.T, amu_mat**0.5)

    #   compute center of mass and remove from geometry
    amu = np.array(amu)
    xyz_shaped = xyz_ang.reshape((-1, 3))
    com = np.average(xyz_shaped, axis=0, weights=amu)
    xyz_ang = (xyz_shaped - com).flatten()

    return amu_mat, xyz_ang, frq, redmas, L, U, com, atom_number_mat


##############################################################################
### Read the initial geometry in xyz and rotate it into normal coordinates 
### to define the initial phase space displacement
##############################################################################
def get_normal_geo(U, xyz_ang, amu_mat, debug=False):
    # Define amu2au conversion factor as a matrix
    amu2au_mat = np.zeros((nnuc,nnuc))
    for i in range(nnuc):
        amu2au_mat[i,i] = amu2au
    
    # Convert nuclear geometry in angstrom into bohr
    xyz_bohr = xyz_ang * ang2bohr

    # Rotate Cartesian into normal coordinates
    # To do this, do q = M.(L.T).x = sqrt(M).U.x, where M is a diagonal matrix
    # of amu and L.T is the transpose of L, and U is the unitary transformation 
    # matrix. 
    # NOTE: The row of L.T as well as U is reading the GAMESS hessian matrix VERTICALLY.  
    normal_geo = np.matmul(U[6:,:], np.matmul(np.matmul(amu_mat, amu2au_mat)**0.5, xyz_bohr))

    if debug:
        print("U:\n",pandas.DataFrame(U))
        print("U U.T:\n",pandas.DataFrame(np.matmul(U,U.T)))
        print("amu_mat:\n",pandas.DataFrame(amu_mat))
        print("normal coords:\n",normal_geo)  


    return(normal_geo)
        
        
############################################################
### Sampling of initial phase space configurations using ### 
### random number generator.                             ###
############################################################
'''Nuclear phase space variables in harmonic approximation'''
def sample_nuclear(qcenter, frq):
    if(frq >= 0):
        # position
        Q = np.random.normal(loc=qcenter, scale=np.sqrt(1.0/(2.0*frq*np.tanh(beta*frq/2.0))))
        # print("WARNING: Ignoring the nuclear coordinate sampling!!!!!!")
        # Q = qcenter
        # momentum
        P = np.random.normal(loc=pN0, scale=np.sqrt(frq/(2.0*np.tanh(beta*frq/2.0))))
    else:
        Q = qcenter
        P = pN0
    return(Q, P)

'''LSC-IVR with Wigner population estimator'''
def sample_wignerLSC(qN0, frq):
    coord = np.zeros((2, ndof-6))

    # Determine the sampling radius of initially occupied electronic state
    from scipy.optimize import fsolve
    from functools import partial
    def eqn(F, r):
        return 2 ** (F + 1) * (r - 0.5) * np.exp(-(r + 0.5 * (F - 1))) - 1

    root = fsolve(partial(eqn, nel), 2)
    r = root[0] ** 0.5
        
    # Electronic phase space variables
    for i in range(nel):
        theta = random.random()
        if i == init_state-1:
            x = r * np.cos(2.0*pi*theta)
            p = r * np.sin(2.0*pi*theta)
        else:
            x = np.sqrt(1.0/2.0) * np.cos(2.0*pi*theta)
            p = np.sqrt(1.0/2.0) * np.sin(2.0*pi*theta)

        coord[0, i] = x
        coord[1, i] = p
    
    # Nuclear phase space variables
    for i in range(nnuc-6):
        coord[0, i+nel], coord[1, i+nel] = sample_nuclear(qN0[i], frq[i+6])

#    # Nuclear phase space variables
#    for i in range(nnuc-6):
#        # position
#        coord[0,i+nel] = np.random.normal(loc=qN0[i], scale=np.sqrt(1.0/(2.0*frq[i+6]*np.tanh(beta*frq[i+6]/2.0))))
#        # momentum
#        coord[1,i+nel] = np.random.normal(loc=pN0, scale=np.sqrt(frq[i+6]/(2.0*np.tanh(beta*frq[i+6]/2.0))))
    
    return(coord)


'''LSC-IVR with semiclassical population estimator'''
def sample_scLSC(qN0, frq):
    coord = np.zeros((2, ndof-6)) 
    # Electronic phase space variables
    for i in range(nel):
        theta = random.random()
        if i == init_state-1:
            x = np.sqrt(3.0) * np.cos(2.0*pi*theta)
            p = np.sqrt(3.0) * np.sin(2.0*pi*theta)
        else:
            x = np.cos(2.0*pi*theta)
            p = np.sin(2.0*pi*theta)

        coord[0,i] = x
        coord[1,i] = p
    
    # Nuclear phase space variables
    for i in range(nnuc-6):
        coord[0, i+nel], coord[1, i+nel] = sample_nuclear(qN0[i], frq[i+6])
    
    return(coord)

'''Spin LSC-IVR'''
def sample_spinLSC(qN0, frq):
    coord = np.zeros((2, ndof-6)) 
    # Electronic phase space variables
    for i in range(nel):
        theta = random.random()
        if i == init_state-1:
            x = np.sqrt(8.0/3.0) * np.cos(2.0*pi*theta)
            p = np.sqrt(8.0/3.0) * np.sin(2.0*pi*theta)
        else:
            x = np.sqrt(2.0/3.0) * np.cos(2.0*pi*theta)
            p = np.sqrt(2.0/3.0) * np.sin(2.0*pi*theta)

        coord[0, i] = x
        coord[1, i] = p
    
    # Nuclear phase space variables
    for i in range(nnuc-6):
        coord[0, i+nel], coord[1, i+nel] = sample_nuclear(qN0[i], frq[i+6])
    
    return(coord)


####################################
### Get atomic symbols as a list ###
####################################
def get_atom_label():
    atoms = []
    if(mol_input_format == "gamess"):
      f = open(os.path.join(__location__,'geo_gamess'), 'r')
      f.readline()
    elif(mol_input_format == "terachem"):
      f = open(os.path.join(__location__,fname_tc_xyz), 'r')
      f.readline()
      f.readline()
    for i in range(natom):
        x = f.readline().split()
        atoms.append(x[0])
    return(atoms)


#########################################################################
### Rotate q&p sampled in normal coordinate into Cartesian coordinate ###
#########################################################################
def rotate_norm_to_cart(qN, pN, U, amu_mat):
    au_mass_half, au_mass_halfinv = np.zeros((nnuc,nnuc)), np.zeros((nnuc,nnuc))
    for i in range(nnuc):
        au_mass_half[i,i] = np.sqrt(amu_mat[i,i] * amu2au)
        au_mass_halfinv[i,i] = 1.0/np.sqrt(amu_mat[i,i] * amu2au)
    
    # To rotate normal coords q into Cartesian x, do x = Lq = (M**-1/2).(U.T).q 
    # where L is the matrix in GAMESS hessian output and U.T is the transpose
    # of the unitary transformation matrix. Likewise, to rotate normal coords 
    # momentum pN into Cartesian pCart, do pCart = (M**1/2).(U.T).pN
    # NOTE: The row of L as well as U.T is reading the GAMESS hessian output 
    # HORIZONTALLY.
    qCart = np.matmul(au_mass_halfinv, np.matmul(U[6:,:].T, qN))
    pCart = np.matmul(au_mass_half, np.matmul(U[6:,:].T, pN))
    return(qCart, pCart)


#####################################################
### Record the nuclear geometry at each time step ###
#####################################################
def record_nuc_geo(restart, total_time, atoms, qCart, com_ang=None, logger:SimulationLogger=None):
    if logger is not None:
        return logger._nuc_geo_logger.write(total_time, atoms, qCart/ang2bohr, com_ang)
    f = open(os.path.join(__location__, 'nuc_geo.xyz'), 'a')
    
    if com_ang is None:
        com_ang = np.zeros(3)

    qCart_ang = qCart/ang2bohr
    f.write('%d \n' %natom)
    f.write('%f \n' %total_time)
    for i in range(natom):
        f.write('{:<5s}{:>12.6f}{:>12.6f}{:>12.6f} \n'.format(
            atoms[i],
            qCart_ang[3*i+0] + com_ang[0],
            qCart_ang[3*i+1] + com_ang[1],
            qCart_ang[3*i+2] + com_ang[2]))
    f.close()
    return()


###########################################
### Update geo_gamess with new geometry ###
###########################################
def update_geo_gamess(atom_symbols, amu_mat, qCart):
    # Convert Bohr to Ang
    qCart_ang = qCart/ang2bohr
    #with open(os.path.join(__location__, 'progress.out'), 'a') as g:
    #    g.write('Check if geometry is in Ang \n')
    #    for i in range(natom):
    #        g.write('{:<16.10f}{:<16.10f}{:<16.10f} \n'.format(qCart[3*i+0],qCart[3*i+1],qCart[3*i+2]))
    #    g.write('\n')

    # Write a new geo_gamess
    f = open(os.path.join(__location__, 'geo_gamess'), 'w')
    f.write('%i \n' %natom)
    for i in range(natom):
        f.write('%s %6.1f %16.10f %16.10f %16.10f \n' 
                %(atom_symbols[i],amu_mat[3*i,3*i],qCart_ang[3*i+0],qCart_ang[3*i+1],qCart_ang[3*i+2]))
    f.close()
    return()


################################
### Write GAMESS input file  ###
################################
def write_gms_input(input_name, opt, atoms, AN_mat, cart_ang):
    """
    Write GAMESS input file
    """
    input_file = input_name + '.inp'
    if os.path.exists(os.path.join(__location__, input_file)) == True:
        os.system('mv ' + input_file + ' ' + input_name + '_old.inp')
        f = open(os.path.join(__location__, input_file), 'w')
    else:
        f = open(os.path.join(__location__, input_file), 'w')

    cards = list(opt.keys())
    for card in cards:
        if card != 'data':
            if opt[card] != '':
                variables = list(opt[card].keys())
                line = ''
                nvar = 0
                for var in variables:
                    nvar += 1
                    line += '{:s}={:s} '.format(var, opt[card][var])
                    if nvar == 4:
                        f.write(' ${:s} '.format(card) + line + ' $end\n')
                        nvar = 0
                        line = ''
                    elif var == variables[-1]:
                        f.write(' ${:s} '.format(card) + line + ' $end\n')

    f.write(' $data \n')
    f.write("Always blame ab initio calculations, not the dynamcis code I wrote ;) \n")
    f.write(opt['data']['sym']+' \n')
    for i in range(natom):
        f.write('{:<3s}{:<6.1f}{:>12.5f}{:>12.5f}{:>12.5f}\n'.format(atoms[i], AN_mat[3*i,3*i], cart_ang[3*i+0], cart_ang[3*i+1], cart_ang[3*i+2]))
    f.write(' $end \n')

    # Read and write guess orbitals 
    if opt.get('guess', '') != '':
        g = open(os.path.join(__location__, 'vec_gamess'), 'r')
        copy = False
        for line in g:
            if line.strip() == '$VEC':
                copy = True
            if copy:
                f.write(line)
            if line.strip() == '$END':
                copy = False
        g.close()
    f.close()

    return input_file


##########################################
### Write GAMESS job submission script ###
##########################################
def write_subm_script(input_name):
    script_name = 'run_' + input_name
    f = open(os.path.join(__location__, script_name), 'w')
    f.write('#!/bin/bash \n')
    f.write('#SBATCH -J ' + input_name + ' \n')
    f.write('#SBATCH -t 240:00:00 \n')
    f.write('#SBATCH -n ' + str(ncpu) + '\n')
    f.write('#SBATCH -N ' + str(nnode) + '\n')
    f.write('#SBATCH -p ' + partition + '\n')
#    f.write('#SBATCH --exclusive')
    f.write('#SBATCH -o %x.o%j \n')
    f.write('#SBATCH -e %x.e%j \n')
    f.write('scontrol show hostnames $SLURM_JOB_NODELIST > HostFile \n')
    f.write('for node in $(scontrol show hostnames $SLURM_NODELIST); do \n')
    f.write('   srun -N 1-1 -n 1 -w $node /usr/bin/mkdir -p /tmp/$SLURM_JOB_ID/ \n')
    f.write('done \n')
    f.write('sleep 4 \n')
    f.write('./runG_common_pool ' + input_name + '.inp ' + str(ncpu*nnode))
    f.close()
    return()
    

#####################################
### Call GAMESS NACME calculation ###
#####################################
def run_gms_cas(input_name, opt, atoms, AN_mat, qCart, submit_script_loc=None):
    # Convert Bohr into Angstrom
    qCart_ang = qCart/ang2bohr
    
    # Write an input file
    input_file = write_gms_input(input_name, opt, atoms, AN_mat, qCart_ang)
    
    if submit_script_loc is None:
        # Write a submission script
        write_subm_script(input_name)
        
        # change # ppn per node in 'rungms-pool' script
        with open(os.path.join(__location__, 'rungms-pool'), 'r') as g:
            all_lines = g.readlines()
            all_lines[509] = "      @ NSMPCPU = %d \n" %(ncpu)
                
        # Change the mode of submission script
        sp.run(['chmod', '777', 'run_%s' %input_name])
        
        # Submit the bash submission script to execute GAMESS
        sp.call('./run_%s' %input_name)
    else:
        #   call supplied submission script
        output_file = 'cas.out'
        script_loc = os.path.abspath(submit_script_loc)
        sp.call(f'{script_loc} {input_file} {output_file}'.split())
    print("Done running GAMESS CAS-SCF Calculations")

    return()


def read_gms_out(input_name):
    """
    Read GAMESS output file for the electronic state energies, 
    the energy gradients, NACME, and the transition dipole moment
    """
    flag_grad = np.ones(nel)
    flag_dip  = 1
    flag_nac  = 1
    nac = np.zeros((nel, nel, nnuc))
    grad_exist = np.zeros(nel)
    energy = np.zeros(nel)
    gradient = np.zeros((nel, nnuc))
    out_file = input_name + '.out'
    out = {} # dict of outputs

    nrow = 3*natom / 5
    if nrow % 1 > 0:
        nrow = int(np.ceil(nrow))
    else:
        nrow = int(nrow)
    with open(os.path.join(__location__, out_file), 'r') as f:
        for line in f:
            if len(elab) == 1:
                flag_nac = 0
                if 'TRANSITION DIPOLE' in line:
                    flag_dip = 0
                    f.readline()
                    for i in range(elab[0]):
                        f.readline()
                    x = f.readline().split()[4:7]

                    out['dip'] = [float(k) for k in x]
                    out['flag_dip'] = flag_dip

                if '$VIB' in line:
                    flag_grad[0] = 0
                    x = f.readline().split()
                    energy[0] = float(x[-1])
                    out['elecE'] = energy

                    for irow in range(nrow):
                        if irow == nrow-1:
                            nval = (3*natom) % 5
                        else:
                            nval = 5
                        x = f.readline()[1:]
                        x = [x[i:i+16] for i in range(0, len(x), 16)]
                        for k in range(nval):
                            gradient[0, 5*irow+k] = float(x[k])

                    out['gradient'] = gradient
                    out['flag_grad'] = flag_grad
                out['flag_nac'] = 0
                out['nac'] = nac

            elif len(elab) > 1:
                for i, elab_idx in enumerate(elab):
                    if 'STATE-SPECIFIC GRADIENT OF STATE   ' + str(elab_idx) in line:
                        flag_grad[i] = 0
                        x = f.readline().split()
                        energy[i] = float(x[1])

                        [f.readline() for k in range(2)]

                        for j in range(natom):
                            x = f.readline().split()
                            for k in range(3):
                                gradient[i,3*j+k] = float(x[2+k])

                out['elecE'] = energy
                out['gradient'] = gradient
                out['flag_grad'] = flag_grad
                out['dip'] = np.zeros(nel)
                out['flag_dip'] = 0

                if 'NONADIABATIC COUPLING MATRIX ELEMENT' in line:
                    flag_nac = 0
                    x = f.readline()
                    for i in range(nel-1):
                        j = i + 1
                        while j < nel:
                            if 'STATE  ' + str(elab[j]) in x:
                                if 'STATE  ' + str(elab[i]) in x:
                                    f.readline()
                                    for k in range(natom):
                                        x = f.readline().split()
                                        for l in range(3):
                                            nac[j,i,3*k+l] = float(x[2+l])
                            nac[i,j,:] = -nac[j,i,:]
                            j += 1

                    out['nac'] = nac
                    out['flag_nac'] = flag_nac

    if any([el == 1 for el in out['flag_grad']]):
        with open(os.path.join(__location__, 'progress.out'), 'a') as f:
            f.write('Error: Gradient of energy not found in .out \n')
    if out['flag_dip'] == 1:
        with open(os.path.join(__location__, 'progress.out'), 'a') as f:
            f.write('Error: Transition dipole not found in .out \n')
    if flag_nac == 1:
        with open(os.path.join(__location__, 'progress.out'), 'a') as f:
            f.write('Error: Non-adiabatic couplings not found in .out. \n')
    if nel == 1:
        with open(os.path.join(__location__, 'transition_dipole.out'), 'a') as f:
            f.write('{:<12.6f}{:<12.6f}{:<12.6f}\n'.format(*out['dip']))
    return(out)


###############################################
### Read GAMESS NACME calculation .dat file ###
###############################################
def read_gms_dat(input_name):
    flag_orb = 1
    dat_file = input_name + '.dat'
    with open(os.path.join(__location__, dat_file), 'r') as f:
        for line in f:
            if 'OPTIMIZED MCSCF' in line or 'MCSCF OPTIMIZED' in line:
                flag_orb = 0
                [f.readline() for i in range(2)]
                with open(os.path.join(__location__, 'vec_gamess'), 'w') as g:
                    copy = False
                    reading = True
                    while reading:
                        x = f.readline()
                        if '$VEC' in x:
                            copy = True
                        if copy:
                            g.write(x)
                        if '$END' in x:
                            copy = False
                            reading = False

    if flag_orb == 1:
        with open(os.path.join(__location__, 'progress.out'), 'a') as f:
            f.write('Error: Optimized orbitals not found in .dat. \n')
            
    return(flag_orb)


#####################################################################
### Compute equations of motion (mapping variables derivatives)   ###
### of adiabatic MM-ST Hamiltonian with the symmetrized potential ###
#####################################################################
def get_derivatives(au_mas, q, p, nac, grad, elecE):
    der = np.zeros((2, ndof))
    
    # Derivatives of elctronic mapping variables
    for i in range(nel):
        xdpm   = 0 # x*d*p/m
        pdpm   = 0 # p*d*p/m
        sum_DE = 0 # sum(Ei - Ej)
        for j in range(nel):
            if j != i:
                xdpm   += q[j] * np.matmul(nac[j,i,:], p[nel:]/au_mas)
                pdpm   += p[j] * np.matmul(nac[j,i,:], p[nel:]/au_mas)
                sum_DE += elecE[i] - elecE[j]
        # postions
        der[0, i] =  (1.0/nel) * p[i] * sum_DE + xdpm
        # momenta
        der[1, i] = -(1.0/nel) * q[i] * sum_DE + pdpm
    
    # Derivatives of nuclear mapping variables
    for n in range(nnuc):
        # positions
        der[0, nel+n] = p[nel+n]/au_mas[n]
        # momenta
        sum_dEdR   = 0
        p2x2_DdEdR = 0 # (pi^2 - pj^2 + qi^2 - qj^2) * (dEi/dR - dEj/dR)
        ppxx_DEnac = 0 # (pi * pj + qi * qj) * (Ej - Ei) * dij
        for i in range(nel):
            sum_dEdR  += grad[i,n]
            j = i + 1
            while j < nel:
                p2x2_DdEdR += (p[i]**2 - p[j]**2 + q[i]**2 - q[j]**2) * (grad[i,n] - grad[j,n])
                ppxx_DEnac += (p[i]*p[j] + q[i]*q[j]) * (elecE[j] - elecE[i]) * nac[i,j,n]
                j += 1
        der[1, nel+n] = -(1.0/nel) * sum_dEdR - (0.5/nel) * p2x2_DdEdR - ppxx_DEnac
            
    return(der)


##########################################################
### Compute the preductor of modified Euler integrator ###
##########################################################
def compute_ME_predictor(timestep, init_coord, der):
    pred = init_coord + der * timestep
    return(pred)
    

##########################################################
### Compute the corrector of modified Euler integrator ###
##########################################################
def compute_ME_corrector(timestep, init_coord, der1, der2):
    corr = init_coord + 0.5*(der1 + der2) * timestep
    return(corr)


#############################################################################
### Compute the predictor of 4th order Adams-Bashforth-Moulton integrator ###
#############################################################################
def compute_ABM_predictor(timestep, init_coord, f1, f2, f3, f4):
    pred = init_coord + timestep*(55.0*f1 - 59.0*f2 + 37.0*f3 - 9.0*f4)/24.0
    return(pred)


#############################################################################
### Compute the corrector of 4th order Adams-Bashforth-Moulton integrator ###
#############################################################################
def compute_ABM_corrector(timestep, init_coord, f1, f2, f3, f4):
    corr = init_coord + timestep*(9.0*f1 + 19.0*f2 - 5.0*f3 + f4)/24.0
    return(corr)


##############################################################################
### Compute the total energy of adiabatic mapping Hamiltonian defined with ###
### the symmetrized potential
##############################################################################
def get_energy(au_mas, q, p, elecE):
    # Nuclear part (sum of P**2/M)
    p2m_sum = 0
    for n in range(nnuc):
        p2m_temp = 0
        p2m_temp = p[nel+n]**2/au_mas[n]
        p2m_sum += p2m_temp
        
    # Electronic part ((pi2 - p2j + qi2 - qj2) * (Ei - Ej))
    p2x2_DE = 0
    for i in range(nel):
        j = i + 1
        while j < nel:
            p2x2_DE += (p[i]**2 - p[j]**2 + q[i]**2 - q[j]**2) * (elecE[i] - elecE[j])
            j += 1
    
    # Total energy at updated t
    energy = 0.5*p2m_sum + (1.0/nel)*sum(elecE) + (0.5/nel)*p2x2_DE
    return(energy)


# #######################################################
# ### Compute the population of the electronic states ###
# #######################################################
# def get_elec_pop(q, p):
#     total_pop = 0
#     pop = np.zeros(nel)
#     for i in range(nel):
#         pop[i] = 0.5*(q[i]**2 + p[i]**2 - 1.0)
#     total_pop = sum(pop)
#     return(pop, total_pop)


#############################################################################
### Propagate trajectoies using modified-Euler + Adams-Bashforth-Moulton
### integrator.
### Note: Sampling of the initial phase points is done in the normal 
### coordinates but the subsequesnt propagation is made in the Cartesian 
### coordinates. MUST BE CAREFUL on in which coordinate system each variable 
### is defined and especially, that the normal coodinates variables are 
### MASS-WEIGHTED.  
#############################################################################
'''Last edited by by Ken Miyazaki on 05/10/2023'''
def ME_ABM(restart, initq, initp, amu_mat, U, com_ang, AN_mat):
    force = np.zeros((2, ndof, 5))
    der   = np.zeros((2, ndof))
    coord = np.zeros((2, ndof, nstep+1))
    qpred, ppred, qcorr, pcorr = np.zeros(ndof), np.zeros(ndof), np.zeros(ndof), np.zeros(ndof)
    energy = np.zeros(nstep+1)
    input_name = 'cas'
    flag_energy, flag_orb, flag_grad, flag_nac = 0, 0, 0, 0
    
    # Format descriptor depending on the number of electronic states
    total_format = '{:>12.4f}{:>12.5f}'
    for i in range(nel):
        total_format += '{:>12.5f}'
    total_format += '\n'
    
    # Initialization
    if restart == 0: # If this is not a restart run
        x, initial_time = 0.0, 0.0
        q, p    = np.zeros(ndof), np.zeros(ndof)          # collections of all mapping variables
        q[:nel], p[:nel] = initq[:nel], initp[:nel]
        qN, pN  = initq[nel:], initp[nel:]                # collections of nuclear variables in normal coordinate
        qC, pC  = rotate_norm_to_cart(qN, pN, U, amu_mat) # collections of nuclear variables in Cartesian coordinate
        q[nel:], p[nel:] = qC, pC

        # Get atom labels
        atoms = get_atom_label()
        
        # Write initial nuclear geometry in the output file
        record_nuc_geo(restart, x, atoms, qC, com_ang)
        
    elif restart == 1: # If this is a restart run
        q, p    = np.zeros(ndof), np.zeros(ndof)
        # Read the restart file
        with open(os.path.join(__location__, 'restart.out'), 'r') as ff:
            ff.readline() 
            for i in range(ndof):
                x = ff.readline().split()
                q[i], p[i] = float(x[0]), float(x[1]) # Mapping variables already in Cartesian coordinate
            [ff.readline() for i in range(2)]
            for t in range(4):
                ff.readline()
                for i in range(ndof):
                    x = ff.readline().split()
                    force[0,i,t], force[1,i,t] = float(x[0]), float(x[1]) # derivative of each MV
                ff.readline()
            ff.readline()
            init_energy = float(ff.readline()) # Total energy
            [ff.readline() for i in range(2)]
            initial_time = float(ff.readline()) # Total simulation time at the beginning of restart run
            x = initial_time       
 
        qC, pC  = q[nel:], p[nel:]
        
        # Get atom labels
        atoms = get_atom_label()
    
    X = []
    X.append(x)
    proceed = True # Error detector
    terminate = False # Detect the end of the loop
    while proceed and not terminate:
        au_mas = np.diag(amu_mat) * amu2au # masses of atoms in atomic unit
        
        # Update geo_gamess with qC
        update_geo_gamess(atoms, AN_mat, qC)
        
        # Call GAMESS to compute E, dE/dR, and NAC
        run_gms_cas(input_name, opt, atoms, AN_mat, qC)

        # Read GAMESS out file
        elecE, grad, nac, flag_grad, flag_nac = read_gms_out(input_name)
        if any([el == 1  for el in flag_grad]) or flag_nac == 1:
            proceed = False
            break
        
        # Read GAMESS dat file
        flag_orb = read_gms_dat(input_name)
        if flag_orb == 1:
            proceed = False
            break
        
        # Record the initial coordinate
        coord[0,:,0] = q
        coord[1,:,0] = p
        
        with open(os.path.join(__location__, 'progress.out'), 'a') as f:
            f.write('Total number of steps in the simulation: %s \n' %nstep)
            f.write('\n')
        
        if restart == 0:
            ##############################
            ### Modified-Euler routine ###
            ##############################
            # Timestep is made intentionally small because this is a preliminary propagation.
            for t in range(3):
                # Get derivatives
                der = get_derivatives(au_mas, q, p, nac, grad, elecE)
                force[:,:,-(t+2)] = der
                
                # Make predictions
                for i in range(ndof):
                    qpred[i] = compute_ME_predictor(-timestep/200, q[i], force[0,i,-(t+2)])
                    ppred[i] = compute_ME_predictor(-timestep/200, p[i], force[1,i,-(t+2)])
                qC = qpred[nel:]
                
                # Update geo_gamess with qC
                update_geo_gamess(atoms, AN_mat, qC)
                
                # Call GAMESS to compute E, dE/dR, and NAC
                run_gms_cas(input_name, opt, atoms, AN_mat, qC)
                
                # Read GAMESS out file
                elecE, grad, nac, flag_grad, flag_nac = read_gms_out(input_name)
                if any([el == 1 for el in flag_grad]) or flag_nac == 1:
                    proceed = False
                    break
                
                # Read GAMESS dat file
                flag_orb = read_gms_dat(input_name)
                if flag_orb == 1:
                    proceed = False
                    break
                
                # Get derivatives at the predicted coordinates
                der = get_derivatives(au_mas, qpred, ppred, nac, grad, elecE)
                force[:,:,-(t+3)] = der
                
                # Make corrections
                for i in range(ndof):
                    qcorr[i] = compute_ME_corrector(-timestep/200, q[i], force[0,i,-(t+2)], force[0,i,-(t+3)])
                    pcorr[i] = compute_ME_corrector(-timestep/200, p[i], force[1,i,-(t+2)], force[1,i,-(t+3)])
                qC = qcorr[nel:]
                
                # Save the corrected coordinates
                q, p = qcorr, pcorr
                
                # Update geo_gamess with qC
                update_geo_gamess(atoms, AN_mat, qC)
                
                # Call GAMESS to compute E, dE/dR, and NAC
                run_gms_cas(input_name, opt, atoms, AN_mat, qC)
                
                # Read GAMESS out file
                elecE, grad, nac, flag_grad, flag_nac = read_gms_out(input_name)
                if any([el == 1 for el in flag_grad]) or flag_nac == 1:
                    proceed = False
                    break
                
                # Read GAMESS dat file
                flag_orb = read_gms_dat(input_name)
                if flag_orb == 1:
                    proceed = False
                    break
        
        ###################
        ### ABM routine ###
        ###################
        if restart == 0:
            # Reset the mapping variables for the initial condition at t=0
            q, p    = np.zeros(ndof), np.zeros(ndof)          
            q[:nel], p[:nel] = initq[:nel], initp[:nel]
            qN, pN  = initq[nel:], initp[nel:]
            qC, pC  = rotate_norm_to_cart(qN, pN, U, amu_mat)
            q[nel:], p[nel:] = qC, pC
            
            # Total initial energy at t=0
            init_energy = get_energy(au_mas, q, p, elecE)
            with open(os.path.join(__location__, 'energy.out'), 'a') as g:
                g.write(total_format.format(x, init_energy, *elecE))
        
        flag_energy = 0
        old_energy = init_energy
        energy[0] = init_energy
        
        qcorr, pcorr = np.zeros(ndof), np.zeros(ndof)
        
        '''Propagation loop'''
        with open(os.path.join(__location__, 'progress.out'), 'a') as f:
            for t in range(1,nstep+1,1):
                f.write('t index = %d \n' %t)
                
                # Make a prediction of phase space variables using 4 preceding derivatives
                for j in range(ndof):
                    qpred[j] = compute_ABM_predictor(timestep, q[j], force[0,j,-2], force[0,j,-3], force[0,j,-4], force[0,j,-5])
                    ppred[j] = compute_ABM_predictor(timestep, p[j], force[1,j,-2], force[1,j,-3], force[1,j,-4], force[1,j,-5])
                qC = qpred[nel:]
                
                # Update geo_gamess with qC
                update_geo_gamess(atoms, AN_mat, qC)
                
                # Call GAMESS to compute E, dE/dR, and NAC
                run_gms_cas(input_name, opt, atoms, AN_mat, qC)
                
                # Read GAMESS out file
                elecE, grad, nac, flag_grad, flag_nac = read_gms_out(input_name)
                if any([el == 1 for el in flag_grad]) or flag_nac == 1:
                    proceed = False
                    break
                
                # Read GAMESS dat file
                flag_orb = read_gms_dat(input_name)
                if flag_orb == 1:
                    proceed = False
                    break
                
                # Get derivatives at the predicted coordinates
                der = get_derivatives(au_mas, qpred, ppred, nac, grad, elecE)
                force[:,:,-1] = der # update the most recent force
                
                # Compute correctors using the predicted derivatives and 3 preceding derivatives
                for j in range(ndof):
                    qcorr[j] = compute_ABM_corrector(timestep, q[j], force[0,j,-1], force[0,j,-2], force[0,j,-3], force[0,j,-4])
                    pcorr[j] = compute_ABM_corrector(timestep, p[j], force[1,j,-1], force[1,j,-2], force[1,j,-3], force[1,j,-4])
                qC, pC = qcorr[nel:], pcorr[nel:]
                
                # Update geo_gamess with qC
                update_geo_gamess(atoms, AN_mat, qC)
                
                # Call GAMESS to compute E, dE/dR, and NAC
                run_gms_cas(input_name, opt, atoms, AN_mat, qC)
                
                # Read GAMESS out file
                elecE, grad, nac, flag_grad, flag_nac = read_gms_out(input_name)
                if any([el == 1 for el in flag_grad]) or flag_nac == 1:
                    proceed = False
                    break
                
                # Read GAMESS dat file
                flag_orb = read_gms_dat(input_name)
                if flag_orb == 1:
                    proceed = False
                    break
                
                # Compute total energy
                new_energy = get_energy(au_mas, qcorr, pcorr, elecE)
                f.write('Energy = {:<12.6f} \n'.format(new_energy))
                
                # Redefine the newest derivatives using the corrected phase space values
                der = get_derivatives(au_mas, qcorr, pcorr, nac, grad, elecE)
                force[:,:,-1] = der # update the most recent force
                
                # Check energy conservation
                if (init_energy-new_energy)/init_energy > 0.02: # 2% deviation = terrible without doubt
                    flag_energy = 1
                    proceed = False
                    break
                
                # Update energy, derivatives, coordinates, and total time
                old_energy    = new_energy
                energy[t]     = new_energy
                q             = qcorr
                p             = pcorr
                coord[0,:,t]  = qcorr
                coord[1,:,t]  = pcorr
                force[:,:,-5] = force[:,:,-4]
                force[:,:,-4] = force[:,:,-3]
                force[:,:,-3] = force[:,:,-2]
                force[:,:,-2] = force[:,:,-1]
                x            += timestep
                X.append(x)
                
                # Record nuclear geometry in angstrom
                record_nuc_geo(restart, x, atoms, qC, com_ang)
                
                # Record the electronic state energies
                with open(os.path.join(__location__, 'energy.out'), 'a') as g:
                    g.write(total_format.format(x, new_energy, *elecE))

                if t == nstep:
                    terminate = True
                    f.write('Propagated to the final time step.\n')
                    break  

    if any([el == 1  for el in flag_grad]) or flag_nac == 1:
        with open(os.path.join(__location__, 'progress.out'), 'a') as f:
            f.write('CAS gradient failure. \n.')
    elif flag_orb == 1: 
        with open(os.path.join(__location__, 'progress.out'), 'a') as f:
            f.write('CAS orbital not obtained. \n.')
    elif flag_energy == 1: # If E conservation error is raised
        with open(os.path.join(__location__, 'progress.out'), 'a') as f:
            f.write('Energy deviated by more than 2%; Energy conservation failed.\n')  
    elif flag_energy == 0: # If no E conservation error is raised
        if all([el == 0 for el in flag_grad]) and flag_nac == 0:
            ############################
            ### Write a restart file ###
            ############################
            with open(os.path.join(__location__, 'restart.out'), 'w') as gg:
                # Write the coordinates
                gg.write('Coordinates (a.u.) at the last update: \n')
                for i in range(ndof):
                    gg.write('{:>16.10f}{:>16.10f} \n'.format(coord[0,i,t], coord[1,i,t]))
                gg.write('\n')
                
                # Write the forces
                gg.write('Forces (a.u.) at the last 4 time steps: \n')
                gg.write('t = -3 \n')
                for i in range(ndof):
                    gg.write('{:>16.10f}{:>16.10f} \n'.format(force[0,i,-4], force[1,i,-4]))
                gg.write('\n')
                gg.write('t = -2 \n')
                for i in range(ndof):
                    gg.write('{:>16.10f}{:>16.10f} \n'.format(force[0,i,-3], force[1,i,-3]))
                gg.write('\n')
                gg.write('t = -1 \n')
                for i in range(ndof):
                    gg.write('{:>16.10f}{:>16.10f} \n'.format(force[0,i,-2], force[1,i,-2]))
                gg.write('\n')
                gg.write('t = 0 \n')
                for i in range(ndof):
                    gg.write('{:>16.10f}{:>16.10f} \n'.format(force[0,i,-1], force[1,i,-1]))
                gg.write('\n')
                
                # Record the energy and the total time
                gg.write('Energy at the last time step \n')
                gg.write('{:>16.10f} \n'.format(energy[t]))
                gg.write('\n')
                
                # Record the total time
                gg.write('Total time in a.u. \n')
                gg.write('{:>16.10f} \n'.format(X[-1]))
                gg.write('\n')
        
    return(np.array(X), coord, flag_energy, flag_grad, flag_nac, flag_orb, initial_time)



# =============================================================================
#           Modified Midpoint Method for Bulirsch-Stoer integrator
# yStop = integrate(F,x,y,xStop,tol=1.0e-6)
#
# Modified midpoint method for solving the initial value problem y’ = F(x,y}.
#     x,y = initial conditions (y: 2*ndof dimensional vector)
#   xStop = terminal value of x
#   yStop = y(xStop)
#       F = user-supplied function that returns the array F(x,y)={y’[0],y’[1],...,y’[n-1]}.
# =============================================================================
def integrate(F, xvar, yvar, xStop, tol, input_name, atoms, amu_mat, qC):

   def midpoint(F, x, y, xStop, nSteps):
      proceed = True

      ### Midpoint formula ###
      au_mas = np.diag(amu_mat) * amu2au
      h  = (xStop - x)/nSteps
      y0 = y.copy()
      y1 = np.zeros(2*ndof)
      y2 = np.zeros(2*ndof)
      for i in range(2*ndof):
         if i < ndof:
            y1[i] = y0[i] + h*F[0,i] # position
         elif i >= ndof:
            y1[i] = y0[i] + h*F[1,i-ndof] # momentum

      # ES calculation at y1
      qC = y1[nel:ndof]
#      update_geo_gamess(atoms, amu_mat, qC)
      with open(os.path.join(__location__, 'progress.out'), 'a') as aa:
         aa.write('CAS calculation at the beginning of midpoint routine (t=%.4f)\n' %(x+h))
      run_gms_cas(input_name, opt, atoms, amu_mat, qC)
      elecE, grad, nac, flag_grad, flag_nac = read_gms_out(input_name)
      if any([el == 1 for el in flag_grad]) or flag_nac == 1:
         proceed = False
      flag_orb = read_gms_dat(input_name)
      if flag_orb == 1:
         proceed = False

      if proceed:
         # Get derivatives at y1
         F = get_derivatives(au_mas, y1[:ndof], y1[ndof:], nac, grad, elecE)

         # General steps for yn
         for nn in range(nSteps-1):
            if not proceed:
               break
            else:
               x += h
               for i in range(2*ndof):
                  if i < ndof:
                     y2[i] = y0[i] + 2.0*h*F[0,i]
                  elif i >= ndof:
                     y2[i] = y0[i] + 2.0*h*F[1,i-ndof]
                  y0[i] = y1[i]
                  y1[i] = y2[i]
 
               # ES calculation at yn
               qC = y2[nel:ndof]
#               update_geo_gamess(atoms, amu_mat, qC)
               run_gms_cas(input_name, opt, atoms, amu_mat, qC)
               elecE, grad, nac, flag_grad, flag_nac = read_gms_out(input_name)
               if any([el == 1 for el in flag_grad]) or flag_nac == 1:
                  proceed = False
               flag_orb = read_gms_dat(input_name)
               if flag_orb == 1:
                  proceed = False
               if proceed:
                  # Get derivatives at yn
                  F = get_derivatives(au_mas, y2[:ndof], y2[ndof:], nac, grad, elecE)

      if not proceed:
         sys.exit("Electronic structure calculation failed in midpoint algorithm. Exitting.")

      # Compute the coordinates at t=x+H
      result = np.zeros(2*ndof)
      for i in range(2*ndof):
         if i < ndof:
            result[i] = 0.5*(y1[i] + y0[i] + h*F[0,i])
         elif i >= ndof:
            result[i] = 0.5*(y1[i] + y0[i] + h*F[0,i-ndof])

      return(result)
  
   def richardson(r, k):
      ### Richardson's extrapolation ###
      for j in range(k-1,0,-1):
         const = (k/(k-1.0))**(2.0*(k-j))
         r[j]  = (const*r[j+1]-r[j])/(const-1.0)
      return

   ###########################################
   ### Here starts the "integrate" routine ###
   ###########################################
   with open(os.path.join(__location__, 'progress.out'), 'a') as g:
      g.write('Inside the subroutine "integrate"\n')
   kMax = 9 
   n    = 2*ndof
   r    = np.zeros((kMax,n))
   # Start with 2 integration steps
   nSteps = 2
   with open(os.path.join(__location__, 'progress.out'), 'a') as g:
      g.write('\n')
      g.write('Midpoint method with nSteps = %d\n' %(nSteps))
   r[1]   = midpoint(F,xvar,yvar,xStop,nSteps) # Coordinate at t=x+H through 2 steps
   r_old  = r[1].copy()
   # Increase the number of integration points by 2, do midpoint method,
   # and refine the result by Richardson extrapolation
   for k in range(2,kMax):
      nSteps = 2*k
      with open(os.path.join(__location__, 'progress.out'), 'a') as g:
         g.write('\n')
         g.write('Midpoint method with nSteps = %d\n' %(nSteps))
      r[k]   = midpoint(F,xvar,yvar,xStop,nSteps) # Coordinates at t=x+H through 2*k steps
      with open(os.path.join(__location__, 'progress.out'), 'a') as g:
         g.write('Rechardson Extrapolation to refine the midpoint step\n')
      richardson(r,k)
      # Compute RMS change in the solution
      er = np.sqrt(sum((r[1]-r_old)**2)/n)
      with open(os.path.join(__location__, 'progress.out'), 'a') as g:
         g.write('ERROR = %.3e\n' %(er))
      # Check for convergence
      if er < tol: 
         return(r[1])
      r_old = r[1].copy()
      if k == kMax-1:
         sys.exit("nSteps > 16. Consider to reduce H.")

   sys.exit("Midpoint method did not converge.")   


# =============================================================================
# X, Y = BulStoer(x,q,p,xStop,H,tol=10**-6)

# Simplified Bulirsch-Stoer method for solving the initial value problem
# {y}' = {F(x,{y})}, where {y}={y[0],y[1],...,y[n-1]}

# x, y  = initial conditions
# xStop = terminal value of x
# H     = increment of x at which results are stored
# F     = user-supplied function that returns the array F(x,y)={y'[0],y'[1],...,y'[n-1]}
# =============================================================================
def BulStoer(initq, initp, xStop, H, tol, restart, amu_mat, U, com_ang, AN_mat):
   proceed      = True
   input_name   = 'cas'
   au_mas = np.diag(amu_mat) * amu2au # masses of atoms in atomic unit (vector)

   # Format descriptor depending on the number of electronic states
   total_format = '{:>12.4f}{:>12.5f}' # "time" "total"
   for i in range(nel):
      total_format += '{:>12.5f}' # "elec1" "elec2" ...
   total_format += '\n'

   ### Initial-time property calculation ###
   with open(os.path.join(__location__, 'progress.out'), 'a') as f:
      f.write("Initial property evaluation started.\n")
   if restart == 0:
      x       = 0.0
      initial_time = 0.0
      q, p    = np.zeros(ndof), np.zeros(ndof)          # collections of all mapping variables
      q[:nel], p[:nel] = initq[:nel], initp[:nel]
      qN, pN  = initq[nel:], initp[nel:]                # collections of nuclear variables in normal coordinate
      qC, pC  = rotate_norm_to_cart(qN, pN, U, amu_mat) # collections of nuclear variables in Cartesian coordinate
      q[nel:], p[nel:] = qC, pC
      y = np.concatenate((q, p))

      # Get atom labels
      atoms = get_atom_label()

      # Write initial nuclear geometry in the output file
      record_nuc_geo(restart, x, atoms, qC, com_ang)

      # Update geo_gamess with qC
      update_geo_gamess(atoms, AN_mat, qC)
  
      # Call GAMESS to compute E, dE/dR, and NAC
      run_gms_cas(input_name, opt, atoms, AN_mat, qC)
      elecE, grad, nac, flag_grad, flag_nac = read_gms_out(input_name)
      if any([el == 1  for el in flag_grad]) or flag_nac == 1:
         proceed = False
      flag_orb = read_gms_dat(input_name)
      if flag_orb == 1:
         proceed = False

      if not proceed:
         sys.exit("Electronic structure calculation failed at initial time. Exitting.")

      # Total initial energy at t=0
      init_energy = get_energy(au_mas, q, p, elecE)
      with open(os.path.join(__location__, 'energy.out'), 'a') as g:
         g.write(total_format.format(x, init_energy, *elecE))

      # Get derivatives at t=0
      F = get_derivatives(au_mas, q, p, nac, grad, elecE)

   elif restart == 1:
      q, p = np.zeros(ndof), np.zeros(ndof)
      F = np.zeros((2,ndof))
      # Read the restart file
      with open(os.path.join(__location__, 'restart.out'), 'r') as ff:
         ff.readline() 
         for i in range(ndof):
            x = ff.readline().split()
            q[i], p[i] = float(x[0]), float(x[1]) # Mapping variables already in Cartesian coordinate
         [ff.readline() for i in range(2)]

         for i in range(ndof):
            x = ff.readline().split()
            F[0,i], F[1,i] = float(x[0]), float(x[1]) # derivative of each MV
         [ff.readline() for i in range(2)]

         init_energy = float(ff.readline()) # Total energy
         [ff.readline() for i in range(2)]

         initial_time = float(ff.readline()) # Total simulation time at the beginning of restart run
         x = initial_time       
 
      qC, pC = q[nel:], p[nel:]
      y = np.concatenate((q, p))
      
      # Get atom labels
      atoms = get_atom_label()
   
   X,Y = [],[]
   X.append(x)
   Y.append(y)
   flag_energy = 0
   flag_grad   = 0
   flag_nac    = 0
   flag_orb    = 0
   energy      = [init_energy]
   with open(os.path.join(__location__, 'progress.out'), 'a') as f:
      f.write("Initilization done. Move on to propagation routine.\n")
   
   while x < xStop:
      if not proceed:
         sys.exit("Electronic structure calculation failed in Bulirsch-Stoer routine. Exitting.")
      else:
         H  = min(H, xStop-x)
         with open(os.path.join(__location__, 'progress.out'), 'a') as f:
            f.write('\n')
            f.write('Starting modified midpoint + Richardson extrapolation routine.\n')
         y  = integrate(F, x, y, x+H, tol, input_name, atoms, amu_mat, qC) # midpoint method
         x += H
         X.append(x)
         Y.append(y)
         
         # ES calculation at new y
         with open(os.path.join(__location__, 'progress.out'), 'a') as f:
            f.write('\n')
            f.write('Midpoint+Richardson step has been accepted.\n')
         qC = y[nel:ndof]
         update_geo_gamess(atoms, AN_mat, qC)
         run_gms_cas(input_name, opt, atoms, AN_mat, qC)
         elecE, grad, nac, flag_grad, flag_nac = read_gms_out(input_name)
         if any([el == 1 for el in flag_grad]) or flag_nac == 1:
            proceed = False
         flag_orb = read_gms_dat(input_name)
         if flag_orb == 1:
            proceed = False

         if proceed:
            # Get derivatives at new y
            F = get_derivatives(au_mas, y[:ndof], y[ndof:], nac, grad, elecE)
         
            # Compute energy
            new_energy = get_energy(au_mas, y[:ndof], y[ndof:], elecE)
            with open(os.path.join(__location__, 'progress.out'), 'a') as f:
               f.write('Energy = {:<12.6f} \n'.format(new_energy))

            # Check energy conservation
            if (init_energy-new_energy)/init_energy > 0.02: # 2% deviation = terrible without doubt
               flag_energy = 1
               proceed = False
               sys.exit("Energy conservation failed during the propagation. Exitting.")

            # Update energy, derivatives, coordinates, and total time
            old_energy  = new_energy
            energy.append(new_energy)

            # Record nuclear geometry in angstrom
            record_nuc_geo(restart, x, atoms, qC, com_ang)

            # Record the electronic state energies
            with open(os.path.join(__location__, 'energy.out'), 'a') as g:
               g.write(total_format.format(x, new_energy, *elecE))

            if x == xStop:
               with open(os.path.join(__location__, 'progress.out'), 'a') as f:
                  f.write('Propagated to the final time step.\n')

   coord = np.zeros((2,ndof,len(Y)))
   for i in range(len(Y)):
      coord[0,:,i] = Y[i][:ndof]
      coord[1,:,i] = Y[i][ndof:]

   ############################
   ### Write a restart file ###
   ############################
   with open(os.path.join(__location__, 'restart.out'), 'w') as gg:
       # Write the coordinates
       gg.write('Coordinates (a.u.) at the last update: \n')
       for i in range(ndof):
           gg.write('{:>16.10f}{:>16.10f} \n'.format(coord[0,i,-1], coord[1,i,-1]))
       gg.write('\n')

       # Write the deerivatives
       gg.write('Forces (a.u.) at the last update: \n')
       for i in range(ndof):
           gg.write('{:>16.10f}{:>16.10f} \n'.format(F[0,i],F[1,i]))
       gg.write('\n')

       # Record the energy and the total time
       gg.write('Energy at the last time step \n')
       gg.write('{:>16.10f} \n'.format(energy[-1]))
       gg.write('\n')

       # Record the total time
       gg.write('Total time in a.u. \n')
       gg.write('{:>16.10f} \n'.format(x))
       gg.write('\n')

   return(np.array(X), coord, initial_time)


'''
Function to take one integration step by 4th-order Runge-Kutta
'''
def integrate_rk4(elecE, grad, nac, xvar, yvar, xStop, amu_mat):
   au_mas = np.diag(amu_mat) * amu2au
   h  = xStop - xvar
   y0 = yvar.copy()
   y1 = np.zeros(2*ndof)
   y2 = np.zeros(2*ndof)
   y3 = np.zeros(2*ndof)

   ### 4th-order Runge-Kutta routine ###
   # Get derivatives (k1) at y0
   k1 = get_derivatives(au_mas, y0[:ndof], y0[ndof:], nac, grad, elecE)
   k1 = k1.flatten()
   for i in range(2*ndof):
        y1[i] = y0[i] + 0.5*h*k1[i]

   # Get derivatives (k2) at y1
   k2 = get_derivatives(au_mas, y1[:ndof], y1[ndof:], nac, grad, elecE)
   k2 = k2.flatten()
   # Take an intermediate step to y2
   for i in range(2*ndof):
        y2[i] = y0[i] + 0.5*h*k2[i] # position

   # Get derivatives (k3) at y2
   k3 = get_derivatives(au_mas, y2[:ndof], y2[ndof:], nac, grad, elecE)
   k3 = k3.flatten()
   # Take an intermediate step to y3
   for i in range(2*ndof):
        y3[i] = y0[i] + h*k3[i] # position

   # Get derivatives (k4) at y3
   k4 = get_derivatives(au_mas, y3[:ndof], y3[ndof:], nac, grad, elecE)
   k4 = k4.flatten()

   # Compute the coordinates at t=x+H
   result = np.zeros(2*ndof)
   for i in range(2*ndof):
      result[i] = y0[i] + h*(k1[i] + 2*k2[i] + 2*k3[i] + k4[i])/6.0

   return(result)

def scipy_rk4(elecE, grad, nac, yvar, dt, au_mas):
    def get_deriv(t, y0):
        der = get_derivatives(au_mas, y0[:ndof], y0[ndof:], nac, grad, elecE)
        der = der.flatten()
        return(der)
    result = it.solve_ivp(get_deriv, (0,dt), yvar, method='RK45', max_step=dt, t_eval=[dt], rtol=1e-10, atol=1e-10)
    return(result.y.flatten())

'''
Main driver of RK4 and electronic structure 
'''
def rk4(initq, initp, tStop, H, restart, amu_mat, U, com_ang, AN_mat):
    logger = SimulationLogger(nel, dir=logging_dir, save_jobs=tcr_log_jobs)

    if QC_RUNNER == 'terachem':
        from qcRunners.TeraChem import TCRunner, format_output_LSCIVR
        logger.state_labels = [f'S{x}' for x in tcr_state_options['grads']]
    
    trans_dips = None
    job_results = {}
    qc_timings = {}
    proceed      = True
    input_name   = 'cas'
    au_mas = np.diag(amu_mat) * amu2au # masses of atoms in atomic unit (vector)

    # Format descriptor depending on the number of electronic states
    total_format = '{:>12.4f}{:>12.5f}' # "time" "total"
    for i in range(nel):
            total_format += '{:>12.5f}' # "elec1" "elec2" ...
    total_format += '\n'

    #   very first step does not need a GAMESS guess
    opt['guess'] = ''

    # History length for nonadiabatic coupling vector and transition dipole moments
    hist_length = 2 #this is the only implemented length

    ### Initial-time property calculation ###
    with open(os.path.join(__location__, 'progress.out'), 'a') as f:
        f.write("Initial property evaluation started.\n")
    if restart == 0:
        t       = 0.0
        initial_time = 0.0
        q, p    = np.zeros(ndof), np.zeros(ndof)          # collections of all mapping variables
        q[:nel], p[:nel] = initq[:nel], initp[:nel]
        qN, pN  = initq[nel:], initp[nel:]                # collections of nuclear variables in normal coordinate
        qC, pC  = rotate_norm_to_cart(qN, pN, U, amu_mat) # collections of nuclear variables in Cartesian coordinate
        q[nel:], p[nel:] = qC, pC
        y = np.concatenate((q, p))

        # Arrays for history of nonadiabatic coupling vectors (nac) and transition dipole moments (tdm)
        nac_hist = np.zeros((nel,nel,nnuc,hist_length)) # history of nonadiabatic coupling vectors for extrapolation in nac sign flip correction
        tdm_hist = np.zeros((nel,nel,3,hist_length)) # history of transition dipole moments for extrapolation in nac sign flip correction

        # Get atom labels
        atoms = get_atom_label()

        # Write initial nuclear geometry in the output file
        record_nuc_geo(restart, t, atoms, qC, com_ang, logger)

        if QC_RUNNER == 'gamess':
            # Update geo_gamess with qC
            update_geo_gamess(atoms, AN_mat, qC)
        
            # Call GAMESS to compute E, dE/dR, and NAC
            run_gms_cas(input_name, opt, atoms, AN_mat, qC, sub_script)
            elecE, grad, nac, flag_grad, flag_nac = read_gms_out(input_name)
            if any([el == 1  for el in flag_grad]) or flag_nac == 1:
                proceed = False
            flag_orb = read_gms_dat(input_name)
            if flag_orb == 1:
                proceed = False
            if not proceed:
                sys.exit("Electronic structure calculation failed at initial time. Exitting.")
        else:
            tc_runner = TCRunner(tcr_host, tcr_port, atoms, tcr_job_options, server_roots=tcr_server_root, run_options=tcr_state_options, tc_spec_job_opts=tcr_spec_job_opts, tc_initial_job_options=tcr_initial_frame_opts, start_new=False)
            job_results, qc_timings = tc_runner.run_TC_new_geom(qC/ang2bohr)
            # import pickle
            # pickle.dump([job_results, qc_timings], open('_tmp.pkl', 'wb'))
            # job_results, qc_timings = pickle.load( open('_tmp.pkl', 'rb'))
            elecE, grad, nac, trans_dips = format_output_LSCIVR(job_results)


        # Total initial energy at t=0
        init_energy = get_energy(au_mas, q, p, elecE)
        # with open(os.path.join(__location__, 'energy.out'), 'a') as g:
        #     g.write(total_format.format(t, init_energy, *elecE))

        # Create nac history for sign-flip extrapolation
        for it in range(0,hist_length):
            nac_hist[:,:,:,it] = nac
            if trans_dips is not None:
                tdm_hist[:,:,:,it] = trans_dips
   
    elif restart == 1:
        opt['guess'] = 'moread'

        q, p, nac_hist, tdm_hist, init_energy, initial_time = read_restart(file_loc=restart_file_in, ndof=ndof)
        t = initial_time

        ## Read the restart file
        #q, p = np.zeros(ndof), np.zeros(ndof)
        #F = np.zeros((2,ndof))
        #with open(os.path.join(__location__, 'restart.out'), 'r') as ff:
        #        ff.readline() 
        #        for i in range(ndof):
        #            x = ff.readline().split()
        #            q[i], p[i] = float(x[0]), float(x[1]) # Mapping variables already in Cartesian coordinate
        #        [ff.readline() for i in range(2)]

        #        init_energy = float(ff.readline()) # Total energy
        #        [ff.readline() for i in range(2)]

        #        initial_time = float(ff.readline()) # Total simulation time at the beginning of restart run
        #        t = initial_time   
        #        [ff.readline() for i in range(2)]

        #        nac_hist_shape = tuple(map(int, ff.readline().strip().split()))  
        #        flat_nac_hist = [float(num) for num in ff.readline().strip().split()]  
        #        nac_hist = np.array(flat_nac_hist).reshape(nac_hist_shape)  
        #        print("nac_hist",nac_hist)
                
    
        qC, pC = q[nel:], p[nel:]
        y = np.concatenate((q, p))

        # Get atom labels
        atoms = get_atom_label()

        # write_restart('restart_init.json', [y[:ndof], y[ndof:]], init_energy, t, nel, 'rk4')

        if QC_RUNNER == 'gamess':
            # Call GAMESS to compute E, dE/dR, and NAC
            run_gms_cas(input_name, opt, atoms, AN_mat, qC, sub_script)
            elecE, grad, nac, flag_grad, flag_nac = read_gms_out(input_name)
            if any([el == 1  for el in flag_grad]) or flag_nac == 1:
                proceed = False
            flag_orb = read_gms_dat(input_name)
            if flag_orb == 1:
                proceed = False
            if not proceed:
                sys.exit("Electronic structure calculation failed at initial time. Exitting.")
        else:
            tc_runner = TCRunner(tcr_host, tcr_port, atoms, tcr_job_options, server_roots=tcr_server_root, run_options=tcr_state_options, tc_spec_job_opts=tcr_spec_job_opts, tc_initial_job_options=tcr_initial_frame_opts)
            job_results, qc_timings = tc_runner.run_TC_new_geom(qC/ang2bohr)
            # import json
            # json.dump(tc_runner.cleanup_multiple_jobs(job_results), open('tmp.json', 'w'), indent=4)
            elecE, grad, nac, trans_dips  = format_output_LSCIVR(job_results)
        
        # If nac_hist and tdm_hist array does not exist yet, create it as zeros array
        if nac_hist.size == 0:
            nac_hist = np.zeros((nel,nel,nnuc,hist_length))
            # fill array with current nac
            for it in range(0,hist_length):
                nac_hist[:,:,:,it] = nac
        if tdm_hist.size == 0:
            tdm_hist = np.zeros((nel,nel,3,hist_length))
            # fill array with current tdm (if available)
            if trans_dips is not None:
                for it in range(0,hist_length):
                    tdm_hist[:,:,:,it] = trans_dips
        
        nac, nac_hist, tdm_hist = correct_nac_sign(nac,nac_hist,trans_dips,tdm_hist)

    # pops = compute_CF_single(q[0:nel], p[0:nel])
    logger.atoms = atoms
    qc_timings['Wall_Time'] = 0.0
    logger.write(t, init_energy, elecE,  grad, nac, qc_timings, elec_p=p[0:nel], elec_q=q[0:nel], nuc_p=p[nel:], jobs_data=job_results)

    opt['guess'] = 'moread'
    X,Y = [],[]
    X.append(t)
    Y.append(y)
    flag_energy = 0
    flag_grad   = 0
    flag_nac    = 0
    flag_orb    = 0
    energy      = [init_energy]
    with open(os.path.join(__location__, 'progress.out'), 'a') as f:
        f.write("Initilization done. Move on to propagation routine.\n")

    ### Runge-Kutta routine ###
    while t < tStop:
        start_time = time.time()
        if not proceed:
            sys.exit("Electronic structure calculation failed in Runge-Kutta routine. Exitting.")
        else:
            H  = min(H, tStop-t)
            with open(os.path.join(__location__, 'progress.out'), 'a') as f:
                f.write('\n')
                f.write('Starting 4th-order Runge-Kutta routine.\n')
            #y  = integrate_rk4(elecE,grad,nac,t,y,t+H,amu_mat) 
            y  = scipy_rk4(elecE,grad,nac,y,H,au_mas)
            t += H
            X.append(t)
            Y.append(y)
            
            print(f"##### Performing MD Step Time: {t:8.2f} a.u. ##### ")
    
            # ES calculation at new y
            with open(os.path.join(__location__, 'progress.out'), 'a') as f:
                f.write('\n')
                f.write('Runge-Kutta step has been accepted.\n')

            qC = y[nel:ndof]

            if QC_RUNNER == 'gamess':
                update_geo_gamess(atoms, AN_mat, qC)
                run_gms_cas(input_name, opt, atoms, AN_mat, qC, sub_script)
                elecE, grad, nac, flag_grad, flag_nac = read_gms_out(input_name)
                if any([el == 1 for el in flag_grad]) or flag_nac == 1:
                    proceed = False
                flag_orb = read_gms_dat(input_name)
                if flag_orb == 1:
                    proceed = False
            else:
                job_results, qc_timings = tc_runner.run_TC_new_geom(qC/ang2bohr)
                elecE, grad, nac, trans_dips  = format_output_LSCIVR(job_results)
            #correct nac sign
            nac, nac_hist, tdm_hist = correct_nac_sign(nac,nac_hist,trans_dips,tdm_hist)

        if proceed:
            # Compute energy
            new_energy = get_energy(au_mas, y[:ndof], y[ndof:], elecE)
            with open(os.path.join(__location__, 'progress.out'), 'a') as f:
                f.write('Energy = {:<12.6f} \n'.format(new_energy))

            # Check energy conservation
            if (init_energy-new_energy)/init_energy > 0.02: # 2% deviation = terrible without doubt
                flag_energy = 1
                proceed = False
                sys.exit("Energy conservation failed during the propagation. Exitting.")

            # Update & store energy
            old_energy  = new_energy
            energy.append(new_energy)

            # Record nuclear geometry in angstrom
            record_nuc_geo(restart, t, atoms, qC, com_ang, logger)

            # Record the electronic state energies
            # with open(os.path.join(__location__, 'energy.out'), 'a') as g:
            #     g.write(total_format.format(t, new_energy, *elecE))

            #   New logging information
            # pops = compute_CF_single(y[0:nel], y[ndof:ndof+nel])
            end_time = time.time()
            qc_timings['Wall_Time'] = end_time - start_time
            logger.write(t, total_E=new_energy, elec_E=elecE,  grads=grad, NACs=nac, timings=qc_timings, elec_q=y[0:nel], elec_p=y[ndof:ndof+nel], nuc_p=y[-natom*3:], jobs_data=job_results)
            write_restart('restart.json', [Y[-1][:ndof], Y[-1][ndof:]], nac_hist, tdm_hist, new_energy, t, nel, 'rk4')

            if t == tStop:
                with open(os.path.join(__location__, 'progress.out'), 'a') as f:
                    f.write('Propagated to the final time step.\n')

    coord = np.zeros((2,ndof,len(Y)))
    for i in range(len(Y)):
        coord[0,:,i] = Y[i][:ndof]
        coord[1,:,i] = Y[i][ndof:]

    ############################
    ### Write a restart file ###
    ############################
    with open(os.path.join(__location__, 'restart.out'), 'w') as gg:
        # Write the coordinates
        gg.write('Coordinates (a.u.) at the last update: \n')
        for i in range(ndof):
            gg.write('{:>16.10f}{:>16.10f} \n'.format(coord[0,i,-1], coord[1,i,-1]))
        gg.write('\n')

    #       # Write the ES variables at the final step
    #       gg.write('Energies (a.u.) at the last update: \n')
    #       form = ''
    #       for i in range(nel):
    #           form += '{:>16.10f}'
    #        form += '\n'
    #       gg.write(form.format(*elecE))
    #       gg.write('\n')
    #
    #       gg.write('Gradients (a.u.) at the last update: \n')
    #       form = ''
    #       for i in range(nnuc):
    #           form += '{:>16.10f}'
    #       form += '\n'
    #       for i in range(nel):
    #          gg.write(form.format(*grad[i]))
    #       gg.write('\n')

        # Record the energy and the total time
        gg.write('Energy at the last time step \n')
        gg.write('{:>16.10f} \n'.format(energy[-1]))
        gg.write('\n')

        # Record the total time
        gg.write('Total time in a.u. \n')
        gg.write('{:>16.10f} \n'.format(t))
        gg.write('\n')

        if len(nac_hist) > 0:
            gg.write('NAC History:\n')
            gg.write(' '.join(map(str, nac_hist.shape)) + '\n')
            gg.write(np.array2string(nac_hist, separator=',').replace('[', '').replace(']', '') + '\n')

    return(np.array(X), coord, initial_time)


def compute_CF_single(q, p):
   ### Compute the estimator of electronic state population ###
   pop = np.zeros(nel)

   common_TCF = 2**(nel+1) * np.exp(-np.dot(q, q) - np.dot(p, p))
   for i in range(nel):
        final_state_TCF = q[i]**2 + p[i]**2 - 0.5
        pop[i] = common_TCF * final_state_TCF

   return pop


def compute_CF_wigner(X, Y):
   '''
   Function to compute electronic population correlation function after
   trajectory propagation for Wigner population estimator.
   X = time array
   Y = coordinate array
   '''
   ### Compute the estimator of electronic state population ###
   pop = np.zeros((nel, len(X)))
   total_format = '{:>12.4f}'
   for i in range(nel+1):
       total_format += '{:>16.10f}'
   total_format += '\n'
   corr_file = 'corr.out'

   with open(os.path.join(__location__, corr_file), 'a') as f:
       for t in range(len(X)):
           # The common term for all electronic state projection operators
           common_TCF = 2**(nel+1) * np.exp(-np.dot(Y[0,:nel,t], Y[0,:nel,t])\
                                         -np.dot(Y[1,:nel,t], Y[1,:nel,t]))

           # The specific term for each final electronic state projection operator
           for i in range(nel):
               final_state_TCF = Y[0,i,t]**2 + Y[1,i,t]**2 - 0.5
               pop[i,t] += common_TCF * final_state_TCF

           if restart == 0:
               f.write(total_format.format(X[t], sum(pop[:,t]), *pop[:,t]))
           elif restart == 1:
               if t != 0:
                   f.write(total_format.format(X[t], sum(pop[:,t]), *pop[:,t]))
   return()


def compute_CF_sc(X, Y):
    '''
    Function to compute electronic population correlation function after
    trajectory propagation for Semiclassical (SC) population estimator.
    X = time array
    Y = coordinate array
    '''
    ### Compute the estimator of electronic state population ###
    pop = np.zeros((nel, len(X)))
    total_format = '{:>12.4f}'
    for i in range(nel+1):
        total_format += '{:>16.10f}'
    total_format += '\n'
    corr_file = 'corr.out'

    with open(os.path.join(__location__, corr_file), 'a') as f:
        for t in range(len(X)):
            for i in range(nel):
                pop[i,t] += 0.5 * (Y[0,i,t]**2 + Y[1,i,t]**2 - 1)

            if restart == 0:
                f.write(total_format.format(X[t], sum(pop[:,t]), *pop[:,t]))
            elif restart == 1:
                if t != 0:
                    f.write(total_format.format(X[t], sum(pop[:,t]), *pop[:,t]))
    return()


def compute_CF_spin(X, Y):
    '''
    Function to compute electronic population correlation function after
    trajectory propagation for Spin-Mapping (SM) population estimator.
    ### Only the case with 3 electronic states is implemented ###
    X = time array
    Y = coordinate array
    '''
    ### Compute the estimator of electronic state population ###
    pop = np.zeros((nel, len(X)))
    total_format = '{:>12.4f}'
    for i in range(nel+1):
        total_format += '{:>16.10f}'
    total_format += '\n'
    corr_file = 'corr.out'

    with open(os.path.join(__location__, corr_file), 'a') as f:
        for t in range(len(X)):
            R0 = Y[0,0,t]**2 + Y[1,0,t]**2
            R1 = Y[0,1,t]**2 + Y[1,1,t]**2
            R2 = Y[0,2,t]**2 + Y[1,2,t]**2
            pop[0, t] += (1/3) + (2*R0 - R1 - R2) / 6
            pop[1, t] += (1/3) - (R0 - 2*R1 + R2) / 6
            pop[2, t] += (1/3) - (R0 + R1 - 2*R2) / 6

            if restart == 0:
                f.write(total_format.format(X[t], sum(pop[:,t]), *pop[:,t]))
            elif restart == 1:
                if t != 0:
                    f.write(total_format.format(X[t], sum(pop[:,t]), *pop[:,t]))
    return()


def compute_anisotropy_correlation(D0, Dt):
    '''
    Function to compute transition dipole anisotropy after
    trajectory propagation.
    D0 : Transition dipole moment at initial time
    Dt : Transition dipole moment at time t
    '''
    ### Compute the estimator of electronic state population ###
    aniso_file = 'anisotropy.out'

    with open(os.path.join(__location__, aniso_file), 'a') as f:
        # Compute the angle between the initial and the present dipole moment
        cosine = np.dot(D0, Dt)/(np.linalg.norm(D0) * np.linalg.norm(Dt))
        # The common term for all electronic state projection operators
        anisotropy = (3.0 * cosine**2 - 1) / 5
        f.write('{:<12.6f} \n'.format(anisotropy))
    return()
    
'''
Check which sign for the nac is expected and correct artificial sign flips
'''
def correct_nac_sign(nac, nac_hist, tdm, tdm_hist, hist_length=None, debug=False):
    # Predict d(t) from d(t-1),d(t-2) with a linear extrapolation 
    #   The line through the points (-2,k),(-1,l)
    #   is p(x) = (l-k)*x + 2*l - k
    #   Extrapolation to the next time step yields
    #   p(0) = 2*l - k
    # Do that for all NACs and all vector components
    # If available, countercheck if transition dipole moment has also flipped sign
    # One can also do a higher degree polynomial or choose more points, which uses numpy polyfit then.
    # Right now, the degree is hard coded to 1 with 2 history points

    # If there is not history, do not correct artificial sign flips
    #if len(nac_hist) == 0:
    #    return nac, []
    # if tdm is not None:
    #     use_tdm = True
    # else:
    #     use_tdm = False


    if tdm is None:
        use_tdm = False
    elif len(tdm) == 0:
        use_tdm = False
    else:
        use_tdm = True
    if debug: print("TDM HIST: ", tdm_hist, hist_length)


    if hist_length is None:
        hist_length = nac_hist.shape[3]

    polynom_degree = 1 # hardcoded. 1 is usually sufficient. 2 is in principle better but could lead to artificial oscillations

    # Allocate array for extrapolated vector
    nac_expol = np.empty_like(nac)
    if (polynom_degree == 1):
        # default
        # uses only the last 2 points
        nac_expol = 2.0*nac_hist[:,:,:,-1] - 1.0*nac_hist[:,:,:,-2]
    else:
        # for scientific purposes only
        # uses the whole history
        timesteps = np.arange(hist_length)
        for i in range(0, nel):
            for j in range(0, nel):
                for ix in range(0,nac.shape[2]):
                    coefficients = np.polyfit(timesteps,nac_hist[i,j,ix,:], polynom_degree)
                    nac_expol[i,j,ix] = np.polyval(coefficients,hist_length)

    # Do similar with transition dipole moment
    if use_tdm:
        tdm_expol = np.empty_like(tdm)
        if (polynom_degree == 1):
            tdm_expol = 2.0*tdm_hist[:,:,:,-1] - 1.0*tdm_hist[:,:,:,-2]
        else:
            timesteps = np.arange(hist_length)
            for i in range(0, nel):
                for j in range(0, nel):
                    for ix in range(0,3):
                        coefficients = np.polyfit(timesteps,tdm_hist[i,j,ix,:], polynom_degree)
                        tdm_expol[i,j,ix] = np.polyval(coefficients,hist_length)


    # check whether the TC/GAMESS vector goes in the same or opposite direction
    # (means an angle with more than 90 degree) as the estimation
    # if the angle is < 90 degree -> np.sign(dot_product)== 1 -> no flip
    # if the angle is > 90 degree -> np.sign(dot_product)==-1 -> flip
    for i in range(0, nel):
        for j in range(0, nel):
            nac_dot_product = np.dot(nac[i,j,:],nac_expol[i,j,:])
            # if tdm is available: check if it also flips sign. if not, no correction
            # if tdm is not available rely only on nac
            if use_tdm:
                tdm_dot_product = np.dot(tdm[i,j,:],tdm_expol[i,j,:])
                sign_tdm = np.sign(tdm_dot_product)
                sign_nac = np.sign(nac_dot_product)
                if sign_tdm == sign_nac:
                    nac[i,j,:] = sign_nac*nac[i,j,:]
                    tdm[i,j,:] = sign_tdm*tdm[i,j,:]
            else:
                sign = np.sign(nac_dot_product)
                nac[i,j,:] = sign*nac[i,j,:]


    if debug:
        print("nac_hist vor roll: ")
        print("nh[:,:,0]")
        print(nac_hist[:,:,:,0])
        print("nh[:,:,1]")
        print(nac_hist[:,:,:,1])
        print("nh[:,:,2]")
        print(nac_hist[:,:,:,2])

    # roll array and update newest entry
    nac_hist = np.roll(nac_hist,-1,axis=3)
    nac_hist[:,:,:,hist_length-1] = nac
    if use_tdm:
        tdm_hist = np.roll(tdm_hist,-1,axis=3)
        tdm_hist[:,:,:,hist_length-1] = tdm

    if debug:
        print("nac_hist nach roll: ")
        print("nh[:,:,0]")
        print(nac_hist[:,:,:,0])
        print("nh[:,:,1]")
        print(nac_hist[:,:,:,1])
        print("nh[:,:,2]")
        print("nac_dot[0,1] history post roll:",nac_dot_hist[0,1,0],nac_dot_hist[0,1,1],nac_dot_hist[0,1,2])
        
   
    if debug: print("nac_dot[0,1] history: post upda",nac_dot_hist[0,1,0],nac_dot_hist[0,1,1],nac_dot_hist[0,1,2])

    return (nac,nac_hist,tdm_hist)







'''End of file'''
