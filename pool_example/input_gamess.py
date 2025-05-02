#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 2025

@author: KM
"""

"""
Library of GAMESS input variables 
"""

option = {
    # $system card
    'system': {'mwords': '400', 'memddi': '10', 'parall': '.t.'},
    # $contrl card
    'contrl': {'exetyp': 'run', 'runtyp': 'gradient', 'scftyp': 'rhf', 
               'dfttyp': 'b3lyp', 'units': 'angs', 'mult': '1', 'mplevl': '0', 
               'ispher': '1', 'maxit': '200', 'inttyp': 'rysquad', 'nosym': '1', 
               'nprint': '-5', 'tddft': 'excite'},
    # $basis card
    'basis': {'gbasis': 'n31', 'ngauss': '6', 'ndfunc': '1', 'diffsp': '.f.'},
    # $scf card
    'scf': {'npunch': '2', 'conv': '1.0d-06', 'dirscf': '.t.',
            'soscf': '.f.', 'diis': '.t.', 'ethrsh': '10'},
    # $tddft card
    'tddft': {'nstate': '2', 'iroot': '1', 'mult': '1'},
    # $elmom card
    'elmom': {'iemom': '1', 'iemint': '1'},
    # $guess card
    'guess': {'guess': 'moread', 'norb': '34'},
    # $data card
    'data': {'sym': 'c1'},
    }
