#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  3 19:28:55 2023

@author: user
"""

"""
Library of GAMESS input variables for NACME calculation
"""

nacme_option = {
    # $system card
    'mwords': 400, 
    'memddi': 10, 
    'parall': '.t.',
    # $contrl card
    'exetyp': 'run', 
    'runtyp': 'nacme', 
    'scftyp': 'mcscf', 
    'dfttyp': 'none',
    'units': 'angs',
    'mult': 1, 
    'mplevl': 0, 
    'ispher': 1, 
    'maxit': 200, 
    'inttyp': 'rysquad',
    'nosym': 1, 
    'nprint': 9,
    # $basis card
    'gbasis': 'n31',
    'ngauss': 6,
    'ndfunc': 1,
    'diffsp': '.f.',
    # $scf card
    'conv': '1.0d-06',
    'dirscf': '.t.',
    'soscf': '.f.',
    'diis': '.t.',
    'ethrsh': 10,
    # $mcscf card
    'cistep': 'aldet',
    'diabat': '.f.',
    'soscf': '.f.',
    'fullnr': '.t.',
    'fors': '.t.',
    'finci': 'mos',
    'acurcy': '1.0d-6',
    'maxit': 200,
    # $det 
    'ncore': 7,
    'nstate': 4,
    'pures': '.t.',
    'nact': 2,
    'nels': 2,
    'iroot': 1,
    'wstate': '1,1,1',
    'itermx': 100,
    # $cpmchf card
    'gcro': '.t.',
    'micit': 50,
    'kicit': 100,
    'prcchg': '.t.',
    'prctol': 1.0,
    'napick': '.t.',
    'nacst': '2,3, 1,3, 1,2',
    # $guess card
    'guess': 'moread',
    'norb': 34,
    # $data card
    'sym': 'c1',
    }
