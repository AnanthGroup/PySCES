#   number of atoms in the molecule
natom = 6
nel = 3 # number of electronic states
temp = 300 # simulation temperature in Kelvin

# Maximum propagation time (a.u.), one Runge-Kutta step (a.u.) (Only relevant for RK4)
tmax_rk4, Hrk4 = 8, 2.0

# Index of initially occupied electronic state
init_state = 2

# Restart request: 0 = no restart, 1 = restart
restart = 1
restart_file_in = 'restart_start.json'

#   type of QC runner, either 'gamess or 'terachem'
QC_RUNNER = 'terachem'

#   TeraChem runner options
import os
if os.path.isfile('../host_ports.txt'):
    from numpy import loadtxt
    host_ports = loadtxt('../host_ports.txt', dtype=str)
    tcr_host = host_ports[:, 0]
    tcr_port = host_ports[:, 1].astype(int)
    tcr_server_root = ['.']*len(tcr_host)
else:
    tcr_host = ['localhost', 'localhost']
    tcr_port = [1234, 1235]
    tcr_server_root = ['', '']
tcr_job_options = {
        'method': 'hf',
        'basis': 'sto-3g',
        'charge': 0,
        'spinmult': 1,
        'closed_shell': True,
        'restricted': True,
        'precision': 'mixed',
        'convthre': 1E-6,
        'sphericalbasis': 'yes',

        #   TD-DFT
        'cis': 'yes',
        'cisnumstates': 2
}
tcr_state_options = {
    'max_state': 2
}

# Terachem files
fname_tc_xyz      = "freq/mol.xyz"
fname_tc_geo_freq = "freq/Geometry.frequencies.dat"
fname_tc_redmas   = "freq/Reduced.mass.dat"
fname_tc_freq     = "freq/Frequencies.dat"

input_seed = 123456
