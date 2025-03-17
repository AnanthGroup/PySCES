#   number of atoms in the molecule
natom = 6
nel = 3 # number of electronic states
temp = 300 # simulation temperature in Kelvin

# Maximum propagation time (a.u.), one Runge-Kutta step (a.u.) (Only relevant for RK4)
tmax_rk4, Hrk4 = 6, 2.0

# Index of initially occupied electronic state
init_state = 2

# Restart request: 0 = no restart, 1 = restart
restart = 0
restart_file_in = 'restart_start.json'

#   type of QC runner, either 'gamess or 'terachem'
QC_RUNNER = 'terachem'

#   TeraChem runner options
try:
     import json
     with open('../host_ports.json') as file:
        data = json.load(file)
        print("DATA: ", data)
        tcr_host = data['hosts']
        tcr_port = data['ports']
        tcr_server_root = data['server_roots']
except:
    tcr_host = ['localhost']
    tcr_port = [1234]
    tcr_server_root = ['/scratch/cmyers7/single_servers/']
tcr_job_options = {
        'method': 'hf',
        'basis': 'sto-3g',
        'charge': 0,
        'spinmult': 1,
        'closed_shell': True,
        'restricted': True,

        #   TD-DFT
        'cis': 'yes',
        'cisnumstates': 2,

        #   thresholds
        'precision': 'double',
        'convthre': 1e-8,
        'cisconvtol': 1e-8,
        'maxiter': 100,
        'cismaxiter': 100,
        'threall': 1e-20,
        'threcl': 1e-20,
        'pqthre': 1e-20,
        'threoe': 1e-20,
        'thregr': 1e-20,
        'threex': 1e-20,


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
