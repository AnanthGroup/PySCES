from pysces.qcRunners.DebugRunner import DebugRunner

#   number of atoms in the molecule
natom = 2
nel = 2 # number of electronic states
temp = 300 # simulation temperature in Kelvin

# Maximum propagation time (a.u.), one Runge-Kutta step (a.u.) (Only relevant for RK4)
tmax_rk4, Hrk4 = 165, 5.0

# Index of initially occupied electronic state
init_state = 2

# Restart request: 0 = no restart, 1 = restart
restart = 1
restart_file_in = 'init.json'

#   type of QC runner, either 'gamess or 'terachem'
QC_RUNNER = DebugRunner(['H', 'F'], {})

input_seed = 123456

hdf5_logging = True
 
logging_mode = 'w'
