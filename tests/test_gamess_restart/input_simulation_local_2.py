nel, natom = 3, 6 # number of electronic states, number of atoms in the molecule
nnuc = 3*natom # number of nuclear DOFs
ndof = nel + nnuc 
temp = 300 # simulation temperature in Kelvin

# Initial sampling function ('conventioanl', 'modified', or 'spin' LSC-IVR)
sampling = 'wigner' 

# Centers of initial electronic coherent states (positions q0 & momenta p0)
# From left, electronic state 1, 2, 3, ...
q0 = [0.0, 0.0, 0.0] 
p0 = [0.0, 0.0, 0.0]

# Center of initial momentum of nuclear modes (same value for all nuc DOFs)
# NOTE: the centers of initial position are determined by normal coordinates
pN0 = 0.0

# ELectronic coherent state width parameters
# From left, electronic state 1, 2, 3, ...
width = [1.0, 1.0, 1.0]

# Specify an integrator (Choose from 'ABM', 'BSH', and 'RK4')
integrator = 'RK4'
# Size of time step (a.u.), number of steps (Only relevant for ABM)
timestep, nstep = 1.0, 16700
# Maximum propagation time (a.u.), BSH step to be tried (a.u.), error tolerance (ratio) (Only relevant for BSH)
tmax_bsh, Hbsh, tol = 10, 3.0, 0.01 
# Maximum propagation time (a.u.), one Runge-Kutta step (a.u.) (Only relevant for RK4)
tmax_rk4, Hrk4 = 50, 10.0

# Scaling factor of normal mode frequancies
frq_scale = 0.967

# Number of CPUs and nodes in each internal NACME calculation
ncpu, nnode = 1, 1

# Partition of Pool cluster (astra or common)
partition = 'astra'

# Indices of electronic states for which NACME to be read from GAMESS output file
elab = [1, 2, 3]

# Index of initially occupied electronic state
init_state = 2

# Restart request: 0 = no restart, 1 = restart
restart = 1
restart_file_in = 'restart.json'

#   type of QC runner, either 'gamess or 'terachem'
QC_RUNNER = 'gamess'
sub_script = 'run_gamess.sh'

input_seed = 12345
