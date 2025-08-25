import numpy as np
import os
import sys

from pysces.fileIO import SimulationLogger, write_restart, read_restart
from pysces.interpolation import SignFlipper
from pysces.common import PhaseVars, ESVarsHistory, ESVars
from pysces import timers
from pysces import input_simulation as opts
from pysces.subroutines import (
    get_atom_label,
    rotate_norm_to_cart,
    record_nuc_geo,
    run_gamess_at_geom,
    get_energy,
    integrate_rk4_main,
    verlet_Uprop_step_first_half,
    verlet_Uprop_step_second_half,
    print_energy_summary,
    __location__,
    amu2au,
)
from pysces.input_gamess import nacme_option as opt 


def rk4(initq, initp, tStop, H, restart, amu_mat, U, AN_mat):

    #   Initialize variables
    qc_runner = opts.qc_runner
    nel = opts.nel
    natom = opts.natom
    nnuc = 3 * natom
    ndof = nnuc + nel

    tc_runner       = None
    input_name      = 'cas'
    au_mas          = np.diag(amu_mat) * amu2au # masses of atoms in atomic unit (vector)
    t               = 0.0
    initial_time    = 0.0
    X,Y             = [],[]
    q, p            = np.zeros(ndof), np.zeros(ndof)  # collections of all mapping variables
    es_vars         = ESVars()  # electronic structure variables
    es_history      = ESVarsHistory()
    sign_flipper    = SignFlipper(nel, 2, nnuc, 'LSC')
    logger          = SimulationLogger(dir=opts.logging_dir, hdf5=opts.hdf5_logging)

    #   very first step does not need a GAMESS guess
    opt['guess'] = ''

    # Get atom labels
    atoms = get_atom_label()
    logger.atoms = atoms

    if qc_runner == 'terachem':
        from pysces.qcRunners.TeraChem import TCRunner
        logger.state_labels = [f'S{x}' for x in opts.tcr_state_options['grads']]
        tc_runner = TCRunner(atoms, opts.tc_runner_opts)
        tc_runner._prev_ref_job = opts.tcr_ref_job
        if opts.tcr_log_jobs:
            tc_runner.set_logger_file(logger._h5_file)
    elif qc_runner != 'gamess':
        qc_runner.set_logger_file(logger._h5_file)

    #   Initialization
    if restart == 1:
        q, p, nac_hist, tdm_hist, init_energy, initial_time, es_vars = read_restart(opts.restart_file_in, tc_runner=tc_runner, qc_runner=qc_runner)
        t = initial_time
        qC, pC = q[nel:], p[nel:]
        y = np.concatenate((q, p))

    elif restart == 0:
        q[:nel], p[:nel] = initq[:nel], initp[:nel]
        qN, pN  = initq[nel:], initp[nel:]                # collections of nuclear variables in normal coordinate
        qC, pC  = rotate_norm_to_cart(qN, pN, U, amu_mat) # collections of nuclear variables in Cartesian coordinate
        q[nel:], p[nel:] = qC, pC
        y = np.concatenate((q, p))
        nac_hist, tdm_hist = np.empty(0), np.empty(0)

    timers.traj_timer.update(t, tStop)

    #   Run first electronic structure calculation if we are not restarting,
    #   or if we are restarting and the electronic structure information is missing
    if restart == 0 or not es_vars.complete:
        if qc_runner == 'gamess':
            es_vars = run_gamess_at_geom(input_name, AN_mat, qC, atoms)
        elif qc_runner == 'terachem':
            es_vars = tc_runner.run_new_geom(PhaseVars(time=t, nuc_q0=qC))
        else:
            es_vars = qc_runner.run_new_geom(PhaseVars(time=t, nuc_q0=qC))

        # Total initial energy at t=0
        init_energy = get_energy(au_mas, q, p, es_vars.elecE)

        # Record nuclear geometry in angstrom and log the rest
        record_nuc_geo(restart, t, atoms, qC, logger)
        logger.write(t, init_energy, es_vars, y)


    # Create nac history for sign-flip extrapolation
    sign_flipper.set_history(es_vars.nacs, nac_hist, es_vars.trans_dips, tdm_hist)
    
    #   the first frame of the trajectory has a restart file written
    if t == 0.0:
        write_restart(y, sign_flipper, init_energy, t, es_vars, tc_runner, qc_runner, file_loc='initial_restart.json')

    es_vars.time = t
    opt['guess'] = 'moread'
    X.append(t)
    Y.append(y)
    energy      = [init_energy]
    with open(os.path.join(__location__, 'progress.out'), 'a') as f:
        f.write("Initilization done. Move on to propagation routine.\n")

    ### Runge-Kutta routine ###
    while t < tStop:
        print(f"##### Performing MD Step Time: {t:8.2f} a.u. ##### ")

        H  = min(H, tStop-t)
        with open(os.path.join(__location__, 'progress.out'), 'a') as f:
            f.write('\n')
            f.write('Starting 4th-order Runge-Kutta routine.\n')

        es_history.append(es_vars)
        y = integrate_rk4_main(y, H, au_mas, t, es_history, es_vars)
        t += H
        X.append(t)
        Y.append(y)
        timers.traj_timer.update(t, tStop)
        
        # ES calculation at new y
        with open(os.path.join(__location__, 'progress.out'), 'a') as f:
            f.write('\n')
            f.write('Runge-Kutta step has been accepted.\n')

        qC = y[nel:ndof]

        if qc_runner == 'gamess':
            es_vars = run_gamess_at_geom(input_name, AN_mat, qC, atoms)
        elif qc_runner == 'terachem':
            es_vars = tc_runner.run_new_geom(PhaseVars(time=t, nuc_q0=qC))
        else:
            es_vars = qc_runner.run_new_geom(PhaseVars(time=t, nuc_q0=qC))
        es_vars.time = t

        #correct nac sign
        es_vars.nacs = sign_flipper.correct_nac_sign(es_vars.nacs, es_vars.trans_dips)

        # Compute energy
        new_energy = get_energy(au_mas, y[:ndof], y[ndof:], es_vars.elecE)
        with open(os.path.join(__location__, 'progress.out'), 'a') as f:
            f.write('Energy = {:<12.6f} \n'.format(new_energy))

        # Check energy conservation
        if (init_energy-new_energy)/init_energy > 0.02: # 2% deviation = terrible without doubt
            sys.exit("Energy conservation failed during the propagation. Exitting.")

        # Update & store energy
        energy.append(new_energy)

        # Record nuclear geometry, logs, and restarts
        record_nuc_geo(restart, t, atoms, qC, logger)
        logger.write(t, new_energy, es_vars, y)
        write_restart(y, sign_flipper, new_energy, t, es_vars, tc_runner, qc_runner)

        if t == tStop:
            with open(os.path.join(__location__, 'progress.out'), 'a') as f:
                f.write('Propagated to the final time step.\n')
                
    write_restart(y, sign_flipper, energy[-1], t, es_vars, tc_runner, qc_runner)
    if qc_runner == 'terachem':
        tc_runner.cleanup()

    coord = np.zeros((2,ndof,len(Y)))
    for i in range(len(Y)):
        coord[0,:,i] = Y[i][:ndof]
        coord[1,:,i] = Y[i][ndof:]

    return (np.array(X), coord, initial_time)

def verlet_main(initq, initp, tStop, H, restart, amu_mat, U, AN_mat):

    #   Initialize variables
    qc_runner = opts.qc_runner
    nel = opts.nel
    natom = opts.natom
    nnuc = 3 * natom
    ndof = nnuc + nel

    tc_runner       = None
    input_name      = 'cas'
    au_mas          = np.diag(amu_mat) * amu2au # masses of atoms in atomic unit (vector)
    t               = 0.0
    initial_time    = 0.0
    X,Y             = [],[]
    q, p            = np.zeros(ndof), np.zeros(ndof)  # collections of all mapping variables
    es_vars         = ESVars()  # electronic structure variables
    es_history      = ESVarsHistory()
    sign_flipper    = SignFlipper(nel, 2, nnuc, 'LSC')
    logger          = SimulationLogger(dir=opts.logging_dir, hdf5=opts.hdf5_logging)
    from pprint import pprint

    #   very first step does not need a GAMESS guess
    opt['guess'] = ''

    # Get atom labels
    atoms = get_atom_label()
    logger.atoms = atoms

    if qc_runner == 'terachem':
        from pysces.qcRunners.TeraChem import TCRunner
        logger.state_labels = [f'S{x}' for x in opts.tcr_state_options['grads']]
        tc_runner = TCRunner(atoms, opts.tc_runner_opts)
        tc_runner._prev_ref_job = opts.tcr_ref_job
        if opts.tcr_log_jobs:
            tc_runner.set_logger_file(logger._h5_file)
    elif qc_runner != 'gamess':
        qc_runner.set_logger_file(logger._h5_file)

    #   Initialization
    if restart == 1:
        q, p, nac_hist, tdm_hist, init_energy, initial_time, es_vars = read_restart(opts.restart_file_in, tc_runner=tc_runner, qc_runner=qc_runner, integrator='verlet-uprop')
        t = initial_time
        qC, pC = q[nel:], p[nel:]
        y = np.concatenate((q, p))

    elif restart == 0:
        q[:nel], p[:nel] = initq[:nel], initp[:nel]
        qN, pN  = initq[nel:], initp[nel:]                # collections of nuclear variables in normal coordinate
        qC, pC  = rotate_norm_to_cart(qN, pN, U, amu_mat) # collections of nuclear variables in Cartesian coordinate
        q[nel:], p[nel:] = qC, pC
        y = np.concatenate((q, p))
        nac_hist, tdm_hist = np.empty(0), np.empty(0)

    timers.traj_timer.update(t, tStop)

    #   Run first electronic structure calculation if we are not restarting,
    #   or if we are restarting and the electronic structure information is missing
    if restart == 0 or not es_vars.complete:
        if qc_runner == 'gamess':
            es_vars = run_gamess_at_geom(input_name, AN_mat, qC, atoms)
        elif qc_runner == 'terachem':
            es_vars = tc_runner.run_new_geom(PhaseVars(time=t, nuc_q0=qC))
        else:
            es_vars = qc_runner.run_new_geom(PhaseVars(time=t, nuc_q0=qC))
        # Total initial energy at t=0
        init_energy = print_energy_summary(au_mas, q, p, es_vars.elecE)

        # Record nuclear geometry in angstrom and log the rest
        record_nuc_geo(restart, t, atoms, qC, logger)
        logger.write(t, init_energy, es_vars, y)

    # Create nac history for sign-flip extrapolation
    sign_flipper.set_history(es_vars.nacs, nac_hist, es_vars.trans_dips, tdm_hist)
    
    #   the first frame of the trajectory has a restart file written
    if t == 0.0:
        write_restart(y, sign_flipper, init_energy, t, es_vars, tc_runner, qc_runner, file_loc='initial_restart.json')

    es_vars.time = t
    opt['guess'] = 'moread'
    X.append(t)
    Y.append(y)

    print('In Verlet Main')
    while t < tStop:
        print(f"##### Performing MD Step Time: {t:8.2f} a.u. ##### ")

        H  = min(H, tStop-t)
        es_history.append(es_vars)

        #   First half of Verlet step
        #   Everything is a full step update, except the nuclear momentum (only a half step)
        y_half = verlet_Uprop_step_first_half(es_vars.elecE, es_vars.grads, es_vars.nacs, y, H, au_mas)
        t += H

        # ES calculation
        qC = y_half[nel:ndof]
        if qc_runner == 'gamess':
            es_vars = run_gamess_at_geom(input_name, AN_mat, qC, atoms)
        elif qc_runner == 'terachem':
            es_vars = tc_runner.run_new_geom(PhaseVars(time=t, nuc_q0=qC))
        else:
            es_vars = qc_runner.run_new_geom(PhaseVars(time=t, nuc_q0=qC))
        es_vars.time = t
        timers.traj_timer.update(t, tStop)

        #correct nac sign
        es_vars.nacs = sign_flipper.correct_nac_sign(es_vars.nacs, es_vars.trans_dips)

        #   Second half of Verlet step
        #   Nuclear momentum is updated at the second half of the Verlet step
        y = verlet_Uprop_step_second_half(es_vars.elecE, es_vars.grads, es_vars.nacs, y_half, H, au_mas)
        X.append(t)
        Y.append(y)

        # Compute energy
        new_energy = print_energy_summary(au_mas, y[:ndof], y[ndof:], es_vars.elecE)

        # Record nuclear geometry, logs, and restarts
        record_nuc_geo(restart, t, atoms, qC, logger)
        logger.write(t, new_energy, es_vars, y)
        write_restart(y, sign_flipper, new_energy, t, es_vars, tc_runner, qc_runner)

                
    write_restart(y, sign_flipper, new_energy, t, es_vars, tc_runner, qc_runner)
    if qc_runner == 'terachem':
        tc_runner.cleanup()

    coord = np.zeros((2,ndof,len(Y)))
    for i in range(len(Y)):
        coord[0,:,i] = Y[i][:ndof]
        coord[1,:,i] = Y[i][ndof:]

    return (np.array(X), coord, initial_time)
