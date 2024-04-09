#!/usr/bin/env python
# Basic energy calculation
import os
import numpy as np
from tcpb import TCProtobufClient as TCPBClient
from tcpb.exceptions import ServerError
from tcpb import terachem_server_pb2 as pb
import time
import warnings
import shutil
import socket
import subprocess
import time
import psutil
import concurrent.futures


_server_processes = {}

class TCServerStallError(Exception):
    def __init__(self, message):            
        # Call the base class constructor with the parameters it needs
        super().__init__(message)

def _val_or_iter(x):
    try:
        iter(x)
    except:
        x = [x]
    return x

def _convert(value):
    if isinstance(value, list):
        new = []
        for i in range(len(value)):
            new.append(_convert(value[i]))
        return new
    elif isinstance(value, bytes):
        return value.decode('utf-8')
    elif isinstance(value, np.ndarray):
        # new = []
        # for i in range(len(value)):
        #     new.append(_convert(value[i]))
        new = value.tolist()
        return new
    elif isinstance(value, dict):
        new = {}
        for key, v in value.items():
            new[key] = _convert(v)
        return new
    else:
        return value

def start_TC_server(port: int):
    '''
        Start a new TeraChem server. 
        
        Parameters
        ----------
        port: int, port number for the server to use

        Returns
        -------
        host: string, the host of the server being run
    '''
    host = socket.gethostbyname(socket.gethostname())

    #   make sure terachem executable is found
    tc_bin = shutil.which('terachem')
    if tc_bin is None:
        raise RuntimeError("executable 'terachem' not fond")
    
    #   make sure the server isn't already running with this host:port combo
    try:
        tmp_client = TCPBClient(host, port)
        tmp_client.connect()
        if tmp_client.is_available():
            raise Exception(f'Python does not controll TC Server at {host}:{port}')
    except ServerError:
        #   A server error indicates that it is NOT running.
        #   This is GOOD, now we can continue to start our own process.
        pass
    
    #   start TC process
    command = f'terachem -s {port}'
    process = subprocess.Popen(command.split(), shell=False)
    time.sleep(10)

    #   set up a temporary client to make sure it is open
    try:
        tmp_client = TCPBClient(host, port)
        TCRunner.wait_until_available(tmp_client)
        _server_processes[(host, port)] = process
        tmp_client.disconnect()
    except TimeoutError as e:
        raise RuntimeError(f'Could not start TeraChem server on {host} with port {port}')
    
    return host

def stop_TC_server(host: str, port: int):
    '''
        Stop the TC Server. Python must have started the server in the first place.
    '''
    key = (host, port)
    if key not  in _server_processes:
        raise ValueError(f'Host:port {host}:{port} not found in current process list. Python must own the server process.')
    
    parent_pid = _server_processes[(host, port)].pid
    parent = psutil.Process(parent_pid)
    for child in parent.children(recursive=True):  # or parent.children() for recursive=False
        child.kill()
    parent.kill()

def _start_TC_server(port: int):
    ip = socket.gethostbyname(socket.gethostname())

    #   make sure terachem executable is found
    tc_bin = shutil.which('terachem')
    if tc_bin is None:
        raise RuntimeError("executable 'terachem' not fond")
    
    try:
        command = f'terachem -s {port}'
        process = subprocess.Popen(command.split())
        # process = subprocess.run(command.split())
        _server_processes[port] = process
    except:
        return None
    
    return ip

def compute_job_sync(client: TCPBClient, jobType="energy", geom=None, unitType="bohr", **kwargs):
    """Wrapper for send_job_async() and recv_job_async(), using check_job_complete() to poll the server.
    Main funcitonality is coppied from TCProtobufClient.send_job_async() and
    TCProtobufClient.check_job_complete(). This is mostly to change the time in which the server is pinged. 
    
    Args:
        jobType:    Job type key, as defined in the pb.JobInput.RunType enum (defaults to 'energy')
        geom:       Cartesian geometry of the new point
        unitType:   Unit type key, as defined in the pb.Mol.UnitType enum (defaults to 'bohr')
        **kwargs:   Additional TeraChem keywords, check _process_kwargs for behaviour

    Returns:
        dict: Results mirroring recv_job_async
    """

    print("Submitting Job...")
    accepted = client.send_job_async(jobType, geom, unitType, **kwargs)
    while accepted is False:
        time.sleep(0.1)
        accepted = client.send_job_async(jobType, geom, unitType, **kwargs)

    completed = client.check_job_complete()
    while completed is False:
        time.sleep(0.1)
        client._send_msg(pb.STATUS, None)
        status = client._recv_msg(pb.STATUS)

        if status.WhichOneof("job_status") == "completed":
            completed = True
        elif status.WhichOneof("job_status") == "working":
            completed = False
        else:
            raise ServerError(
                "Invalid or no job status received, either no job submitted before check_job_complete() or major server issue",
                client,
            )
    return client.recv_job_async()



class TCRunner():
    def __init__(self, 
                 hosts: str, 
                 ports: int,
                 atoms: list,
                 tc_options: dict,
                 tc_spec_job_opts: dict = None,
                 server_roots = '.',
                 start_new:  bool=False,
                 run_options: dict={}, 
                 max_wait=20,
                 ) -> None:

        if isinstance(hosts, str):
            hosts = [hosts]
        if isinstance(ports, int):
            ports = [ports]
        if isinstance(server_roots, str):
            server_roots = [server_roots]

        #   set up the servers
        if start_new:
            hosts = start_TC_server(ports)

        self._host_list = hosts
        self._port_list = ports
        self._server_root_list = server_roots
        self._client_list = []
        for h, p in zip(hosts, ports):
            client = TCPBClient(host=h, port=p)
            self.wait_until_available(client, max_wait=max_wait)
            self._client_list.append(client)

        self._client = self._client_list[0]
        self._host = hosts[0]
        self._port = ports[0]

        self._atoms = np.copy(atoms)
        self._base_options = tc_options.copy()

        self._cas_guess = None
        self._scf_guess = None
        self._ci_guess = None

        #   timings and server stalling
        self._max_time_list = []
        self._max_time = None
        self._restart_on_stall = True
        self._n_calls = 0

        #   state options
        if run_options.get('grads', False):
            self._grads = run_options['grads']
            self._max_state = max(self._grads)
        elif run_options.get('max_state', False):
            self._max_state = run_options['max_state']
            self._grads = list(range(self._max_state + 1))
        else:
            raise ValueError('either "max_state" or a list of "grads" must be specified')
        # self._max_state = run_options.get('max_state', 0)
        # self._grads = run_options.get('grads', False)
        self._NACs = run_options.pop('nacs', False)

        #   job options for specific jobs
        self._spec_job_opts = tc_spec_job_opts
        

    @staticmethod
    def wait_until_available(client: TCPBClient, max_wait=10.0, time_btw_check=1.0):
        total_wait = 0.0
        avail = False
        while not avail:
            try:
                client.connect()
                client.is_available()
                avail = True
            except:
                print(f'TeraChem server not available: \
                        trying again in {time_btw_check} seconds')
                time.sleep(time_btw_check)
                total_wait += time_btw_check
                if total_wait >= max_wait:
                    raise TimeoutError('Maximum time allotted for checking for TeraChem server')
        print('Terachem Server is available')
        print(avail)

    @staticmethod
    def append_results_file(results: dict):
        results_file = os.path.join(results['job_scr_dir'], 'results.dat')
        results['results.dat'] = open(results_file).readlines()
        return results
    
    @staticmethod
    def append_output_file(results: dict):
        output_file = os.path.join(results['job_dir'], 'tc.out')
        # results['tc.out'] = open(results_file).readlines()
        lines = open(output_file).readlines()
        for n in range(len(lines)):
            lines[n] = lines[n][0:-1]
        results['tc.out'] = lines
        return results
    
    @staticmethod
    def remove_previous_job_dir(client: TCPBClient):
        results = client.prev_results
        TCRunner.remove_previous_scr_dir(client)
        job_dir = results['job_dir']
        shutil.rmtree(job_dir)

    @staticmethod
    def remove_previous_scr_dir(client: TCPBClient):
        results = client.prev_results
        scr_dir = results['job_scr_dir']
        shutil.rmtree(scr_dir)

    @staticmethod
    def cleanup_multiple_jobs(results: list, *remove: str):
        new_results = []
        for res in results:
            new_results.append(TCRunner.cleanup_job(res, *remove))
        return new_results

    @staticmethod
    def cleanup_job(results: dict, *remove: str):
        '''
            Removes unwanted entries in the TC output dictionary and converts
            all numpy arrays to lists
        '''
        
        if remove:
            if len(remove) == 1 and remove[0] == '':
                remove = []
        else:
            remove = ['orb_energies', 'bond_order', 'orb_occupations', 'spins']

        #   remove unwanted entries
        for r in remove:
            if r in results:
                results.pop(r)

        #   convert all numpy arrays to lists
        cleaned = {}
        for key in results:
            cleaned[key] = _convert(results[key])
            # if isinstance(results[key], np.ndarray):
            #     results[key] = results[key].tolist()
            # if isinstance(results[key], list) and len(results[key]):
            #     if isinstance(results[key][0], np.ndarray):
            #         results[key] = np.array(results[key]).tolist()

        return cleaned

    @staticmethod
    def run_TC_single(client: TCPBClient, geom, atoms: list[str], opts: dict):
        opts['atoms'] = atoms
        start = time.time()
        results = client.compute_job_sync('energy', geom, 'angstrom', **opts)
        end = time.time()
        print(f"Job completed in {end - start: .2f} seconds")

        return results
    
    def compute_job_sync_with_restart(self, jobType="energy", geom=None, unitType="bohr", **kwargs):
        """Wrapper for send_job_async() and recv_job_async(), using check_job_complete() to poll the server.
           Main funcitonality is coppied from TCProtobufClient.send_job_async()

        Parameters
        ----------
            jobType:    Job type key, as defined in the pb.JobInput.RunType enum (defaults to 'energy')
            geom:       Cartesian geometry of the new point
            unitType:   Unit type key, as defined in the pb.Mol.UnitType enum (defaults to 'bohr')
            **kwargs:   Additional TeraChem keywords, check _process_kwargs for behaviour

        Returns
        -------
            dict: Results mirroring recv_job_async
        """

        max_time = self._max_time
        if max_time is None and self._restart_on_stall:
            return self._client.compute_job_sync(jobType, geom, unitType, **kwargs)
        else:
            total_time = 0.0
            accepted = self._client.send_job_async(jobType, geom, unitType, **kwargs)
            while accepted is False:
                start_time = time.time()
                time.sleep(0.5)
                accepted = self._client.send_job_async(jobType, geom, unitType, **kwargs)
                end_time = time.time()
                total_time += (end_time - start_time)
                if total_time > max_time and max_time >= 0.0:
                    print("FAILING: ", total_time, max_time)
                    raise TCServerStallError('TeraChem server might have stalled')

            completed = self._client.check_job_complete()
            while completed is False:
                start_time = time.time()
                time.sleep(0.5)
                completed = self._client.check_job_complete()
                
                # if self._n_calls == 2:
                #     max_time = 12
                #     print("STALLING: ", total_time, max_time)
                #     completed = False
                # else:
                #     completed = self._client.check_job_complete()

                end_time = time.time()
                total_time += (end_time - start_time)
                if total_time > max_time and max_time >= 0.0:
                    print("FAILING: ", total_time, max_time)
                    raise TCServerStallError('TeraChem server might have stalled')

            return self._client.recv_job_async()

    def run_TC_new_geom(self, geom):

        try:
            result = self._run_TC_new_geom_kernel(geom)
        except TCServerStallError as error:
            host, port = self._host, self._port
            print('TC Server stalled: attempting to restart server')
            stop_TC_server(host, port)
            time.sleep(2.0)
            start_TC_server(port)
            self._client = TCPBClient(host=host, port=port)
            self.wait_until_available(self._client, max_wait=20)
            print('Started new TC Server: re-running current step')
            result = self.run_TC_new_geom(geom)
                
        return result
    
    def set_avg_max_times(self, times: dict):
        max_time = np.max(list(times.values()))
        self._max_time_list.append(max_time)
        self._max_time = np.mean(self._max_time_list)*5
    

    def _run_TC_new_geom_kernel(self, geom):

        self._n_calls += 1
        client = self._client
        atoms = self._atoms
        opts = self._base_options.copy()
        max_state = self._max_state
        gradients = self._grads
        couplings = self._NACs

        self._job_counter = 0
    
        times = {}

        #   convert all keys to lowercase
        orig_opts = {}
        for key, val in opts.items():
            orig_opts[key.lower()] = val

        #   Determine the type of excited state calculations to run, if any.
        #   Also remove any non-excited state options for the 'base_options'
        excited_options = {}
        base_options = {}
        excited_type = None
        cis_possible_opts = ['cis', 'cisrestart', 'cisnumstates', 'cistarget']
        cas_possible_opts = ['casscf', 'casci', 'closed', 'active', 'cassinglets', 'castarget', 'castargetmult', 'fon']

        if orig_opts.get('cis', '') == 'yes':
            #   CI and TDDFT
            excited_type = 'cis'
            for key, val in orig_opts.items():
                if key in cis_possible_opts:
                    excited_options[key] = val
                else:
                    base_options[key] = val
            self._ci_guess = 'cis_restart_' + str(os.getpid())
        elif orig_opts.get('casscf', '') == 'yes' or orig_opts.get('casci', '') == 'yes':
            excited_type = 'cas'
            #   CAS-CI and CAS-SCF
            for key, val in orig_opts.items():
                if key in cas_possible_opts:
                    excited_options[key] = val
                else:
                    base_options[key] = val
        else:
            base_options.update(orig_opts)

        #   determine the range of gradients and couplings to compute
        grads = []
        NACs = []
        if gradients:
            if gradients == 'all':
                grads = list(range(max_state+1))
            else: grads = gradients

        if couplings:
            if couplings == 'all':
                for i in sorted(grads):
                    for j in sorted(grads):
                        if j <= i:
                            continue
                        NACs.append((i, j))

        # if couplings:
        #     NACs = []
        #     for i in range(max_state+1):
        #         for j in range(i+1, max_state+1):
        #             NACs.append((i, j))

        #   make sure we are computing enough states
        if excited_type == 'cis':
            if excited_options.get('cisnumstates', 0) < max_state:
                warnings.warn(f'Number of states in TC options is less than `max_state`. Increasing number of states to {max_state}')
                excited_options['cisnumstates'] = max_state + 1
        if excited_type == 'cas':
            if excited_options.get('cassinglets', 0) < max_state:
                warnings.warn(f'Number of states in TC options is less than `max_state`. Increasing number of states to {max_state}')
                excited_options['cassinglets'] = max_state + 1

        if max_state > 0:
            base_options['cisrestart'] = 'cis_restart_' + str(os.getpid())
        base_options['purify'] = False
        base_options['atoms'] = atoms


        n_clients = len(self._client_list)
        jobs_to_run = [{} for i in range(n_clients)]
        job_num = 0

        #   run energy only if gradients and NACs are not requested
        all_results = []
        if len(grads) == 0 and len(NACs) == 0:
            job_opts = base_options.copy()
            start = time.time()
            results = self.compute_job_sync_with_restart('energy', geom, 'angstrom', **job_opts)
            times[f'energy'] = time.time() - start
            results['run'] = 'energy'
            results.update(job_opts)
            all_results.append(results)

        #   create gradient job properties
        #   gradient computations have to be separated from NACs
        for job_i, state in enumerate(grads):
            name = f'gradient_{state}'
            job_opts = base_options.copy()
            if name in self._spec_job_opts:
                job_opts.update(self._spec_job_opts[name])

            if excited_type == 'cas':
                job_opts.update(excited_options)
                job_opts['castarget'] = state

            elif state > 0:
                if excited_type == 'cis':
                    job_opts.update(excited_options)
                    job_opts['cistarget'] = state
                    
            # self._set_guess(job_opts, excited_type, all_results, state)

            jobs_to_run[job_num % n_clients][name] = {
                'opts': job_opts.copy(), 
                'type': 'gradient', 
                'state': state
                }
            job_num += 1


        #   create NAC job properties
        for job_i, (nac1, nac2) in enumerate(NACs):
            name = f'nac_{nac1}_{nac2}'
            job_opts = base_options.copy()
            job_opts.update(excited_options)
            if name in self._spec_job_opts:
                job_opts.update(self._spec_job_opts[name])

            if excited_type == 'cis':
                job_opts.update(excited_options)
                job_opts['cistarget'] = state
            elif excited_type == 'cas':
                job_opts.update(excited_options)
                job_opts['castarget'] = state

            job_opts['nacstate1'] = nac1
            job_opts['nacstate2'] = nac2

            jobs_to_run[job_num % n_clients][name] = {
                'opts': job_opts.copy(),
                'type': 'coupling',
                'state': max(nac1, nac2)
                }
            job_num += 1


        #   if only one client is being used, don't open up threads, easier to debug
        if n_clients == 1:
            start = time.time()
            results, times = _run_jobs(self._client_list[0], jobs_to_run[0], geom, excited_type, self._server_root_list[0], 0)
            all_results += results
            end = time.time()

        else:
            start = time.time()
            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = []
                for i in range(n_clients):
                    jobs = jobs_to_run[i]
                    args = (self._client_list[i], jobs, geom, excited_type, self._server_root_list[i], i)
                    future = executor.submit(_run_jobs, *args)
                    futures.append(future)
                for f in futures:
                    batch_results, batch_times = f.result()
                    all_results += batch_results
                    times.update(batch_times)
            end = time.time()

    
        self.set_avg_max_times(times)

        return all_results, times
    
    def _set_guess(self, job_opts: dict, excited_type: str, all_results: list[dict], state):
        return _set_guess(job_opts, excited_type, all_results, state)

def _run_jobs(client: TCPBClient, jobs, geom, excited_type, server_root, client_ID=0):
    times = {}
    all_results = []
    for job_name, job_props in jobs.items():
        job_opts = job_props['opts']
        job_type = job_props['type']
        job_state = job_props['state']

        _set_guess(job_opts, excited_type, all_results, job_state, server_root)
        print(f"\nRunning {job_name} on client ID {client_ID}")

        start = time.time()
        # results = client.compute_job_sync(job_type, geom, 'angstrom', **job_opts)
        results = compute_job_sync(client, job_type, geom, 'angstrom', **job_opts)
        times[job_name] = time.time() - start
        results['run'] = job_type
        results.update(job_opts)
        all_results.append(results)

    return all_results, times

def _set_guess(job_opts: dict, excited_type: str, all_results: list[dict], state: int, server_root='.'):
    import json
    cas_guess = None
    scf_guess = None
    if state > 0:
        if excited_type == 'cas':
            for prev_job in reversed(all_results):
                if prev_job.get('castarget', 0) >= 1:
                    prev_orb_file = prev_job['orbfile']
                    if prev_orb_file[-6:] == 'casscf':
                        cas_guess = os.path.join(server_root, prev_orb_file)
                        break
        #   we do not consider cis file because it is not stored in each job dorectory; we set it ourselves

    for prev_job in reversed(all_results):
        if 'orbfile' in prev_job:
            scf_guess = os.path.join(server_root, prev_job['orbfile'])
            #   This is to fix a bug in terachem that still sets the c0.casscf file as the
            #   previous job's orbital file
            if scf_guess[-6:] == 'casscf':
                scf_guess = scf_guess[0:-7]
            break

    if os.path.isfile(str(cas_guess)):
        job_opts['casguess'] = cas_guess
    if os.path.isfile(str(scf_guess)):
        job_opts['guess'] = scf_guess

def format_output_LSCIVR(job_data: list[dict]):
    atoms = job_data[0]['atoms']
    n_atoms = len(atoms)
    
    # energies = np.zeros(n_elec)
    # grads = np.zeros((n_elec, n_atoms*3))
    # nacs = np.zeros((n_elec, n_elec, n_atoms*3))
    energies = {}
    grads = {}
    nacs = {}
    trans_dips = {}
    for job in job_data:
        if job['run'] == 'gradient':
            state = job.get('cistarget', job.get('castarget', 0))
            grads[state] = np.array(job['gradient']).flatten()
            if isinstance(job['energy'], float):
                energies[state] = job['energy']
            else:
                energies[state] = job['energy'][state]
        elif job['run'] == 'coupling':
            state_1 = job['nacstate1']
            state_2 = job['nacstate2']
            nacs[(state_1, state_2)] = np.array(job['nacme']).flatten()
            nacs[(state_2, state_1)] = - nacs[state_1, state_2]

            x = len(job['cis_transition_dipoles'])
            N = int((1 + int(np.sqrt(1+8*x)))/2) - 1
            count = 0
            for i in range(0, N):
                for j in range(i+1, N+1):
                    if (i, j) == (state_1, state_2):
                        trans_dips[(state_1, state_2)] = job['cis_transition_dipoles'][count]
                        trans_dips[(state_2, state_1)] = job['cis_transition_dipoles'][count]
                    count += 1


    #   make sure there are the correct number of gradients and NACs
    n_states = len(grads)
    if n_states*(n_states-1) != len(nacs):
        raise RuntimeError('LSC-IVR requires a NAC vector for each gradient pair')
    
    #   make sure each grad pair has a NAC vector
    for grad_i in grads:
        for grad_j in grads:
            if grad_i == grad_j:
                continue
            if (grad_i, grad_j) not in nacs:
                raise RuntimeError('LSC-IVR requires a NAC vector for each gradient pair')

    #   print mapping
    ivr_energies = np.zeros(n_states)
    ivr_grads = np.zeros((n_states, n_atoms*3))
    ivr_nacs  = np.zeros((n_states, n_states, n_atoms*3))
    ivr_trans_dips = np.zeros((n_states, n_states, 3))
    print(" --------------------------------")
    print(" LSC-IVR to quantum chemistry")
    print(" state number mapping")
    print(" ---------------------------------")
    print(" LSC-IVR -->   QC  ")
    grads_in_order = sorted(list(grads.keys()))
    for i in range(n_states):
        qc_i = grads_in_order[i]
        print(f"   {i:2d}    -->  {qc_i:2d}")
        ivr_grads[i] = grads[qc_i]
        ivr_energies[i] = energies[qc_i]
        for j in range(n_states):
            if i <= j:
                continue
            qc_idx_j = grads_in_order[j]
            ivr_nacs[i, j] = nacs[(qc_i, qc_idx_j)]
            ivr_nacs[j, i] = nacs[(qc_idx_j, qc_i)]

            ivr_trans_dips[i, j] = trans_dips[(qc_i, qc_idx_j)]
            ivr_trans_dips[j, i] = trans_dips[(qc_idx_j, qc_i)]
    print(" ---------------------------------")

    return ivr_energies, ivr_grads, ivr_nacs, trans_dips
    # return energies, grads, nacs
