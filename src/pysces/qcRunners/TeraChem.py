from __future__ import annotations 

from tcpb import TCProtobufClient as TCPBClient
from tcpb.exceptions import ServerError
from tcpb import terachem_server_pb2 as pb

from pysces.input_simulation import logging_dir, TCRunnerOptions
from pysces.common import PhaseVars, PhaseVarHistory, QCRunner, ESResults, ESResultsHistory
from pysces.interpolation import GradientInterpolation, NACnterpolation

from datetime import datetime
import functools
import os
import threading
import numpy as np
import time
import warnings
import shutil
import socket
import subprocess
import time
import psutil
import concurrent.futures
import copy
import pickle
import json
from copy import deepcopy
from typing import Literal
import itertools
from collections import deque
import qcelemental as qcel
import base64
from pprint import pprint
from scipy.interpolate import interp1d
import scipy.linalg as la

from pprint import pprint

_server_processes = {}

#   debug flags
_DEBUG = bool(int(os.environ.get('DEBUG', False)))
_DEBUG_TRAJ = os.environ.get('DEBUG_TRAJ', False)
_SAVE_BATCH = os.environ.get('SAVE_BATCH', False)
_SAVE_DEBUG_TRAJ = os.environ.get('SAVE_DEBUG_TRAJ', False)
_RESULTS_ID = 0

ANG_2_BOHR = 1.8897259886         # angstroms to bohr
AMU_2_AU = 1.822888486*10**3      # atomic mass unit to a.u. of mass

def synchronized(function):
    ''''Decorator to make sure that only one thread can access the function at a time'''
    lock = threading.Lock()
    @functools.wraps(function)
    def wrapper(self, *args, **kwargs):
        with lock:
            return function(self, *args, **kwargs)
    return wrapper


class TCClientExtra(TCPBClient):
    host_port_to_ID: dict = {}

    def __init__(self, host: str = "127.0.0.1", port: int = 11111, debug=False, trace=False, log=True, server_root='.') -> None:
        """
        Initializes a TCClientExtra object.

        Args:
            host (str): The host IP address. Defaults to "127.0.0.1".
            port (int): The port number. Defaults to 11111.
            debug (bool): Whether to enable debug mode. Defaults to False.
            trace (bool): Whether to enable trace mode. Defaults to False.
            log (bool): Whether to enable logging. Defaults to True.
        """
        
        
        self._log = None
        if log:
            log_file_loc = os.path.join(logging_dir, f'{host}_{port}.log')
            self._log = open(log_file_loc, 'a')
            self.log_message(f'Client started on {host}:{port}')
        self.server_root = server_root
        self._possible_files_to_remove: set[str] = {'exciton_overlap.dat', 'exciton_overlap.dat.1', 'exciton.dat'}
        self._last_known_curr_dir = None
        self._results_history = deque(maxlen=10)
        self._exciton_overlap_data = None
        self._exciton_data = None
        self._scf_guess_file = None
        self._cas_guess_file = None
        self._cis_guess_file = None
        self.prev_job = None


        super().__init__(host, port, debug, trace)
        self._add_to_host_port_to_ID()

    def cleanup(self):
        if self._log:
            self._log.close()
        for file in self._possible_files_to_remove:
            full_file = os.path.join(self.server_root, file)
            if os.path.isfile(full_file):
                os.remove(full_file)

    def __del__(self):
        if self._log:
            self._log.close()

    def __repr__(self) -> str:
        """
        Returns a string representation of the TCClientExtra object.

        Returns:
            str: The string representation of the object.
        """
        out_str = f'TCClientExtra(host={self.host}, port={self.port}, id={self.get_ID()})'
        return out_str
    
    def __getstate__(self):
        state = self.__dict__.copy()
        # for k, v in state.items():
        #     print('ITEM: ', k)
        state.pop('tcsock')
        state.pop('_log')
        return state

    def startup(self, max_wait=10.0, time_btw_check=1.0):
        print('Setting up new client')
        total_wait = 0.0
        avail = False
        while not avail:
            try:
                self.connect()
                self.is_available()
                avail = True
            except:
                print(f'TeraChem server {self.host}:{self.port} not available: \n\
                        trying again in {time_btw_check} seconds', flush=True)
                time.sleep(time_btw_check)
                total_wait += time_btw_check
                if total_wait >= max_wait:
                    raise TimeoutError('Maximum time allotted for checking for TeraChem server')
        print(f'Terachem server {self.host}:{self.port} is available and connected')
    
    def restart(self, max_wait=10.0, time_btw_check=1.0):
        server_root = self.server_root
        if (self.host, self.port) in _server_processes:
            process: TCServerProcess = _server_processes[(self.host, self.port)]
            process.kill()
            time.sleep(8.0)
            start_TC_server(self.port, server_root=server_root, gpus=process.gpus)
        self.disconnect()
        self.startup()

    def disconnect(self):
        super().disconnect()
        self.host_port_to_ID.pop((self.host, self.port))

    def log_message(self, message):
        """
        Logs a message to the log file.

        Args:
            message (str): The message to be logged.
        """
        if self._log:
            current_time = datetime.now()
            formatted_time = current_time.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3] 
            self._log.write(f'{formatted_time}: {message}\n')
            self._log.flush()

    @synchronized
    def _add_to_host_port_to_ID(self):
        """
        Adds the host and port combination to the host_port_to_ID dictionary.
        """
        new_ID = 0
        while new_ID in self.host_port_to_ID.values():
            new_ID += 1
        TCClientExtra.host_port_to_ID[(self.host, self.port)] = new_ID

    def get_ID(self):
        """
        Returns the ID associated with the current host and port combination.

        Returns
        -------
            int: The ID associated with the host and port combination.
        """
        return self.host_port_to_ID[(self.host, self.port)]
    
    def get_curr_job_dir(self):
        return os.path.join(self.server_root, self.curr_job_dir)
    
    @property
    def prev_results(self):
        '''overwrite the getter for TCPBClient.prev_results to add the results to the history'''
        if len(self._results_history) == 0:
            return None
        return self._results_history[-1]

    @prev_results.setter
    def prev_results(self, value):
        '''
            Overwrite the setter for TCPBClient.prev_results to add the results to the history.
            This is used as a trigger to run routines directly after a job has been sent back
            from TeraChem.
        '''
        if value is None:
            return
        
        if not os.path.isdir(self.server_root):
            print('Warning: Server root directory does not exist, cannot remove old jobs')

        if len(self._results_history) == self._results_history.maxlen:
            oldest_res = self._results_history[0]
            job_dir = os.path.join(self.server_root, oldest_res['job_dir'])
            

            if os.path.isdir(job_dir):
                shutil.rmtree(job_dir)
            else:
                print('Warning: no job directory found at')
                print(job_dir)
                print('Check that "tcr_server_root" is properly set to remove old jobs')

        self._results_history.append(value)

    def copy_guess_files(self, prev_results_hist: list[dict]):
        '''
            Not used yet. This is to copy the guess files from the previous job t
        '''
        scf_guess, cis_guess, cas_guess = self.get_guess_file_locs(prev_results_hist)
        guess_data = {}
        if os.path.isfile(str(cas_guess)):
            with open(cas_guess, 'rb') as file:
                data = file.read()
                guess_data['casguess'] = base64.b64encode(data).decode('utf-8')
        if os.path.isfile(str(scf_guess)):
            with open(scf_guess, 'rb') as file:
                data = file.read()
                guess_data['scfguess'] = base64.b64encode(data).decode('utf-8')
        if os.path.isfile(str(cis_guess)):
            with open(cis_guess, 'rb') as file:
                data = file.read()
                guess_data['cisguess'] = base64.b64encode(data).decode('utf-8')

    def set_guess_files_from_job(self, prev_job: TCJob):
        '''
            Set the guess files for the next job based on the history of results
        '''
        res_history = self._results_history

        server_root = self.server_root
        if len(res_history) != 0:
            prev_job_res = res_history[-1]
            job_dir = os.path.join(server_root, prev_job_res['job_dir'])
            if not os.path.isdir(job_dir):
                print("Warning: no job directory found at")
                print(job_dir)
                print("Check that 'tcr_server_root' is properly set in order to use previous job guess orbitals")

        cas_guess = None
        scf_guess = None
        cis_guess = prev_job.opts.get('cisrestart', None)
        if prev_job.state > 0:
            if prev_job.excited_type == 'cas':
                for prev_job_res in reversed(res_history):
                    if prev_job_res.get('castarget', 0) >= 1:
                        prev_orb_file = prev_job_res['orbfile']
                        if prev_orb_file[-6:] == 'casscf':
                            cas_guess = os.path.join(server_root, prev_orb_file)
                            break


        for i, prev_job_res in enumerate(reversed(res_history)):
            if 'orbfile' in prev_job_res:
                scf_guess = os.path.join(server_root, prev_job_res['orbfile'])
                #   This is to fix a bug in terachem that still sets the c0.casscf file as the
                #   previous job's orbital file
                if scf_guess[-6:] == 'casscf':
                    scf_guess = scf_guess[0:-7]
                break

        self._cis_guess_file = cis_guess
        self._scf_guess_file = scf_guess
        self._cas_guess_file = cas_guess

        return scf_guess, cis_guess, cas_guess

    def assign_guess_files(self, new_job: TCJob):
        '''
            Assign the guess files to the new job
        '''
        if self._cas_guess_file is not None:
            new_job.opts['casguess'] = self._cas_guess_file 
        if self._scf_guess_file is not None:
            new_job.opts['guess'] = self._scf_guess_file
        if self._cis_guess_file is not None:
            new_job.opts['cisrestart'] = self._cis_guess_file 

    @property
    def results_history(self):
        return self._results_history

    @property
    def curr_job_dir(self):
        return self.__dict__['curr_job_dir']
    
    @curr_job_dir.setter
    def curr_job_dir(self, value):
        self.__dict__['curr_job_dir'] = value
        self._last_known_curr_dir = value

    def server_file(self, file_name):
        return os.path.join(self.server_root, file_name)

    def print_end_of_file(self, n_lines=30):
        lines  = []

        if self.curr_job_dir is not None:
            job_dir = os.path.join(self.server_root, self.curr_job_dir)
        elif self._last_known_curr_dir is not None:
            job_dir = os.path.join(self.server_root, self._last_known_curr_dir)
        else:
            print('No job directory set, cannot print end of tc.out file')
            return
        
        tc_out_file_loc = os.path.join(job_dir, 'tc.out')
        if not os.path.isfile(tc_out_file_loc):
            print('No tc.out file found at:')
            print(job_dir)
            return

        with open(tc_out_file_loc, 'r') as file:
            lines = file.readlines()
        print('End of tc.out file at:')
        print(job_dir)
        print('\n START OF FILE .... \n')
        for line in lines[-n_lines:]:
            if '\n' in line:
                print(line[0:-1])
            else:
                print(line)
        print('\n ... END OF FILE \n')

    def compute_job(self, job: TCJob):
        
        #   assign the guess files to the current job
        self.assign_guess_files(job)

        #   send the job to the terachem server
        job.start_time = time.time()
        results = self.compute_job_sync(job.job_type, job.geom, 'angstrom', **job.opts)
        job.end_time = time.time()

        #   set the results of the job
        results['run'] = job.job_type
        results.update(job.opts)
        job.results = results.copy()
        self.prev_job = job

        #   set the guess files for the next job
        self.set_guess_files_from_job(job)

        return results

    #   TODO: add a counter for jobs submitted.
    #   If the number of jobs submitted is the first job, remove the old restart file
    def compute_job_sync(self, jobType="energy", geom=None, unitType="bohr", **kwargs):
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

        # print("Submitting Job...")
        self.log_message("Submitting Job...")

        accepted = self.send_job_async(jobType, geom, unitType, **kwargs)
        while accepted is False:
            time.sleep(0.5)
            accepted = self.send_job_async(jobType, geom, unitType, **kwargs)

        self.log_message("Job Accepted")
        self.log_message(f"    Job Type: {jobType}")
        self.log_message(f"    Current Job Dir: {self.get_curr_job_dir()}")

        completed = self.check_job_complete()
        while completed is False:
            time.sleep(0.5)
            self._send_msg(pb.STATUS, None)
            status = self._recv_msg(pb.STATUS)

            if status.WhichOneof("job_status") == "completed":
                completed = True
            elif status.WhichOneof("job_status") == "working":
                completed = False
            else:
                raise ServerError(
                    "Invalid or no job status received, either no job submitted before check_job_complete() or major server issue",
                    self,
                )
        results = self.recv_job_async()
        self.log_message(f"Job Complete with {len(results)} dictionary entries")
        time.sleep(0.1)

        self._finalize_run(results, kwargs)

        return results
            
    def _finalize_run(self, results: dict, opts: dict):

        #   try to refresh the file system metadata
        #   sometimes solves the issue of the file not being found
        os.listdir(self.server_root)

        def _ensure_file_exists(file_loc):
            #   if the file is still not found, raise an error
            if not os.path.isfile(file_loc):
                out_str = f'{file_loc} file not found in server root directory\n'
                out_str += 'Current files in server root directory:\n'
                out_str += f'    {self.server_root}\n'
                for x in os.listdir(self.server_root):
                    out_str += f'        {x}\n'
                exit(out_str)


        
        if opts.get('cisexcitonoverlap', 'no') == 'yes' and opts.get('cis', 'no') == 'yes':
            self.expect_exciton_overlap_dat = True

            exciton_overlap_file = os.path.join(self.server_root, 'exciton_overlap.dat')
            exciton_overlap_file_1 = os.path.join(self.server_root, 'exciton_overlap.dat.1')
            exciton_file = os.path.join(self.server_root, 'exciton.dat')

            #   make sure the files exist, exit if it does not
            _ensure_file_exists(exciton_overlap_file)
    
            #   read in previous data
            with open(exciton_overlap_file, 'rb') as file:
                self._exciton_overlap_data = file.read()

            #   if exciton_overlap_file_1 also exists, then exciton.dat must have been created
            if os.path.isfile(exciton_overlap_file_1):
                _ensure_file_exists(exciton_file)
                overlap_data = []
                with open(exciton_file, 'r') as file:
                    for line in file:
                        sp = line.split()
                        if sp[0] == 'Overlap:':
                            data = [float(x) for x in sp[1:]]
                            overlap_data.append(data)
                overlap_data = np.array(overlap_data)
                self._exciton_data = overlap_data
                results['exciton_overlap'] = overlap_data
                os.remove(exciton_file)
            else:
                #   exciton_overlap.dat exists but exciton_overlap.dat.1 does not
                #   this occus the first time exciton_overlap.dat is read by TeraChem, so we
                #   assume that it's the first frame that does so. We rename to .1 to keep
                #   the wavefunctiosn consistent
        
                os.rename(exciton_overlap_file, exciton_overlap_file_1)

            if opts.get('cisrestart', None):
                self._possible_files_to_remove.add(opts.get('cisrestart', None))

            #   attempt to make sure the file system is updated
            dir_fd = os.open(self.server_root, os.O_DIRECTORY)
            os.fsync(dir_fd)
            os.close(dir_fd)

        os.listdir(self.server_root)
            

class TCServerProcess(subprocess.Popen):
    def __init__(self, port, gpus=[]):
        self.gpus = gpus
        self.port = port
        self.start()

    def start(self):
        if len(self.gpus) == 0:
            command = f'terachem -s {self.port}'
        else:
            gpus_str = ''.join([str(x) for x in self.gpus])
            command = f'terachem -s {self.port} --gpus {gpus_str}'

        super().__init__(command.split(), shell=False)

    def restart(self):
        self.kill()
        self.start()

    def kill(self):
        proc = psutil.Process(self.pid)
        for child in proc.children(recursive=True):  # or parent.children() for recursive=False
            child.kill()
        proc.kill()

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

def start_TC_server(port: int, gpus=[], server_root='.'):
    '''
        Start a new TeraChem server. 
        
        Parameters
        ----------
        port: int, port number for the server to use
        gpus: list, list of GPU numbers to use
        server_root: string, the root directory to run the server in

        Returns
        -------
        host: string, the host of the server being run
    '''

    #   change to server root directory to start the server
    original_curr_dir = os.path.abspath(os.curdir)
    os.chdir(server_root)

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
    process = TCServerProcess(port, gpus)
    time.sleep(10)
    #   set up a temporary client to make sure it is open
    try:
        tmp_client = TCPBClient(host, port)
        tmp_client.connect()
        tmp_client.disconnect()
    except TimeoutError as e:
        raise RuntimeError(f'Could not start TeraChem server on {host} with port {port}')
    
    _server_processes[(host, port)] = process
    #   return to original directory
    os.chdir(original_curr_dir)
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

class TCJob():
    __job_counter = 0
    def __init__(self, geom, opts, job_type, excited_type, state, name='', client=None) -> None:
        self.geom = geom
        self.excited_type: Literal['cas', 'cis'] = excited_type
        self.opts = opts
        self.job_type = job_type
        self.state = state
        self.name = name
        self.start_time = 0
        self.end_time = 0
        self._results = {}
        self.client: TCClientExtra = client
        self.batch: TCJobBatch = None

        if job_type == 'gradient_est' or job_type == 'coupling_est':
            pass
        elif job_type not in ['energy', 'gradient', 'coupling']:
            raise ValueError('TCJob job_type must be either "energy", "gradient", or "coupling"')

        assert self.excited_type in ('cas', 'cis')

        TCJob.__job_counter += 1
        self.__jobID = TCJob.__job_counter

    def __repr__(self) -> str:
        out_str = '('
        for k in ['name', 'excited_type', 'job_type', 'state']:
            out_str += f'{k}={self.__dict__[k]}, '
        out_str += f'jobID={self.jobID}, complete={self.complete}'
        out_str += ')'
        return out_str

    def new_from_old(self, new_geom=None):
        if new_geom is None:
            new_geom = self.geom
        new_job = TCJob(new_geom, self.opts, self.job_type, self.excited_type, self.state, self.name, self.client)
        return new_job

    @classmethod
    def get_ID_counter(cls):
        return cls.__job_counter
    
    @classmethod
    def set_ID_counter(cls, value):
        if not isinstance(value, int):
            raise ValueError('TCJob ID counter must be an integer')
        if value < cls.__job_counter:
            raise ValueError('TCJob ID counter must be greater than the current value')
        cls.__job_counter = value

    @property
    def total_time(self):
        return self.end_time - self.start_time

    @property
    def complete(self):
        if self.results: 
            return True
        else: 
            return False

    @property
    def jobID(self):
        return self.__jobID
    
    @property
    def results(self):
        return self._results
    
    @results.setter
    def results(self, value):
        self._results = value  
        self.batch._update_completion()
        # self.batch._completed_jobs[self.jobID] = value

class _CustomJobList(list):
    ''' POSSIBLE REMOVE THIS CLASS '''
    def __init__(self, parent_batch, *args):
        super().__init__(*args)
        self.parent_batch = parent_batch

    def append(self, job: TCJob, allow_duplicates=True):
        if not allow_duplicates:
            if job.jobID in [j.jobID for j in self]:
                raise ValueError(f'JobID {job.jobID} already in batch')
        super().append(job)
        # Link the job to the batch
        job.batch = self.parent_batch

class TCJobBatch():
    '''
        Combines multiple TCJobs into one wrapper
    '''
    __batch_counter = 0
    def __init__(self, jobs: list[TCJob] | list['TCJobBatch'] = []) -> None:
        if len(jobs) == 0:
            self.jobs = _CustomJobList(self)
        
        if len(jobs) > 0:
            if isinstance(jobs[0], TCJobBatch):
                jobs = list(itertools.chain(*[j.jobs for j in jobs]))
            
            # Sort by ID
            id_list = [j.jobID for j in jobs]
            order = np.argsort(id_list)
            self.jobs = _CustomJobList(self, [jobs[i] for i in order])

        # TCJobBatch.__batch_counter += 1
        # self.__batchID = TCJobBatch.__batch_counter
        self._complete = False

        caller_info = self._get_caller_info()
        

    def _get_caller_info(self):
        # This helps identify the function that created the object
        import inspect
        frame = inspect.currentframe()
        outer_frame = inspect.getouterframes(frame, 2)
        caller_frame = outer_frame[2]
        return caller_frame


    def __getstate__(self):
        state = self.__dict__.copy()
        state['__batchID'] = self.__batch_counter
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        TCJobBatch.__batch_counter = max(TCJobBatch.__batch_counter, state['__batchID'])

    def __repr__(self) -> str:
        out_str = ''
        for j in self.jobs:
            out_str += f'{j}\n'
        return out_str
    
    def __len__(self):
        return len(self.jobs)
    
    def _update_completion(self):
        if self._complete:
            return
        else:
            if self.check_complete():
                self._complete = True
                TCJobBatch.__batch_counter += 1

    @classmethod
    def get_ID_counter(cls):
        return cls.__batch_counter
    
    @classmethod
    def set_ID_counter(cls, value):
        if not isinstance(value, int):
            raise ValueError('TCJobBatch ID counter must be an integer')
        if value < cls.__batch_counter:
            raise ValueError('TCJobBatch ID counter must be greater than the current value')
        cls.__batch_counter = value
    
    @property
    def batchID(self):
        return self.__batch_counter

    def get_by_id(self, id):
        for j in self.jobs:
            if j.jobID == id:
                return j
        raise ValueError(f'JobID {id} not found in jobs batch')
    
    def get_by_type(self, job_type: Literal['energy', 'gradient', 'coupling']):
        if job_type not in ['energy', 'gradient', 'coupling']:
            raise ValueError('TCJob job_type must be either "energy", "gradient", or "coupling"')
        jobs = [x for x in self.jobs if x.job_type == job_type]
        return TCJobBatch(jobs)
    
    def get_by_name(self, names: list | str):
        if isinstance(names, str):
            names = [names]
        jobs = [x for x in self.jobs if x.name in names]
        return TCJobBatch(jobs)
    
    def get_by_client(self, client: TCClientExtra | int):
        if isinstance(client, int):
            jobs = [x for x in self.jobs if x.client.get_ID() == client]
        elif isinstance(client, TCClientExtra):
            jobs = [x for x in self.jobs if x.client == client]
        else:
            raise ValueError('client must be either an integer or a TCClientExtra object')
        return TCJobBatch(jobs)
    
    def get_by_options(self, opts: dict, match_type: Literal['all', 'any'] = 'all'):
        jobs = []
        if match_type == 'all':
            for j in self.jobs:
                if set(j.opts.items()).issubset(opts.items()):
                    jobs.append(j)
        elif match_type == 'any':
            for j in self.jobs:
                if len(set(j.opts.items()).intersection(opts.items())) > 0:
                    jobs.append(j)
        else:
            raise ValueError('match_type must be either "all" or "any"')
        
        return TCJobBatch(jobs)

    def set_client(self, client: TCClientExtra):
        for j in self.jobs:
            j.client = client

    def append(self, job: TCJob, allow_duplicates=True):
        if not allow_duplicates:
            if job.jobID in [j.jobID for j in self.jobs]:
                raise ValueError(f'JobID {job.jobID} already in batch')
        self.jobs.append(job)
    
    def sorted_jobs_by_state(self):
        states = [x.state for x in self.jobs]
        order = np.argsort(states)
        return [self.jobs[n] for n in order]
    
    def check_complete(self):
        for j in self.jobs:
            if not j.complete:
                return False
        return True
    
    def check_client(self):
        for j in self.jobs:
            if j.client is None:
                return False
        return True

    def wait_for_complete(self, max_wait=20):
        start = time.time()
        while not self.check_complete():
            time.sleep(0.25)
            if time.time() - start > max_wait:
                raise TimeoutError('Maximum time allotted for waiting for job completion')

    def _remove_clients(self):
        ''' DEBUGGING ONLY 
            Clients can't be pickled
        '''
        for j in self.jobs:
            j.client = None

    @property
    def results_list(self) -> list[dict]:
        return [j.results for j in self.jobs]
    
    @property
    def timings(self) -> dict:
        timings = {j.name: j.total_time for j in self.jobs}
        min_time = np.min([j.start_time for j in self.jobs])
        max_time = np.max([j.end_time for j in self.jobs])
        timings['Wall_Time'] = max_time - min_time
        # timings['Wall_Time'] = np.sum(list(timings.values()))
        return timings


class TCRunner(QCRunner):
    def __init__(self, atoms: list, tc_opts: TCRunnerOptions, max_wait=20) -> None:
        super().__init__()
        
        # Atoms and max_wait
        # self._atoms = np.copy(atoms)
        self._atoms = tuple(atoms)
        self._max_wait = max_wait

        # Hosts, ports, and server roots
        self._hosts = tc_opts.host
        self._ports = tc_opts.port
        self._server_roots = tc_opts.server_root
        self._prepare_server_info()  # Validate and process server information

        # Job-related options
        self._spec_job_opts = {}
        self._orig_options = {}
        self._initial_frame_options = {}
        self._initialize_job_options(tc_opts)  # Initialize job options

        # State-related options
        self._tc_client_assignments = tc_opts.client_assignments
        self._grads = None
        self._max_state = None
        self._NACs = None
        self._initialize_state_options(tc_opts)  # Initialize state options

        #   Base and excited state options
        self._excited_type = None
        self._base_options = {}
        self._excited_options = {}
        self._initialize_base_and_excited_options()

        #   Gradient and NAC specific job options
        self.grad_job_options = {}
        self.nac_job_options = {}
        self._initialize_grad_nac_options()

        # Server and client management
        self._server_root_list = []
        self._client_list: list[TCClientExtra] = []
        self._client = None
        self._host = None
        self._port = None
        self._setup_servers_and_clients(tc_opts)  # Set up servers and clients

        # Guesses for SCF/CAS/CI calculations
        # self._cas_guess = None
        # self._scf_guess = None
        # self._ci_guess = None
        self._exciton_overlap_data = None

        # Timing and stall handling
        self._max_time_list = []
        self._max_time = None
        self._restart_on_stall = True

        # Job tracking and debugging
        self._prev_results = []
        self._prev_jobs: list[TCJob] = []
        self._frame_counter = 0
        self._prev_ref_job = None
        self._prev_job_batch: TCJobBatch = None
        # self.job_batch_history = deque(maxlen=4)

        # Debug trajectory
        self._debug_traj = []
        self._load_debug_trajectory()  # Load debug trajectory if applicable
        self._initial_ref_nacs = tc_opts._initial_ref_nacs

        #   inteprolation
        # self._energy_history = deque(maxlen=10)
        self._masses = np.array([[qcel.periodictable.to_mass(symbol)]*3 for symbol in atoms]).flatten()
        self._es_history = ESResultsHistory()
        self._phase_var_history = PhaseVarHistory()
        # self._grad_estimator = {g: GradEstimator(order=2, interval=3.0, name=f'grad_{g}') for g in self._grads}
        self._grad_estimates = {g: None for g in self._grads}
        self._nac_estimates = {n: None for n in self._NACs}
        self._interpolation = tc_opts.interpolation

        self._grad_interpolator = GradientInterpolation(self._grads, self._masses)
        self._nac_interpolator = NACnterpolation(self._NACs, self._masses)
        self._probe_job_results = None

        # Print options summary
        self._print_options_summary()
        self._cleanup_stale_files()
        self._coordinate_exciton_overlap_files(tc_opts.fname_exciton_overlap_data)
        
        

    def _prepare_server_info(self):
        """Ensure server roots are valid paths and check server configuration."""
        if isinstance(self._hosts, str):
            self._hosts = [self._hosts]
        if isinstance(self._ports, int):
            self._ports = [self._ports]
        if isinstance(self._server_roots, str):
            self._server_roots = [self._server_roots]

        for i, root in enumerate(self._server_roots):
            os.makedirs(root, exist_ok=True)
            self._server_roots[i] = os.path.abspath(root)

        if len({len(self._hosts), len(self._ports), len(self._server_roots)}) != 1:
            raise ValueError('Number of servers must match the number of port numbers and root locations')

    def _initialize_job_options(self, tc_opts: TCRunnerOptions):
        """Initialize job options."""
        # self._base_options = tc_opts.job_options.copy()
        for k, v in tc_opts.job_options.items():
            self._orig_options[k.lower()] = v

        if tc_opts.spec_job_opts is not None:
            for k, v in tc_opts.spec_job_opts.items():
                self._spec_job_opts[k.lower()] = v

        if tc_opts.initial_frame_opts is not None:
            for k, v in tc_opts.initial_frame_opts.items():
                self._initial_frame_options[k.lower()] = v

    def _initialize_state_options(self, tc_opts: TCRunnerOptions):
        """Initialize state-related options."""
        if tc_opts.state_options.get('grads', False):
            self._grads = tc_opts.state_options['grads']
            self._max_state = max(self._grads)
        elif tc_opts.state_options.get('max_state', False):
            self._max_state = tc_opts.state_options['max_state']
            self._grads = list(range(self._max_state + 1))
        else:
            raise ValueError('either "max_state" or a list of "grads" must be specified')

        self._NACs = tc_opts.state_options.pop('nacs', 'all')
        if self._NACs == 'all':
            self._NACs = list(itertools.combinations(self._grads, 2))
    
    def _initialize_base_and_excited_options(self):
        orig_opts = self._orig_options.copy()
        max_state = self._max_state
        excited_options = {}
        base_options = {}
        excited_type = None

        cas_possible_opts = [
            "closed", "active", "casnumalpha", "casnumbeta",
            "cassinglets", "casdoublets", "castriplets", "casquartets",
            "casquintets", "cassextets", "casseptets",
            "castargetmult", "castarget", "casweights"]
        cas_possible_opts += [
            "casscf", "casscfmicromaxiter", "casscfmacromaxiter",
            "casscfmaxiter", "casscfmicroconvthre",
            "casscfmacroconvthre", "casscfconvthre",
            "dynamicweights", "cpsacasscfmaxiter",
            "cpsacasscfconvthre", 'casci', 'fon']



        if orig_opts.get('cis', '') == 'yes':
            #   CI and TDDFT
            excited_type = 'cis'
            for key, val in orig_opts.items():
                if key[0:3] == 'cis':
                    excited_options[key] = val
                else:
                    base_options[key] = val
            # self._ci_guess = 'cis_restart_' + str(os.getpid())
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
            excited_options['cisrestart'] = 'cis_restart_' + str(os.getpid())
        base_options['purify'] = False
        base_options['atoms'] = self._atoms

        self._base_options = base_options.copy()
        self._excited_options = excited_options.copy()
        self._excited_type = excited_type

    def _initialize_grad_nac_options(self):
        #   create gradient job properties

        base_options = self._base_options.copy()
        excited_options = self._excited_options.copy()

        #   gradient computations have to be separated from NACs
        for job_i, state in enumerate(self._grads):
            name = f'gradient_{state}'
            job_opts = base_options.copy()
            if name in self._spec_job_opts:
                job_opts.update(self._spec_job_opts[name])

            if self._excited_type == 'cas':
                job_opts.update(excited_options)
                job_opts['castarget'] = state

            elif state > 0:
                if self._excited_type == 'cis':
                    job_opts.update(excited_options)
                    job_opts['cistarget'] = state
                    job_opts['cisexcitonoverlap'] = 'yes'

            self.grad_job_options[name] = job_opts

        #   create NAC job properties
        for job_i, (nac1, nac2) in enumerate(self._NACs):
            name = f'nac_{nac1}_{nac2}'
            job_opts = base_options.copy()
            job_opts.update(excited_options)
            if name in self._spec_job_opts:
                job_opts.update(self._spec_job_opts[name])

            if self._excited_type == 'cis':
                job_opts.update(excited_options)
                job_opts['cistarget'] = state
                job_opts['cisexcitonoverlap'] = 'yes'
            elif self._excited_type == 'cas':
                job_opts.update(excited_options)
                job_opts['castarget'] = state

            job_opts['nacstate1'] = nac1
            job_opts['nacstate2'] = nac2

            self.nac_job_options[name] = job_opts

    def _setup_servers_and_clients(self, tc_opts: TCRunnerOptions):
        """Set up servers and clients."""
        if len(tc_opts.server_gpus) != 0:
            self._hosts = []
            if len(tc_opts.server_gpus) != len(self._ports):
                raise ValueError('Number of GPUs must match the number of servers')

            for i, port in enumerate(self._ports):
                host = start_TC_server(port, tc_opts.server_gpus[i], self._server_roots[i])
                self._hosts.append(host)

        print('Starting TCRunner')
        print('')
        print('          Server Information          ')
        print('------------------------------------------------------------')
        for i in range(len(self._hosts)):
            print(f'Client {i+1}')
            print(f'    Host:        {self._hosts[i]}')
            print(f'    Port:        {self._ports[i]}')
            print(f'    Server Root: {self._server_roots[i]}')


        self._server_root_list = self._server_roots

        for h, p, s in zip(self._hosts, self._ports, self._server_roots):
            if _DEBUG_TRAJ:
                self._client_list.append(None)
                print('DEBUG_TRAJ set, TeraChem clients will not be opened')
                break
            client = TCClientExtra(host=h, port=p, server_root=s)
            client.startup(max_wait=self._max_wait)
            self._client_list.append(client)

        self._client = self._client_list[0]
        self._host = self._hosts[0]
        self._port = self._ports[0]

    def _load_debug_trajectory(self):
        """Load debug trajectory if _DEBUG_TRAJ is set."""
        if _DEBUG_TRAJ:
            with open(_DEBUG_TRAJ, 'rb') as file:
                self._debug_traj = pickle.load(file)

    def _print_options_summary(self):
        exclude_keys = ['atoms', 'cisrestart']
        print('------------------------------------------------------------')
        print()
        print('            Job Information           ')
        print('--------------------------------------')
        print(f'Number of Atoms: {len(self._atoms)}')
        print(f'Max State:       {self._max_state}')
        grad_str = ''
        for g in self._grads:
            grad_str += f'{g}, '
        grad_str = grad_str[0:-2]
        print( 'Gradients:      ',grad_str)
        print('')
        print('TC Job Base Options:')
        for k, v in self._base_options.items():
            if k in ['atoms', 'cisrestart']:
                continue
            # print(f'    {k:20s}: {v}')
            print(f'    {k + " ":.<24s} {v}')

        print('\n TC Job Excited Options:')
        for k, v in self._excited_options.items():
            print(f'    {k + " ":.<24s} {v}')

        print('\n TC Job Specific Options:')
        if len(self._spec_job_opts) == 0:
            print('    None')
        else:
            for k, v in self._spec_job_opts.items():
                print(f'    {k}:')
                for kk, vv in v.items():
                    print(f'        {kk + " ":.<24s} {vv}')

        print('\n TC Initial Frame Options:')
        if len(self._initial_frame_options) == 0:
            print('    None')
        else:
            for k, v in self._initial_frame_options.items():
                print(f'    {k + " ":.<24s} {v}')

        print('\n TC Gradient Specific Options:')
        for k, v in self.grad_job_options.items():
            print(f'    {k}:')
            for kk, vv in v.items():
                if (kk in exclude_keys) or (kk in self._excited_options) or (kk in self._base_options):
                    continue
                print(f'        {kk + " ":.<20s} {vv}')

        print('\n TC NAC Specific Options:')
        for k, v in self.nac_job_options.items():
            print(f'    {k}:')
            for kk, vv in v.items():
                if (kk in exclude_keys) or (kk in self._excited_options) or (kk in self._base_options):
                    continue
                print(f'        {kk + " ":.<20s} {vv}')
        print('--------------------------------------')
        print()

    def _cleanup_stale_files(self):
        for client in self._client_list:
            for file in ['exciton.dat', 'exciton_overlap.dat', 'exciton_overlap.dat.1']:
                file_loc = os.path.join(client.server_root, file)
                if os.path.isfile(file_loc):
                    print('Removing stale file:', file_loc)
                    os.remove(file_loc)

    def report(self):
        return self._prev_job_batch

    def cleanup(self):
        if _SAVE_DEBUG_TRAJ:
            with open(_SAVE_DEBUG_TRAJ, 'wb') as file:
                pickle.dump(self._debug_traj, file)
        if _DEBUG_TRAJ:
            return
        for client in self._client_list:
            client.disconnect()
            client.cleanup()

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.cleanup()

    def __eq__(self, o: object) -> bool:
        if isinstance(o, str):
            if o == 'TCRunner' or o.lower() == 'terachem':
                return True
        return super().__eq__(o)

    def _conenct_user_clients(self):
        pass

  

    '''
    @staticmethod
    def get_new_client(host: str, port: int, max_wait=10.0, time_btw_check=1.0, server_root=''):
        
        #   start a new TeraChem server if gpus are supplied
        # if len(gpus) != 0:
        #     host = start_TC_server(port, gpus, server_root)
        
        print('Setting up new client')
        client = TCClientExtra(host=host, port=port, server_root=server_root)
        total_wait = 0.0
        avail = False
        while not avail:
            try:
                client.connect()
                client.is_available()
                avail = True
            except:
                print(f'TeraChem server {host}:{port} not available: \n\
                        trying again in {time_btw_check} seconds')
                time.sleep(time_btw_check)
                total_wait += time_btw_check
                if total_wait >= max_wait:
                    raise TimeoutError('Maximum time allotted for checking for TeraChem server')
        print('Terachem server is available and connected')
        return client
    
    @staticmethod
    def restart_client(client: TCClientExtra, max_wait=10.0, time_btw_check=1.0):
        host = client.host
        port = client.port
        server_root = client.server_root
        if (host, port) in _server_processes:
            process: TCServerProcess = _server_processes[(host, port)]
            
            process.kill()
            time.sleep(8.0)
            start_TC_server(port, server_root=server_root, gpus=process.gpus)
        client.disconnect()
        del client
        return TCRunner.get_new_client(host, port, max_wait=max_wait, time_btw_check=time_btw_check, server_root=server_root)

    '''        

    @staticmethod
    def append_results_file(results: dict):
        results_file = os.path.join(results['job_scr_dir'], 'results.dat')
        results['results.dat'] = open(results_file).readlines()
        return results
    
    @staticmethod
    def append_output_file(results: dict, server_root=''):
        output_file = os.path.join(server_root, results['job_dir'], 'tc.out')
        if os.path.isfile(output_file):
            with open(output_file, 'r') as file:
                lines = file.readlines()
            #   remove line breaks
            for n in range(len(lines)):
                lines[n] = lines[n][0:-1]
            results['tc.out'] = lines
        else:
            print("Warning: Output file not found at ", output_file)
            
        return results
    
    @staticmethod
    def remove_previous_job_dir(client: TCClientExtra):
        results = client.prev_results
        TCRunner.remove_previous_scr_dir(client)
        job_dir = results['job_dir']
        shutil.rmtree(job_dir)

    @staticmethod
    def remove_previous_scr_dir(client: TCClientExtra):
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

        return cleaned

    @staticmethod
    def run_TC_single(client: TCClientExtra, geom, atoms: list[str], opts: dict):
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
                time.sleep(0.25)
                accepted = self._client.send_job_async(jobType, geom, unitType, **kwargs)
                end_time = time.time()
                total_time += (end_time - start_time)
                if total_time > max_time and max_time >= 0.0:
                    print("FAILING: ", total_time, max_time)
                    raise TCServerStallError('TeraChem server might have stalled')

            completed = self._client.check_job_complete()
            while completed is False:
                start_time = time.time()
                time.sleep(0.25)
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
        
    def _set_avg_max_times(self, times: dict):
        max_time = np.max(list(times.values()))
        self._max_time_list.append(max_time)
        self._max_time = np.mean(self._max_time_list)*5

    def run_new_geom(self, phase_vars: PhaseVars=None, geom=None) -> ESResults:
        if phase_vars is not None:
            geom = phase_vars.nuc_q/ANG_2_BOHR
            self._phase_var_history.append(phase_vars)
        elif geom is not None:
            #   legacy support for geom, assumed to be in angstroms
            pass
        else:
            raise ValueError('Either phase_vars or geom must be provided')

        try:
            job_batch = self._run_TC_new_geom_kernel(geom)
        except TCServerStallError as error:
            host, port = self._host, self._port
            print('TC Server stalled: attempting to restart server')
            stop_TC_server(host, port)
            time.sleep(2.0)
            start_TC_server(port)

            self._client.restart(max_wait=20)
            print('Started new TC Server: re-running current step')
            job_batch = self.run_new_geom(geom)


        if _SAVE_DEBUG_TRAJ:
            print("SAVING DEBUG TRAJ FILE: ", _SAVE_DEBUG_TRAJ)
            with open(_SAVE_DEBUG_TRAJ, 'wb') as file:
                pickle.dump(self._debug_traj, file)

        all_energies, elecE, grad, nac, trans_dips = format_output_LSCIVR(job_batch.results_list)
        self._prev_job_batch = job_batch
        
        #   DEBUG: correct signs of NACs for the first frame
        if self._initial_ref_nacs is not None:
            print('SETTING INITIAL NACS FROM SETTINGS')
            for i in range(nac.shape[0]):
                for j in range(1, nac.shape[1]):
                    sign = np.sign(np.dot(nac[i, j], self._initial_ref_nacs[i, j]))
                    nac[i, j] = sign*nac[i, j]
                    nac[j, i] = -nac[i, j]
            self._initial_ref_nacs = None

                
        # return (all_energies, elecE, grad, nac, trans_dips, job_batch.timings)
    
        all_energies, energies, grads, nacs, trans_dips = format_output_LSCIVR(job_batch.results_list)
        es_result = ESResults(None, all_energies, energies, grads, nacs, trans_dips, job_batch.timings)

        return es_result
    
    def run_new_geom_LEGACY(self, phase_vars: PhaseVars=None, geom=None):
        res = self.run_new_geom(phase_vars, geom)
        return (res.all_energies, res.elecE, res.grads, res.nacs, res.trans_dips, res.timings)


    def _run_TC_new_geom_kernel(self, geom):
        
        #   frame specific job options
        frame_opts = {}
        if self._initial_frame_options is not None:
            if self._frame_counter < self._initial_frame_options['n_frames']:
                frame_opts.update(self._initial_frame_options)

        #   run energy only if gradients and NACs are not requested
        if len(self._grads) == 0 and len(self._NACs) == 0:
            job_opts = self._base_options.copy()
            results = self.compute_job_sync_with_restart('energy', geom, 'angstrom', **job_opts)
            results['run'] = 'energy'
            results.update(job_opts)
            return TCJobBatch([results])

        #   create gradient jobs
        job_batch = TCJobBatch()
        for name, opts in self.grad_job_options.items():
            state = opts.get(f'{self._excited_type}target', 0)
            job_batch.append(TCJob(geom, opts | frame_opts, 'gradient', self._excited_type, state, name))

        #   create NAC jobs
        for name, opts in self.nac_job_options.items():
            state = max(opts['nacstate1'], opts['nacstate2'])
            job_batch.append(TCJob(geom, opts | frame_opts, 'coupling', self._excited_type, state, name))

        if len(job_batch) == 0:
            raise ValueError('No jobs to run')


        self._interpolation_setup(job_batch)

        job_batch = self._send_jobs_to_clients(job_batch)
        self._frame_counter += 1
        self._prev_jobs = job_batch.jobs

        self._interpolation_finalize(job_batch)

                

        return job_batch
    
    def _interpolation_setup(self, job_batch: TCJobBatch):
        if not self._interpolation:
            return        

        #   first find a job to copy the options from for the energy job
        #   couplings are guarenteed to have excited_options
        ref_job = job_batch.get_by_type('coupling').jobs[0]
        eng_opts = ref_job.opts.copy()
        for x in ('nacstate1', 'nacstate2', 'couplings', 'gradients'):
            eng_opts.pop(x, None)
        eng_results = self._client.compute_job_sync('energy', ref_job.geom, 'angstrom', **eng_opts)
        # eng_results = self.compute_job_sync_with_restart('energy', ref_job.geom, 'angstrom', **eng_opts)
        all_energies = eng_results['energy']
        exciton_overlap = eng_results.get('exciton_overlap', None)
        curr_time = self._phase_var_history.time[-1]
        self._es_history.append(ESResults(time=curr_time, all_energies=all_energies))
        self._probe_job_results = eng_results

        grads_to_run = self._grad_interpolator.get_gradient_states(all_energies, self._es_history[-2].all_energies, self._es_history.time, self._phase_var_history.nuc_p)
        nacs_to_run = self._nac_interpolator.get_nac_states(exciton_overlap, self._es_history.time, self._phase_var_history.nuc_p)

        for g, estimate in self._grad_interpolator.get_guesses(curr_time):
            if not grads_to_run[g][0]:
                job = job_batch.get_by_name(f'gradient_{g}').jobs[0]
                job_batch.jobs.remove(job)
                self._grad_estimates[g] = estimate
            else:
                self._grad_estimates[g] = None

        for x, estimate in self._nac_interpolator.get_guesses(curr_time):
            if not nacs_to_run[x][0]:
                job = job_batch.get_by_name(f'nac_{x[0]}_{x[1]}').jobs[0]
                job_batch.jobs.remove(job)
                self._nac_estimates[x] = estimate
            else:
                self._nac_estimates[x] = None


    def _interpolation_finalize(self, job_batch: TCJobBatch):
        if not self._interpolation:
            return 
        
        # geom = job_batch.get_by_type('coupling').jobs[0].geom
        geom = self._probe_job_results['geom']
        curr_time = self._phase_var_history.time[-1]
        for g, grad_est in self._grad_estimates.items():
            if self._grad_estimates[g] is None:
                job = job_batch.get_by_name(f'gradient_{g}').jobs[0]
                self._grad_interpolator.update_history(g, curr_time, job.results['gradient'])
            else:
                est_job = TCJob(geom, {}, 'gradient_est', self._excited_type, -1, f'gradient_{g}')
                job_batch.jobs.append(est_job)
                results = self._probe_job_results.copy()
                energies = self._probe_job_results['energy']
                if g == 0:
                    energies = energies[0:1]
                results.update({'run': 'gradient_est', 
                                   'energy': energies, 
                                   'gradient': grad_est.reshape((-1, 3)),
                                   f'{self._excited_type}target': g,
                })
                est_job.results = results
        self._grad_interpolator.print_update_messages()

        #   DEBUG
        T = np.zeros((3, 3))
        U = np.zeros((3, 3))
        overlaps = np.zeros((3, 3))

        for x, nac_est in self._nac_estimates.items():
            if self._nac_estimates[x] is None:
                job = job_batch.get_by_name(f'nac_{x[0]}_{x[1]}').jobs[0]
                self._nac_interpolator.update_history(x, curr_time, job.results['nacme'])
            
                #   DEBUG
                x2 = tuple(reversed(x))
                overlaps = np.array(job.results['exciton_overlap'])
                U = overlaps @ la.inv(la.sqrtm(overlaps.T @ overlaps))
                nac = np.array(job.results['nacme'])
                if curr_time == 0:
                    vel = self._phase_var_history.nuc_p[-1]/(self._masses*AMU_2_AU)
                else:
                    vel = self._phase_var_history.nuc_q[-1] - self._phase_var_history.nuc_q[-2]
                T[x] = np.dot(nac.flatten(), vel)
                T[x2] = -T[x]

                mass_coord = self._masses * self._phase_var_history.nuc_q[-1]/np.sum(self._masses)
                com = np.sum(mass_coord.reshape(-1, 3), axis=0)

            else:
                est_job = TCJob(geom, {}, 'coupling_est', self._excited_type, -1, f'nac_{x[0]}_{x[1]}')
                job_batch.jobs.append(est_job)
                results = self._probe_job_results.copy()
                results.update({'run': 'coupling_est', 
                                   'energy': self._es_history.all_energies[-1], 
                                   'cis_transition_dipoles': self._probe_job_results['cis_transition_dipoles'],
                                   'nacme': nac_est.reshape((-1, 3)), 
                                   'nacstate1': x[0],
                                   'nacstate2': x[1],
                })
                est_job.results = results
        self._nac_interpolator.print_update_messages()
        
        #   DEBUG
        # np.save(f'overlaps.{int(curr_time)}.npy', job.results['exciton_overlap'])
        # np.save(f'U.{int(curr_time)}.npy', U)
        # np.save(f'T.{int(curr_time)}.npy', T)


    def _send_jobs_to_clients(self, jobs_batch: TCJobBatch):

        # DEBUG ONLY
        if _SAVE_BATCH:
            batch_file = os.path.join(_SAVE_BATCH, f'_jobs_{jobs_batch.batchID}.pkl')
            if os.path.isfile(batch_file):
                print('DEBUG OVERWRITING JOBS WITH FILE ', f'{batch_file}')
                with open(batch_file, 'rb') as file:
                    completed_batch = pickle.load(file)
                for i in range(len(jobs_batch.jobs)):
                    jobs_batch.jobs[i] = completed_batch.jobs[i]
                return completed_batch
            else:
                print('COULD NOT FIND FILE ', f'{batch_file}')


        #   debug mode
        if self._debug_traj and not _SAVE_DEBUG_TRAJ:
            completed_batch = self._debug_traj.pop(0)

            for i in range(len(jobs_batch.jobs)):
                jobs_batch.jobs[i] = completed_batch.jobs[i]

            # jobs_batch = completed_batch
            n_comp = len(completed_batch.jobs)        
            n_req = len(jobs_batch.jobs)
            if n_comp != n_req:
                raise ValueError(f'DEBUG MODE: Number of jobs requested ({n_req}) does not match the next batch of jobs in the trajectory ({n_comp})')
            time.sleep(0.025)
            return completed_batch
        

        n_clients = len(self._client_list)
        if len(self._tc_client_assignments) > 0:
            all_job_names = [j.name for j in jobs_batch.jobs]
            clients_IDs_for_other = []

            #   assign clients specified in the client_assignments
            
            for i, names in enumerate(self._tc_client_assignments):
                names_list = names.copy()
                if 'other' in names_list:
                    clients_IDs_for_other.append(i)
                    names_list.remove('other')

                sub_jobs = jobs_batch.get_by_name(names_list)
                sub_jobs.set_client(self._client_list[i])
                for name in names_list:
                    if name in all_job_names:
                        all_job_names.remove(name)

            #   everything else gets assigned to the 'other' clients
            if len(clients_IDs_for_other) > 0:
                n_other_clients = len(clients_IDs_for_other)
                for i, name in enumerate(all_job_names):
                    sub_batch = jobs_batch.get_by_name(name)
                    client_id = clients_IDs_for_other[i % n_other_clients]
                    client = self._client_list[client_id]
                    sub_batch.set_client(client)

        else:
            #   equally distribute jobs among clients
            for j in jobs_batch.jobs:
                j.client = self._client_list[j.jobID % n_clients]


        if not jobs_batch.check_client():
            raise ValueError('Not all jobs have been assigned a client')

        if len(jobs_batch.jobs) == 0:
            self._coordinate_exciton_overlap_files()
            return jobs_batch

        #   if only one client is being used, don't open up threads, easier to debug
        if n_clients == 1:
            jobs_batch.jobs[0].client = self._client_list[0]
            _run_batch_jobs(jobs_batch)

        #   submit jobs as separate threads, one per client
        else:
            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = []
                for client in self._client_list:
                    jobs = jobs_batch.get_by_client(client)
                    args = (jobs, )
                    future = executor.submit(_run_batch_jobs, *args)
                    futures.append(future)
                for f in futures:
                    f.result()

        self._prev_results = jobs_batch.results_list
        self._set_avg_max_times(jobs_batch.timings)
        self._coordinate_exciton_overlap_files()

        if _SAVE_DEBUG_TRAJ:
            print("APPENDING DEBUG TRAJ ", jobs_batch)
            jobs_batch._remove_clients()
            self._debug_traj.append(jobs_batch)

        if _SAVE_BATCH:
            print(f"SAVING BATCH FILE: {batch_file}")
            for job in jobs_batch.jobs:
                job.client = None
            with open(batch_file, 'wb') as file:
                pickle.dump(jobs_batch, file)

        return jobs_batch        

    def _coordinate_exciton_overlap_files(self, overlap_file_loc=None, overlap_data=None):
        '''
            Copy a single exciton overlap file to all server roots
        '''

        if self._excited_type != 'cis':
            return

        exciton_overlap_data = None
        if overlap_data is not None:
            exciton_overlap_data = overlap_data
        elif overlap_file_loc is not None:
            with open(overlap_file_loc, 'rb') as file:
                exciton_overlap_data = file.read()
        else:
            for client in self._client_list:
                overlap_file_loc = client.server_file('exciton_overlap.dat')
                if not os.path.isfile(overlap_file_loc):
                    continue
                with open(overlap_file_loc, 'rb') as file:
                    exciton_overlap_data = file.read()
                break


        #   copy file to all other server roots
        if exciton_overlap_data is not None:
            self._exciton_overlap_data = exciton_overlap_data
            for client in self._client_list:
                new_file_loc = client.server_file('exciton_overlap.dat.1')
                with open(new_file_loc, 'wb') as file:
                    file.write(self._exciton_overlap_data)

    def _run_numerical_derivatives(self, ref_job: TCJob, n_points=3, dx=0.01, overlap=False):
        '''
            dx is in bohr
        '''

        base_opts = ref_job.opts.copy()
        num_deriv_jobs = []
        indicies = []
        for n in range(len(ref_job.geom)):
            for i in [0, 1, 2]:
                shift_multiples = (np.arange(n_points) - n_points//2).tolist()
                shift_multiples.pop(n_points//2)
                for j in shift_multiples:
                    indicies.append((n, i, j))

        for n, i, j in indicies:
            new_geom = np.copy(ref_job.geom)
            new_geom[n, i] += j*dx/1.8897259886 # angstrom to bohr
            job = TCJob(new_geom, base_opts, 'energy', ref_job.excited_type, ref_job.state, str((n, i, j)), self._client_list[0])
            num_deriv_jobs.append(job)


        if not _DEBUG:
            batch = self._send_jobs_to_clients([num_deriv_jobs])
        else:
            if os.path.isfile('_num_deriv_jobs.pkl'):
                with open('_num_deriv_jobs.pkl', 'rb') as file: num_deriv_jobs = pickle.load(file)
            else:
                _run_batch_jobs(num_deriv_jobs, self._server_root_list[0], 0, self._prev_results)
                self._send_jobs_to_clients(num_deriv_jobs)
                with open('_num_deriv_jobs.pkl', 'wb') as file: pickle.dump(num_deriv_jobs, file)        

        if overlap:
            overlap_jobs = []
            job_pairs = []
            for deriv_job in num_deriv_jobs:
                if ref_job.excited_type != 'cas':
                    raise ValueError('Wavefunction overlaps are only available for CAS methods in TeraChem')
                opts = copy.copy(base_opts)
                job_results = deriv_job.results
                ref_results = ref_job.results
                opts['cvec1file'] = os.path.join(self._server_root_list[0], ref_results["job_scr_dir"], "CIvecs.Singlet.dat")
                opts['cvec2file'] = os.path.join(self._server_root_list[0], job_results["job_scr_dir"], "CIvecs.Singlet.dat")
                opts['orb1afile'] = os.path.join(self._server_root_list[0], ref_results["job_scr_dir"], "c0")
                opts['orb2afile'] = os.path.join(self._server_root_list[0], job_results["job_scr_dir"], "c0")
                opts['old_coors'] = os.path.join(self._server_root_list[0], ref_results["job_dir"], "geom.xyz")
                job = TCJob(job.geom, opts, 'ci_vec_overlap', ref_job.excited_type, ref_job.state, f'overlap_{deriv_job.name}', self._client_list[0])
                overlap_jobs.append(job)
                # deriv_to_overlap[deriv_job.jobID] = job.jobID
                job_pairs.append((deriv_job, job))

            if not _DEBUG:
                _run_batch_jobs(overlap_jobs, self._server_root_list[0], 0, self._prev_results)
            else:
                if os.path.isfile('_overlap_pairs.pkl'):
                    with open('_overlap_pairs.pkl', 'rb') as file: job_pairs = pickle.load(file)
                else:
                    _run_batch_jobs(overlap_jobs, self._server_root_list[0], 0, self._prev_results)
                    with open('_overlap_pairs.pkl', 'wb') as file: pickle.dump(job_pairs, file)
            

            #   make sure each returned job is in the same order as in job_pairs
            return_dict = {'num_deriv_jobs': [], 'overlap_jobs': []}
            for deriv_job, overlap_job in job_pairs:
                return_dict['num_deriv_jobs'].append(deriv_job)
                return_dict['overlap_jobs'].append(overlap_job)
            return return_dict
        
        else:
            return_dict = {'num_deriv_jobs': num_deriv_jobs, 'overlap_jobs': None}
            return return_dict

def _correct_signs_from_overlaps(job: TCJob, overlap_job: TCJob):
    '''
        Correct the sings of transition dipole moments and NACs based 
        on the overlaps with previous jobs.
    '''

    signs = np.sign(np.diag(overlap_job.results['ci_overlap']))
    n_states = len(job.results['energy'])

    #   correct the transition dipole moments
    for k in ['cas', 'cis']:
        key = f'{k}_transition_dipoles' 
        if key not in job.results:
            continue
        count = 0
        for i in range(n_states):
            for j in range(i+1, n_states):
                dipole = job.results[key][count]
                job.results[key][count] = (np.array(dipole)*signs[i]*signs[j]).tolist()
                count += 1

    #   correct the nonadibatic coupling
    if 'nacme' in job.results:
        idx1 = job.results['nacstate1']
        idx2 = job.results['nacstate2']
        job.results['nacme'] = (np.array(job.results['nacme'])*signs[idx1]*signs[idx2]).tolist()
    
def _correct_signs(job: TCJob, ref_job: TCJob):
    '''
        Correct the sings of transition dipole moments and NACs based 
        on the overlaps with previous jobs.
    '''
    _debug_print = False

    n_states = len(job.results['energy'])

    if 'cas_tr_resp_charges' in job.results:
        quant_key = 'cas_tr_resp_charges'
        exc_type = 'cas'
    elif 'cis_tr_resp_charges' in job.results:
        quant_key = 'cis_tr_resp_charges'
        exc_type = 'cis'
    elif 'cas_transition_dipoles' in job.results:
        quant_key = 'cas_transition_dipoles'
        exc_type = 'cas'
    elif 'cis_transition_dipoles' in job.results:
        quant_key = 'cis_transition_dipoles'
        exc_type = 'cis'
    else:
        warnings.warn(f'No transition charges or dipoles found in job {job}, cannot correct transition charges')
        # for key in job.results:
        #     print('    ', key)
        # input()
        return
    

    # if 'cas_tr_resp_charges' in job.results:
    #     exc_type = 'cas'
    # elif 'cis_tr_resp_charges' in job.results:
    #     exc_type = 'cis'
    # else:
    #     warnings.warn(f'cas_tr_resp_charges or cis_tr_resp_charges not found in job {job}, cannot correct transition charges')
    #     return

    #   Get the correct dipole key. Some CAS jobs don't have an 's' at the end
    # dipole_key = None
    dipole_key = f'{exc_type}_transition_dipoles'
    charge_key = f'{exc_type}_tr_resp_charges'
    # if f'{exc_type}_transition_dipole' in job.results:
    #     dipole_key = f'{exc_type}_transition_dipole'

    # elif f'{exc_type}_transition_dipoles' in job.results:
    #     dipole_key = f'{exc_type}_transition_dipoles' 

    # charges_key = f'{exc_type}_tr_resp_charges'
    quant_orig = np.array(job.results.get(quant_key, []))
    quant_ref = np.array(ref_job.results.get(quant_key, []))
    if quant_ref.shape[0] == 0:
        raise ValueError(f'{quant_key} not found in reference job {job}')

    min_RRMSE_all = []
    min_RRMSE = 1e20
    min_signs = None
    combos = itertools.combinations_with_replacement(range(n_states), n_states)
    used_signs = []

    #   we exhaustively search for the combination of sign flips that minimizes
    #   the agreement with the reference job. This is not a problem for only a
    #   small number of states, but will quickly become expensive with many states
    for c in combos:
        c_unique = set(c)

        used_signs.append(c_unique)

        #   each indix indicates a the state that is negated
        signs = np.ones(n_states, dtype=int)
        for i in c_unique:
            signs[i] = -1

        RRMSE = 0.0
        count = 0
        for i in range(0, n_states):
            for j in range(i+1, n_states):
                q_new = quant_orig[count]*signs[i]*signs[j]
                q_ref = quant_ref[count]
                RRMSE += np.sqrt(np.mean((q_new - q_ref)**2) / np.mean(q_ref**2))
                count += 1

        if RRMSE < 1e-04 and _debug_print:
            print('Found Low RRMSE: ', RRMSE, signs)
            count = 0
            for i in range(0, n_states):
                for j in range(i+1, n_states):
                    q_new = quant_orig[count]*signs[i]*signs[j]
                    q_ref = quant_ref[count]
                    print(  f'{i} {j} {q_ref} {q_new}')
                    count += 1



        min_RRMSE_all.append((signs, RRMSE))
        if RRMSE < min_RRMSE:
            min_signs = signs.copy()
            min_RRMSE = RRMSE


    if _debug_print:
        print('Min Signs: ', min_signs)
        print("Transition dipoles BEFORE sign corrections")
        for i in range(len(job.results[dipole_key])):
            fmt_str = '{:10.6f} '*3 + '|' + '{:10.6f} '*3 + '|' + '{:10.6f}'
            ref_d, job_d = ref_job.results[dipole_key][i], job.results[dipole_key][i]
            proj = np.dot(ref_d, job_d)/(np.linalg.norm(ref_d)*np.linalg.norm(job_d))
            print(fmt_str.format(*ref_d, *job_d, proj))

    #   correct the transition charges, dipoles, and dipole derivatives
    for key in [dipole_key, charge_key, 'cis_transition_dipole_deriv']:
        count = 0
        if key not in job.results:
            continue
        for i in range(0, n_states):
            for j in range(i+1, n_states):
                orig_val = np.array(job.results[key][count])
                job.results[key][count] = orig_val*min_signs[i]*min_signs[j]
                count += 1
                
    if _debug_print:
        print("Transition dipoles AFTER sign corrections")
        print("minimum RRMSE: ", min_RRMSE, min_signs)
        for i in range(len(job.results[dipole_key])):
            ref_d, job_d = ref_job.results[dipole_key][i], job.results[dipole_key][i]
            proj = np.dot(ref_d, job_d)/(np.linalg.norm(ref_d)*np.linalg.norm(job_d))
            print(fmt_str.format(*ref_d, *job_d, proj))


    #   correct the nonadibatic coupling
    if 'nacme' in job.results:
        idx1 = job.results['nacstate1']
        idx2 = job.results['nacstate2']
        job.results['nacme'] = (np.array(job.results['nacme'])*min_signs[idx1]*min_signs[idx2]).tolist()

    

def _run_batch_jobs(jobs_batch: TCJobBatch):
    for j in jobs_batch.jobs:
        j: TCJob

        client: TCClientExtra = j.client
        client.log_message(f"Running {j.name}")
        print(f"Running {j.name}")

        max_tries = 5
        try_count = 0
        try_again = True
        while try_again:
            try:
                # results = client.compute_job_sync(j.job_type, j.geom, 'angstrom', **job_opts)
                client.compute_job(j)
                try_again = False
            except Exception as e:
                try_count += 1
                if try_count == max_tries:
                    try_again = False
                    print(e)
                    print("Server error recieved")
                    client.print_end_of_file()
                    print("    Will not try again")
                    exit()
                else:
                    try_again = True
                    print(e)
                    current_time = datetime.now()
                    formatted_time = current_time.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3] 
                    print(f"Server error recieved at {formatted_time} on client {client}\n")
                    client.print_end_of_file()
                    print("    Trying to run job once more")
                    client.log_message(f"Server error recieved; trying to run job once more")
                    time.sleep(10)
                    client.restart()

        TCRunner.append_output_file(j.results, client.server_root)


def format_output_LSCIVR(job_data: list[dict]):
    atoms = job_data[0]['atoms']
    n_atoms = len(atoms)
    
    energies = {}
    all_energies = []
    grads = {}
    nacs = {}
    trans_dips = {}
    for job in job_data:
        all_energies = job['energy']
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

            if 'cis_transition_dipoles' in job:
                x = len(job['cis_transition_dipoles'])
                N = int((1 + int(np.sqrt(1+8*x)))/2)
                count = 0
                for i in range(0, N):
                    for j in range(i+1, N):
                        if (i, j) == (state_1, state_2):
                            trans_dips[(state_1, state_2)] = job['cis_transition_dipoles'][count]
                            trans_dips[(state_2, state_1)] = job['cis_transition_dipoles'][count]
                        count += 1
        elif job['run'] == 'gradient_est':
            state = job.get('cistarget', job.get('castarget', 0))
            grads[state] = np.array(job['gradient']).flatten()
            energies[state] = job['energy'][state]

        elif job['run'] == 'coupling_est':
            state_1 = job['nacstate1']
            state_2 = job['nacstate2']
            nacs[(state_1, state_2)] = np.array(job['nacme']).flatten()
            nacs[(state_2, state_1)] = - nacs[state_1, state_2]
            if 'cis_transition_dipoles' in job:
                x = len(job['cis_transition_dipoles'])
                N = int((1 + int(np.sqrt(1+8*x)))/2)
                count = 0
                for i in range(0, N):
                    for j in range(i+1, N):
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
    # print(" --------------------------------")
    # print(" LSC-IVR to TeraChem")
    # print(" state number mapping")
    # print(" ---------------------------------")
    # print(" LSC-IVR -->   QC  ")
    grads_in_order = sorted(list(grads.keys()))
    for i in range(n_states):
        qc_i = grads_in_order[i]
        # print(f"   {i:2d}    -->  {qc_i:2d}")
        ivr_grads[i] = grads[qc_i]
        ivr_energies[i] = energies[qc_i]
        for j in range(n_states):
            if i <= j:
                continue
            qc_idx_j = grads_in_order[j]
            ivr_nacs[i, j] = nacs[(qc_i, qc_idx_j)]
            ivr_nacs[j, i] = nacs[(qc_idx_j, qc_i)]

            td = trans_dips.get((qc_i, qc_idx_j), None)
            if td is not None:
                ivr_trans_dips[i, j] = td
                ivr_trans_dips[j, i] = td
            else:
                ivr_trans_dips = None
    # print(" ---------------------------------")

    return all_energies, ivr_energies, ivr_grads, ivr_nacs, ivr_trans_dips
