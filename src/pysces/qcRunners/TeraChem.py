from __future__ import annotations 

from tcpb import TCProtobufClient as TCPBClient
from tcpb.exceptions import ServerError
from tcpb import terachem_server_pb2 as pb

from pysces.input_simulation import logging_dir, TCRunnerOptions
from pysces.common import PhaseVars, QCRunner
from pysces.qcRunners.LoadBalancing import balance_tasks_optimum, ESDerivTasks, ServerBenchmark
from pysces.h5file import H5File, H5Group, h5py
from pysces.fileIO import BaseLogger, LoggerData

_HAS_TCPARSE = True
try:
    from tcparse import parse_from_list, TCJobData
except ImportError:
    _HAS_TCPARSE = False

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
from typing import Literal
import itertools
from collections import deque
import qcelemental as qcel
import base64
from pprint import pprint


_server_processes = {}

#   debug flags
_DEBUG = bool(int(os.environ.get('DEBUG', False))) # used with numerical derivatives
_DEBUG_LOAD_TRAJ = os.environ.get('DEBUG_LOAD_TRAJ', False)
_DEBUG_SAVE_TRAJ = os.environ.get('DEBUG_SAVE_TRAJ', False)


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

    def __init__(self, host: str = "127.0.0.1", port: int = 11111, debug=False, trace=False, log=True, server_root='.'):
        """
        Initializes a TCClientExtra object.

        Args:
            host (str): The host IP address. Defaults to "127.0.0.1".
            port (int): Port number (must be above 1023). Defaults to 11111.
            debug (bool): If True, assumes connections work (used for testing with no server). Defaults to False.
            trace (bool): If True, packets are saved to .bin files (which can then be used for testing). Defaults to False.
            log (bool): Whether to enable logging. Defaults to True.
        """
        
        
        self._log = None
        if log and os.path.isdir(logging_dir):
            log_file_loc = os.path.join(logging_dir, f'{host}_{port}.log')
            self._log = open(log_file_loc, 'a')
            self.log_message(f'Client started on {host}:{port}')
        self.server_root = server_root
        self._possible_files_to_remove: set[str] = {'exciton_overlap.dat', 'exciton_overlap.dat.1', 'exciton.dat', 'cispropertyfile', 'scf_guess', 'cas_guess', 'cis_guess'}
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
            self.remove_file(file, raise_error=False)

    def __exit__(self):
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
                        trying again in {time_btw_check} seconds')
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
        return self.get_dir_loc(self.curr_job_dir)
    
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
        if prev_job.state > 0 and prev_job.excited_type == 'cas':
            for prev_job_res in reversed(res_history):
                if prev_job_res.get('castarget', 0) < 1:
                    continue
                prev_orb_file = prev_job_res['orbfile']
                if prev_orb_file[-6:] != 'casscf':
                    continue
                # cas_guess = os.path.join(server_root, prev_orb_file)
                cas_guess = self.get_file_loc(prev_orb_file)
                break


        for i, prev_job_res in enumerate(reversed(res_history)):
            if 'orbfile' in prev_job_res:
                # scf_guess = os.path.join(server_root, prev_job_res['orbfile'])
                scf_guess = self.get_file_loc(prev_job_res['orbfile'])
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

    def print_end_of_file(self, n_lines=30):
        lines  = []

        if self.curr_job_dir is not None:
            job_dir = self.get_dir_loc(self.curr_job_dir)
        elif self._last_known_curr_dir is not None:
            job_dir = self.get_dir_loc(self._last_known_curr_dir)
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

    def compute_job(self, job: TCJob, append_tc_out=False, use_guess_files=True, start_fresh=False):
        
        if start_fresh:
            self.clean_up_stale_files()

        #   assign the guess files to the current job
        if use_guess_files:
            self.assign_guess_files(job)

        #   send the job to the terachem server
        job.start_time = time.time()
        results = self.compute_job_sync(job.job_type, job.geom, 'angstrom', **job.opts)
        job.end_time = time.time()

        #   set the results of the job
        results['run'] = job.job_type
        results.update(job.opts)
        if append_tc_out:
            self._append_output_file(results)
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
            accepted = self.send_job_async(jobType, geom, unitType, **kwargs)
            time.sleep(0.1)

        self.log_message("Job Accepted")
        self.log_message(f"    Job Type: {jobType}")
        self.log_message(f"    Current Job Dir: {self.get_curr_job_dir()}")

        completed = self.check_job_complete()
        while completed is False:
            time.sleep(0.1)
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
            if not self.is_file(file_loc): 
                out_str = f'{file_loc} file not found in server root directory\n'
                out_str += 'Current files in server root directory:\n'
                out_str += f'    {self.server_root}\n'
                for x in os.listdir(self.server_root):
                    out_str += f'        {x}\n'
                exit(out_str)
        
        if opts.get('cisexcitonoverlap', 'no') == 'yes' and opts.get('cis', 'no') == 'yes':
            self.expect_exciton_overlap_dat = True


            #   make sure the files exist, exit if it does not
            _ensure_file_exists('exciton_overlap.dat')
    
            #   read in previous data
            self._exciton_overlap_data = self.get_file('exciton_overlap.dat')

            #   if exciton_overlap_file_1 also exists, then exciton.dat must have been created
            if self.is_file('exciton_overlap.dat.1'):
                _ensure_file_exists('exciton.dat')
                overlap_data = []

                exciton_data = self.get_file('exciton.dat', 'r')
                for line in exciton_data.splitlines():
                    sp = line.split()
                    if sp[0] == 'Overlap:':
                        data = [float(x) for x in sp[1:]]
                        overlap_data.append(data)
                overlap_data = np.array(overlap_data)
                self._exciton_data = overlap_data
                results['exciton_overlap'] = overlap_data
                self.remove_file('exciton.dat')
            else:
                #   exciton_overlap.dat exists but exciton_overlap.dat.1 does not
                #   this occus the first time exciton_overlap.dat is read by TeraChem, so we
                #   assume that it's the first frame that does so. 
                pass
            self.rename_file('exciton_overlap.dat', 'exciton_overlap.dat.1')

            if opts.get('cisrestart', None):
                self._possible_files_to_remove.add(opts.get('cisrestart', None))

            #   attempt to make sure the file system is updated
            dir_fd = os.open(self.server_root, os.O_DIRECTORY)
            os.fsync(dir_fd)
            os.close(dir_fd)

        os.listdir(self.server_root)

    def _append_output_file(self, results: dict):
        output_file = os.path.join(self.server_root, results['job_dir'], 'tc.out')
        if os.path.isfile(output_file):
            with open(output_file, 'r') as file:
                lines = file.readlines()
            #   remove line breaks
            for n in range(len(lines)):
                lines[n] = lines[n][0:-1]
            results['tc.out'] = lines
        else:
            print("Warning: Output file not found at ", output_file)

    def clean_up_stale_files(self):
        for file in ['exciton.dat', 'exciton_overlap.dat', 'exciton_overlap.dat.1']:
            if self.remove_file(file, raise_error=False):
                print('Removing stale file:', file)
        for file in os.listdir(self.server_root):
            if 'XYZia_CPCIS_' in os.path.basename(file):
                print('Removing stale file:', file)
                os.remove(os.path.join(self.server_root, file))
            if 'XYZia_CPHF_' in os.path.basename(file):
                print('Removing stale file:', file)
                os.remove(os.path.join(self.server_root, file))


    def _convert_file_path(self, file_name):
        if not os.path.isabs(file_name):
            return os.path.join(self.server_root, file_name)
        else:
            return file_name

    def is_file(self, file_name):
        return os.path.isfile(os.path.join(self.server_root, file_name))

    def get_file_loc(self, file_name):
        file_loc = self._convert_file_path(file_name)

        if os.path.isfile(file_loc):
            return file_loc
        else:
            raise FileNotFoundError(f'File {file_name} not found in server root directory {self.server_root}')

    def get_dir_loc(self, dir_name):
        dir_loc = self._convert_file_path(dir_name)
            
        if os.path.isdir(dir_loc):
            return dir_loc
        else:
            raise FileNotFoundError(f'Directory {dir_name} not found in server root directory {self.server_root}')
            
    def get_file(self, file_name, mode='rb'):
        file_loc = self.get_file_loc(file_name)
        with open(file_loc, mode) as file:
            data = file.read()
        return data

    def set_file(self, file_name, data, mode='wb'):
        file_loc = self._convert_file_path(file_name)
            
        with open(file_loc, mode) as file:
            file.write(data)

    def remove_file(self, file_name, raise_error=True):
        file_loc = self._convert_file_path(file_name)
        if os.path.isfile(file_loc):
            os.remove(file_loc)
            return True
        elif raise_error:
            raise FileNotFoundError(f'File {file_name} not found in server root directory {self.server_root}')
        return False

    def rename_file(self, old_name, new_name):
        old_file_loc = self._convert_file_path(old_name)
        new_file_loc = os.path.join(self.server_root, new_name)
        if os.path.isfile(old_file_loc):
            os.rename(old_file_loc, new_file_loc)
        else:
            raise FileNotFoundError(f'File {old_name} not found in server root directory {self.server_root}')


class TCCLientExtraDebug(TCClientExtra):

    debug_data: dict[int, dict]= None

    def __init__(self, host: str = "127.0.0.1", port: int = 11111, debug=False, trace=False, log=True, server_root='.'):
        super().__init__(host, port, debug, trace, log, server_root)

    def startup(self, max_wait=10, time_btw_check=1):
        print('DEBUG MODE: Setting up new client ', self.host, self.port)

    def compute_job(self, job: TCJob, append_tc_out=False, use_guess_files=True, start_fresh=False):
        print('DEBUG MODE: Computing job')

        results = self.debug_data[job.jobID].results

        if start_fresh:
            self.clean_up_stale_files()

        #   assign the guess files to the current job
        if use_guess_files:
            self.assign_guess_files(job)

        #   set the results of the job
        results['run'] = job.job_type
        results.update(job.opts)
        if append_tc_out:
            self._append_output_file(results)
        job.results = results.copy()
        self.prev_job = job

        #   set the guess files for the next job
        self.set_guess_files_from_job(job)

        return results


    

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
        self.geom = np.array(geom)
        self.excited_type: Literal['cas', 'cis'] = excited_type
        self.opts = dict(opts)
        self.job_type = job_type
        self.state = state
        self.name = name
        self.start_time = 0
        self.end_time = 0
        self._results = {}
        self.client: TCClientExtra = client

        if job_type not in ['energy', 'gradient', 'coupling']:
            raise ValueError('TCJob job_type must be either "energy", "gradient", or "coupling"')

        assert self.excited_type in ('cas', 'cis')

        TCJob.__job_counter += 1
        self.__jobID = TCJob.__job_counter

    def __hash__(self) -> int:
        return hash(self.__jobID)
    
    def __eq__(self, other) -> bool:
        if not isinstance(other, TCJob):
            return False
        return self.__jobID == other.__jobID

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

class TCJobBatch():
    '''
        Combines multiple TCJobs into one wrapper
    '''
    __batch_counter = 0
    def __init__(self, jobs: list[TCJob] | list['TCJobBatch'] = []) -> None:
        if len(jobs) == 0:
            self.jobs = []
        
        if len(jobs) > 0:
            if isinstance(jobs[0], TCJobBatch):
                jobs = list(itertools.chain(*[j.jobs for j in jobs]))
        
        #   sort by ID
        id_list = [j.jobID for j in jobs]
        order = np.argsort(id_list)
        self.jobs: list[TCJob] = [jobs[i] for i in order]

        TCJobBatch.__batch_counter += 1
        self.__batchID = TCJobBatch.__batch_counter

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
        state['__batchID'] = self.__batchID
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        #   not sure if this is needed yet
        # TCJobBatch.__batch_counter = max(TCJobBatch.__batch_counter, self.__batchID)

    def __repr__(self) -> str:
        out_str = ''
        for j in self.jobs:
            out_str += f'{j}\n'
        return out_str
    
    def __len__(self):
        return len(self.jobs)

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
        return self.__batchID

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
    def __init__(self, atoms: list[str], tc_opts: TCRunnerOptions, max_wait=20) -> None:
        super().__init__()
        
        # Atoms and max_wait
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
        self._setup_servers_and_clients(tc_opts)  # Set up servers and clients

        # Guesses for SCF/CAS/CI calculations
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
        self._debug_traj = {}
        self._load_debug_trajectory()  # Load debug trajectory if applicable
        self._initial_ref_nacs = tc_opts._initial_ref_nacs

        # Print options summary
        self._print_options_summary()
        self._coordinate_exciton_overlap_files(tc_opts.fname_exciton_overlap_data)
        # time.sleep(60)

        #   interpolation options
        self._interpolate_grads = False
        self._interpolate_NACs = False

        #   use condensed jobs
        self._combine_jobs = _HAS_TCPARSE
        
        #   logging
        self._logger = None
        
        #   load balancing
        self._task_benchmarks = self._get_default_task_benchmarks()
        
    def _get_default_task_benchmarks(self):
        # single_benchmark = ServerBenchmark(GS_grad=0.23, EX_grad=2.22, GS_EX_NAC=2.01, EX_EX_NAC=3.22, GS_dipole=6.93, EX_dipole=46.07, GS_EX_dipole=39.32)
        single_benchmark = ServerBenchmark(GS_grad=0.12, EX_grad=1.16, GS_EX_NAC=1.05, EX_EX_NAC=1.68, GS_dipole=3.61, EX_dipole=23.98, GS_EX_dipole=20.46, EX_EX_dipole=47.95)
        max_n_benchmark = max(1, len(self._client_list)) # to ensure debug loading works
        benchmarks = [single_benchmark for _ in range(max_n_benchmark)]
        for i, bm in enumerate(benchmarks):
            bm.name = f'Server_{i}'
            bm.address = f'{self._hosts[i]}'
            bm.port = self._ports[i]
        
        return benchmarks

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
            excited_options['cisexcitonoverlap'] = 'yes'
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

        if max_state > 0 and excited_type == 'cis':
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
            # if _DEBUG_LOAD_TRAJ:
            #     # self._client_list = [None]*len(self._hosts)
            #     print('DEBUG_TRAJ set, TeraChem clients will not be opened')
            #     break

            if _DEBUG_LOAD_TRAJ:
                client = TCCLientExtraDebug(h, p, s)
            else:
                client = TCClientExtra(host=h, port=p, server_root=s)
            client.startup(max_wait=self._max_wait)
            self._client_list.append(client)
            client.clean_up_stale_files()

    def _load_debug_trajectory(self):
        """Load debug trajectory if _DEBUG_TRAJ is set."""
        if _DEBUG_LOAD_TRAJ:
            with open(_DEBUG_LOAD_TRAJ, 'rb') as file:
                self._debug_traj = pickle.load(file)

        TCCLientExtraDebug.debug_data = self._debug_traj

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

    def set_logger_file(self, h5_file: H5File):
        if self._combine_jobs:
            self._logger = TCJobsLogger(None, h5_file)
        else:
            self._logger = TCJobsLoggerSequential(None, h5_file)

    def report(self):
        return self._prev_job_batch

    def cleanup(self):
        if _DEBUG_SAVE_TRAJ:
            with open(_DEBUG_SAVE_TRAJ, 'wb') as file:
                pickle.dump(self._debug_traj, file)
        if _DEBUG_LOAD_TRAJ:
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

        client = self._client_list[0]
        max_time = self._max_time
        if max_time is None and self._restart_on_stall:
            return client.compute_job_sync(jobType, geom, unitType, **kwargs)
        else:
            total_time = 0.0
            accepted = client.send_job_async(jobType, geom, unitType, **kwargs)
            while accepted is False:
                start_time = time.time()
                time.sleep(0.25)
                accepted = client.send_job_async(jobType, geom, unitType, **kwargs)
                end_time = time.time()
                total_time += (end_time - start_time)
                if total_time > max_time and max_time >= 0.0:
                    print("FAILING: ", total_time, max_time)
                    raise TCServerStallError('TeraChem server might have stalled')

            completed = client.check_job_complete()
            while completed is False:
                start_time = time.time()
                time.sleep(0.25)
                completed = client.check_job_complete()
                
                # if self._n_calls == 2:
                #     max_time = 12
                #     print("STALLING: ", total_time, max_time)
                #     completed = False
                # else:
                #     completed = client.check_job_complete()

                end_time = time.time()
                total_time += (end_time - start_time)
                if total_time > max_time and max_time >= 0.0:
                    print("FAILING: ", total_time, max_time)
                    raise TCServerStallError('TeraChem server might have stalled')

            return client.recv_job_async()
        
    def _set_avg_max_times(self, times: dict):
        max_time = np.max(list(times.values()))
        self._max_time_list.append(max_time)
        self._max_time = np.mean(self._max_time_list)*5

    def _run_with_restart(self, geom):
        ''' Legacy: This is here to handle server stalls. It's very likely this will not be needed again'''
        try:
            job_batch = self._run_TC_new_geom_kernel(geom)
        except TCServerStallError as error:
            print('TC Server stalled: attempting to restart server')
            stop_TC_server(self._hosts[0], self._ports[0])
            time.sleep(2.0)
            start_TC_server(self._ports[0])

            self._client_list[0].restart(max_wait=20)
            print('Started new TC Server: re-running current step')
            job_batch = self.run_new_geom(geom)

    def run_new_geom(self, phase_vars: PhaseVars=None, geom=None):

        if phase_vars is not None:
            geom = phase_vars.nuc_q*qcel.constants.bohr2angstroms
        elif geom is not None:
            #   legacy support for geom, assumed to be in angstroms
            pass
        else:
            raise ValueError('Either phase_vars or geom must be provided')


        #   step 1
        job_batch = self.create_jobs(geom, False, self._grads, self._NACs)
        job_batch = self._send_jobs_to_clients(job_batch)

        #   step 2
        if self._interpolate_grads or self._interpolate_NACs:
            new_grads, new_nacs = self._check_new_grads_nacs_to_run(job_batch)
            job_batch_2 = self.create_jobs(geom, False, new_nacs, new_grads)
            job_batch_2 = self._send_jobs_to_clients(job_batch_2)

            #   step 3: combine both batches
            job_batch.jobs += job_batch_2.jobs

        self._log_jobs(job_batch, phase_vars.time)

        all_energies, elecE, grad, nac, trans_dips, mu_deriv_matrix = self._extract_results(job_batch)
        self._initialize_nac_sign(nac)
        self._finalize_frame(job_batch)

        return (all_energies, elecE, grad, nac, trans_dips, job_batch.timings)
    
    # def set_logger():

    def _log_jobs(self, job_batch: TCJobBatch, time):
        if self._logger is None:
            return
        
        data_to_save = []
        if isinstance(self._logger, TCJobsLogger):
            for j in job_batch.jobs:
                res = dict(j.results)
                res['timestep'] = time
                data_to_save.append(res)
            
            self._logger.write(data_to_save)
        elif isinstance(self._logger, TCJobsLoggerSequential):
            self._logger.write(job_batch, time)
        
        self._print_timings(job_batch)

    def _print_timings(self, job_batch: TCJobBatch):
        #   print timings
        print()
        print('Terachem Job Timings:\n')
        print('   Job        Client              Name        Time (s)')
        print('--------------------------------------------------------')
        for i, job in enumerate(job_batch.jobs):
            address = f'{job.client.host}:{job.client.port}'
            print(f'  {i:3d}   {address:22s}  {job.name:10s}   {job.total_time:8.3f}')
        print()


    def _extract_results(self, job_batch: TCJobBatch):
        if self._combine_jobs:
            return format_combo_job_results(job_batch.results_list, self._grads)
        else:
            return format_output_LSCIVR(job_batch.results_list)
        
    def _initialize_nac_sign(self, nac: np.ndarray):
        #   TODO: split this into two functions
        if (nac is not None) and (self._initial_ref_nacs is not None):
            print('SETTING INITIAL NACS FROM SETTINGS')
            for i in range(nac.shape[0]):
                for j in range(1, nac.shape[1]):
                    sign = np.sign(np.dot(nac[i, j], self._initial_ref_nacs[i, j]))
                    nac[i, j] = sign*nac[i, j]
                    nac[j, i] = -nac[i, j]
            self._initial_ref_nacs = None


    def _finalize_frame(self, job_batch: TCJobBatch):
        self._prev_job_batch = job_batch
        self._frame_counter += 1
        self._prev_jobs = job_batch.jobs

    def _check_initial_grads_nacs_to_run(self):
        if not self._interpolate_grads and not self._interpolate_NACs:
            return self._grads, self._NACs
        else:
            raise NotImplementedError('Interpolation of gradients and NACs is not yet implemented')

    def _check_new_grads_nacs_to_run(self, job_batch: TCJobBatch):
        if not self._interpolate_grads and not self._interpolate_NACs:
            return (), ()
        else:
            raise NotImplementedError('Interpolation of gradients and NACs is not yet implemented')

    def create_jobs(self, geom, energy_only = False, grads = [], nacs = [], dipoles = [], tr_dipoles = []):
        if self._combine_jobs:
            return self._create_jobs_bulk(geom, energy_only, grads, nacs, dipoles, tr_dipoles)
        else:
            return self._create_jobs_singles(geom, energy_only, grads, nacs)

    def _create_jobs_bulk(self, geom, energy_only = False, grads: list[int]=[], nacs: list[int, int]=[], dipoles: list[int]=[], tr_dipoles: list[int, int]=[]):

        if self._excited_type != 'cis':
            raise ValueError('Bulk jobs are only supported for CIS excited state calculations')
        
        job_batch = TCJobBatch()

        #   run the job balancing algorithm
        es_tasks = ESDerivTasks(grads, nacs, dipoles, tr_dipoles)
        balanced = balance_tasks_optimum(self._task_benchmarks, es_tasks, len(self._client_list))
        self._tc_client_assignments = [[f'combo_{i}'] for i in range(len(self._client_list))]

        #   distribute the balanced tasks across all clients
        for i, client in enumerate(self._client_list):
            client_tasks = balanced[i]
            job = TCJob(geom, self._base_options, 'energy', self._excited_type, 0, name=f'combo_{i}')
            prop_file_contents = ''

            #   handle ground state gradient differently
            for state in client_tasks.gs_grad:
                job.job_type = 'gradient'
                prop_file_contents += f'gradient {state}\n'

            for state in client_tasks.ex_grads:
                job.opts.update({'cis': 'yes', 'cisgradients': 'yes'})
                prop_file_contents += f'gradient {state}\n'

            for state_1, state_2 in client_tasks.gs_ex_nacs:
                job.opts.update({'cis': 'yes', 'ciscouplings': 'yes'})
                prop_file_contents += f'coupling {state_1} {state_2}\n'

            for state_1, state_2 in client_tasks.ex_ex_nacs:
                job.opts.update({'cis': 'yes', 'ciscouplings': 'yes'})
                prop_file_contents += f'coupling {state_1} {state_2}\n'

            for state in client_tasks.gs_dipole_grad:
                job.opts.update({'cis': 'yes', 'dipolederivative': 'yes'})
                # prop_file_contents += f'dipolederivative {state}\n'

            for state in client_tasks.ex_dipole_grads:
                job.opts.update({'cis': 'yes', 'cisdipolederiv': 'yes'})
                prop_file_contents += f'dipolederiv {state}\n'

            for state_1, state_2 in client_tasks.gs_ex_dipole_grads:
                job.opts.update({'cis': 'yes', 'cistransdipolederiv': 'yes'})
                prop_file_contents += f'transdipolederiv {state_1} {state_2}\n'

            for state_1, state_2 in client_tasks.ex_ex_dipole_grads:
                job.opts.update({'cis': 'yes', 'cistransdipolederiv': 'yes'})
                prop_file_contents += f'transdipolederiv {state_1} {state_2}\n'

            #   add in the rest of the excited state options
            job.opts.update(self._excited_options)

            self._apply_initial_frame_options(job)
            client.set_file('cispropertyfile', prop_file_contents, 'w')
            job.opts['cispropertyfile'] = client.get_file_loc('cispropertyfile')

            job_batch.append(job)

        return job_batch
        

    def _create_jobs_singles(self, geom, energy_only = False, grads = [], nacs = []):
        job_batch = TCJobBatch()

        #   run energy only if gradients and NACs are not requested
        if energy_only:
            job_batch.append(TCJob(geom, self._base_options, 'energy', self._excited_type, 0))

        for name, opts in self.grad_job_options.items():
            state = opts.get(f'{self._excited_type}target', 0)
            if state not in grads:
                continue
            job_batch.append(TCJob(geom, opts, 'gradient', self._excited_type, state, name))

        #   create NAC jobs
        for name, opts in self.nac_job_options.items():
            state = (opts['nacstate1'], opts['nacstate2'])
            if state not in nacs:
                continue
            job_batch.append(TCJob(geom, opts, 'coupling', self._excited_type, max(state), name))

        #   frame specific job options
        for job in job_batch.jobs:
            self._apply_initial_frame_options(job)

        return job_batch


    def combine_jobs(self, job_batch: TCJobBatch, keep_tr_off_diags=False):
        #   make sure there are CIS jobs
        for job in job_batch.jobs:
            if job.excited_type != 'cis':
                print('Warining: Only CIS jobs can be combined')
                return job_batch
            
        gs_gradient = False
        ex_gradients = []
        gs_ex_nacs = []
        ex_ex_nacs = []
        gs_dipole_deriv = False
        ex_dipole_derivs = []
        tr_dipole_derivs = []

        found_cis_dipole_deriv = False
        found_cis_tr_dipole_deriv = False
        cis_states = []

        for job in job_batch.jobs:
            #   group by GS and EX gradients
            if job.job_type == 'gradient':
                if job.state == 0:
                    gs_gradient = True
                else:
                    ex_gradients.append(job.state)
                    cis_states.append(job.state)

            #   group by GS-EX and EX-EX nacs
            elif job.job_type == 'coupling':
                state_1 = min(job.opts['nacstate1'], job.opts['nacstate2'])
                state_2 = max(job.opts['nacstate1'], job.opts['nacstate2'])
                if state_1 == 0:
                    gs_ex_nacs.append((state_1, state_2))
                else:
                    ex_ex_nacs.append((state_1, state_2))

            #   dipole derivatives
            if 'dipolederivative' in job.opts:
                gs_dipole_deriv = True
            elif 'cisdipolederiv' in job.opts:
                found_cis_dipole_deriv = True
            elif 'cistransdipolederiv' in job.opts:
                found_cis_tr_dipole_deriv = True

        cis_states = sorted(set(cis_states))
        if found_cis_dipole_deriv:
            for state in cis_states:
                ex_dipole_derivs.append(state)
        if found_cis_tr_dipole_deriv:
            for state in cis_states:
                tr_dipole_derivs.append((0, state))
            if keep_tr_off_diags:
                for i in cis_states:
                    for j in cis_states[i+1:]:
                        tr_dipole_derivs.append((i, j))

    def _apply_initial_frame_options(self, job: TCJob):
        if self._initial_frame_options is not None:
            if self._frame_counter < self._initial_frame_options['n_frames']:
                job.opts.update(self._initial_frame_options)

    def _load_debug_batch(self, jobs_batch: TCJobBatch):
        if self._debug_traj and not _DEBUG_SAVE_TRAJ:
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
        else:
            return None

    def _save_debug_batch(self, jobs_batch: TCJobBatch):
        if _DEBUG_SAVE_TRAJ:
            print("APPENDING DEBUG TRAJ ", jobs_batch)

            for job in jobs_batch.jobs:
                job_copy = copy.deepcopy(job)
                job_copy.client = None
                self._debug_traj[job_copy.jobID] = job_copy

            with open(_DEBUG_SAVE_TRAJ, 'wb') as file:
                pickle.dump(self._debug_traj, file)

    def _assign_clients_by_request(self, jobs_batch: TCJobBatch):
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

    def _assign_clients_equally(self, jobs_batch: TCJobBatch):
        n_clients = len(self._client_list)
        for j, job in enumerate(jobs_batch.jobs):
            job.client = self._client_list[j % n_clients]
    
    def _send_jobs_to_clients(self, jobs_batch: TCJobBatch):

        #   debug mode
        # debug_batch =  self._load_debug_batch(jobs_batch)
        # if debug_batch is not None: 
        #     return debug_batch
        
        if len(self._tc_client_assignments) > 0:
            self._assign_clients_by_request(jobs_batch)
        else:
            self._assign_clients_equally(jobs_batch)

        if not jobs_batch.check_client():
            raise ValueError('Not all jobs have been assigned a client')

        #   if only one client is being used, don't open up threads, easier to debug
        if len(self._client_list) == 1:
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

        self._save_debug_batch(jobs_batch)
        self._parse_tc_out_data(jobs_batch)
        self._prev_results = jobs_batch.results_list
        self._set_avg_max_times(jobs_batch.timings)
        self._coordinate_exciton_overlap_files()

        return jobs_batch


    def _parse_tc_out_data(self, jobs_batch: TCJobBatch):
        if _HAS_TCPARSE:
            for j in jobs_batch.jobs:
                parsed = parse_from_list(j.results['tc.out'])
                parsed = parsed.model_dump(mode='json')
                for key, v in list(parsed.items()):
                    if v is None:
                        parsed.pop(key)
                parsed.update(j.results)
                j.results = parsed
                # j.results.update(parsed)


    def _coordinate_exciton_overlap_files(self, overlap_file_loc=None, overlap_data=None):
        '''
            Copy a single exciton overlap data file to all clients file systems.
            Precedence is given to the overlap_data argument, then to the overlap_file_loc.
            If neither is provided, the data it attempted to be read from the first client
            that has the exciton_overlap.dat.1 file. If no such file is found, nothing is done.
        '''

        if self._excited_type != 'cis':
            return

        #   first establish which data we are using
        exciton_overlap_data = None
        if overlap_data is not None:
            exciton_overlap_data = overlap_data

        elif overlap_file_loc is not None:
            with open(overlap_file_loc, 'rb') as file:
                exciton_overlap_data = file.read()

        else:
            for client in self._client_list:                
                if client.is_file('exciton_overlap.dat.1'):
                    exciton_overlap_data = client.get_file('exciton_overlap.dat.1', 'rb')
                    break

        #   then copy data to all other server roots
        if exciton_overlap_data is not None:
            self._exciton_overlap_data = exciton_overlap_data
            for client in self._client_list:
                client.set_file('exciton_overlap.dat.1', self._exciton_overlap_data, 'wb')


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

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)
    

class TCJobsLogger(BaseLogger):
    name = 'tc_job_data'

    def __init__(self, file_loc: str = None, h5_group: H5Group = None) -> None:
        # if hasattr(h5_group, 'filename'):
        #     raise TypeError('TCJobsLogger cannot be used with a global H5File object')
        super().__init__(file_loc, h5_group)
        
    def _initialize(self, data: LoggerData):
        print('Initializing TCJobsLogger: ', self._h5_group)
        if self._h5_group:
            dt = h5py.string_dtype(encoding='utf-8')
            self._h5_dataset = self._h5_group.create_dataset(self.name, shape=(0,), maxshape=(None,), dtype=dt)
            print('Created dataset: ', self._h5_dataset, type(self._h5_dataset))

        if self._file:
            raise NotImplementedError('TCJobsLogger can only write to HDF5 files')

    def write(self, data: list[dict]):
        super().write(None)

        if self._h5_dataset:
            H5File.append_dataset(self._h5_dataset, json.dumps(data, cls=NumpyEncoder).encode('utf-8'))

    # def write(self, data: list[dict]):
    #     #   Dummy function: we don't want to write every time step. Instead, we want the
    #     #   TeraChem runner to call when to log data, which uses _write instead.
    #     pass

class TCJobsLoggerSequential():
    name = 'tc_job_data_sequential'
    def __init__(self, file: str = None, h5_file=None) -> None:
        self._file = None
        self._file_loc = None
        self._next_dataset = None
        self.setup(None, h5_file)

    def setup(self, log_dir: str, file: str | h5py.File):

        if isinstance(file, str):
            if log_dir is not None:
                self._file_loc = os.path.join(log_dir, file)
            else:
                self._file_loc = file
            self._file = H5File(self._file_loc, 'w')

        elif isinstance(file, h5py.File):
            self._file_loc = None
            self._file = file
        # else:
        #     raise ValueError('file must be a string or h5py.File object')

        self._data_fields = [
            'energy', 'gradient', 'dipole_moment', 'dipole_vector', 'nacme', 'cis_', 'cas_'
        ]
        self._units_from_field = {'energy': 'a.u.', 'dipole_moment': 'Debye'}
        self._job_datasets = {}
        self._group_name = 'tc_job_data'

        
    def __del__(self):
        if self._file is not None:
            self._file.close()

    def _initialize(self, cleaned_batch: TCJobBatch):
        #   TODO: Add a fix for when the simulation has been restarted, but
        #   the next few jobs are missing some keywords. This happens with dipole
        #   moments and gradients, since an extrapolation scheme doesn't need to
        #   compute a new one every frame. 
        self._file.create_group(self._group_name)
        
        str_dt = h5py.string_dtype(encoding='utf-8')

        for job in cleaned_batch.jobs:
            group = self._file[self._group_name].create_group(name=job.name)
            group.create_dataset(name='timestep', shape=(0,1), maxshape=(None, 1))
            for key, value in job.results.items():
                for k in self._data_fields:
                    if k in key:
                        if isinstance(value, list):
                            shape = (0,) + np.shape(value)
                        else:   #   assume it is a single value
                            shape = (0,1)
                        ds = group.create_dataset(name=key, shape=shape, maxshape=(None,) + shape[1:])
                        if k in self._units_from_field:
                            ds.attrs['units'] = self._units_from_field[k]

            #   couldn't figure out how to initialize with an empty shape when using strings,
            #   so I just resized afterwards
            ds = group.create_dataset(name='tc.out', shape=(1,1), maxshape=(None, 1), data='', dtype=str_dt)
            ds.resize((0, 1))
            ds = group.create_dataset(name='other', shape=(1,1), maxshape=(None, 1), data='', dtype=str_dt)
            ds.resize((0, 1))


        self._file.create_dataset(name = f'{self._group_name}/atoms', 
                                  data = cleaned_batch.results_list[0]['atoms'], dtype=str_dt)
        geom = np.array(cleaned_batch.results_list[0]['geom'])
        geom_ds = self._file.create_dataset(name = f'{self._group_name}/geom', 
                                            shape=(0,) + geom.shape,
                                            maxshape=(None,) + geom.shape)
        geom_ds.attrs.create('units', 'angstroms')
    
    def set_next_dataset(self, jobs_batch: TCJobBatch):
        self._next_dataset = jobs_batch

    def write(self, jobs_data: TCJobBatch, time: float):
        if self._file is None:
            return

        # if data.jobs_data is not None:
        #     jobs_data = data.jobs_data
        if self._next_dataset is not None:
            jobs_data = self._next_dataset
            self._next_dataset = None
        elif jobs_data is None:
            print(self, ': No jobs data found')
            return
        
        results: list[dict] = copy.deepcopy(jobs_data.results_list)
        #   'cis_excitations' are not guarenteed to be the same size,
        #   so this can't be added to an H5 dataset.
        #   TODO: add a check to make sure all jobs have the same number of cis_excitations,
        #   or convert to a string?
        cleaned_results = TCRunner.cleanup_multiple_jobs(results, 'cis_excitations', 'orb_energies', 'bond_order', 'orb_occupations', 'spins')
        # cleaned_batch = deepcopy(data.jobs_data)
        cleaned_batch = jobs_data
        for job, res in zip(cleaned_batch.jobs, cleaned_results):
            job.results = res

        #   the first job is used to establish dataset sizes
        if self._group_name not in self._file:
            print('initializing HDF5 file for TC job data')
            self._initialize(cleaned_batch)

        group = self._file[self._group_name]
        H5File.append_dataset(group['geom'], cleaned_batch.results_list[0]['geom'])
        for job in cleaned_batch.jobs:
            results = job.results.copy()
            results.pop('geom')
            results.pop('atoms')
            for key in group[job.name]:
                if key in ['other', 'timestep', 'tc.out']:
                    continue
                if key in results:
                    H5File.append_dataset(group[job.name][key], results[key])
                    results.pop(key)
                else:
                    print(f'Warning: "{key}" not found in job results, using zeros instead')
                    prev_vals = group[job.name][key][:][-1]
                    fill_result = np.zeros_like(prev_vals)
                    H5File.append_dataset(group[job.name][key], fill_result)
            if 'tc.out' in results:
                H5File.append_dataset(group[job.name]['tc.out'], json.dumps(results['tc.out']))
                results.pop('tc.out')
            H5File.append_dataset(group[job.name]['timestep'], time)

            #   everything else goes into 'other'
            other_data = json.dumps(results)
            H5File.append_dataset(group[job.name]['other'], other_data)
            
        self._file.flush()     

        

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
                client.compute_job(j, append_tc_out=True)
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



def _extract_subset_transition_data(in_data: np.array | list, states: list[int], sym: str):
    if sym.lower() == 's':
        scale = 1.0
    elif sym.lower() == 'a':
        scale = -1.0
    else:
        raise ValueError(f'Invalid symmetry type {sym}, must be "s" or "a"')

    #   First we put all of the transition data into a matrix that is referenced by (state_i, state_j)
    max_states = max(states) + 1
    all_matrix_data = [[None for _ in range(max_states)] for _ in range(max_states)]
    state_pairs = [(i, j) for i in range(max_states) for j in range(i+1, max_states)]
    data_shape = None
    for pair, values in zip(state_pairs, in_data):
        i, j = pair
        all_matrix_data[i][j] = all_matrix_data[j][i] = values
        if values is not None:
            data_shape = np.shape(values)
    
    #   Now we only keep the data for the states we are interested in
    subset_matrix_data = np.zeros((len(states), len(states), *data_shape))

    for i, state1 in enumerate(states):
        for j, state2 in enumerate(states):
            if i >= j:
                continue
            subset_matrix_data[i, j] =         np.array(all_matrix_data[state1][state2])
            subset_matrix_data[j, i] = scale * np.array(all_matrix_data[state2][state1])

    return subset_matrix_data

def dipole_sign_loss(signs, dipole_matrix, ref_dipole_matrix):
    N = dipole_matrix.shape[0]
    S_mat = np.outer(signs, signs)
    corrected_ref = ref_dipole_matrix * S_mat[:, :, np.newaxis]  # shape (N, N, 3)
    diff = dipole_matrix - corrected_ref  # shape (N, N, 3)
    squared_diff = np.sum(diff**2, axis=-1)  # shape (N, N)

    # Use only upper triangle (i < j), excluding diagonal
    i_upper = np.triu_indices(N, k=1)
    return np.sum(squared_diff[i_upper])

def get_signs_from_dipole_matrix(dipole_matrix: np.ndarray, ref_dipole_matrix: np.ndarray):
    '''
        Minimizes the differences between the dipole matrix and the reference dipole matrix
        by adjusting the signs of the states. The first state is always assumed to be positive.

        Parameters
        ----------
        dipole_matrix : np.ndarray
            The dipole matrix for the current job, shape (n_states, n_states, 3)
        ref_dipole_matrix : np.ndarray
            The dipole matrix for the reference job, shape (n_states, n_states, 3)

        Returns
        -------
        np.ndarray
            An array of signs for each state, shape (n_states,)
    '''
    
    dim = dipole_matrix.shape[0]

    from scipy.optimize import minimize
    res = minimize(
        dipole_sign_loss, 
        np.ones(dim), 
        args=(dipole_matrix, ref_dipole_matrix), 
        method='Nelder-Mead', 
        options={'maxiter': 200, 'disp': False, 'xatol': 1e-4, 'fatol': 1e-3}
    )
    signs = np.array([1 if s > 0 else -1 for s in res.x])
    return signs

def format_combo_job_results(job_data: list[dict], states: list[int], ref_dipole_matrix: np.ndarray = None):
    '''
        Combine the job data from multiple jobs into a single dictionary.

        Note
        ----
            Only works with CIS job data for now
    '''

    DEBYE_2_AU = 0.3934303
    validated_list = [TCJobData.model_validate(data) for data in job_data]
    validated: TCJobData = validated_list[0]
    if len(validated_list) > 1:
        validated.append_results(*validated_list[1:])

    #   energies
    all_energies = np.array(validated.energy)
    energies = all_energies[states]

    max_s = max(states) + 1
    n_states = len(states)
    n_atoms = len(validated.geom)
    gradients = np.zeros((max_s, n_atoms*3))
    nacs = np.zeros((max_s, max_s, n_atoms*3))
    mu_deriv_matrix = np.zeros((max_s, max_s, n_atoms*3, 3))
    mu_matrix = np.zeros((max_s, max_s, 3))

    #   extend to lists of None so the following for-loop structure
    #   is compatable with lists of actual arrays
    max_state = max(states)
    if validated.cis_gradients is None:
        validated.cis_gradients = [None] *max_state
    if validated.cis_unrelaxed_dipole_deriv is None:
        validated.cis_unrelaxed_dipole_deriv = [None] * max_state
    if validated.cis_transition_dipole_deriv is None:
        validated.cis_transition_dipole_deriv = [None] * (max_state * (max_state + 1) // 2)
    if validated.cis_couplings is None:
        validated.cis_couplings = [None] * (max_state * (max_state + 1) // 2)

    tc_gradients = [validated.gs_gradient] + validated.cis_gradients
    tc_dipole_derivs = [validated.dipole_deriv] + validated.cis_unrelaxed_dipole_deriv
    tc_dipoles = [np.array(validated.dipole_vector)*DEBYE_2_AU] + validated.cis_unrelaxed_dipoles

    for i, state_i in enumerate(states):
        #   gradients
        g = tc_gradients[state_i]
        if g is not None:
            gradients[state_i] = np.array(g).flatten()

    for i in range(max_s):

        #   diagonal dipoles
        mu = tc_dipoles[i]
        if mu is not None:
            mu_matrix[i, i] = np.array(mu).flatten()
        
        #   diagonal dipole derivatives
        mu_grad = tc_dipole_derivs[i]
        if mu_grad is not None:
            mu_deriv_matrix[i, i] = np.array(mu_grad).reshape((3, n_atoms*3)).T

    #   NACs and transition dipole information are stored in a triangular matrix
    #   of size N(N-1)/2
    for i in range(max_s):
        for j in range(i+1, max_s):

            #   NACs
            nac = validated.cis_couplings.pop(0)
            if nac is not None:
                nacs[i, j] = np.array(nac).flatten()
                nacs[j, i] = -nacs[i, j]

            #   transition dipole derivatives
            mu_tr_deriv = validated.cis_transition_dipole_deriv.pop(0)
            if mu_tr_deriv is not None:
                mu_tr_deriv = np.array(mu_tr_deriv).reshape((3, n_atoms*3)).T
                mu_deriv_matrix[i, j] = mu_tr_deriv
                mu_deriv_matrix[j, i] = mu_tr_deriv

            #   transition dipoles
            mu_tr = validated.cis_transition_dipoles.pop(0)
            if mu_tr is not None:
                mu_matrix[i, j] = mu_tr
                mu_matrix[j, i] = mu_tr

    #   make sure the shapes are correctcorrect for sign flips
    if ref_dipole_matrix is not None:
        signs = get_signs_from_dipole_matrix(mu_matrix, ref_dipole_matrix)
        for i in range(n_states):
            for j in range(n_states):
                mu_matrix[i, j]         *= signs[i] * signs[j]
                mu_deriv_matrix[i, j]   *= signs[i] * signs[j]
                nacs[i, j]              *= signs[i] * signs[j]

    #   make sure the shapes are correct based on the states
    if len(states) != max_s:
        gradients = gradients[states]
        nacs = nacs[states][:, states]
        mu_matrix = mu_matrix[states][:, states]
        mu_deriv_matrix = mu_deriv_matrix[states][:, states]
    

    return (all_energies, energies, gradients, nacs, mu_matrix, mu_deriv_matrix)

def _print_ES_values(energies, gradients, nacs, mu_matrix, mu_deriv_matrix):
    '''
        Print the energies, gradients, NACs, transition dipoles, and transition dipole derivatives
        in a human-readable format.
    '''
    print('Energies:')
    print(energies)
    print('\nGradients:')
    for i, g in enumerate(gradients):
        print(f'State {i}\n:{g}')

    print('\nNACs:')
    for i in range(nacs.shape[0]):
        for j in range(i+1, nacs.shape[1]):
            print(f'NAC {i} - {j}\n:{nacs[i, j]}')

    print('\nDipole Matrix:')
    for i in range(mu_matrix.shape[0]):
        for j in range(i, mu_matrix.shape[1]):
            print(f'Dipole {i} - {j}\n:{mu_matrix[i, j]}')

    print('\nDipole Derivative Matrix:')
    for i in range(mu_deriv_matrix.shape[0]):
        for j in range(i, mu_deriv_matrix.shape[1]):
            print(f'Dipole Derivative {i} - {j}\n:{mu_deriv_matrix[i, j]}')

def format_combo_job_results_OLD(job_data: list[dict], states: list[int]):
    '''
        Combine the job data from multiple jobs into a single dictionary.
    '''
    validated_list = [TCJobData.model_validate(data) for data in job_data]
    validated: TCJobData = validated_list[0]
    validated.append_results(*validated_list[1:])
    
    #   extract the gradients
    all_gradients = [validated.gs_gradient] + validated.cis_gradients
    gradients = np.array([all_gradients[i] for i in states])
    gradients = np.reshape(gradients, (len(states), -1))

    #   extract the energies
    all_energies = np.array(validated.energy)
    energies = all_energies[states]
    
    #   extract the NACs
    nacs = _extract_subset_transition_data(validated.cis_couplings, states, 'a')
    nacs = np.reshape(nacs, nacs.shape[0:-2] + (-1,))

    #   extract the transition dipoles
    all_mu_tr = np.array(validated.cis_transition_dipoles)
    mu_tr = _extract_subset_transition_data(all_mu_tr, states, 's')

    #   extract the transition dipole derivatives
    mu_deriv_matrix = _extract_subset_transition_data(validated.cis_transition_dipole_deriv, states, 's')

    #   form the final dipole derivative matrix
    mu_diagonal_grads = [validated.dipole_deriv] + validated.cis_unrelaxed_dipole_deriv
    for i, state in enumerate(states):
        mu_deriv_matrix[i, i] = mu_diagonal_grads[state]
    mu_deriv_matrix = np.reshape(mu_deriv_matrix, mu_deriv_matrix.shape[0:-2] + (-1,))
    mu_deriv_matrix = mu_deriv_matrix.transpose((0, 1, 3, 2))
    

    return (all_energies, energies, gradients, nacs, mu_tr, mu_deriv_matrix)
    


def format_output_LSCIVR(job_data: list[dict]):
    atoms = job_data[0]['atoms']
    n_atoms = len(atoms)
    
    # energies = np.zeros(n_elec)
    # grads = np.zeros((n_elec, n_atoms*3))
    # nacs = np.zeros((n_elec, n_elec, n_atoms*3))
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
    print(" LSC-IVR to TeraChem")
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

            # ivr_trans_dips[i, j] = trans_dips[(qc_i, qc_idx_j)]
            # ivr_trans_dips[j, i] = trans_dips[(qc_idx_j, qc_i)]

            td = trans_dips.get((qc_i, qc_idx_j), None)
            if td is not None:
                ivr_trans_dips[i, j] = td
                ivr_trans_dips[j, i] = td
            else:
                ivr_trans_dips = None
    print(" ---------------------------------")

    return all_energies, ivr_energies, ivr_grads, ivr_nacs, ivr_trans_dips, None

