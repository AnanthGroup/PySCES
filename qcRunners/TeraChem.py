#!/usr/bin/env python
# Basic energy calculation
import sys, os
import numpy as np
from qcelemental.models import AtomicInput, Molecule
from tcpb import TCProtobufClient as TCPBClient
from tcpb.exceptions import ServerError
import subprocess, time


def _wait_until_available(client: TCPBClient, max_wait=10.0, time_btw_check=1.0):
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

def _print_times(times):
    total = 0.0
    print()
    print("Timings")
    print("-------------------------------")
    for key, value in times.items():
        print(f'{key:20s} {value:10.3f}')
        total += value
    print()
    print(f'{"total":20s} {total:10.3f}')
    print("-------------------------------")

def _val_or_iter(x):
    try:
        iter(x)
    except:
        x = [x]
    return x

class TCRunner():
    def __init__(self, host: str, port: int, atoms: list, tc_options: dict, run_options: dict={}, max_wait=20) -> None:
        self._client = client = TCPBClient(host=host, port=port)
        _wait_until_available(client, max_wait=max_wait)
        self._atoms = np.copy(atoms)

        self._base_options = tc_options.copy()

        self._max_state = run_options.get('max_state', 0)
        self._grads = run_options.get('grads', [])
        self._NACs = run_options.pop('NACs', [])


    def run_TC(self, geom):
        return self.run_TC_with_options(self._client, geom, self._atoms, self._base_options, self._max_state, self._grads, self._NACs)

    @staticmethod
    def run_TC_with_options(client, geom, atoms, opts: dict, 
            max_state:int=0, 
            grads:list[int]|int|str=[], 
            NACs:list[int]|float|str=[]):
        
        times = {}

        #   convert grads and NACs to list format 
        if isinstance(grads, str):
            if grads.lower() == 'all':
                grads = list(range(max_state+1))
            else:
                raise ValueError('grads must be an iterable in ints or "all"')
        else:
            grads = _val_or_iter(grads)
        if isinstance(NACs, str):
            if NACs.lower() == 'all':
                NACs = []
                for i in range(max_state+1):
                    for j in range(i+1, max_state+1):
                        NACs.append((i, j))
            else:
                raise ValueError('NACs must be an iterable in ints or "all"')
        else:
            NACs = _val_or_iter(NACs)

        #   make sure we are computing enough states
        base_options = opts.copy()
        if max_state > 0:
            if 'cisnumstates' not in base_options:
                base_options['cisnumstates'] = max_state + 2
            elif base_options['cisnumstates'] < max_state:
                raise ValueError('"cisnumstates" is less than requested electronic state')
        base_options['cisrestart'] = 'cis_restart_' + str(os.getpid())
        base_options['purify'] = False
        base_options['atoms'] = atoms
        
        #   run energy only if gradients and NACs are not requested
        all_results = []
        if len(grads) == 0 and len(NACs) == 0:
            start = time.time()
            results = client.compute_job_sync('energy', geom, 'angstrom', **job_opts)
            times[f'energy_{state}'] = time.time() - start
            results['run'] = 'energy'
            all_results.append(results)

        #   gradient computations have to be separated from NACs
        for job_i, state in enumerate(grads):
            print("Grad ", job_i+1)
            job_opts = base_options.copy()
            if state > 0:
                job_opts['cis'] = 'yes'
                if 'cisnumstates' not in job_opts:
                    job_opts['cisnumstates'] = max_state + 2
                elif job_opts['cisnumstates'] < max_state:
                    raise ValueError('"cisnumstates" is less than requested electronic state')
                job_opts['cistarget'] = state

            if job_i > 0:
                job_opts['guess'] = all_results[-1]['orbfile']

            start = time.time()
            results = client.compute_job_sync('gradient', geom, 'angstrom', **job_opts)
            times[f'gradient_{state}'] = time.time() - start
            results['run'] = 'gradient'
            all_results.append(results)

        #   run NAC jobs
        for job_i, (nac1, nac2) in enumerate(NACs):
            print("NAC ", job_i+1)
            job_opts = base_options.copy()
            job_opts['nacstate1'] = nac1
            job_opts['nacstate2'] = nac2
            job_opts['cis'] = 'yes'
            job_opts['cisnumstates'] = max_state + 2
            if job_i > 0:
                job_opts['guess'] = all_results[-1]['orbfile']

            start = time.time()
            results = client.compute_job_sync('coupling', geom, 'angstrom', **job_opts)
            times[f'nac_{nac1}_{nac2}'] = time.time() - start
            results['run'] = 'coupling'
            all_results.append(results)


        _print_times(times)
        return all_results

   