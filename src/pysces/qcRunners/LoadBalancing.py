import matplotlib.pyplot as plt
import numpy as np
from pprint import pprint
import time
from dataclasses import dataclass, fields, asdict
from typing import Optional
import copy
import enum

# plt.style.use('my_style_2')

_assignments_optimum: '_BalancedCollection' = None
_assignments_round_robin: '_BalancedCollection'  = None


@dataclass
class TaskTimes():
    ''' Holds the time for each electronic structure derivative task. '''
    GS_grad: float = 0.0
    EX_grad: float = 0.0
    GS_EX_NAC: float = 0.0
    EX_EX_NAC: float = 0.0
    GS_dipole: float = 0.0
    EX_dipole: float = 0.0
    GS_EX_dipole: float = 0.0
    EX_EX_dipole: float = 0.0


    def __post_init__(self):
        if all(value == 0 for value in self.__dict__.values()):
            raise ValueError("At least one task time must be provided")
        #   check if all values are floats
        for value in self.__dict__.values():
            if value != 0 and not isinstance(value, float):
                raise ValueError(f"Task time must be a float, got {type(value)}")
            
@dataclass
class ServerBenchmark(TaskTimes):
    ''' Holds the benchmark times for each electronic structure derivative task. '''
    name: Optional[str] = None
    address: Optional[str] = None
    port: Optional[int] = None

    def __post_init__(self):
        pass

class ElectronicStates:
    def __init__(self, states=None):
        self._states = set()
        if states is not None:
            for s in states:
                self.add(s)
        
    def add(self, state: int):
        if not isinstance(state, int) or state < 0:
            raise ValueError(f"State must be an integer greater than 0, got: {state}")
        self._states.add(state)
        
    def remove(self, state: int):
        self._states.discard(state)

    def pop(self):
        return self._states.pop()
        
    @property
    def states(self) -> list:
        return sorted(self._states)
    
    def __iter__(self):
        return iter(self.states)
    
    def __len__(self):
        return len(self._states)
        
    def __repr__(self):
        return f"ElectronicStates({self.states})"
    
class ElectronicStatePairs:
    def __init__(self, states=None):
        self._states = set()
        if states is not None:
            for s in states:
                self.add(s)
        
    def add(self, state: tuple[int, int]):
        if len(state) != 2 or (type(state[0]) != int and type(state[1]) != int):
            raise ValueError(f"State must be a pair of two integers, got: {state}")
        if min(state) < 0:
            raise ValueError(f"State must be greater than 0, got: {state}")
        if state[0] > state[1]:
            state = (state[1], state[0])
        self._states.add(state)
        
    def remove(self, state: tuple[int, int]):
        self._states.discard(state)

    def pop(self):
        return self._states.pop()
        
    @property
    def states(self) -> list:
        return sorted(self._states)
    
    def __iter__(self):
        return iter(self.states)
    
    def __len__(self):
        return len(self._states)
        
    def __repr__(self):
        return f"ElectronicStatePair({self.states})"

class ESDerivTasks():
    def __init__(self, grads: list[int]=[], nacs: list[int, int]=[], dipoles: list[int]=[], tr_dipoles: list[int, int]=[]):

        self.gs_grad = ElectronicStates()
        self.ex_grads = ElectronicStates()
        self.gs_ex_nacs = ElectronicStatePairs()
        self.ex_ex_nacs = ElectronicStatePairs()

        self.gs_dipole_grad = ElectronicStates()
        self.ex_dipole_grads = ElectronicStates()
        self.gs_ex_dipole_grads = ElectronicStatePairs()
        self.ex_ex_dipole_grads = ElectronicStatePairs()

        for state in grads:
            if state == 0:
                self.gs_grad.add(state)
            else:
                self.ex_grads.add(state)

        for state_pair in nacs:
            if min(state_pair) == 0:
                self.gs_ex_nacs.add(state_pair)
            else:
                self.ex_ex_nacs.add(state_pair)

        for state in dipoles:
            if state == 0:
                self.gs_dipole_grad.add(state)
            else:
                self.ex_dipole_grads.add(state)

        for state_pair in tr_dipoles:
            if min(state_pair) == 0:
                self.gs_ex_dipole_grads.add(state_pair)
            else:
                self.ex_ex_dipole_grads.add(state_pair)

class TaskCollection:
    ''' A load balanced configurarion of tasks and times for a single server'''
    def __init__(self):
        self.task_types: list[str] = []
        self.times: list[float] = []
        self.total_time = 0

    def add_task(self, task_type: str, time: float):
        self.task_types.append(task_type)
        self.times.append(time)
        self.total_time += time
    
    def copy(self):
        new_collection = TaskCollection()
        new_collection.task_types = self.task_types.copy()
        new_collection.total_time = self.total_time
        new_collection.times = self.times.copy()
        return new_collection

class _BalancedCollection:
    '''
        wrapper class with methods to help balance the tasks across multiple servers
    '''
    def __init__(self, n_servers):
        self.collections = [TaskCollection() for _ in range(n_servers)]
        self.order_added = []
        self.n_servers = n_servers
        self.ES_tasks = {}

        self._possible_times = None
    
    def get_max_load(self):
        return max([server.total_time for server in self.collections])
    
    def copy(self):
        new_collection = _BalancedCollection(self.n_servers)
        new_collection.collections = [server.copy() for server in self.collections]
        new_collection.order_added = self.order_added.copy()
        new_collection._possible_times = self._possible_times.copy()
        return new_collection

    def assign_ES_tasks(self, tasks: ESDerivTasks):
        tasks_reduced = copy.deepcopy(tasks)
        for collection in self.collections:
            for task_type in collection.task_types:
                tasks_reduced.grads
    
    def _set_server_benchmarks(self, benchmarks: list[ServerBenchmark]):
        possible_task_times = {}
        task_names = [f.name for f in fields(TaskTimes)]

        for task in task_names:
            possible_task_times[task] = []
            for bm in benchmarks:
                if task not in possible_task_times:
                    raise ValueError(f"Task {task} not found in task_times")
                possible_task_times[task].append(getattr(bm, task))
        self._possible_times = possible_task_times
    
    # def _add_server(self, server):
    #     self.collections.append(server)

    def _add_task(self, server_idx, task):
        time = self._possible_times[task][server_idx]
        self.collections[server_idx].add_task(task, time)
        self.order_added.append((server_idx, task, time))

    def _find_server_with_min_time_after_add(self, task: str):
        min_time = float('inf')
        min_idx = -1

        times = self._possible_times[task]
        for i, server in enumerate(self.collections):
            new_collection = self.copy()
            new_collection.collections[i].add_task(task, times[i])
            new_time = new_collection.get_max_load()
            if new_time < min_time:
                min_time = new_time
                min_idx = i

        return min_idx


def _flatten_tasks(benchmarks: list[ServerBenchmark], es_tasks: ESDerivTasks):
    flattened_tasks = []
    flattened_tasks.extend(['GS_grad'] * len(es_tasks.gs_grad))
    flattened_tasks.extend(['EX_grad'] * len(es_tasks.ex_grads))
    flattened_tasks.extend(['GS_EX_NAC'] * len(es_tasks.gs_ex_nacs))
    flattened_tasks.extend(['EX_EX_NAC'] * len(es_tasks.ex_ex_nacs))
    flattened_tasks.extend(['GS_dipole'] * len(es_tasks.gs_dipole_grad))
    flattened_tasks.extend(['EX_dipole'] * len(es_tasks.ex_dipole_grads))
    flattened_tasks.extend(['GS_EX_dipole'] * len(es_tasks.gs_ex_dipole_grads))
    flattened_tasks.extend(['EX_EX_dipole'] * len(es_tasks.ex_ex_dipole_grads))

    # Order flat tasks by the first server benchmark
    ordered_tasks = sorted(flattened_tasks, key=lambda t: -getattr(benchmarks[0], t))
    return ordered_tasks

def _partition_jobs(balanced_collection: '_BalancedCollection', es_tasks: ESDerivTasks) -> list[ESDerivTasks]:
    depleted_tasks = copy.deepcopy(es_tasks)

    collections = []
    for coll in balanced_collection.collections:
        sub_tasks = ESDerivTasks()
        for name in coll.task_types:
            if name == 'GS_grad':
                sub_tasks.gs_grad.add(depleted_tasks.gs_grad.pop())

            elif name == 'EX_grad':
                sub_tasks.ex_grads.add(depleted_tasks.ex_grads.pop())

            elif name == 'GS_EX_NAC':
                sub_tasks.gs_ex_nacs.add(depleted_tasks.gs_ex_nacs.pop())
            
            elif name == 'EX_EX_NAC':
                sub_tasks.ex_ex_nacs.add(depleted_tasks.ex_ex_nacs.pop())

            elif name == 'GS_dipole':
                sub_tasks.gs_dipole_grad.add(depleted_tasks.gs_dipole_grad.pop())

            elif name == 'EX_dipole':
                sub_tasks.ex_dipole_grads.add(depleted_tasks.ex_dipole_grads.pop())

            elif name == 'GS_EX_dipole':
                sub_tasks.gs_ex_dipole_grads.add(depleted_tasks.gs_ex_dipole_grads.pop())

            elif name == 'EX_EX_dipole':
                sub_tasks.ex_ex_dipole_grads.add(depleted_tasks.ex_ex_dipole_grads.pop())

            else:
                raise ValueError(f"Unknown task type: {name}")
            
        collections.append(sub_tasks)

    #   check if all tasks are depleted
    for task_type in ['gs_grad', 'ex_grads', 'gs_ex_nacs', 'ex_ex_nacs', 'gs_dipole_grad', 'ex_dipole_grads', 'gs_ex_dipole_grads', 'ex_ex_dipole_grads']:
        if len(getattr(depleted_tasks, task_type)) != 0:
            raise ValueError(f"Not all tasks are depleted: {task_type} has {len(getattr(depleted_tasks, task_type))} tasks left")

    return collections

def balance_tasks_optimum(benchmarks: list[ServerBenchmark], tasks: ESDerivTasks, num_workers: int, n_trials: int=500) -> list[ESDerivTasks]:
    '''
        Load balances the tasks across the servers using a greedy algorithm.
        The algorithm assigns tasks to the server with the least load after adding the task,
        and performs this `n_trials` times to find the best configuration. Each trial
        starts with a random ordering of the tasks.
        The algorithm is not guaranteed to find the optimal solution, but it should be
        close to it.

        Parameters
        ----------
        benchmarks : list[ServerBenchmark]
            A list of server benchmarks, each with a set of task times.
        tasks : ESDerivTasks
            Contains the tasks to be balanced.
        num_workers : int
            The number of workers (servers) to balance the tasks across.
        n_trials : int
            The number of trials to run to find the best configuration.

        Returns
        -------
        balanced_collection: List[ESDerivTasks]
            A list of load balanced configurations of tasks and times for multiple servers

    '''

    flattened_tasks = _flatten_tasks(benchmarks, tasks)

    best_time = float('inf')
    best_balanced = None
    base_collection = _BalancedCollection(num_workers)
    base_collection._set_server_benchmarks(benchmarks)

    #   try random starting points
    start_time = time.time()
    np.random.seed(0)
    for k in range(n_trials):
        collection = base_collection.copy()

        # Assign tasks to workers
        for i, task in enumerate(flattened_tasks):
            server_idx = collection._find_server_with_min_time_after_add(task)
            collection._add_task(server_idx, task)

        max_time = collection.get_max_load()
        if max_time < best_time:
            best_time = max_time
            best_balanced = collection.copy()

        rand_order = np.random.permutation(len(flattened_tasks))
        flattened_tasks = [flattened_tasks[i] for i in rand_order]


    #   for debugging
    global _assignments_optimum
    _assignments_optimum = best_balanced
    # print('Time:', time.time() - start_time)

    return _partition_jobs(best_balanced, tasks)
    
def balance_tasks_round_robin(benchmarks: list[ServerBenchmark], tasks: ESDerivTasks, num_workers) -> list[ESDerivTasks]:
    '''
        Load balances the tasks across the servers using a round-robin algorithm.
        The algorithm assigns tasks to the servers in a round-robin fashion.

        Parameters
        ----------
        benchmarks : list[ServerBenchmark]
            A list of server benchmarks, each with a set of task times.
        tasks : ESDerivTasks
            Contains the tasks to be balanced.
        num_workers : int
            The number of workers (servers) to balance the tasks across.

        Returns
        -------
        balanced_collection: List[ESDerivTasks]
            A list of load balanced configurations of tasks and times for multiple servers
    '''

    flattened_tasks = _flatten_tasks(benchmarks, tasks)

    collection = _BalancedCollection(num_workers)
    collection._set_server_benchmarks(benchmarks)

    # Assign tasks to workers
    for i, task in enumerate(flattened_tasks):
        server_idx = i % num_workers
        collection._add_task(server_idx, task)
        collection

    #   debugging only
    global _assignments_round_robin
    _assignments_round_robin = collection

    return _partition_jobs(collection, tasks)


def _debug_plot_assignments_new(task_times, collections: _BalancedCollection):
    
    fig, ax = plt.subplots()
    height = 0.8
    for i, server in enumerate(collections.collections):
        left = 0.0
        for task, time in zip(server.task_types, task_times):
            time = task_times[task]
            ax.barh(i, time, height, left)
            left += time
            name = task
            ax.text(left - time/2, i, name, ha='center', va='center', color='white')
            # ax.barh(i, task_times[task], left=sum(task_times[t] for t in worker_tasks[:worker_tasks.index(task)]), color='C0')
    
    ax.set_yticks(range(len(collections.collections)))
    ax.set_yticklabels([f"Worker {i+1}" for i in range(len(collections.collections))])
    ax.set_xlabel("Time")
    ax.set_ylabel("Worker")
    ax.set_xlim(0, 200)
    fig.savefig("task_assignments.png")
    plt.show()

def _debug_creat_plot(collections_1: _BalancedCollection, collections_2: _BalancedCollection):

    max_time_1 = collections_1.get_max_load()
    max_time_2 = collections_2.get_max_load()
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12), sharex=True)

    default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color'] + ['pink']
    colors_by_task = {}
    task_names = ['GS_grad', 'EX_grad', 'GS_EX_NAC', 'EX_EX_NAC', 'GS_dipole', 'EX_dipole', 'GS_EX_dipole']
    for i, task in enumerate(task_names):
        colors_by_task[task] = default_colors[i % len(default_colors)]

    max_of_max = max(max_time_1, max_time_2)

    diff = abs(max_time_1- max_time_2)
    pct_savings = 100 * diff / max_of_max
    ax1.set_title(f"Load Balancing (max time: {max_time_1:.2f}s)")
    ax2.set_title(f"Round-Robin (max time: {max_time_2:.2f}s)")
    fig.suptitle(f"Load Balancing Comparison ({pct_savings:.2f}% savings)")
    
    n_workers = len(collections_1.collections)
    for ax in (ax1, ax2):
        ax.set_ylim(-1 + 0.4, n_workers - 0.4)
        ax.set_xlim(0, max_of_max*1.1)
        ax.set_yticks(range(n_workers))
        ax.set_yticklabels([f"Server {i+1}" for i in range(n_workers)])
        ax.set_xlabel("Time")
        ax.set_ylabel("Server")

        #   add a legend between each plot for each job type
    for task, color in colors_by_task.items():
        ax1.bar(0, 0, color=color, label=task)
    ax1.legend(loc='center', ncol=3, bbox_to_anchor=(0.5, -0.25), title='Task Type')
    fig.tight_layout()
    plt.subplots_adjust(hspace=0.5)

    return fig, ax1, ax2, colors_by_task

def _debug_create_annimation(n_workers, collections_1: _BalancedCollection, collections_2: _BalancedCollection):
        
    fig, ax1, ax2, colors_by_task = _debug_creat_plot(collections_1, collections_2)

    height = 0.8
    count = 0

    for j in range(2):
        ax = (ax1, ax2)[j]
        order = collections_1.order_added if j == 0 else collections_2.order_added

        left_sums = np.zeros(n_workers)
        for (worker, task, time) in order:
            print(count)
            ax.barh(worker, time, height, left_sums[worker], color=colors_by_task[task], edgecolor='white', linewidth=3.5)
            left_sums[worker] += time

            file_name = f"figs/fig_{count:03d}.png"
            fig.savefig(file_name)
            plt.close(fig)
            count += 1

def _debug_plot_comparisons(name: dict):
    global _assignments_optimum, _assignments_round_robin

    fig, ax1, ax2, colors_by_task = _debug_creat_plot(_assignments_optimum, _assignments_round_robin)

    height = 0.8
    for (ax, collections) in zip([ax1, ax2], [_assignments_optimum, _assignments_round_robin]):
        for i, server in enumerate(collections.collections):
            print(i, server.task_types, server.times)
            left = 0.0
            for task, time in zip(server.task_types, server.times):
                ax.barh(i, time, height, left, color=colors_by_task[task], edgecolor='white', linewidth=3.5)
                left += time

    fig.savefig(f"{name}png")
    plt.show()

def _debug_run_test():
    num_workers = 3

    task_times = ServerBenchmark(GS_grad=0.23, EX_grad=2.22, GS_EX_NAC=2.01, EX_EX_NAC=3.22, GS_dipole=6.93, EX_dipole=46.07*0.8, GS_EX_dipole=39.32*0.65)

    benchmarks = []
    for i in range(0, num_workers):
        if i == 0:
            task_times = copy.copy(task_times)
            task_names = [f.name for f in fields(TaskTimes)]
            for k in task_names:
                if k[0] == '_':
                    continue
                task_times.__dict__[k] *= 1.0

            benchmarks.append(task_times)
        else:
            benchmarks.append(copy.copy(task_times))

    tasks = ESDerivTasks(
        grads=[0, 1, 2, 3, 4], 
        nacs=[(0, 1), (0, 2), (0, 3), (1, 2), (3, 4)], 
        dipoles=[0, 1, 2], 
        tr_dipoles=[(0, 2), (0, 3)]
        )

    pprint(benchmarks)

    balance_tasks_optimum(benchmarks, tasks, num_workers, 500)
    balance_tasks_round_robin(benchmarks, tasks, num_workers)
    _debug_plot_comparisons("comparison")

if __name__ == "__main__":

    _debug_run_test()

