import matplotlib.pyplot as plt
import numpy as np
from pprint import pprint
import time
from dataclasses import dataclass, fields, asdict
from typing import Optional
import copy

# plt.style.use('my_style_2')

class DictLikeBase:
    ''' A base class that provides dictionary-like access to attributes. 
        DELETE ME    
    '''
    def __getitem__(self, key):
        if not hasattr(self, key):
            raise KeyError(f"Key '{key}' not found!")
        return getattr(self, key)

    def __setitem__(self, key, value):
        if not hasattr(self, key):
            raise KeyError(f"Key '{key}' not found!")
        setattr(self, key, value)

    def __iter__(self):
        return iter(self.__dict__)

    def items(self):
        return self.__dict__.items()

    def keys(self):
        return self.__dict__.keys()

    def values(self):
        return self.__dict__.values()

    def __contains__(self, key):
        return key in self.__dict__

    def as_dict(self):
        return {field.name: getattr(self, field.name) for field in fields(self)}


@dataclass
class _TaskCounts():
    ''' Holds the counts for each electronic structure derivative task. '''
    GS_grad: int = 0
    EX_grad: int = 0
    GS_EX_NAC: int = 0
    EX_EX_NAC: int = 0
    GS_dipole: int = 0
    EX_dipole: int = 0
    GS_EX_dipole: int = 0

    def __post_init__(self):
        # if all(value == 0 for value in self.__dict__.values()):
        #     raise ValueError("At least one task count must be provided")
        #   check if all values are ints
        for value in self.__dict__.values():
            if value != 0 and not isinstance(value, int):
                raise ValueError(f"Task count must be an int, got {type(value)}")
            

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


    def __post_init__(self):
        if all(value == 0 for value in self.__dict__.values()):
            raise ValueError("At least one task time must be provided")
        #   check if all values are floats
        for value in self.__dict__.values():
            if value != 0 and not isinstance(value, float):
                raise ValueError(f"Task time must be a float, got {type(value)}")

class ElectronicState:
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
        
    @property
    def states(self) -> list:
        return sorted(self._states)
    
    def __iter__(self):
        return iter(self.states)
        
    def __repr__(self):
        return f"ElectronicStates({self.states})"
    
class ElectronicStatePair:
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
        
    @property
    def states(self) -> list:
        return sorted(self._states)
    
    def __iter__(self):
        return iter(self.states)
        
    def __repr__(self):
        return f"ElectronicStatePair({self.states})"

class ESDerivTasks():
    def __init__(self, grads: list[int]=[], nacs: list[int, int]=[], dipoles: list[int]=[], tr_dipoles: list[int, int]=[]):

        self.grads = ElectronicState(states=grads)
        self.nacs = ElectronicStatePair(states=nacs)
        self.dipoles = ElectronicState(states=dipoles)
        self.tr_dipoles = ElectronicStatePair(states=tr_dipoles)


    def _check_and_filter(self):
        pass
        '''

        filtered_grads = set()
        for g in self.grads:
            if g < 0:
                raise ValueError(f"Gradient {g} must be greater than or equal to 0")
            filtered_grads.add(g)
        self.grads = list(filtered_grads)

        filtered_nacs = set()
        for n in self.nacs.copy():
            if min(n) < 0:
                raise ValueError(f"NAC {n} must be greater than or equal to 0")
            if n[0] == n[1]:
                print('Warning: found a NAC with two of the same states: this is always zero and will be removed')
            elif n[0] > n[1]:
                filtered_nacs.add((n[1], n[0]))
            else:
                filtered_nacs.add(n)
        self.nacs = list(filtered_nacs)

        filtered_dipoles = set()
        for d in self.dipoles:
            if min(d) < 0:
                raise ValueError(f"Dipole {d} must be greater than or equal to 0")
            filtered_dipoles.add(d)
        self.dipoles = list(filtered_dipoles)


        filtered_tr_dipoles = set()
        for d in self.tr_dipoles.copy():
            if min(d) < 0:
                raise ValueError(f"TrDipole {d} must be greater than or equal to 0")
            if d[0] == d[1]:
                print('Warning: found a TrDipole with two of the same states: this is not a "transition" dipole and will be removed')
                self.tr_dipoles.remove(d)
            elif d[0] > d[1]:
                filtered_tr_dipoles.add((d[1], d[0]))
            else:
                filtered_tr_dipoles.add(d)
        self.tr_dipoles = list(filtered_tr_dipoles)
        '''

    def get_task_times(self):
        self._check_and_filter()
        task_counts = _TaskCounts()

        #   gradients
        for g in self.grads:
            if g == 0:
                task_counts.GS_grad = 1
            else:
                task_counts.EX_grad += 1

        #   nacs
        for n in self.nacs:
            if min(n) == 0:
                task_counts.GS_EX_NAC += 1
            else:
                task_counts.EX_EX_NAC += 1

        #   dipole derivatives
        for d in self.dipoles:
            if d == 0:
                task_counts.GS_dipole += 1
            else:
                task_counts.EX_dipole += 1

        #   transition dipole derivatives
        for d in self.tr_dipoles:
            if min(d) == 0:
                task_counts.GS_EX_dipole += 1
            else:
                raise NotImplementedError('Excited-Excited dipole moment derivatives not implemented yet')

        return task_counts

@dataclass
class ServerBenchmark(TaskTimes):
    ''' Holds the benchmark times for each electronic structure derivative task. '''
    name: Optional[str] = None
    address: Optional[str] = None
    port: Optional[int] = None

    def __post_init__(self):
        pass


class TaskCollection:
    ''' A load balanced configurarion of tasks and times for a single server'''
    def __init__(self):
        self.tasks = []
        self.times = []
        self.total_time = 0

    def add_task(self, task, time):
        self.tasks.append(task)
        self.times.append(time)
        self.total_time += time
    
    def copy(self):
        new_collection = TaskCollection()
        new_collection.tasks = self.tasks.copy()
        new_collection.total_time = self.total_time
        new_collection.times = self.times.copy()
        return new_collection
    
# @dataclass
# class BalancedCollection:
#     ''' A collection of load balanced configurations of tasks and times for multiple servers'''
#     def __init__(self, n_servers):
#         self.collections = [TaskCollection() for _ in range(n_servers)]
#         self.order_added = []
#         self.n_servers = n_servers

class BalancedCollection:
    '''
        wrapper class with methods to help balance the tasks across multiple servers
    '''
    def __init__(self, n_servers):
        self.collections = [TaskCollection() for _ in range(n_servers)]
        self.order_added = []
        self.n_servers = n_servers

        self._possible_times = None

    # def get_result(self):
    #     result = BalancedCollection(self.n_servers)
    #     for i, server in enumerate(self.collections):
    #         result.collections[i].tasks = server.tasks
    #         result.collections[i].times = server.times
    #         result.collections[i].total_time = server.total_time
    #     result.order_added = self.order_added
    #     return result

    def _set_server_benchmarks(self, benchmarks: list[ServerBenchmark]):
        possible_task_times = {}
        task_names = [f.name for f in fields(_TaskCounts)]

        for task in task_names:
            possible_task_times[task] = []
            for bm in benchmarks:
                if task not in possible_task_times:
                    raise ValueError(f"Task {task} not found in task_times")
                possible_task_times[task].append(getattr(bm, task))
        self._possible_times = possible_task_times
    
    def get_max_load(self):
        return max([server.total_time for server in self.collections])
    
    def copy(self):
        new_collection = BalancedCollection(self.n_servers)
        new_collection.collections = [server.copy() for server in self.collections]
        new_collection.order_added = self.order_added.copy()
        new_collection._possible_times = self._possible_times.copy()
        return new_collection
    
    def _add_server(self, server):
        self.collections.append(server)

    def _add_task(self, server_idx, task):
        time = self._possible_times[task][server_idx]
        self.collections[server_idx].add_task(task, time)
        self.order_added.append((server_idx, task, time))

    def _find_server_with_min_time_after_add(self, task: list):
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


def _flatten_tasks(benchmarks: list[ServerBenchmark], task_counts: _TaskCounts):
    flattened_tasks = []
    for task, count in asdict(task_counts).items():
        flattened_tasks.extend([task] * count)

    #   order flat tasks by the first server benchmark
    ordered_tasks = sorted(flattened_tasks, key=lambda t: -getattr(benchmarks[0], t))
    flattened_tasks = ordered_tasks
    return flattened_tasks

def tasks_to_task_counts(tasks):
    pass

def balance_tasks_optimum(benchmarks: list[ServerBenchmark], tasks: ESDerivTasks, num_workers: int, n_trials: int=500):
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
        BalanceResult
            A collection of load balanced configurations of tasks and times for multiple servers

    '''
    task_counts = tasks.get_task_times()
    flattened_tasks = _flatten_tasks(benchmarks, task_counts)

    best_time = float('inf')
    best_balanced = None
    base_collection = BalancedCollection(num_workers)
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

    print('Time:', time.time() - start_time)

    return best_balanced
    
def balance_tasks_round_robin(benchmarks: list[ServerBenchmark], tasks: ESDerivTasks, num_workers):
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
        MockServerCollection
            A collection of "server" objects with the tasks assigned to them.
    '''
    task_counts = tasks.get_task_times()
    flattened_tasks = _flatten_tasks(benchmarks, task_counts)

    collection = BalancedCollection(num_workers)
    collection._set_server_benchmarks(benchmarks)

    # Assign tasks to workers
    for i, task in enumerate(flattened_tasks):
        server_idx = i % num_workers
        collection._add_task(server_idx, task)
        collection

    return collection


def _debug_plot_assignments_new(task_times, collections: BalancedCollection):
    
    fig, ax = plt.subplots()
    height = 0.8
    for i, server in enumerate(collections.collections):
        left = 0.0
        for task, time in zip(server.tasks, task_times):
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

def _debug_creat_plot(collections_1: BalancedCollection, collections_2: BalancedCollection):

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

def _debug_create_annimation(n_workers, collections_1: BalancedCollection, collections_2: BalancedCollection):
        
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

def _debug_plot_comparisons(name: dict, collections_1: BalancedCollection, collections_2: BalancedCollection):

    fig, ax1, ax2, colors_by_task = _debug_creat_plot(collections_1, collections_2)

    height = 0.8
    for (ax, collections) in zip([ax1, ax2], [collections_1, collections_2]):
        for i, server in enumerate(collections.collections):
            print(i, server.tasks, server.times)
            left = 0.0
            for task, time in zip(server.tasks, server.times):
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

    assignments_1 = balance_tasks_optimum(benchmarks, tasks, num_workers, 500)
    assignments_2 = balance_tasks_round_robin(benchmarks, tasks, num_workers)
    _debug_plot_comparisons("comparison", assignments_1, assignments_2)

if __name__ == "__main__":

    _debug_run_test()

