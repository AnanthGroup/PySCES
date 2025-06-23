import time

_AU_2_FS = 0.024188843666478416

class _TrajectoryTimer:
    '''
    A simple timer to track the elapsed time of a simulation trajectory.
    It calculates the total runtime elapsed, trajectory time elapsed, trajectory 
    time-rate (time-step per hour), trajectory time left, and estimated runtime left.
    '''
    def __init__(self):
        self.start_time = None
        self.end_time = None
        self.start_traj_time = None

    def update(self, traj_time: float, end_traj_time: float):
        '''
        Update the trajectory timer with the current trajectory time.

        Parameters
        ----------
        traj_time : float
            The current trajectory time in atomic units (a.u.).
        end_traj_time : float
            The trajectory time in which the trajectory will end(a.u.).
        '''

        #   this must be the first time we are calling update
        if self.start_traj_time is None:
            self.start_time = time.time()
            self.start_traj_time = traj_time
            return

        diff_traj_time = traj_time - self.start_traj_time
        if diff_traj_time <= 0:
            raise ValueError("Simulation time must be greater than the initial simulation time.")

        total_time = time.time()
        total_time_elapsed = total_time - self.start_time
        traj_time_left = (end_traj_time - traj_time)* _AU_2_FS
        traj_time_elapsed = (traj_time - self.start_traj_time)* _AU_2_FS

        if traj_time_left < 0:
            traj_time_rate = 0
            est_time_left = 0
        else:
            traj_time_rate = traj_time_elapsed / total_time_elapsed
            est_time_left = traj_time_left / traj_time_rate


        print()
        print('         Trajectory Timer:')
        print('-------------------------------------------')
        print(f'{"Total runtime elapsed:":<24} {total_time_elapsed:>8.2f} seconds')
        print(f'{"Trajectory time elapsed:":<24} {traj_time_elapsed:>8.2f} fs')
        print(f'{"Trajectory time rate:":<24} {traj_time_rate * 3600:>8.2f} fs/hr')
        print(f'{"Trajectory time left:":<24} {traj_time_left:>8.2f} fs')
        print(f'{"Estimated runtime left:":<24} {est_time_left / 3600:>8.2f} hrs')
        print('-------------------------------------------')
        print()

traj_timer = _TrajectoryTimer()