import gym
import copy
import subprocess
import os
import shutil
import numpy as np
import csv

import io

class RingBuffer():
    "A 1D ring buffer using numpy arrays"
    def __init__(self, length):
        self.data = np.zeros(length, dtype='f')  # Initialise ring array 'data' as length-array of floats
        self.index = 0  # Initialise InPointer as 0 (where new data begins to be written)

    def extend(self, x):
        "adds array x to ring buffer"
        x_indices = (self.index + np.arange(x.size)) % self.data.size  # Find indices that x will occupy in 'data' array
        self.data[x_indices] = x  # Input the new array into ring buffer ('data')
        self.index = x_indices[-1] + 1  # Find new index for next new data

    def get(self):
        "Returns the first-in-first-out data in the ring buffer (returns data in order of introduction)"
        idx = (self.index + np.arange(self.data.size)) % self.data.size
        return self.data[idx]

class IBMEnv(gym.Env):
    
    @classmethod
    def GenerateRestart(cls, reset_dir, exec_dir, reset_iterations):
        if os.path.exists(reset_dir):
            print('Restart environment directory already exists, aborting... Delete the restart directory if you want to generate a new restart file.')
            exit()
        
        os.makedirs(reset_dir)
        shutil.copy2(os.path.join(exec_dir, 'incompact3d.prm'), reset_dir)
        cls.ModifyConfigForReset(reset_dir, reset_iterations)
        os.rename(os.path.join(reset_dir, 'incompact3d_reset.prm'), os.path.join(reset_dir, 'incompact3d.prm'))
        shutil.copy2(os.path.join(exec_dir, 'incompact3d'), reset_dir)
        shutil.copy2(os.path.join(exec_dir, 'probe_layout.txt'), reset_dir) # Copy over the probe layout so we can get the initial observation in the right shape
        # Write 0 angles to alpha transitions
        actions = open(os.path.join(reset_dir, 'alpha_transitions.prm'), 'w')
        actions.write('0.0\n'*4)
        actions.close()

        subprocess.run("./incompact3d", check=True, cwd=reset_dir) # Run the solver to generate the restart file
    
    @classmethod
    def ModifyConfigForReset(cls, cwd, reset_iterations):
        config = open(os.path.join(cwd, 'incompact3d.prm'), 'r')
        reset_config = open(os.path.join(cwd, 'incompact3d_reset.prm'), 'w')

        config_lines = config.readlines()
        config_lines[21] = f'{reset_iterations}  # Last iteration (ifin)\n'
        config_lines[35] = '0              # Read initial flow field ?\n' # Don't read from a reset file
        config_lines[37] = f'{reset_iterations + 1}       # Index of first snapshot file\n' # Don' store snapshots
        reset_config.writelines(config_lines)

        config.close()
        reset_config.close()

    #def __init__(self, env_number, cwd, exec_dir, reset_dir, probe_layout, solver_params, env_params, logdir=None):
    def __init__(self, env_config):
        super().__init__()

        self.env_number = env_config.worker_index if hasattr(env_config, 'worker_index') else env_config['index']

        cwd = env_config['cwd']
        exec_dir = env_config['exec_dir']
        reset_dir = env_config['reset_dir']

        probe_layout = env_config['probe_layout']
        solver_params = env_config['solver_params']
        env_params = env_config['env_params']
        logdir = env_config['logdir']
        self.rl_output = env_config['rl_output']

        self._is_eval = env_config['eval']

        if self._is_eval and logdir is not None:
            logdir += "eval/"

        self.env_params = env_params

        self.prev_angles = np.array(env_params['default_action'], dtype=np.float)
        self.probe_layout = probe_layout
        self.solver_params = solver_params
        self.total_reward = 0

        # For render
        self.plt = None
        self.fig = None

        if logdir is not None:
            self.logdir = logdir
            if not os.path.exists(self.logdir):
                os.makedirs(self.logdir, exist_ok=True)
            self.scalar_output_file = open(self.logdir + f'scalars_env_{self.env_number}.csv', 'w')
            self.scalar_writer = csv.writer(self.scalar_output_file, delimiter=';', lineterminator='\n')
            self.scalar_writer.writerow(['##NaNs encountered for 2 steps at every action steps are not a bug!\n##Solver just doesn\'t provide forces for first 2 steps!'])
            self.scalar_writer.writerow(['episode', 'step', 'lift', 'drag', 'top_angle_deg', 'bottom_angle_deg', 'reward'])
            assert env_params['dump_debug'] % self.solver_params.step_iter == 0, 'dumb_debug has to be divisible by step iteration' # TODO: Implement so this doesnt have to be the case

            if self.env_params['obs_normalisation']:
                 if not os.path.exists(self.logdir + 'norm_vars/'):
                     os.makedirs(self.logdir + 'norm_vars/')
        else:
            self.logdir = None

        self.max_iter = env_params['max_iter']
        self.cur_iter = 0
        self.episode = 0
        self.dump_debug = env_params['dump_debug'] if not type(env_params['dump_debug']) == bool else 0
        self.hbuffer_length = max(env_params['reward_f']['avg_window']*2, 500)
        self.mean_drag_no_control = env_params['baseline_drag']
        self.flap_behaviour = env_params['behaviour_type'] # snaking, clapping or free

        if self.flap_behaviour == 'free':
            self.action_shape = 2
        else:
            self.action_shape = 1
            start_angle = self.env_params['default_action']
            if self.flap_behaviour == 'snaking':
                assert start_angle[0] == start_angle[1], f'When snaking, default angles have to be the same for both flaps. Received: {start_angle}'
            elif self.flap_behaviour == 'clapping':
                assert start_angle[0] == -start_angle[1], f'When clapping, default angles have to be the opposite for each flaps. Received: {start_angle}'

        self.top_flap_limits = env_params['abs_limits']['top']
        self.bottom_flap_limits = env_params['abs_limits']['bottom']

        self.out_of_bounds_penalty = env_params['out_of_bounds_penalty']

        self.history_buffer = {'drag': RingBuffer(self.hbuffer_length), 'lift': RingBuffer(self.hbuffer_length),
                                'top_angle': RingBuffer(self.hbuffer_length), 'bottom_angle': RingBuffer(self.hbuffer_length)}
        self.reset_dir = reset_dir
        # Make a new directory for this env and copy over the necessary files
        env_dir = f'env_{self.env_number}'
        env_dir = "eval/" + env_dir if self._is_eval else env_dir
        if os.path.exists(env_dir):
            shutil.rmtree(env_dir) # Remove the directory if it already exists
        os.makedirs(env_dir)
        
        if(not os.path.exists("episode_averages")):

            os.makedirs("episode_averages",exist_ok=True)

        self.cwd = cwd + '/' + env_dir + '/'
        # Copy over the necessary files TODO: Cleanup the duplicate code between here and self.reset()
        shutil.copy2(os.path.join(exec_dir, 'incompact3d'), self.cwd)
        shutil.copy2(os.path.join(exec_dir, 'incompact3d.prm'), self.cwd)
        shutil.copy2(os.path.join(exec_dir, 'probe_layout.txt'), self.cwd)

        state_shape = self.probe_layout.get_n_probes()

        if self.env_params['obs_normalisation']:
             self.obs_means = np.zeros(state_shape, dtype=float) # Empirical mean
             self.obs_stds = np.ones(state_shape, dtype=float) # Empirical standard deviation
             self.obs_sse = np.zeros(state_shape, dtype=float) # Empirical sum of squares of deviations from mean

        if self.env_params['include_angles_in_state']:
            state_shape = state_shape + 2 # Shape is number of probes + top and bottom angles
        if self.env_params['include_angle_change_in_state']:
            state_shape = state_shape + 2 # Shape is number of probes + top and bottom angle change

        self.observation_space = gym.spaces.Box(shape=(33,), low=-np.inf, high=np.inf) # Shape is number of probes
        
        if self.rl_output == 'angle_change':
            self.action_space = gym.spaces.Box(shape=(self.action_shape,), low=float(self.env_params['delta_limits'][0])*np.pi/180, high=float(self.env_params['delta_limits'][1])*np.pi/180)
        elif self.rl_output == 'angle':
            self.action_space = gym.spaces.Box(shape=(self.action_shape,), low=float(self.top_flap_limits[0])*np.pi/180, high=float(self.top_flap_limits[1])*np.pi/180)
        else:
            assert 'The rl output in code launch_parallel_sb3.py is not in correct format'
        #self.spec.max_episode_steps = int(np.ceil(self.max_iter / self.solver_params.step_iter))

    # Optional
    def close(self):
        super().close()
        if self.logdir is not None:
            self.scalar_output_file.flush()
            self.scalar_output_file.close()

            if self.env_params['obs_normalisation']:
                np.save(self.logdir + f'norm_vars/obs_means_{self.env_number}.npy', self.obs_means) # Store means
                np.save(self.logdir + f'norm_vars/obs_stds_{self.env_number}.npy', self.obs_stds) # Store standard deviations

    def reset(self):
        self.prev_angles = np.array(self.env_params['default_action'], dtype=np.float)
        self.cur_iter = 0
        
        self.episode += 1
        self.history_buffer = {'drag': RingBuffer(self.hbuffer_length), 'lift': RingBuffer(self.hbuffer_length),
                                'top_angle': RingBuffer(self.hbuffer_length), 'bottom_angle': RingBuffer(self.hbuffer_length)}
        if self.logdir is not None:
            self.scalar_output_file.flush()
            if self.env_params['obs_normalisation']:
                 np.save(self.logdir + f'norm_vars/obs_means_{self.env_number}.npy', self.obs_means) # Store means
                 np.save(self.logdir + f'norm_vars/obs_stds_{self.env_number}.npy', self.obs_stds) # Store standard deviations

        if self.env_params['restart_num'] == -1:
            shutil.copy2(os.path.join(self.reset_dir, 'restart'), self.cwd) # Copy over the default reset
            shutil.copy2(os.path.join(self.reset_dir, 'probe.dat'), self.cwd) # Copy over initial observation
        else:
            reset_env = np.random.randint(low=0, high=self.env_params['restart_num'])

            shutil.copy2(os.path.join(self.reset_dir, f'reset_{reset_env}/restart'), self.cwd)
            shutil.copy2(os.path.join(self.reset_dir, f'reset_{reset_env}/probe.dat'), self.cwd)

            angle_file = open(os.path.join(self.reset_dir, f'reset_{reset_env}/alpha_transitions.prm'), 'r')
            angles = angle_file.readlines()
            angle_file.close()

            # Set the starting angles to the angles from the reset alpha_transitions file
            self.prev_angles[0] = float(angles[1])
            self.prev_angles[1] = float(angles[3])

        # Reset flaps to 0 angles
        actions = open(os.path.join(self.cwd, 'alpha_transitions.prm'), 'w')
        actions.write(f'{self.prev_angles[0]}\n{self.prev_angles[0]}\n{self.prev_angles[1]}\n{self.prev_angles[1]}')
        actions.close()
        
        name2 = "RewardsPlot.csv"
        if(not os.path.exists("episode_averages/"+name2)):
                with open("episode_averages/"+name2, "w") as csv_file:
                    spam_writer=csv.writer(csv_file, delimiter=";", lineterminator="\n")
                    spam_writer.writerow(["Episode", "Rewards" ])
                    spam_writer.writerow([self.episode, self.total_reward])
        else:
            with open("episode_averages/"+name2, "a") as csv_file:
                spam_writer=csv.writer(csv_file, delimiter=";", lineterminator="\n")
                spam_writer.writerow([self.episode, self.total_reward])

        if self.plt is not None:
            self.reset_plot()

        state = self.get_next_state(np.zeros(2))
        self.total_reward = 0
        return state

    def reset_plot(self):
        fig, ax = self.plt.subplots(4, sharex=True)
        self.fig = fig
        self.ax = ax

        ax[0].axhline(y=self.mean_drag_no_control, linewidth=1, color='r',
                label='baseline')
        ax[0].grid()
        ax[0].set(ylabel='Drag')
        l_drag = ax[0].plot(np.nan, np.nan, label='control')

        ax[1].axhline(y=0, linewidth=1.5, color='r')
        ax[1].grid()
        ax[1].set(ylabel='Angle')
        l_angle_top = ax[1].plot(np.nan, np.nan, label='top')
        l_angle_bot = ax[1].plot(np.nan, np.nan, label='bottom')

        ax[2].grid()
        ax[2].set(ylabel='Lift')
        l_lift = ax[2].plot(np.nan, np.nan)

        ax[3].grid()
        ax[3].set(xlabel='Time', ylabel='Force Reward')
        l_reward = ax[3].plot(np.nan, np.nan)

        self.lines = {"drag" : l_drag[0],
                        "top_angle" : l_angle_top[0],
                        "bot_angle" : l_angle_bot[0],
                        "lift" : l_lift[0],
                        "reward" : l_reward[0]}

    #def _render(self):
    #    if self.plt is None:
    #        import matplotlib.pyplot as p
    #        self.plt = p
    #        self.plt.ion()

    #        self.reset_plot()

    #    drag, lift = self.read_force_output()
    #    delta = self.solver_params.step_iter
    #    step_range = np.arange(self.cur_iter - delta + 2,
    #            self.cur_iter, dtype=float)/delta

    #    self.lines["drag"].set_data(
    #            np.append(self.lines["drag"].get_xdata(orig=True), step_range),
    #            np.append(self.lines["drag"].get_ydata(orig=True), drag))

    #    self.lines["lift"].set_data(
    #            np.append(self.lines["lift"].get_xdata(orig=True), step_range),
    #            np.append(self.lines["lift"].get_ydata(orig=True), lift))

    #    self.lines["top_angle"].set_data(
    #            np.append(self.lines["top_angle"].get_xdata(orig=True),
    #                self.cur_iter/delta),
    #            np.append(self.lines["top_angle"].get_ydata(orig=True),
    #                self.prev_angles[0]))

    #    self.lines["bot_angle"].set_data(
    #            np.append(self.lines["bot_angle"].get_xdata(orig=True),
    #                self.cur_iter/delta),
    #            np.append(self.lines["bot_angle"].get_ydata(orig=True),
    #                self.prev_angles[1]))

    #    self.lines["reward"].set_data(
    #            np.append(self.lines["reward"].get_xdata(orig=True),
    #                self.cur_iter/delta),
    #            np.append(self.lines["reward"].get_ydata(orig=True),
    #                self.get_reward(np.zeros(2))))

    #    for a in self.ax:
    #        a.autoscale(enable=True)

    #    return self.fig
        # with io.BytesIO() as buff:
        #     self.fig.savefig(buff, format='raw')
        #     buff.seek(0)
        #     data = np.frombuffer(buff.getvalue(), dtype=np.uint8)
        #     w, h = self.fig.canvas.get_width_height()
        #     return data.reshape((int(h), int(w), -1))

        # w, h = self.fig.canvas.get_width_height()
        # # Buffer creation failed, return empty image
        # return np.zeros((int(h), int(w), 3), dtype=np.uint8)


    def step(self, actions_ref):
        # Action represents the change in angle in radians from previous execution
        """	1. Output actions into a file
            2. Execute simulation step
            3. Compute reward
            4. Read probe output
        """
        actions = copy.deepcopy(actions_ref)

        alpha_transitions = open(os.path.join(self.cwd, 'alpha_transitions.prm'), 'w')
        
        if self.flap_behaviour == 'snaking':
            actions = np.append(actions, actions[0]) # When snaking, both flaps have the same angle
        elif self.flap_behaviour == 'clapping':
            actions = np.append(actions, -actions[0]) # When clapping, flap angles are opposite
            
        if self.rl_output == 'angle_change':
            clipped_actions, penalty = self._clip_angles(actions*180/np.pi) # Clip the angles into a valid range
            alpha_transitions.write(f'{self.prev_angles[0]}\n{self.prev_angles[0] + clipped_actions[0]}\n{self.prev_angles[1]}\n{self.prev_angles[1] + clipped_actions[1]}')
            alpha_transitions.close()
        elif self.rl_output == 'angle':
            actions = actions*(180/np.pi)
            alpha_transitions.write(f'{self.prev_angles[0]}\n{actions[0]}\n{self.prev_angles[1]}\n{actions[1]}')
            alpha_transitions.close()
        else:
            assert 'The rl output in code launch_parallel_sb3.py is not in correct format'
            
        self.run_iters_with_dt() # Run the solver   
        
        (drag, lift) = self.read_force_output() #read the drag and lift, and output it to an csv file
        
        self.cur_iter += self.solver_params.step_iter
        if self.rl_output == 'angle_change':
            self.prev_angles += clipped_actions # Action is change in angle, thus current angle is prev_angle + action
        elif self.rl_output == 'angle':
            self.prev_angles = actions # Action is the angle
        else:
            assert 'The rl output in code launch_parallel_sb3.py is not in correct format'
            
       
        self.history_buffer['lift'].extend(lift)
        self.history_buffer['drag'].extend(drag)

        force_rw = self.get_reward(actions)
        if self.logdir is not None:
            # TODO: Implement data dumps only every N steps
            if not self.dump_debug == 0 and self.cur_iter % self.dump_debug == 0:
                row = [self.episode - 1, # episode
                self.cur_iter, # step
                self.history_buffer['lift'].get()[-1], # lift
                self.history_buffer['drag'].get()[-1], # drag
                self.prev_angles[0], # top_angle
                self.prev_angles[1], # bottom_angle
                force_rw] #reward
                self.scalar_writer.writerow(row)

        self.history_buffer['top_angle'].extend(self.prev_angles[0])
        self.history_buffer['bottom_angle'].extend(self.prev_angles[1])

        next_state = self.get_next_state(actions)
        terminal = False #self.cur_iter >= self.max_iter !! We should not be setting terminal to true when reaching max timestep
        if self.rl_output == 'angle_change':
            reward = force_rw + penalty
        elif self.rl_output == 'angle':
            reward = force_rw
        else:
            assert 'The rl output in code launch_parallel_sb3.py is not in correct format'
        self.total_reward = self.total_reward + reward
            
        return next_state, reward, terminal, {}

    def run_iters_with_dt(self, iters=None, dt=None):

        assert (iters is None and dt is None) or (iters is not None and dt is not None), f'Both parameters need to be None or both need to have value, got {iters} and {dt}'  # Ensure both parameters are none or not none together

        if iters is not None and dt is not None:
            solver_prm = open(os.path.join(self.cwd, 'incompact3d.prm'), 'r')
            prm_file = solver_prm.readlines()

            prm_file[21] = str(int(iters)) + '\n'
            prm_file[11] = str(float(dt)) + '  # Time step\n'

            solver_prm = open(os.path.join(self.cwd, 'incompact3d.prm'), 'r')
            solver_prm.writelines(prm_file)
            solver_prm.close()

        subprocess.run("./incompact3d", check=True, cwd=self.cwd, stdout=subprocess.DEVNULL) # Run the solver and wait for it to finish

    def read_probe_output(self):
        # TODO: Convert to binary file
        probe_f = open(os.path.join(self.cwd, 'probe.dat'), 'r')
        probe_vals = np.array([float(line) for line in probe_f.readlines()])
        return probe_vals

    def read_force_output(self):
        # TODO: Convert to binary file
        #force_f = open(os.path.join(self.cwd, 'aerof6.dat'), 'r')
        file_path = os.path.join(self.cwd, 'aerof6.dat')
        file_path_top = os.path.join(self.cwd, 'aerof7.dat')
        file_path_bottom = os.path.join(self.cwd, 'aerof8.dat')
        
        drag_vals = []
        lift_vals = []
        top_normal = []
        bottom_normal = []
        
    
        with open(file_path, 'r') as force_f:
            for line in force_f:
                    values = line.split()
                    try:
                        drag_vals.append(float(values[0]))
                        lift_vals.append(float(values[1]))
                    except (IndexError, ValueError):  # catches lines with not enough values or non-numeric values
                        # Handle or log the error here
                        pass

        with open(file_path, 'r') as force_top:
            for line in force_top:
                    values = line.split()
                    try:
                        top_normal.append(float(values[0]))
                        lift_vals.append(float(values[1]))
                    except (IndexError, ValueError):  # catches lines with not enough values or non-numeric values
                        # Handle or log the error here
                        pass
        

        file_path_angle = os.path.join(self.cwd, 'alpha_transitions.prm')
        with open(file_path_angle, 'r') as angle_a:
            lines = angle_a.readlines()
        
            try:
                top_flap_prev_angle = float(lines[0].strip())
                top_flap_future_angle = float(lines[1].strip())
                bottom_flap_prev_angle = float(lines[2].strip())
                bottom_flap_future_angle = float(lines[3].strip())
            except (IndexError, ValueError):
                # Handle or log the error here. Perhaps set angles to None or another default value.
                top_flap_prev_angle, top_flap_future_angle, bottom_flap_prev_angle, bottom_flap_future_angle = None, None, None, None
                
        drag = np.array(drag_vals)
        lift = np.array(lift_vals)

        output_name = f'debug_{self.env_number}'
        output_path = os.path.join(self.cwd, output_name)
        file_exists = os.path.exists(output_path)
        mode = 'a' if file_exists else 'w'  # Open in append mode if file exists, otherwise write mode

        steps = self.cur_iter
        counter = 0
        with open(output_path, mode) as csv_file:
            spam_writer = csv.writer(csv_file, delimiter=";", lineterminator="\n")

            
            # If file didn't exist, write the header
            if not file_exists:
                spam_writer.writerow(["Episode", "Steps", "Drag", "Lift", "Top Flap Angle", "Bottom Flap Angle"])

            for d, l in zip(drag, lift):
                steps += 1
                counter +=1
                
                #linear interpolation between previous and futrue angle, consistent with incompact3d setting which assume uniform angular velocity
                topflap_current = (counter/self.solver_params.step_iter)*(top_flap_future_angle-top_flap_prev_angle) + top_flap_prev_angle 
                bottomflap_current = (counter/self.solver_params.step_iter)*(bottom_flap_future_angle-bottom_flap_prev_angle) + bottom_flap_prev_angle

                
                spam_writer.writerow([self.episode, steps, d, l, topflap_current, bottomflap_current])

        return drag, lift


    def get_next_state(self, actions_rad):
        next_state = self.read_probe_output() # Get the pressure field information

        if self.env_params['obs_normalisation']:
            step = self.cur_iter // self.solver_params.step_iter + 1
            deviation = next_state - self.obs_means
            temp_mean = self.obs_means + deviation/step # Update the mean
            self.obs_sse += deviation * (next_state - temp_mean) # Update sum of squares of deviations
            next_state = np.divide(next_state - self.obs_means, np.clip(self.obs_stds, a_min=1E-6, a_max=None)) # Normalise the observation according to the previously seen means and variances
            self.obs_means = temp_mean
            if step > 1:
                self.obs_stds = np.sqrt(self.obs_sse/(step - 1))

        if self.env_params['include_angles_in_state']:
            normalised_angles = np.zeros(2, dtype=float)
            normalised_angles[0] = self.linear_transform(self.prev_angles[0], self.top_flap_limits)
            normalised_angles[1] = self.linear_transform(self.prev_angles[1], self.bottom_flap_limits)
            next_state = np.append(next_state, normalised_angles) # Append current angles to state information

        if self.env_params['include_angle_change_in_state']:
            normalised_change = self.linear_transform(actions_rad*180/np.pi, self.env_params['delta_limits'])
            next_state = np.append(next_state, normalised_change)

        return next_state

    def linear_transform(self, x, bounds):
         return 2*(x - bounds[1])/(bounds[1] - bounds[0]) + 1

    def get_reward(self, actions):
        # TODO: Allow user to select from a range of different reward functions
        reward_f = self.env_params['reward_f']['type']
        gamma = self.env_params['reward_f']['gamma']
        avg_window = self.env_params['reward_f']['avg_window']
        angle_weight = self.env_params['reward_f']['angle_weight']
        angle_change_weight = self.env_params['reward_f']['angle_change_weight']

        drag_window = self.history_buffer['drag'].get()[-min(self.cur_iter - 2 * round(self.cur_iter / self.solver_params.step_iter), avg_window):] # Mean drag over the window length defined in params.json
        lift_window = self.history_buffer['lift'].get()[-min(self.cur_iter - 2 * round(self.cur_iter / self.solver_params.step_iter), avg_window):] # Mean drag over the window length defined in params.json

        if self.env_params['reward_f']['filter']:
            window = self.env_params['reward_f']['filter_window']
            polyorder = 1
            
        if reward_f == 'quad_drag':
            reward = -np.mean(drag_window**2) + self.mean_drag_no_control**2
        elif reward_f == 'drag_avg_abs_lift':
            reward = -np.mean(drag_window) + self.mean_drag_no_control - gamma * np.mean(np.absolute(lift_window))
        elif reward_f == 'drag_abs_avg_lift':
            reward = -np.mean(drag_window) + self.mean_drag_no_control - gamma * np.absolute(np.mean(lift_window))
        elif reward_f == 'drag_quad_lift_quad':
            reward = -(np.mean(drag_window))**2 + self.mean_drag_no_control - gamma * (np.mean(lift_window))**2
        elif reward_f == 'drag_lift_angle':
            reward = -np.mean(drag_window) + self.mean_drag_no_control - gamma * np.absolute(np.mean(lift_window)) - 2.5E-4 * (np.sum(np.absolute(self.prev_angles)))
        elif reward_f == 'drag_lift_angle':
             reward = -np.mean(drag_window) + self.mean_drag_no_control - gamma * np.absolute(np.mean(lift_window)) - angle_weight * (np.sum(np.absolute(self.prev_angles)))
        elif reward_f == 'drag_lift_angle_change':
             reward = -np.mean(drag_window) + self.mean_drag_no_control - gamma * np.absolute(np.mean(lift_window)) + angle_change_weight * np.sum(np.absolute(actions))

        return reward / 4

    def _clip_angles(self, actions_deg): # Ensure both flaps are in a valid range, accepts angles in DEGREES
        new_angles = self.prev_angles + actions_deg

        penalty = 0
        clipped_actions = actions_deg

        if ((new_angles[0] < self.top_flap_limits[0] or new_angles[0] > self.top_flap_limits[1]) or 
            (new_angles[1] < self.bottom_flap_limits[0] or new_angles[1] > self.bottom_flap_limits[1])):
            
            clipped_new_angles = np.zeros(2)
            # Clip angles between the desired range
            clipped_new_angles[0] = max(min(new_angles[0], self.top_flap_limits[1]), self.top_flap_limits[0])
            clipped_new_angles[1] = max(min(new_angles[1], self.bottom_flap_limits[1]), self.bottom_flap_limits[0])

            clipped_actions = (clipped_new_angles - self.prev_angles)

            # Discourage this action
            penalty = self.out_of_bounds_penalty

        return (clipped_actions, penalty)
    
    def compute_angles_at_step(self, actions, step):
        bounded_step = step
        # Calculate the angles over 1 action assuming linear hold
        top_angles = actions[0] * bounded_step / self.solver_params.step_iter + self.prev_angles[0]
        bottom_angles = actions[1] * bounded_step / self.solver_params.step_iter + self.prev_angles[1]

        return (top_angles, bottom_angles)
