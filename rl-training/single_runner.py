import argparse
import os
import json
import numpy as np
import pandas as pd
from tqdm import trange

import ray
from ray import tune
from ray.rllib.agents.sac import SACTrainer, DEFAULT_CONFIG
from ray.tune.integration.wandb import WandbLogger
from ray.tune.logger import pretty_print

import probes
from IBM_env import IBMEnv
from Incompact3D_params import SolverParams

def one_run(duration, solver_params, baseline):
    nb_actuations = int(np.ceil(env_params['max_iter'] / solver_params.step_iter))
    output_dir = 'baseline_run/' if baseline_run else 'test_run/' 
    
    if os.path.exists(output_dir):
        print(f'Test result directory already exists. Delete {os.path.join(os.getcwd(), output_dir)} before rerunning. Aborting...')
        exit()

    os.makedirs(example_environment.cwd + 'snapshots')

    print("Running baseline") if baseline else print("Start simulation")
    state = example_environment.reset()
    dt = solver_params.dt
    simulation_duration = example_environment.max_iter * dt

    action_step_size = simulation_duration / nb_actuations  # Duration of 1 train episode / actions in 1 episode
    action_steps = int(duration / action_step_size)
    solver_step_iter = int(action_step_size / dt)
    episode_forces = np.empty([action_steps*(solver_step_iter - 2), 2]) # drag idx 0, lift idx 1
    angles = np.empty([action_steps, 2])
    rewards = np.empty((action_steps,))
    t = trange(action_steps)

    for iter in t:
        
        action = agent.compute_action(state, explore=False) if not baseline else np.array(env_params['default_action'])
        state, rw, done, _ = example_environment.step(action)

        if os.path.exists(example_environment.cwd + 'snapshots/snapshot0000.vtr'):
            # Rename snapshots so the solver does not overwrite them
            os.rename(example_environment.cwd + 'snapshots/snapshot0000.vtr', example_environment.cwd + f'snapshots/snapshot{iter}.vtr')
        
        (drag, lift) = example_environment.read_force_output()
        episode_forces[iter*(solver_step_iter - 2):(iter + 1)*(solver_step_iter - 2), 0] = drag
        episode_forces[iter*(solver_step_iter - 2):(iter + 1)*(solver_step_iter - 2), 1] = lift
        angles[iter] = example_environment.prev_angles
        rewards[iter] = rw
        
        t.set_postfix(reward=rw, angles=angles[iter])

    print(f'Average drag: {np.mean(episode_forces[:, 0])}\nAverage lift: {np.mean(episode_forces[:, 1])}')

    df_forces = pd.DataFrame(episode_forces, columns=['Drag', 'Lift'])
    df_angles = pd.DataFrame(angles, columns=['Top Flap', 'Bottom Flap'])
    df_rewards = pd.DataFrame(rewards, columns=['Reward'])

    os.makedirs(output_dir)
    df_forces.to_csv(output_dir + 'forces.csv', sep=';', index=False)
    df_angles.to_csv(output_dir + 'angles.csv', sep=';', index=False)
    df_rewards.to_csv(output_dir + 'rewards.csv', sep=';', index=False)


# Argument Parsing
ap = argparse.ArgumentParser()
ap.add_argument("-r", "--reset-iterations", required=False, help="solver iterations for reset state generation. 0 for no reset generation", type=int, default=0)
ap.add_argument("-b", "--baseline-run", required=False, help="If true, compute values for a run with no control output", type=bool, default=False)
ap.add_argument("-s", "--savedir", required=False, help="Directory into which to save the NN. Defaults to 'saver_data'.", type=str, default='saver_data')
args = vars(ap.parse_args())

params = json.load(open('params.json', 'r'))

reset_iterations = args["reset_iterations"]
baseline_run = args["baseline_run"]
savedir = args["savedir"]

cwd = os.getcwd()
exec_dir = os.path.join(cwd, '../Incompact3d_Flaps')
reset_dir = os.path.join(cwd, 'reset_env')

ray.init(address='auto')

env_params = params['env_params']
solver_params = SolverParams(exec_dir)
probe_layout = probes.ProbeLayout(env_params['probe_layout'], solver_params)
probe_layout.generate_probe_layout_file(os.path.join(exec_dir, 'probe_layout.txt'))

if not reset_iterations == 0:
    IBMEnv.GenerateRestart(reset_dir, exec_dir, reset_iterations)

nb_actuations = int(np.ceil(env_params['max_iter'] / solver_params.step_iter))

config = json.load(open(savedir + '/../params.json', 'r'))
config['callbacks'] = ray.rllib.agents.callbacks.DefaultCallbacks
config['num_workers'] = 0

env_config = {
    'cwd': cwd,
    'exec_dir': exec_dir,
    'reset_dir': reset_dir,
    'probe_layout': probe_layout,
    'solver_params': solver_params,
    'env_params': env_params,
    'logdir': None,
    'index': 512
}

config['env_config'] = env_config

rl_params = params['rl_params']
if baseline_run:
    pass
else:
    example_environment = IBMEnv(env_config)
    agent = SACTrainer(config=config, env=IBMEnv)
    agent.restore(savedir + '/checkpoint-5')

one_run(50, solver_params, baseline_run)
os._exit(0)

        



