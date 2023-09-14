import argparse
import copy
import json
import os
import sys

import numpy as np
from sb3_contrib import TQC
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize, VecFrameStack
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import Logger, HumanOutputFormat, DEBUG
from stable_baselines3.sac import SAC

from gym.wrappers.time_limit import TimeLimit

from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback

import probes
from IBM_env import IBMEnv
from Incompact3D_params import SolverParams


def create_env(config, horizon, n_env):

    def _init():
        cf = copy.deepcopy(config)
        cf['index'] = n_env

        return Monitor(TimeLimit(IBMEnv(cf), max_episode_steps=horizon))

    return _init

if __name__ == '__main__':
    # Argument Parsing
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", "--number-servers", required=True, help="number of servers to spawn", type=int)
    ap.add_argument("-r", "--reset-iterations", required=False, help="solver iterations for reset state generation. 0 for no reset generation", type=int, default=0)
    ap.add_argument("-l", "--logdir", required=False, help="Directory into which to write the output debug logs. Defaults to None.", type=str, default=None)
    ap.add_argument("-s", "--savedir", required=False, help="Directory into which to save the NN. Defaults to 'saver_data'.", type=str, default='saver_data')
    ap.add_argument("-t", "--restore", required=False, help="Checkpoint directory from which to restore", type=str, default=None)
    args = vars(ap.parse_args())

    number_servers = args["number_servers"] # Number of parallel environments
    reset_iterations = args["reset_iterations"]
    logdir = args["logdir"]
    savedir = args["savedir"]
    restoredir = args["restore"]


    params = json.load(open('params.json', 'r'))
    env_params = params['env_params']

    cwd = os.getcwd()
    exec_dir = os.path.join(cwd, 'Incompact3d_Flaps')
    reset_dir =  os.path.join(cwd, 'reset_env')
    # reset_dirs = '/rds/general/user/pm519/home/IBM-Flaps-RL/restarts' #os.path.join(cwd, 'restarts')
    print(reset_dir)
    solver_params = SolverParams(exec_dir)
    (eq_params, key_diff) = solver_params.compare_to_json(params['solver_params'])
    assert eq_params, f'Parameters defined in params.json are not the same as the ones in incompact3d.prm. Difference found for keys: {key_diff}'

    reset_solver_params = SolverParams(exec_dir)
    (eq_params, key_diff) = SolverParams.compare(solver_params, reset_solver_params)
    assert eq_params, f'Solver parameters are not the same as reset env parameters. Produce a new reset environment with the proper settings. Difference found for keys: {key_diff}'

    probe_layout = probes.ProbeLayout(env_params['probe_layout'], solver_params)
    probe_layout.generate_probe_layout_file(os.path.join(exec_dir, 'probe_layout.txt'))

    # quit()

    if not reset_iterations == 0:
        IBMEnv.GenerateRestart(reset_dir, exec_dir, reset_iterations)

    rl_output = 'angle_change' #('angle_change' or 'angle')

    env_config = {
        'cwd': cwd,
        'exec_dir': exec_dir,
        'reset_dir': reset_dir,
        'probe_layout': probe_layout,
        'solver_params': solver_params,
        'env_params': env_params,
        'logdir': logdir,
        'rl_output': rl_output,
        'eval': False
    }

    nb_actuations = int(np.ceil(env_params['max_iter'] / solver_params.step_iter))

    config = {}

    config["learning_rate"] = 1e-4
    config["learning_starts"] = (1 * nb_actuations * number_servers) if restoredir is None else 0
    config["batch_size"] = 128

    config["tau"] = 5e-3
    config["gamma"] = 0.99
    config["train_freq"] = 1
    config["target_update_interval"] = 1
    config["gradient_steps"] = 48

    config["buffer_size"] = int(10e5)
    config["optimize_memory_usage"] = False

    config["ent_coef"] = "auto_0.01"
    config["target_entropy"] = "auto"
    policy_kwargs = dict(net_arch=dict(pi=[512,512,512], qf=[512,512,512]))

    #config["verbose"] = 2 # DEBUG

    mode = 'train'
    checkpoint_callback = CheckpointCallback(
                                            save_freq=max(20, 1),
                                            #num_to_keep=5,
                                            #save_buffer=True,
                                            #save_env_stats=True,
                                            #save_replay_buffer=True, # This is not tested on 31 Oct 2022, may be useful for resume
                                            save_vecnormalize=True,
                                            save_path=savedir,
                                            name_prefix='TQC_Flaps_Model_Snaking_VecFrameStack')

    env = SubprocVecEnv([create_env(env_config, nb_actuations, i) for i in range(number_servers)],
            start_method='spawn')
    env = VecFrameStack(env, n_stack=55)
    env = VecNormalize(env, gamma=0.99)

    if mode=='train':
        model = TQC('MlpPolicy', env, policy_kwargs=policy_kwargs,  tensorboard_log=savedir, device='auto', **config)
        model.learn(10000000, callback=[checkpoint_callback], log_interval=1)
    elif mode == 'restart':
        checkpoint_callback = CheckpointCallback(
                                            save_freq=max(20, 1),
                                            #num_to_keep=5,
                                            #save_buffer=True,
                                            #save_env_stats=True,
                                            #save_replay_buffer=True, # This is not tested on 31 Oct 2022, may be useful for resume
                                            save_vecnormalize=True,
                                            save_path=savedir,
                                            name_prefix='TQC_Flaps_Model_Snaking_restart_VecFrameStack')
        model = TQC.load('/rds/general/user/cx220/home/FlapSnaking_PM_VecFrameStack_Gpu/rl-training/saver_data/TQC_Flaps_Model_Snaking_VecFrameStack_99360_steps.zip')
        model.set_env(env)
        model.learn(10000000, callback=[checkpoint_callback], log_interval=1)


    print("Agent and Runner closed -- Learning complete -- End of script")
    os._exit(0)
