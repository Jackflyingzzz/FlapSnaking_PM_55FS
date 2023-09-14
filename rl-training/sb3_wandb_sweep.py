import wandb

import argparse
import copy
import json
import os
import sys
import torch.nn as nn

from wandb.integration.sb3 import WandbCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
from stable_baselines3.common.logger import Logger, HumanOutputFormat, DEBUG
# from stable_baselines3.sac import SAC
from sb3_contrib import TQC

from wandb_logger_sb3 import WandbOutputFormat
from IBM_env import IBMEnv
from env_utils import *

def run(number_servers):

    params = json.load(open('params.json', 'r'))
    env_params = params['env_params']

    cwd = os.getcwd()
    exec_dir = os.path.join(cwd, '../Incompact3d_Flaps')
    reset_dir = os.path.join(cwd, 'reset_env')

    env_config = create_env_config(0, None, cwd, env_params, params["solver_params"])

    wandb.init()

    config = {}

    config["learning_rate"] = wandb.config.learning_rate
    config["learning_starts"] = 0
    config["batch_size"] = wandb.config.batch_size
    config["top_quantiles_to_drop_per_net"] = wandb.config.top_quantiles_to_drop

    n_critics = wandb.config.n_critics
    n_quantiles = wandb.config.n_quantiles
    n_actor_layers = wandb.config.n_actor_layers
    n_critic_layers = wandb.config.n_critic_layers
    n_actor_neurons = wandb.config.n_actor_neurons
    n_critic_neurons = wandb.config.n_critic_neurons
    activation_fn = wandb.config.activation_fn

    net_arch = {
            "pi": [n_actor_neurons]*n_actor_layers,
            "qf": [n_critic_neurons]*n_critic_layers
    }

    config["policy_kwargs"] = {
            "n_critics": n_critics,
            "n_quantiles": n_quantiles,
            "net_arch": net_arch,
            "activation_fn": {"tanh": nn.Tanh, "relu": nn.ReLU}[activation_fn]
            }

    config["tau"] = wandb.config.tau
    config["gamma"] = 1 - wandb.config.gamma_complement
    config["train_freq"] = int(wandb.config.train_freq)
    config["target_update_interval"] = int(wandb.config.target_update_interval)
    config["gradient_steps"] = int(wandb.config.gradient_steps)

    config["buffer_size"] = int(wandb.config.buffer_size)
    config["optimize_memory_usage"] = False

    config["ent_coef"] = "auto_" + str(wandb.config.initial_alpha)
    config["target_entropy"] = "auto"

    env = SubprocVecEnv([create_env(env_config, env_config["nb_act"], i) for i in range(number_servers)],
            start_method='spawn')

    model = TQC('MlpPolicy', VecNormalize(env, gamma=config["gamma"]), **config)

    logger = Logger(folder=None,
        output_formats=[WandbOutputFormat(), HumanOutputFormat(sys.stdout)])

    logger.set_level(DEBUG)
    model.set_logger(logger)

    model.load_replay_buffer('buffer_env_init/buffer.pkl')

    model.learn(int(wandb.config.max_timesteps))


if __name__ == "__main__":
    # Argument Parsing
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", "--number-servers", required=True, help="number of servers to spawn", type=int)
    ap.add_argument("-i", "--sweep-id", required=True, help="Sweep ID from wandb", type=str)
    args = vars(ap.parse_args())

    number_servers = args["number_servers"] # Number of parallel environments

    wandb.agent(args["sweep_id"], function= lambda : run(number_servers))

    print("Agent and Runner closed -- Learning complete -- End of script")
    os._exit(0)

