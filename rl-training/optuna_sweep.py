import argparse
import pickle as pkl
from typing import Any, Dict

import gym
import json
import numpy as np
import os
import sys
import torch
import torch.nn as nn
import wandb

from sb3_contrib import TQC
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.utils import safe_mean
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
from stable_baselines3.common.logger import Logger, HumanOutputFormat, DEBUG

import optuna
from optuna.pruners import HyperbandPruner
from optuna.samplers import TPESampler

from wandb_logger_sb3 import WandbOutputFormat
from IBM_env import IBMEnv
from env_utils import *

N_STARTUP_TRIALS = 2
N_TIMESTEPS = 9216
TIMEOUT = int(60*60*24 - 60*30)  # 23hrs 30 minutes
CONN_STR = "mysql+pymysql://sCYHuWLSrQ:zrqb1m7Uoa@remotemysql.com:3306/sCYHuWLSrQ"

DEFAULT_HYPERPARAMS = {
    "learning_starts" : 960,
    "buffer_size" : int(1e5),
    "optimize_memory_usage" : False,
    "verbose" : 2
}


def sample_tqc_params(trial: optuna.Trial) -> Dict[str, Any]:
    """Sampler for A2C hyperparameters."""
    learning_rate = trial.suggest_float("lr", 1e-5, 1, log=True)
    batch_size = trial.suggest_int("batch_size", 32, 512, log=True)
    quantiles_to_drop = trial.suggest_int("top_quantiles_to_drop", 1, 8)
    n_critics = trial.suggest_int("n_critics", 1, 8)
    n_quantiles = trial.suggest_int("n_quantiles", 5, 40)

    n_actor_layers = trial.suggest_int("n_actor_layers", 1, 4)
    n_critic_layers = trial.suggest_int("n_critic_layers", 1, 4)
    n_actor_neurons = trial.suggest_int("n_actor_neurons", 64, 1024)
    n_critic_neurons = trial.suggest_int("n_critic_neurons", 64, 1024)
    activation_fn = trial.suggest_categorical("activation_fn", ["tanh", "relu"])

    tau = trial.suggest_float("tau", 5e-5, 5e-2, log=True)
    gamma = 1.0 - trial.suggest_float("gamma", 0.0001, 0.1, log=True)
    train_freq = trial.suggest_int("train_freq", 1, 128)
    target_update_interval = trial.suggest_int("target_update_interval", 1, 64)
    gradient_steps = trial.suggest_int("gradient_steps", 1, 512, log=True)
    alpha_init = trial.suggest_float("ent_coef", 0.00000001, 1, log=True)

    # Display true values
    trial.set_user_attr("gamma_", gamma)

    net_arch = {
            "pi": [n_actor_neurons]*n_actor_layers,
            "qf": [n_critic_neurons]*n_critic_layers
            }

    activation_fn = {"tanh": nn.Tanh, "relu": nn.ReLU}[activation_fn]

    return {
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "top_quantiles_to_drop_per_net": quantiles_to_drop,
        "policy_kwargs": {
            "n_critics": n_critics,
            "n_quantiles": n_quantiles,
            "net_arch": net_arch,
            "activation_fn": activation_fn
        },
        "tau": tau,
        "gamma": gamma,
        "train_freq": train_freq,
        "target_update_interval": target_update_interval,
        "gradient_steps": gradient_steps,
        "ent_coef": "auto_" + str(alpha_init)
    }


class TrialCallback(BaseCallback):
    """Callback used for evaluating and reporting a trial."""

    def __init__(
        self,
        trial: optuna.Trial,
        verbose: int = 0,
    ):
        super().__init__(verbose=verbose)

        self.trial = trial
        self.is_pruned = False
        self.step = 0
        self.last_mean_reward = np.nan

    def _on_step(self) -> bool:
        num_ep = 0
        for done in self.locals["dones"]:
            if done:
                num_ep += 1

        if num_ep > 0:
            self.step += 1
            self.last_mean_reward = safe_mean([ep_info["r"] for ep_info in self.locals["self"].ep_info_buffer])
            self.trial.report(self.last_mean_reward, self.step)
            wandb.log({"tracked_last_mean_reward" : self.last_mean_reward})

        # Prune trial if need
        if self.trial.should_prune():
            self.is_pruned = True
            return False

        return True


def make_objective(env):

    def objective(trial: optuna.Trial) -> float:

        kwargs = DEFAULT_HYPERPARAMS.copy()
        # Sample hyperparameters
        kwargs.update(sample_tqc_params(trial))
        # Create the RL model
        wandb.config.update(kwargs.copy())
        model = TQC('MlpPolicy', VecNormalize(env, gamma=kwargs['gamma']), **kwargs)
        # Create the callback that will periodically evaluate
        # and report the performance
        eval_callback = TrialCallback(
            trial,
            verbose=2
        )

        logger = Logger(folder=None,
            output_formats=[WandbOutputFormat(), HumanOutputFormat(sys.stdout)])

        logger.set_level(DEBUG)
        model.set_logger(logger)

        model.load_replay_buffer('buffer_env_init/buffer.pkl')

        nan_encountered = False
        try:
            model.learn(N_TIMESTEPS, callback=eval_callback)
        except (AssertionError, ValueError) as e:
            # Sometimes, random hyperparams can generate NaN
            print(e)
            nan_encountered = True
        finally:
            # Free memory
            model.env.close()

        # Tell the optimizer that the trial failed
        if nan_encountered:
            return float("nan")

        if eval_callback.is_pruned:
            raise optuna.exceptions.TrialPruned()

        return eval_callback.last_mean_reward

    return objective


if __name__ == "__main__":
    # Argument Parsing
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", "--number-servers", required=True, help="number of servers to spawn", type=int)
    ap.add_argument("-u", "--study-name", required=True, help="Name of the optuna study", type=str, default=None)
    args = vars(ap.parse_args())

    number_servers = args["number_servers"] # Number of parallel environments
    savedir = 'saver_data'
    logdir = None

    params = json.load(open('params.json', 'r'))
    env_params = params['env_params']

    cwd = os.getcwd()
    env_config = create_env_config(0, logdir, cwd, env_params, params["solver_params"])

    success = False

    while not success:
        try:
            run = wandb.init(
                    project="IBM-Flaps-SB3",
                    dir=savedir,
                    name="TQC_" + str(os.getenv('PBS_JOBID')),
                    monitor_gym=True,
                    save_code=True,
                    force=True,
                    reinit=True
            )

            success = True
        except wandb.errors.UsageError as e:
            print("Got exception: ")
            print(e)
            print("Retrying...")
            success = False

    sampler = TPESampler(n_startup_trials=N_STARTUP_TRIALS, constant_liar=True)
    # Do not prune before 1/3 of the max budget is used
    pruner = HyperbandPruner(
        min_resource=2,
        max_resource=5,
        reduction_factor=3
    )

    storage = optuna.storages.RDBStorage(url=CONN_STR, heartbeat_interval=5*60)

    study = optuna.create_study(study_name=args["study_name"], sampler=sampler, pruner=pruner,
            direction="maximize", load_if_exists=True, storage=storage)

    env = SubprocVecEnv([create_env(env_config, env_config["nb_act"], i) for i in range(number_servers)],
            start_method='spawn')

    try:
        study.optimize(make_objective(env), n_trials=1, n_jobs=1, timeout=TIMEOUT)
    except KeyboardInterrupt:
        pass
