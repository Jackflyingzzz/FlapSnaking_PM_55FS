from ray.rllib.agents.callbacks import DefaultCallbacks
import numpy as np

class BetaCallback(DefaultCallbacks):

    def on_train_result(self, *, trainer, result, **kwarg):
        result["current_beta"] = trainer.config["prioritized_replay_beta"].val


    # def on_learn_on_batch(
    #     self, *, policy, train_batch, result, **kwargs):
    #     result["mean_actions_in_train_batch"] = np.mean(train_batch["actions"])
