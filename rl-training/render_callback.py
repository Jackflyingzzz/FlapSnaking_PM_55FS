from ray.rllib.agents.callbacks import DefaultCallbacks
import wandb

class RenderCallback(DefaultCallbacks):

    def on_episode_step(self,
                        *,
                        worker,
                        base_env,
                        policies,
                        episode,
                        **kwargs):


        # Run only on one worker
        if worker.worker_index == 1:
            env = base_env.get_sub_environments()[0]
            env._render()

    def on_episode_end(self,
                        *,
                        worker,
                        base_env,
                        policies,
                        episode,
                        **kwargs):

        if worker.worker_index == 1:
            env = base_env.get_sub_environments()[0]
            img = env._render()
            episode.media["img"] = wandb.Image(img)
        # wandb.log({"render/img" : img}
