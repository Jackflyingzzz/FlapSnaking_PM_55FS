import wandb
from wandb.integration.sb3 import WandbCallback

import time

class WandbTimerCallback(WandbCallback):

    def __init__(
        self,
        verbose: int = 0,
        model_save_path: str = None,
        model_save_freq: int = 0,
        gradient_save_freq: int = 0,
    ):
        super().__init__(verbose,
                            model_save_path,
                            model_save_freq,
                            gradient_save_freq)

        self._rollout_start_time = None
        self._step_start_time = None

    # def _on_rollout_start(self):
    #     self._rollout_start_time = time.time()

    # def _on_step(self):
    #     if self._step_start_time is None:
    #         self._step_start_time = time.time()
    #         return True

    #     now = time.time()
    #     try:
    #         wandb.log({"time/consecutive_step_interval_ms" : 1000*(now - self._step_start_time)})
    #     except Exception as e:
    #         print(e)

    #     self._step_start_time = now

    #     return True

    # def _on_rollout_end(self):
    #     try:
    #         wandb.log({"time/rollout_duration_ms" : 1000*(time.time() - self._rollout_start_time)})
    #     except Exception as e:
    #         print(e)
