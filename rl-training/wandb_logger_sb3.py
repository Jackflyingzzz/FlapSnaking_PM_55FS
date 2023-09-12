import wandb

from stable_baselines3.common.logger import KVWriter, Video, Figure, Image
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch as th

class WandbOutputFormat(KVWriter):

    def write(self, key_values: Dict[str, Any], key_excluded: Dict[str, Union[str, Tuple[str, ...]]], step: int = 0) -> None:

        to_log = {}
        for (key, value), (_, excluded) in zip(sorted(key_values.items()), sorted(key_excluded.items())):

            if excluded is not None and "tensorboard" in excluded:
                continue

            if isinstance(value, np.ScalarType):
                to_log[key] = value
                # wandb.log({key : value}, step=step, commit=False)

            if isinstance(value, th.Tensor):
                to_log[key] = wandb.Histogram(value)
                # wandb.log({key : wandb.Histogram(value)}, step=step, commit=False)

            if isinstance(value, Video):
                to_log[key] = wandb.Video(value.frames.cpu().detach().numpy(), fps=value.fps)
                # wandb.log({key : wandb.Video(value.frames.cpu().detach().numpy(),
                #     fps=value.fps)},
                #     step=step, commit=False)

            if isinstance(value, Figure):
                to_log[key] = wandb.Plotly(value.figure)
                # wandb.log({key : wandb.Plotly(value.figure)}, step=step, commit=False)

            if isinstance(value, Image):
                # Flush the output
                # wandb.log(commit=True)
                raise NotImplementedError

        # Log the output
        wandb.log(to_log, commit=True, step=step)
