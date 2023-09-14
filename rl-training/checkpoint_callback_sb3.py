import os

from stable_baselines3.common.callbacks import CheckpointCallback
from collections import deque
class CheckpointLastN(CheckpointCallback):

    def __init__(self, save_freq: int, num_to_keep: int, save_buffer: bool,
            save_env_stats: bool, save_path: str, name_prefix: str = "rl_model", verbose: int = 0):

        super().__init__(save_freq, save_path, name_prefix, verbose)
        self._cur_saved = deque()
        self._num_to_keep = num_to_keep
        self._save_buffer = save_buffer
        self._save_env_stats = save_env_stats

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            path = os.path.join(self.save_path, f"{self.name_prefix}_{self.num_timesteps}_steps")

            self.model.save(path)
            self._cur_saved.append(path)

            if self.verbose > 1:
                print(f"Saving model checkpoint to {path}")

            if self._save_buffer:
                self.model.save_replay_buffer(path + '_buffer')

            if self._save_env_stats:
                self.training_env.save(path + '_env_stats')

            if len(self._cur_saved) > self._num_to_keep:
                to_remove = self._cur_saved.popleft()

                if os.path.exists(to_remove + '.zip'):
                    os.remove(to_remove + '.zip')
                if os.path.exists(to_remove + '_buffer.pkl'):
                    os.remove(to_remove + '_buffer.pkl')
                if os.path.exists(to_remove + '_env_stats.pkl'):
                    os.remove(to_remove + '_env_stats.pkl')

        return True
