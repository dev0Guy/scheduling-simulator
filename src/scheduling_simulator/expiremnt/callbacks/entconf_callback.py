from stable_baselines3.common.callbacks import BaseCallback

class EntCoefScheduler(BaseCallback):
    def __init__(self, start: float = 0.02, end: float = 0.002, total_timesteps: int = 200_000, hold_fraction: float = 0.0):
        super().__init__()
        self.start = start
        self.end = end
        self.total_timesteps = total_timesteps
        self.hold_fraction = hold_fraction  # keep `start` unchanged for this share of training, then decay linearly

    def _on_step(self) -> bool:
        progress = min(self.num_timesteps / self.total_timesteps, 1.0)
        if self.hold_fraction > 0.0:
            progress = max(0.0, (progress - self.hold_fraction) / (1.0 - self.hold_fraction))
        self.model.ent_coef = self.start + (self.end - self.start) * progress
        return True
