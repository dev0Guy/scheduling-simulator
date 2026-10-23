"""NewSchedulingPolicy with invalid-action masking (Huang and Ontanon, 2020).

The logits of actions that violate a hard constraint are set to a huge negative number before
the categorical distribution is built, so such actions get probability 0, never receive
gradient and are never counted in the entropy bonus. Valid actions:

  * 0 (skip time) is always valid;
  * "start job j on machine m" is valid iff job j is PENDING and its usage fits in machine m's
    free capacity on every resource and every tick (the same test the environment applies, see
    Scheduler.options), so a valid start never fails.
  * optional `forbid_idle_skip=True`: skip is ALSO invalid when no job is running and some pending
    job fits (an idle machine with work that fits). Some deterministic policies keep choosing skip
    there for tens of ticks until the step limit. The exact solver skips in only 2.5% of such states
    (waiting for a shorter job that is about to arrive); in the sub-case where no job is still to
    arrive it never skips (0 of 5,448 states), so the rule is nearly lossless but not exactly.

The mask is computed from the observation inside the policy, so the sampling at rollout time,
the log-probabilities / entropy in the PPO update and the deterministic evaluation all use the
same mask.
"""
import torch as th

from scheduling_simulator.core.job import JobStatus
from scheduling_simulator.expiremnt.policy.new_sched import NewSchedulingPolicy

_NEG = -1e8


class MaskedSchedulingPolicy(NewSchedulingPolicy):

    _action_mask = None
    forbid_idle_skip = False      # class default, so models saved before this option still load

    def __init__(self, *args, forbid_idle_skip: bool = False, **kwargs):
        self.forbid_idle_skip = forbid_idle_skip
        super().__init__(*args, **kwargs)

    def _compute_mask(self, obs) -> th.Tensor:
        """(batch, 1 + n_machines * n_jobs) bool; action = 1 + machine * n_jobs + job."""
        usage = obs["machines_usage"].float()                       # (B, m, R, T)
        cap = obs["machines_capacity"].float()
        jobs = obs["jobs_usage"].float()                            # (B, n, R, T)
        free = cap - usage
        # fits[b, m, j] = job j fits machine m on every resource and tick
        fits = (free[:, :, None] - jobs[:, None]).amin(dim=(3, 4)) >= 0     # (B, m, n)
        pending = obs["status"].round().long() == int(JobStatus.PENDING)    # (B, n)
        valid = fits & pending[:, None, :]
        B = valid.shape[0]
        skip = th.ones(B, 1, dtype=th.bool, device=valid.device)
        if self.forbid_idle_skip:
            nothing_running = ~(obs["status"].round().long() == int(JobStatus.RUNNING)).any(dim=1)   # (B,)
            can_start = valid.reshape(B, -1).any(dim=1)
            skip = (~(nothing_running & can_start)).unsqueeze(1)
        return th.cat([skip, valid.reshape(B, -1)], dim=1)

    def _get_action_dist_from_latent(self, latent_pi):
        logits = self.action_net(latent_pi)
        if self._action_mask is not None:
            logits = logits.masked_fill(~self._action_mask, _NEG)
        return self.action_dist.proba_distribution(action_logits=logits)

    def _with_mask(self, fn, obs, *args, **kwargs):
        self._action_mask = self._compute_mask(obs)
        try:
            return fn(obs, *args, **kwargs)
        finally:
            self._action_mask = None

    def forward(self, obs, deterministic=False):
        return self._with_mask(super().forward, obs, deterministic)

    def evaluate_actions(self, obs, actions):
        return self._with_mask(super().evaluate_actions, obs, actions)

    def get_distribution(self, obs):
        return self._with_mask(super().get_distribution, obs)
