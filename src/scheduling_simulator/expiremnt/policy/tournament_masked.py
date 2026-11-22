"""MaskedSchedulingPolicy for the tournament-in-the-loop environment: skip is a legal pick only in the final slice
of a tournament (observation flag `final_slice`); in the earlier slices a slice winner has to be a job."""
import torch as th

from scheduling_simulator.expiremnt.policy.masked_sched import MaskedSchedulingPolicy


class TournamentMaskedPolicy(MaskedSchedulingPolicy):
    def _compute_mask(self, obs) -> th.Tensor:
        mask = super()._compute_mask(obs)
        if "final_slice" in obs:
            final = obs["final_slice"].reshape(mask.shape[0], -1)[:, 0] > 0.5
            mask = mask.clone()
            mask[:, 0] = mask[:, 0] & final
        return mask
