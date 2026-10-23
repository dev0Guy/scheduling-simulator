"""NewSchedulingPolicy for sb3-contrib's MaskablePPO.

Same network as NewSchedulingPolicy. MaskableActorCriticPolicy comes first in the MRO, so its
`forward` / `evaluate_actions` / `get_distribution` take the action masks that MaskablePPO stores in
the rollout buffer (they come from the environment's own `action_masks()`); the network itself is
built by NewSchedulingPolicy.__init__. Compare with MaskedSchedulingPolicy, which computes the same
mask from the observation inside the policy.
"""
from sb3_contrib.common.maskable.distributions import make_masked_proba_distribution
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy

from scheduling_simulator.expiremnt.policy.new_sched import NewSchedulingPolicy


class MaskableSchedulingPolicy(MaskableActorCriticPolicy, NewSchedulingPolicy):

    # MaskableActorCriticPolicy would otherwise build the stock MlpExtractor (critic on all features)
    _build_mlp_extractor = NewSchedulingPolicy._build_mlp_extractor

    def __init__(self, observation_space, action_space, lr_schedule, **kwargs):
        NewSchedulingPolicy.__init__(self, observation_space, action_space, lr_schedule, **kwargs)
        # the maskable __init__ is skipped, so install its masked categorical distribution here
        self.action_dist = make_masked_proba_distribution(action_space)
