"""Masked PPO from scratch with the tournament inside the training loop (one model per job-pool size N).

    N_MACHINES=1 N_JOBS=20 TRAIN_SEED=3 JOB_USAGE_MODE=random python .../example_masked_tournament_loop.py

The environment has N jobs; the model always sees a 10-slot window and a real decision is a whole knock-out
tournament over the pool (envioremnt/wrappers/tournament_env.py): only the final pick of the tournament is applied
and rewarded. Recipe (optimizer, lr, entropy, 2M steps - here: model picks) as the other models. The model is evaluated
with scripts/tournament/tournament_eval.py (MODEL=models/<run>/final_model.zip, same N).
"""
import os

from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from scheduling_simulator.envioremnt.wrappers.tournament_env import TournamentEnv
from scheduling_simulator.expiremnt.policy.tournament_masked import TournamentMaskedPolicy
from scheduling_simulator.expiremnt.runners.train_runner import ExperimentRunner

n_machines = int(os.environ.get("N_MACHINES", "1"))
n_jobs = int(os.environ.get("N_JOBS", "20"))
eval_episodes = int(os.environ.get("EVAL_EPISODES", "300"))
steps = int(os.environ.get("TRAIN_STEPS", "2000000"))
init_policy = os.environ.get("INIT_POLICY")        # state_dict of a model trained on 10 jobs (fine-tune instead of from scratch)
max_time = 110 if init_policy else 11 * n_jobs     # the pretrained model normalises times by 110

config = {"n_machines": n_machines, "n_jobs": n_jobs, "n_resource": 3, "n_time": 10, "max_capacity": 255}


class TournamentLoopRunner(ExperimentRunner):
    def generate_enviroemnt(self, path, with_video=False, n_env=4, sjf_relative=False, reward_scale=1.0,
                            drop_constant_rewards=False, optimal_gap_reward=False):
        return DummyVecEnv([lambda: Monitor(TournamentEnv(self.config)) for _ in range(n_env)])


warm_steps = int(os.environ.get("CRITIC_WARMUP", "0"))
warm_lr = float(os.environ.get("CRITIC_WARMUP_LR", "3e-4"))
if warm_steps:
    # Critic warm-up: for the first `warm_steps` only the value network (vf_features_extractor + value_net) is trained,
    # with its own learning rate; the actor (pi_features_extractor, action_net) is frozen, so the pretrained policy is
    # not pushed around by advantages computed from a critic that does not know the tournament MDP yet.
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback, CallbackList

    class CriticWarmup(BaseCallback):
        def _on_training_start(self):
            self.model.policy.set_training_mode(True)
            self._sched = self.model.lr_schedule
            self.model.lr_schedule = lambda progress: warm_lr
            self.warm = True
            for name, prm in self.model.policy.named_parameters():
                prm.requires_grad_(name.startswith(("vf_features_extractor", "value_net")))
            print(f"critic warm-up for {warm_steps} steps; trainable:",
                  sorted({n.split('.')[0] for n, q in self.model.policy.named_parameters() if q.requires_grad}))

        def _on_step(self):
            if self.warm and self.num_timesteps >= warm_steps:
                self.warm = False
                self.model.lr_schedule = self._sched
                for prm in self.model.policy.parameters():
                    prm.requires_grad_(True)
                print(f"critic warm-up done at {self.num_timesteps}: everything trainable")
            return True

    _learn = PPO.learn

    def _learn_with_warmup(self, *a, callback=None, **k):
        return _learn(self, *a, callback=CallbackList([callback, CriticWarmup()]), **k)

    PPO.learn = _learn_with_warmup

runner = TournamentLoopRunner(
    config,
    train_steps=steps,
    evalution_steps=eval_episodes,
    policy_kwargs={
        "embedding_dim": 256,
        "n_machines": n_machines,
        "max_time": float(max_time),
        "per_resource_fit": True,
        "pair_features": True,
        "dynamic_machine": True,
        "forbid_idle_skip": True,        # no skip while the cluster is idle and a job fits
    },
    policy_class=TournamentMaskedPolicy,
    max_time=max_time,
    run_with_wandb=True,
    eval_every_steps=100_000,
    n_envs=8,
    n_steps=512,
    init_policy_path=init_policy,
    learning_rate=float(os.environ.get("FT_LR", 1e-5 if init_policy else 2e-5)),
    lr_decay=True,
    target_kl=0.03,
    n_eval_episodes=10,
    ent_coef_start=float(os.environ.get("FT_ENT_START", 0.01 if init_policy else 0.05)),
    ent_coef_end=float(os.environ.get("FT_ENT_END", 0.005 if init_policy else 0.01)),
)

runner.run()
