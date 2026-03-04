"""
Phase 1 Training: Proprioceptive Button-Press
===============================================
Trains with state vector (no vision) to learn muscle coordination.
Fast convergence expected in 200k-500k steps.

Usage:
    cd D:\MyoVision
    python training\train_phase1_proprio.py
    python training\train_phase1_proprio.py --total-timesteps 500000
"""

import os
import sys
import argparse
import numpy as np

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor
from stable_baselines3.common.callbacks import (
    CheckpointCallback, EvalCallback, CallbackList, BaseCallback
)

from envs.button_press_proprio_env import ButtonPressProprioEnv

# --- Wandb (optional) ---
try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False


class SuccessRateCallback(BaseCallback):
    """Logs success rate and distance to wandb/stdout."""
    def __init__(self, log_freq=1000, verbose=0):
        super().__init__(verbose)
        self.log_freq = log_freq
        self.successes = []
        self.distances = []

    def _on_step(self):
        for info in self.locals.get("infos", []):
            if "distance" in info:
                self.distances.append(info["distance"])
            if "success" in info:
                self.successes.append(float(info["success"]))

        if self.num_timesteps % self.log_freq == 0 and self.successes:
            recent = self.successes[-100:]
            rate = np.mean(recent) * 100
            mean_dist = np.mean(self.distances[-100:]) if self.distances else 0

            print(f"  Step {self.num_timesteps:>8d} | "
                  f"Success: {rate:5.1f}% | "
                  f"Mean dist: {mean_dist:.4f}")

            if HAS_WANDB and wandb.run:
                wandb.log({
                    "success_rate": rate,
                    "mean_distance": mean_dist,
                    "timestep": self.num_timesteps,
                })

        return True


class WandbCallback(BaseCallback):
    def _on_step(self):
        if HAS_WANDB and wandb.run and len(self.model.ep_info_buffer) > 0:
            ep = self.model.ep_info_buffer[-1]
            wandb.log({
                "ep_reward": ep["r"],
                "ep_length": ep["l"],
                "timestep": self.num_timesteps,
            })
        return True


def make_env(xml_path, max_steps, randomization, rank, seed=0):
    def _init():
        env = ButtonPressProprioEnv(
            xml_path=xml_path,
            max_steps=max_steps,
            reward_type="staged",
            randomization_level=randomization,
        )
        env.reset(seed=seed + rank)
        return env
    return _init


def train(args):
    print("=" * 60)
    print("Phase 1: Proprioceptive Button-Press Training")
    print("=" * 60)
    print(f"  Obs:           11-dim state vector (no vision)")
    print(f"  Action:        6 muscle activations")
    print(f"  N envs:        {args.n_envs}")
    print(f"  Total steps:   {args.total_timesteps}")
    print(f"  Randomization: {args.randomization}")
    print(f"  Device:        {args.device}")
    print("=" * 60)

    output_dir = os.path.join(args.output_dir, "phase1_proprio")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "best_model"), exist_ok=True)

    # --- Envs ---
    env_fns = [
        make_env(args.xml_path, args.max_steps, args.randomization, i, args.seed)
        for i in range(args.n_envs)
    ]

    if args.n_envs <= 4:
        vec_env = DummyVecEnv(env_fns)
    else:
        vec_env = SubprocVecEnv(env_fns)
    vec_env = VecMonitor(vec_env)

    eval_env = DummyVecEnv([
        make_env(args.xml_path, args.max_steps, "none", 0, args.seed + 1000)
    ])
    eval_env = VecMonitor(eval_env)

    # --- PPO with MLP policy ---
    model = PPO(
        policy="MlpPolicy",
        env=vec_env,
        learning_rate=3e-4,
        n_steps=512,
        batch_size=256,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs=dict(
            net_arch=dict(pi=[256, 256], vf=[256, 256]),
        ),
        verbose=1,
        tensorboard_log=os.path.join(output_dir, "tb_logs"),
        device=args.device,
        seed=args.seed,
    )

    print(f"\nPolicy:\n{model.policy}\n")

    # --- Wandb ---
    if HAS_WANDB and args.wandb:
        wandb.init(
            project="myovision-buttonpress",
            name="phase1-proprio",
            config={
                "phase": 1,
                "obs_type": "proprioceptive",
                "total_timesteps": args.total_timesteps,
                "n_envs": args.n_envs,
                "randomization": args.randomization,
            },
            sync_tensorboard=True,
        )

    # --- Callbacks ---
    callbacks = [
        CheckpointCallback(
            save_freq=max(10000 // args.n_envs, 1),
            save_path=os.path.join(output_dir, "checkpoints"),
            name_prefix="phase1",
        ),
        EvalCallback(
            eval_env,
            best_model_save_path=os.path.join(output_dir, "best_model"),
            log_path=os.path.join(output_dir, "eval_logs"),
            eval_freq=max(5000 // args.n_envs, 1),
            n_eval_episodes=20,
            deterministic=True,
        ),
        SuccessRateCallback(log_freq=2000),
    ]

    if HAS_WANDB and args.wandb:
        callbacks.append(WandbCallback())

    # --- Train ---
    print("Starting Phase 1 training...\n")
    model.learn(
        total_timesteps=args.total_timesteps,
        callback=CallbackList(callbacks),
        progress_bar=True,
    )

    # --- Save ---
    final_path = os.path.join(output_dir, "final_model")
    model.save(final_path)
    print(f"\nPhase 1 complete! Model saved to: {final_path}")
    print(f"Success count: check wandb or tensorboard for success_rate curve")

    if HAS_WANDB and args.wandb:
        wandb.finish()

    vec_env.close()
    eval_env.close()

    return final_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 1: Proprioceptive training")
    parser.add_argument("--xml-path", type=str, default=None)
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--total-timesteps", type=int, default=500_000)
    parser.add_argument("--randomization", type=str, default="none",
                        choices=["none", "medium", "high"])
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="results")
    parser.add_argument("--wandb", action="store_true", help="Enable wandb logging")

    args = parser.parse_args()

    if args.xml_path is None:
        args.xml_path = r"C:\Users\max-r\anaconda3\envs\myovision\lib\site-packages\myosuite\envs\myo\assets\elbow\myoelbow_buttonpress.xml"

    train(args)
