"""
MyoArm Phase 1: Proprioceptive Training
==========================================
Full arm (32 muscles) learns button press with state vector.
Button spawns at random positions on the table each episode.

Usage:
    cd D:\MyoVision
    python training\train_myoarm_phase1.py --wandb
    python training\train_myoarm_phase1.py --total-timesteps 2000000 --n-envs 8 --wandb
"""

import os, sys, argparse
import numpy as np

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor
from stable_baselines3.common.callbacks import (
    CheckpointCallback, EvalCallback, CallbackList, BaseCallback
)
import torch.nn as nn

from envs.myoarm_button_proprio_env import MyoArmButtonPressProprioEnv

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False


class SuccessRateCallback(BaseCallback):
    def __init__(self, log_freq=2000, verbose=0):
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
            recent = self.successes[-200:]
            rate = np.mean(recent) * 100
            mean_dist = np.mean(self.distances[-200:]) if self.distances else 0
            print(f"  Step {self.num_timesteps:>8d} | "
                  f"Success: {rate:5.1f}% | "
                  f"Mean dist: {mean_dist:.4f}")
            if HAS_WANDB and wandb.run:
                wandb.log({"success_rate": rate, "mean_distance": mean_dist,
                           "timestep": self.num_timesteps})
        return True


class WandbCallback(BaseCallback):
    def _on_step(self):
        if HAS_WANDB and wandb.run and len(self.model.ep_info_buffer) > 0:
            ep = self.model.ep_info_buffer[-1]
            wandb.log({"ep_reward": ep["r"], "ep_length": ep["l"],
                        "timestep": self.num_timesteps})
        return True


def make_env(xml_path, max_steps, randomize, rank, seed=0):
    def _init():
        env = MyoArmButtonPressProprioEnv(
            xml_path=xml_path, max_steps=max_steps,
            randomize_button=randomize,
        )
        env.reset(seed=seed + rank)
        return env
    return _init


def train(args):
    print("=" * 60)
    print("MyoArm Phase 1: Proprioceptive Button-Press")
    print("=" * 60)
    print(f"  Model:         MyoArm (32 muscles)")
    print(f"  Obs:           24-dim state vector")
    print(f"  Button random: {args.randomize_button}")
    print(f"  N envs:        {args.n_envs}")
    print(f"  Total steps:   {args.total_timesteps}")
    print("=" * 60)

    output_dir = os.path.join(args.output_dir, "myoarm_phase1")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "best_model"), exist_ok=True)

    env_fns = [make_env(args.xml_path, args.max_steps, args.randomize_button, i, args.seed)
               for i in range(args.n_envs)]

    vec_env = DummyVecEnv(env_fns) if args.n_envs <= 4 else SubprocVecEnv(env_fns)
    vec_env = VecMonitor(vec_env)

    eval_env = DummyVecEnv([make_env(args.xml_path, args.max_steps, False, 0, args.seed + 1000)])
    eval_env = VecMonitor(eval_env)

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
        policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[256, 256]), activation_fn=nn.ReLU),
        verbose=1,
        tensorboard_log=os.path.join(output_dir, "tb_logs"),
        device=args.device,
        seed=args.seed,
    )

    print(f"\nActions: {model.action_space.shape[0]} muscles")
    print(f"Obs: {model.observation_space.shape[0]} dims\n")

    if HAS_WANDB and args.wandb:
        wandb.init(
            project="myovision-buttonpress",
            name="myoarm-phase1-proprio",
            config=vars(args),
            sync_tensorboard=True,
        )

    callbacks = [
        CheckpointCallback(save_freq=max(20000 // args.n_envs, 1),
                           save_path=os.path.join(output_dir, "checkpoints"),
                           name_prefix="myoarm_p1"),
        EvalCallback(eval_env,
                     best_model_save_path=os.path.join(output_dir, "best_model"),
                     log_path=os.path.join(output_dir, "eval_logs"),
                     eval_freq=max(10000 // args.n_envs, 1),
                     n_eval_episodes=20, deterministic=True),
        SuccessRateCallback(log_freq=4000),
    ]
    if HAS_WANDB and args.wandb:
        callbacks.append(WandbCallback())

    print("Starting training...\n")
    model.learn(total_timesteps=args.total_timesteps,
                callback=CallbackList(callbacks), progress_bar=True)

    final_path = os.path.join(output_dir, "final_model")
    model.save(final_path)
    print(f"\nDone! Model saved to: {final_path}")

    if HAS_WANDB and args.wandb:
        wandb.finish()
    vec_env.close()
    eval_env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--xml-path", type=str, default=None)
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--total-timesteps", type=int, default=2_000_000)
    parser.add_argument("--randomize-button", action="store_true", default=True)
    parser.add_argument("--no-randomize-button", dest="randomize_button", action="store_false")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="results")
    parser.add_argument("--wandb", action="store_true")
    args = parser.parse_args()

    if args.xml_path is None:
        args.xml_path = r"C:\Users\max-r\anaconda3\envs\myovision\lib\site-packages\myosuite\envs\myo\assets\arm\myoarm_buttonpress.xml"

    train(args)
