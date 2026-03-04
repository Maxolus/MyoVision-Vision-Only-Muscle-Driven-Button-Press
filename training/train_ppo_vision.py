"""
Training Script: Vision-Only Button-Press with PPO
====================================================
Uses Stable-Baselines3 with a custom CNN policy.

Usage:
    cd D:\MyoVision
    python training\train_ppo_vision.py

    # With options:
    python training\train_ppo_vision.py --total-timesteps 1000000 --n-envs 8 --image-size 64
"""

import os
import sys
import argparse
import numpy as np
import torch
import wandb
from stable_baselines3.common.callbacks import BaseCallback

class WandbCallback(BaseCallback):
    def _on_step(self):
        if len(self.model.ep_info_buffer) > 0:
            ep = self.model.ep_info_buffer[-1]
            wandb.log({
                "ep_reward": ep["r"],
                "ep_length": ep["l"],
                "timestep": self.num_timesteps,
            })
        return True
# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecMonitor
from stable_baselines3.common.callbacks import (
    CheckpointCallback,
    EvalCallback,
    CallbackList,
)
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import gymnasium as gym
import torch.nn as nn

from envs.button_press_env import ButtonPressEnv


# ============================================================
# Custom CNN Feature Extractor (NatureCNN-style)
# ============================================================
class ButtonPressCNN(BaseFeaturesExtractor):
    """
    CNN for stacked RGB frames.
    Input: (batch, frame_stack * H * W * 3) flattened by SB3,
    but we reshape internally.
    """

    def __init__(self, observation_space, features_dim=512,
                 frame_stack=3, image_size=(64, 64)):
        super().__init__(observation_space, features_dim)

        self.frame_stack = frame_stack
        self.image_size = image_size
        n_input_channels = frame_stack * 3  # stacked RGB

        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute CNN output size
        with torch.no_grad():
            sample = torch.zeros(1, n_input_channels, *image_size)
            n_flatten = self.cnn(sample).shape[1]

        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations):
        # observations shape: (batch, frame_stack, H, W, 3) as uint8
        # Reshape to (batch, frame_stack*3, H, W) and normalize
        batch_size = observations.shape[0]
        # (B, stack, H, W, 3) -> (B, stack, 3, H, W) -> (B, stack*3, H, W)
        x = observations.float() / 255.0
        x = x.permute(0, 1, 4, 2, 3)  # (B, stack, 3, H, W)
        x = x.reshape(batch_size, -1, *self.image_size)  # (B, stack*3, H, W)
        return self.linear(self.cnn(x))


# ============================================================
# Environment Factory
# ============================================================
def make_env(xml_path, camera_name, image_size, frame_stack,
             max_steps, randomization_level, rank, seed=0):
    """Factory function for vectorized envs."""
    def _init():
        env = ButtonPressEnv(
            xml_path=xml_path,
            camera_name=camera_name,
            image_size=image_size,
            frame_stack=frame_stack,
            max_steps=max_steps,
            reward_type="staged",
            randomization_level=randomization_level,
        )
        env.reset(seed=seed + rank)
        return env
    return _init


# ============================================================
# Main Training
# ============================================================
def train(args):
    print("=" * 60)
    print("Vision-Only Button-Press Training")
    print("=" * 60)
    print(f"  XML:           {args.xml_path}")
    print(f"  Camera:        {args.camera_name}")
    print(f"  Image size:    {args.image_size}x{args.image_size}")
    print(f"  Frame stack:   {args.frame_stack}")
    print(f"  N envs:        {args.n_envs}")
    print(f"  Total steps:   {args.total_timesteps}")
    print(f"  Randomization: {args.randomization}")
    print(f"  Device:        {args.device}")
    print("=" * 60)

    image_size = (args.image_size, args.image_size)

    # --- Create vectorized environments ---
    env_fns = [
        make_env(
            xml_path=args.xml_path,
            camera_name=args.camera_name,
            image_size=image_size,
            frame_stack=args.frame_stack,
            max_steps=args.max_steps,
            randomization_level=args.randomization,
            rank=i,
            seed=args.seed,
        )
        for i in range(args.n_envs)
    ]

    # Use DummyVecEnv for small n_envs (avoids Windows multiprocessing issues)
    if args.n_envs <= 4:
        vec_env = DummyVecEnv(env_fns)
    else:
        vec_env = SubprocVecEnv(env_fns)

    vec_env = VecMonitor(vec_env)

    # --- Eval environment ---
    eval_env = DummyVecEnv([
        make_env(
            xml_path=args.xml_path,
            camera_name=args.camera_name,
            image_size=image_size,
            frame_stack=args.frame_stack,
            max_steps=args.max_steps,
            randomization_level="none",
            rank=0,
            seed=args.seed + 1000,
        )
    ])
    eval_env = VecMonitor(eval_env)

    # --- Policy kwargs ---
    policy_kwargs = dict(
        features_extractor_class=ButtonPressCNN,
        features_extractor_kwargs=dict(
            features_dim=512,
            frame_stack=args.frame_stack,
            image_size=image_size,
        ),
        net_arch=dict(pi=[256], vf=[256]),
        activation_fn=nn.ReLU,
    )

    # --- PPO ---
    model = PPO(
        policy="MultiInputPolicy" if False else "MlpPolicy",
        # We use MlpPolicy because our custom CNN handles the image processing
        # SB3 will pass the flattened observation to our features extractor
        env=vec_env,
        learning_rate=3e-4,
        n_steps=256,
        batch_size=min(512, 256 * args.n_envs),
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs=policy_kwargs,
        verbose=1,
        tensorboard_log=os.path.join(args.output_dir, "tb_logs"),
        device=args.device,
        seed=args.seed,
    )

    print(f"\nPolicy architecture:\n{model.policy}\n")

    # --- Callbacks ---
    checkpoint_cb = CheckpointCallback(
        save_freq=max(10000 // args.n_envs, 1),
        save_path=os.path.join(args.output_dir, "checkpoints"),
        name_prefix="ppo_buttonpress",
    )
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(args.output_dir, "best_model"),
        log_path=os.path.join(args.output_dir, "eval_logs"),
        eval_freq=max(5000 // args.n_envs, 1),
        n_eval_episodes=10,
        deterministic=True,
    )

    wandb.init(
        project="myovision-buttonpress",
        config={
            "total_timesteps": args.total_timesteps,
            "n_envs": args.n_envs,
            "image_size": args.image_size,
            "randomization": args.randomization,
            "learning_rate": 3e-4,
        },
        sync_tensorboard=True,
    )
    # --- Train ---
    print("Starting training...")
    model.learn(
        total_timesteps=args.total_timesteps,
        callback=CallbackList([checkpoint_cb, eval_cb, WandbCallback()]),
        progress_bar=True,
    )
    wandb.finish()
    # --- Save final model ---
    final_path = os.path.join(args.output_dir, "final_model")
    model.save(final_path)
    print(f"\nTraining complete! Model saved to: {final_path}")

    vec_env.close()
    eval_env.close()


# ============================================================
# CLI
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train vision-only button-press policy")

    parser.add_argument("--xml-path", type=str, default=None,
                        help="Path to MuJoCo XML (default: assets/elbow/myoelbow_buttonpress.xml)")
    parser.add_argument("--camera-name", type=str, default="static_cam")
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--frame-stack", type=int, default=3)
    parser.add_argument("--n-envs", type=int, default=4,
                        help="Number of parallel envs (start with 4 on Windows)")
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--total-timesteps", type=int, default=500_000)
    parser.add_argument("--randomization", type=str, default="medium",
                        choices=["none", "medium", "high"])
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="results")

    args = parser.parse_args()

    # Default XML path
    if args.xml_path is None:
        args.xml_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "assets", "elbow", "myoelbow_buttonpress.xml"
        )

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "best_model"), exist_ok=True)

    train(args)
