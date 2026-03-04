"""
Phase 2: Vision Fine-Tuning
==============================
Takes the trained proprioceptive policy and transfers it to vision input.

Strategy:
  1. Load Phase 1 policy weights (MLP that maps state -> muscle activations)
  2. Create new policy: CNN encoder -> same MLP architecture
  3. Initialize MLP layers from Phase 1 weights
  4. Train the CNN encoder (+ light MLP finetuning) with lower learning rate

Usage:
    cd D:\MyoVision
    python training\train_phase2_vision.py --phase1-model results\phase1_proprio\best_model\best_model.zip
"""

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from stable_baselines3.common.callbacks import (
    CheckpointCallback, EvalCallback, CallbackList, BaseCallback
)
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from envs.button_press_env import ButtonPressEnv

# --- Wandb (optional) ---
try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False


# ============================================================
# CNN Feature Extractor
# ============================================================
class VisionEncoder(BaseFeaturesExtractor):
    """CNN that maps stacked RGB frames to a feature vector."""

    def __init__(self, observation_space, features_dim=256,
                 frame_stack=3, image_size=(64, 64)):
        super().__init__(observation_space, features_dim)

        self.frame_stack = frame_stack
        self.image_size = image_size
        n_channels = frame_stack * 3

        self.cnn = nn.Sequential(
            nn.Conv2d(n_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        with torch.no_grad():
            sample = torch.zeros(1, n_channels, *image_size)
            n_flatten = self.cnn(sample).shape[1]

        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations):
        batch_size = observations.shape[0]
        x = observations.float() / 255.0
        x = x.permute(0, 1, 4, 2, 3)
        x = x.reshape(batch_size, -1, *self.image_size)
        return self.linear(self.cnn(x))


# ============================================================
# Callbacks
# ============================================================
class SuccessRateCallback(BaseCallback):
    def __init__(self, log_freq=1000, verbose=0):
        super().__init__(verbose)
        self.log_freq = log_freq
        self.successes = []

    def _on_step(self):
        for info in self.locals.get("infos", []):
            if "success" in info:
                self.successes.append(float(info["success"]))

        if self.num_timesteps % self.log_freq == 0 and self.successes:
            rate = np.mean(self.successes[-100:]) * 100
            print(f"  Step {self.num_timesteps:>8d} | Success: {rate:5.1f}%")

            if HAS_WANDB and wandb.run:
                wandb.log({"success_rate": rate, "timestep": self.num_timesteps})

        return True


class WandbCallback(BaseCallback):
    def _on_step(self):
        if HAS_WANDB and wandb.run and len(self.model.ep_info_buffer) > 0:
            ep = self.model.ep_info_buffer[-1]
            wandb.log({"ep_reward": ep["r"], "ep_length": ep["l"],
                        "timestep": self.num_timesteps})
        return True


# ============================================================
# Environment Factory
# ============================================================
def make_env(xml_path, camera_name, image_size, frame_stack,
             max_steps, randomization, rank, seed=0):
    def _init():
        env = ButtonPressEnv(
            xml_path=xml_path,
            camera_name=camera_name,
            image_size=image_size,
            frame_stack=frame_stack,
            max_steps=max_steps,
            reward_type="staged",
            randomization_level=randomization,
        )
        env.reset(seed=seed + rank)
        return env
    return _init


# ============================================================
# Weight Transfer
# ============================================================
def transfer_weights(phase1_model_path, phase2_model):
    """
    Transfer MLP weights from Phase 1 (proprio) to Phase 2 (vision).

    Phase 1 architecture: obs(11) -> mlp_extractor.pi[256,256] -> action(6)
    Phase 2 architecture: img -> CNN -> feat(256) -> mlp_extractor.pi[256,256] -> action(6)

    We transfer mlp_extractor and action_net weights.
    """
    phase1 = PPO.load(phase1_model_path, device="cpu")

    p1_params = dict(phase1.policy.named_parameters())
    p2_params = dict(phase2_model.policy.named_parameters())

    transferred = 0
    skipped = 0

    for name, param in p2_params.items():
        # Skip CNN/features_extractor layers
        if "features_extractor" in name:
            skipped += 1
            continue

        if name in p1_params and p1_params[name].shape == param.shape:
            param.data.copy_(p1_params[name].data)
            transferred += 1
        else:
            skipped += 1

    print(f"Weight transfer: {transferred} layers transferred, {skipped} skipped (CNN + mismatched)")
    return phase2_model


# ============================================================
# Main Training
# ============================================================
def train(args):
    print("=" * 60)
    print("Phase 2: Vision Fine-Tuning")
    print("=" * 60)
    print(f"  Phase 1 model: {args.phase1_model}")
    print(f"  Camera:        {args.camera_name}")
    print(f"  Image size:    {args.image_size}x{args.image_size}")
    print(f"  Frame stack:   {args.frame_stack}")
    print(f"  N envs:        {args.n_envs}")
    print(f"  Total steps:   {args.total_timesteps}")
    print(f"  Learning rate: {args.lr}")
    print("=" * 60)

    image_size = (args.image_size, args.image_size)
    output_dir = os.path.join(args.output_dir, "phase2_vision")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "best_model"), exist_ok=True)

    # --- Envs ---
    env_fns = [
        make_env(args.xml_path, args.camera_name, image_size,
                 args.frame_stack, args.max_steps, args.randomization, i, args.seed)
        for i in range(args.n_envs)
    ]
    vec_env = DummyVecEnv(env_fns)
    vec_env = VecMonitor(vec_env)

    eval_env = DummyVecEnv([
        make_env(args.xml_path, args.camera_name, image_size,
                 args.frame_stack, args.max_steps, "none", 0, args.seed + 1000)
    ])
    eval_env = VecMonitor(eval_env)

    # --- Policy ---
    # features_dim=256 must match Phase 1's first hidden layer input
    # Phase 1 used net_arch pi=[256,256], so the mlp_extractor expects 256-dim input
    policy_kwargs = dict(
        features_extractor_class=VisionEncoder,
        features_extractor_kwargs=dict(
            features_dim=256,
            frame_stack=args.frame_stack,
            image_size=image_size,
        ),
        net_arch=dict(pi=[256, 256], vf=[256, 256]),
        activation_fn=nn.ReLU,
    )

    model = PPO(
        policy="MlpPolicy",
        env=vec_env,
        learning_rate=args.lr,
        n_steps=256,
        batch_size=256,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.005,
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs=policy_kwargs,
        verbose=1,
        tensorboard_log=os.path.join(output_dir, "tb_logs"),
        device=args.device,
        seed=args.seed,
    )

    # --- Transfer Phase 1 weights ---
    print("\nTransferring Phase 1 weights...")
    model = transfer_weights(args.phase1_model, model)

    # --- Optionally freeze MLP, train only CNN ---
    if args.freeze_mlp:
        print("Freezing MLP layers (training CNN only)...")
        for name, param in model.policy.named_parameters():
            if "features_extractor" not in name:
                param.requires_grad = False

    trainable = sum(p.numel() for p in model.policy.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.policy.parameters())
    print(f"Trainable parameters: {trainable:,} / {total:,}\n")

    # --- Wandb ---
    if HAS_WANDB and args.wandb:
        wandb.init(
            project="myovision-buttonpress",
            name="phase2-vision",
            config={
                "phase": 2,
                "obs_type": "vision",
                "phase1_model": args.phase1_model,
                "total_timesteps": args.total_timesteps,
                "image_size": args.image_size,
                "freeze_mlp": args.freeze_mlp,
                "learning_rate": args.lr,
            },
            sync_tensorboard=True,
        )

    # --- Callbacks ---
    callbacks = [
        CheckpointCallback(
            save_freq=max(10000 // args.n_envs, 1),
            save_path=os.path.join(output_dir, "checkpoints"),
            name_prefix="phase2",
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
    print("Starting Phase 2 training...\n")
    model.learn(
        total_timesteps=args.total_timesteps,
        callback=CallbackList(callbacks),
        progress_bar=True,
    )

    final_path = os.path.join(output_dir, "final_model")
    model.save(final_path)
    print(f"\nPhase 2 complete! Model saved to: {final_path}")

    if HAS_WANDB and args.wandb:
        wandb.finish()

    vec_env.close()
    eval_env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 2: Vision fine-tuning")
    parser.add_argument("--phase1-model", type=str, required=True,
                        help="Path to Phase 1 trained model .zip")
    parser.add_argument("--xml-path", type=str, default=None)
    parser.add_argument("--camera-name", type=str, default="static_cam")
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--frame-stack", type=int, default=3)
    parser.add_argument("--n-envs", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--total-timesteps", type=int, default=1_000_000)
    parser.add_argument("--randomization", type=str, default="medium")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Lower LR than Phase 1 to avoid catastrophic forgetting")
    parser.add_argument("--freeze-mlp", action="store_true",
                        help="Freeze MLP layers, train only CNN encoder")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="results")
    parser.add_argument("--wandb", action="store_true")

    args = parser.parse_args()

    if args.xml_path is None:
        args.xml_path = r"C:\Users\max-r\anaconda3\envs\myovision\lib\site-packages\myosuite\envs\myo\assets\elbow\myoelbow_buttonpress.xml"

    train(args)
