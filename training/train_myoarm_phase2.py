"""
MyoArm Phase 2: Vision Fine-Tuning
=====================================
Transfer Phase 1 weights, train CNN encoder on camera frames.

Usage:
    cd D:\MyoVision
    python training\train_myoarm_phase2.py --phase1-model results\myoarm_phase1\best_model\best_model.zip --wandb
"""

import os, sys, argparse
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

from envs.myoarm_button_vision_env import MyoArmButtonPressVisionEnv

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False


class VisionEncoder(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=256,
                 frame_stack=3, image_size=(64, 64)):
        super().__init__(observation_space, features_dim)
        self.frame_stack = frame_stack
        self.image_size = image_size
        n_channels = frame_stack * 3

        self.cnn = nn.Sequential(
            nn.Conv2d(n_channels, 32, kernel_size=8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            n_flatten = self.cnn(torch.zeros(1, n_channels, *image_size)).shape[1]
        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations):
        batch_size = observations.shape[0]
        x = observations.float() / 255.0
        x = x.permute(0, 1, 4, 2, 3)
        x = x.reshape(batch_size, -1, *self.image_size)
        return self.linear(self.cnn(x))


class SuccessRateCallback(BaseCallback):
    def __init__(self, log_freq=2000, verbose=0):
        super().__init__(verbose)
        self.log_freq = log_freq
        self.successes = []

    def _on_step(self):
        for info in self.locals.get("infos", []):
            if "success" in info:
                self.successes.append(float(info["success"]))
        if self.num_timesteps % self.log_freq == 0 and self.successes:
            rate = np.mean(self.successes[-200:]) * 100
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


def transfer_weights(phase1_path, phase2_model):
    phase1 = PPO.load(phase1_path, device="cpu")
    p1 = dict(phase1.policy.named_parameters())
    p2 = dict(phase2_model.policy.named_parameters())
    transferred, skipped = 0, 0
    for name, param in p2.items():
        if "features_extractor" in name:
            skipped += 1
            continue
        if name in p1 and p1[name].shape == param.shape:
            param.data.copy_(p1[name].data)
            transferred += 1
        else:
            skipped += 1
    print(f"Transferred {transferred} layers, skipped {skipped}")
    return phase2_model


def make_env(xml_path, camera, image_size, frame_stack, max_steps, randomize, rank, seed=0):
    def _init():
        env = MyoArmButtonPressVisionEnv(
            xml_path=xml_path, camera_name=camera, image_size=image_size,
            frame_stack=frame_stack, max_steps=max_steps, randomize_button=randomize,
        )
        env.reset(seed=seed + rank)
        return env
    return _init


def train(args):
    print("=" * 60)
    print("MyoArm Phase 2: Vision Fine-Tuning")
    print("=" * 60)
    print(f"  Phase 1:    {args.phase1_model}")
    print(f"  Image:      {args.image_size}x{args.image_size}")
    print(f"  N envs:     {args.n_envs}")
    print(f"  Steps:      {args.total_timesteps}")
    print(f"  Freeze MLP: {args.freeze_mlp}")
    print("=" * 60)

    image_size = (args.image_size, args.image_size)
    output_dir = os.path.join(args.output_dir, "myoarm_phase2")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "best_model"), exist_ok=True)

    env_fns = [make_env(args.xml_path, args.camera_name, image_size, args.frame_stack,
                        args.max_steps, True, i, args.seed) for i in range(args.n_envs)]
    vec_env = DummyVecEnv(env_fns)
    vec_env = VecMonitor(vec_env)

    eval_env = DummyVecEnv([make_env(args.xml_path, args.camera_name, image_size,
                                      args.frame_stack, args.max_steps, False, 0, args.seed + 1000)])
    eval_env = VecMonitor(eval_env)

    policy_kwargs = dict(
        features_extractor_class=VisionEncoder,
        features_extractor_kwargs=dict(features_dim=256, frame_stack=args.frame_stack, image_size=image_size),
        net_arch=dict(pi=[256, 256], vf=[256, 256]),
        activation_fn=nn.ReLU,
    )

    model = PPO(
        "MlpPolicy", vec_env, learning_rate=args.lr, n_steps=256, batch_size=256,
        n_epochs=10, gamma=0.99, gae_lambda=0.95, clip_range=0.2, ent_coef=0.005,
        policy_kwargs=policy_kwargs, verbose=1,
        tensorboard_log=os.path.join(output_dir, "tb_logs"), device=args.device, seed=args.seed,
    )

    print("\nTransferring Phase 1 weights...")
    model = transfer_weights(args.phase1_model, model)

    if args.freeze_mlp:
        print("Freezing MLP layers...")
        for name, param in model.policy.named_parameters():
            if "features_extractor" not in name:
                param.requires_grad = False

    trainable = sum(p.numel() for p in model.policy.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.policy.parameters())
    print(f"Trainable: {trainable:,} / {total:,}\n")

    if HAS_WANDB and args.wandb:
        wandb.init(project="myovision-buttonpress", name="myoarm-phase2-vision",
                   config=vars(args), sync_tensorboard=True)

    callbacks = [
        CheckpointCallback(save_freq=max(20000 // args.n_envs, 1),
                           save_path=os.path.join(output_dir, "checkpoints"), name_prefix="myoarm_p2"),
        EvalCallback(eval_env, best_model_save_path=os.path.join(output_dir, "best_model"),
                     log_path=os.path.join(output_dir, "eval_logs"),
                     eval_freq=max(10000 // args.n_envs, 1), n_eval_episodes=20, deterministic=True),
        SuccessRateCallback(log_freq=4000),
    ]
    if HAS_WANDB and args.wandb:
        callbacks.append(WandbCallback())

    print("Starting training...\n")
    model.learn(total_timesteps=args.total_timesteps, callback=CallbackList(callbacks), progress_bar=True)

    final_path = os.path.join(output_dir, "final_model")
    model.save(final_path)
    print(f"\nDone! Model saved to: {final_path}")

    if HAS_WANDB and args.wandb:
        wandb.finish()
    vec_env.close()
    eval_env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase1-model", type=str, required=True)
    parser.add_argument("--xml-path", type=str, default=None)
    parser.add_argument("--camera-name", type=str, default="static_cam")
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--frame-stack", type=int, default=3)
    parser.add_argument("--n-envs", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--total-timesteps", type=int, default=2_000_000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--freeze-mlp", action="store_true")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="results")
    parser.add_argument("--wandb", action="store_true")
    args = parser.parse_args()

    if args.xml_path is None:
        args.xml_path = r"C:\Users\max-r\anaconda3\envs\myovision\lib\site-packages\myosuite\envs\myo\assets\arm\myoarm_buttonpress.xml"
    train(args)
