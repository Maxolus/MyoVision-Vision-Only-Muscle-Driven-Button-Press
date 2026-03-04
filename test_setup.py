import sys
print(f"Python {sys.version}")

print("\n=== Test 1: MuJoCo ===")
import mujoco
print(f"MuJoCo {mujoco.__version__}")

print("\n=== Test 2: MyoSuite ===")
import myosuite
import gymnasium as gym
env = gym.make("myoElbowPose1D6MRandom-v0")
obs, info = env.reset()
print(f"Obs shape: {obs.shape}")
print(f"Action shape: {env.action_space.shape}")
env.close()
print("MyoSuite OK")

print("\n=== Test 3: Offscreen Rendering ===")
import numpy as np
env = gym.make("myoElbowPose1D6MRandom-v0")
env.reset()
model = env.unwrapped.sim.model._model
data = env.unwrapped.sim.data._data
renderer = mujoco.Renderer(model, 64, 64)
mujoco.mj_forward(model, data)
renderer.update_scene(data)
img = renderer.render()
print(f"Image: {img.shape}, mean pixel: {img.mean():.1f}")
assert img.mean() > 0, "BLACK IMAGE - rendering broken!"
env.close()
print("Rendering OK")

print("\n=== Test 4: PyTorch ===")
import torch
print(f"PyTorch {torch.__version__}")
print(f"CUDA: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

print("\n=== Test 5: Stable-Baselines3 ===")
from stable_baselines3 import PPO
print("SB3 OK")

print("\n=== Test 6: OpenCV ===")
import cv2
print(f"OpenCV {cv2.__version__}")

print("\n" + "=" * 40)
print("ALL TESTS PASSED")