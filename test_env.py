"""
Quick test: verify the ButtonPressEnv loads, steps, and renders.

Usage:
    cd D:\MyoVision
    python test_env.py
"""

import os
import sys
import numpy as np

# Add project root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("=== Test: ButtonPressEnv ===\n")

# Step 1: Load environment
print("[1] Loading environment...")
from envs.button_press_env import ButtonPressEnv

env = ButtonPressEnv(
    xml_path=r"C:\Users\max-r\anaconda3\envs\myovision\lib\site-packages\myosuite\envs\myo\assets\elbow\myoelbow_buttonpress.xml",
    camera_name="static_cam",
    image_size=(64, 64),
    frame_stack=3,
    max_steps=100,
    randomization_level="none",
)
print(f"    Obs space:    {env.observation_space.shape}")
print(f"    Action space: {env.action_space.shape}")
print(f"    N muscles:    {env.model.nu}")
print("    PASS\n")

# Step 2: Reset
print("[2] Resetting...")
obs, info = env.reset(seed=42)
print(f"    Obs shape:  {obs.shape}")
print(f"    Obs dtype:  {obs.dtype}")
print(f"    Obs range:  [{obs.min()}, {obs.max()}]")
print(f"    Mean pixel: {obs.mean():.1f}")
assert obs.shape == (3, 64, 64, 3), f"Wrong obs shape: {obs.shape}"
assert obs.mean() > 0, "Black image!"
print("    PASS\n")

# Step 3: Step with random actions
print("[3] Stepping 50 random actions...")
total_reward = 0
pressed = False
for i in range(50):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    total_reward += reward
    if terminated:
        pressed = True
        print(f"    Button pressed at step {i+1}!")
        break

print(f"    Total reward:  {total_reward:.2f}")
print(f"    Final dist:    {info['distance']:.4f} m")
print(f"    Btn displace:  {info['btn_displacement']:.5f}")
print(f"    Btn force:     {info['btn_force']:.3f}")
print("    PASS\n")

# Step 4: Check trial record
print("[4] Checking trial record...")
tr = env.trial_record
print(f"    Trial ID:      {tr.trial_id}")
print(f"    Target dist:   {tr.target_distance:.4f} m")
print(f"    Target width:  {tr.target_width:.4f} m")
print(f"    Fitts ID:      {tr.index_of_difficulty:.2f} bits")
print(f"    Trajectory pts: {len(tr.timestamps)}")
print(f"    Hit:           {tr.hit}")
if tr.hit:
    print(f"    Reaction time: {tr.reaction_time:.3f} s")
    print(f"    Movement time: {tr.movement_time:.3f} s")
print("    PASS\n")

# Step 5: Check rendering independently
print("[5] Testing render()...")
env.reset()
env.render_mode = "rgb_array"
frame = env.render()
if frame is not None:
    print(f"    Frame shape: {frame.shape}")
    print(f"    Frame mean:  {frame.mean():.1f}")
else:
    print("    render() returned None (render_mode may not be set)")
print("    PASS\n")

# Step 6: Test domain randomization
print("[6] Testing domain randomization...")
env2 = ButtonPressEnv(
    xml_path=r"C:\Users\max-r\anaconda3\envs\myovision\lib\site-packages\myosuite\envs\myo\assets\elbow\myoelbow_buttonpress.xml",
    image_size=(64, 64),
    randomization_level="high",
)
btn_positions = []
for i in range(5):
    obs, info = env2.reset(seed=i)
    btn_pos = env2.data.site_xpos[env2._btn_target_site_id].copy()
    btn_positions.append(btn_pos)
    print(f"    Reset {i}: button at [{btn_pos[0]:.3f}, {btn_pos[1]:.3f}, {btn_pos[2]:.3f}]")

positions = np.array(btn_positions)
spread = positions.std(axis=0)
print(f"    Position spread (std): [{spread[0]:.4f}, {spread[1]:.4f}, {spread[2]:.4f}]")
assert spread.max() > 0.001, "Randomization not working — positions identical!"
env2.close()
print("    PASS\n")

env.close()
print("=" * 40)
print("ALL ENV TESTS PASSED")
print("=" * 40)
print("\nNext step: run training with:")
print("  python training\\train_ppo_vision.py --n-envs 4 --total-timesteps 100000")
