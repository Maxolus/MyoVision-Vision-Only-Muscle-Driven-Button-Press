"""
Test: Verify MyoArm button press env loads and runs.

Usage:
    cd D:\MyoVision
    python test_myoarm.py
"""

import os, sys
import numpy as np
import mujoco

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

XML = r"C:\Users\max-r\anaconda3\envs\myovision\lib\site-packages\myosuite\envs\myo\assets\arm\myoarm_buttonpress.xml"

print("=== Test MyoArm ButtonPress ===\n")

# Test 1: Load model
print("[1] Loading MuJoCo model...")
model = mujoco.MjModel.from_xml_path(XML)
data = mujoco.MjData(model)
print(f"    Bodies: {model.nbody}")
print(f"    Joints: {model.njnt}")
print(f"    Actuators (muscles): {model.nu}")
print(f"    Geoms: {model.ngeom}")
print(f"    Sites: {model.nsite}")
print("    PASS\n")

# Test 2: Load proprio env
print("[2] Loading proprioceptive env...")
from envs.myoarm_button_proprio_env import MyoArmButtonPressProprioEnv

env = MyoArmButtonPressProprioEnv(xml_path=XML, randomize_button=True)
obs, info = env.reset(seed=42)
print(f"    Obs shape: {obs.shape}")
print(f"    Action shape: {env.action_space.shape}")
print(f"    Button pos: {info['btn_pos']}")
print("    PASS\n")

# Test 3: Step and check distances
print("[3] Running 100 random steps...")
min_dist = 999
for i in range(100):
    obs, rew, term, trunc, info = env.step(env.action_space.sample())
    if info["distance"] < min_dist:
        min_dist = info["distance"]
    if info.get("success"):
        print(f"    HIT at step {i}!")
        break
print(f"    Min distance: {min_dist:.4f} m ({min_dist*100:.1f} cm)")
print("    PASS\n")

# Test 4: Button randomization
print("[4] Testing button randomization...")
env2 = MyoArmButtonPressProprioEnv(xml_path=XML, randomize_button=True)
positions = []
for i in range(10):
    obs, info = env2.reset(seed=i)
    positions.append(info["btn_pos"].copy())
positions = np.array(positions)
spread = positions.std(axis=0)
print(f"    Position spread: x={spread[0]:.4f}, y={spread[1]:.4f}, z={spread[2]:.4f}")
assert spread[0] > 0.01, "No X randomization!"
assert spread[1] > 0.01, "No Y randomization!"
env2.close()
print("    PASS\n")

# Test 5: Reachability
print("[5] Reachability test (500 episodes)...")
env3 = MyoArmButtonPressProprioEnv(xml_path=XML, randomize_button=False, max_steps=200)
hits = 0
global_min = 999
for ep in range(500):
    env3.reset(seed=ep)
    for step in range(200):
        obs, rew, term, trunc, info = env3.step(env3.action_space.sample())
        if info["distance"] < global_min:
            global_min = info["distance"]
        if info.get("success"):
            hits += 1
            break
    if ep % 100 == 0:
        print(f"    ep {ep}: hits={hits}, min_dist={global_min:.4f}")
print(f"    Total hits: {hits}/500 ({hits/5:.1f}%)")
print(f"    Global min distance: {global_min:.4f} m ({global_min*100:.1f} cm)")
env3.close()

if hits > 0:
    print("    PASS — button is reachable!\n")
else:
    print(f"    WARNING — 0 hits. Min dist {global_min*100:.1f} cm.")
    print("    May need to adjust button position.\n")

# Test 6: Rendering
print("[6] Testing offscreen rendering...")
from envs.myoarm_button_vision_env import MyoArmButtonPressVisionEnv
env4 = MyoArmButtonPressVisionEnv(xml_path=XML, image_size=(64, 64), frame_stack=3)
obs, info = env4.reset(seed=42)
print(f"    Vision obs shape: {obs.shape}")
print(f"    Mean pixel: {obs.mean():.1f}")
assert obs.mean() > 0, "Black image!"
env4.close()
print("    PASS\n")

env.close()
print("=" * 40)
print("ALL MYOARM TESTS COMPLETE")
print("=" * 40)
print("\nNext steps:")
print("  1. Copy XML:  copy myoarm_buttonpress.xml to MyoSuite arm assets folder")
print("  2. Phase 1:   python training\\train_myoarm_phase1.py --wandb")
print("  3. Phase 2:   python training\\train_myoarm_phase2.py --phase1-model results\\myoarm_phase1\\best_model\\best_model.zip --wandb")
