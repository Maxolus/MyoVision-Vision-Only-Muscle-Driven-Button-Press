import sys, numpy as np, mujoco
sys.path.insert(0, ".")
from envs.myoarm_button_proprio_env import MyoArmButtonPressProprioEnv, FINGERTIP_SITES

XML = r"C:\Users\max-r\anaconda3\envs\myovision\lib\site-packages\myosuite\envs\myo\assets\arm\myoarm_buttonpress.xml"
env = MyoArmButtonPressProprioEnv(xml_path=XML, randomize_button=False)

# Run many random episodes and track where fingertips go
print("Sampling fingertip workspace (1000 episodes x 200 steps)...")
all_positions = {name: [] for name in FINGERTIP_SITES}

for ep in range(1000):
    env.reset(seed=ep)
    for step in range(200):
        env.step(env.action_space.sample())
        for i, name in enumerate(FINGERTIP_SITES):
            pos = env.data.site_xpos[env._tip_site_ids[i]].copy()
            all_positions[name].append(pos)

print("\nFingertip workspace ranges:")
for name in FINGERTIP_SITES:
    pts = np.array(all_positions[name])
    print(f"  {name}:")
    print(f"    x: [{pts[:,0].min():.3f}, {pts[:,0].max():.3f}]  mean={pts[:,0].mean():.3f}")
    print(f"    y: [{pts[:,1].min():.3f}, {pts[:,1].max():.3f}]  mean={pts[:,1].mean():.3f}")
    print(f"    z: [{pts[:,2].min():.3f}, {pts[:,2].max():.3f}]  mean={pts[:,2].mean():.3f}")

# Overall workspace center
all_pts = np.concatenate([np.array(v) for v in all_positions.values()])
center = all_pts.mean(axis=0)
print(f"\nWorkspace center (all tips): [{center[0]:.3f}, {center[1]:.3f}, {center[2]:.3f}]")
print(f"Current button at: {env._original_btn_pos}")
print(f"\nSuggested button pos: [{center[0]:.3f}, {center[1]:.3f}, {center[2]:.3f}]")

env.close()