import sys, numpy as np, mujoco
sys.path.insert(0, ".")
from envs.button_press_proprio_env import ButtonPressProprioEnv

XML = r"C:\Users\max-r\anaconda3\envs\myovision\lib\site-packages\myosuite\envs\myo\assets\elbow\myoelbow_buttonpress.xml"
env = ButtonPressProprioEnv(xml_path=XML)

hits = 0
min_geom_dist = 999

for ep in range(500):
    env.reset(seed=ep)
    for step in range(100):
        obs, rew, term, trunc, info = env.step(env.action_space.sample())
        
        # Manually check hand-geom distance
        btn_pos = env.data.site_xpos[env._btn_target_site_id]
        for name in ["arm_r_2mc", "arm_r_3mc", "arm_r_1mc", "arm_r_thumbdist"]:
            gid = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_GEOM, name)
            if gid >= 0:
                d = np.linalg.norm(env.data.geom_xpos[gid] - btn_pos)
                if d < min_geom_dist:
                    min_geom_dist = d
        
        if info.get("success"):
            hits += 1
            break
    
    if ep % 100 == 0:
        print(f"  ep {ep}: hits={hits}, min_geom_dist={min_geom_dist:.4f}")

print(f"\nHits: {hits}/500")
print(f"Min hand-geom to button: {min_geom_dist:.4f} m ({min_geom_dist*100:.1f} cm)")

if min_geom_dist > 0.015:
    print(f"\nClosest approach is {min_geom_dist*100:.1f} cm — threshold is 1.5 cm")
    print("Options:")
    print("  1. Increase threshold to", round(min_geom_dist + 0.005, 3), "m")
    print("  2. Move button closer")

env.close()