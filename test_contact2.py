import sys, numpy as np, mujoco
sys.path.insert(0, ".")
from envs.button_press_proprio_env import ButtonPressProprioEnv

XML = r"C:\Users\max-r\anaconda3\envs\myovision\lib\site-packages\myosuite\envs\myo\assets\elbow\myoelbow_buttonpress.xml"
env = ButtonPressProprioEnv(xml_path=XML)
env.reset()

# Find where the fingertip geoms actually are at various elbow angles
print("=== Fingertip positions across elbow range ===")
jnt_range = env.model.jnt_range[env._elbow_jnt_id]

finger_geom_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_GEOM, "arm_r_2distph")
btn_geom_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_GEOM, "btn_top")

min_dist = 999
best_angle = 0

for angle in np.linspace(jnt_range[0], jnt_range[1], 500):
    mujoco.mj_resetData(env.model, env.data)
    env.data.qpos[env._elbow_qpos_addr] = angle
    mujoco.mj_forward(env.model, env.data)
    
    finger_pos = env.data.geom_xpos[finger_geom_id]
    btn_pos = env.data.geom_xpos[btn_geom_id]
    dist = np.linalg.norm(finger_pos - btn_pos)
    
    if dist < min_dist:
        min_dist = dist
        best_angle = angle
        best_finger = finger_pos.copy()
        best_btn = btn_pos.copy()

print(f"Min finger-to-button: {min_dist:.4f} m ({min_dist*100:.1f} cm)")
print(f"Best angle: {np.degrees(best_angle):.1f} deg")
print(f"Finger at: [{best_finger[0]:.4f}, {best_finger[1]:.4f}, {best_finger[2]:.4f}]")
print(f"Button at: [{best_btn[0]:.4f}, {best_btn[1]:.4f}, {best_btn[2]:.4f}]")
print(f"Delta:     [{best_finger[0]-best_btn[0]:.4f}, {best_finger[1]-best_btn[1]:.4f}, {best_finger[2]-best_btn[2]:.4f}]")

# Show ALL fingertip geom positions at best angle
print(f"\nAll finger geom positions at best angle ({np.degrees(best_angle):.1f} deg):")
mujoco.mj_resetData(env.model, env.data)
env.data.qpos[env._elbow_qpos_addr] = best_angle
mujoco.mj_forward(env.model, env.data)

for name in ["arm_r_thumbdist", "arm_r_2distph", "arm_r_3distph", 
             "arm_r_4distph", "arm_r_5distph", "arm_r_2mc"]:
    gid = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_GEOM, name)
    pos = env.data.geom_xpos[gid]
    d = np.linalg.norm(pos - best_btn)
    print(f"  {name:20s}: [{pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f}] dist={d:.4f}")

print(f"\nButton position: [{best_btn[0]:.4f}, {best_btn[1]:.4f}, {best_btn[2]:.4f}]")
print("\n=> Use the closest finger's position to decide where to move the button!")

env.close()