import sys, numpy as np, mujoco
sys.path.insert(0, ".")
from envs.button_press_proprio_env import ButtonPressProprioEnv

XML = r"C:\Users\max-r\anaconda3\envs\myovision\lib\site-packages\myosuite\envs\myo\assets\elbow\myoelbow_buttonpress.xml"
env = ButtonPressProprioEnv(xml_path=XML)
env.reset()

# Manually set elbow to best angle and simulate
env.data.qpos[env._elbow_qpos_addr] = 1.92
env.data.ctrl[:] = [0, 0, 0, 1, 1, 1]  # full biceps
for _ in range(2000):
    mujoco.mj_step(env.model, env.data)

mujoco.mj_forward(env.model, env.data)

wrist = env.data.site_xpos[env._wrist_site_id]
btn = env.data.site_xpos[env._btn_target_site_id]
print(f"Wrist: {wrist}")
print(f"Button: {btn}")
print(f"Distance: {np.linalg.norm(wrist - btn):.4f} m")
print(f"Contacts: {env.data.ncon}")

for i in range(env.data.ncon):
    c = env.data.contact[i]
    g1 = mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_GEOM, c.geom1)
    g2 = mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_GEOM, c.geom2)
    print(f"  Contact {i}: {g1} <-> {g2}")

# Check: which geoms have contype/conaffinity that can collide with btn_top?
btn_geom_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_GEOM, "btn_top")
btn_contype = env.model.geom_contype[btn_geom_id]
btn_conaff = env.model.geom_conaffinity[btn_geom_id]
print(f"\nbtn_top contype={btn_contype}, conaffinity={btn_conaff}")

print("\nArm geoms that CAN collide with button:")
for i in range(env.model.ngeom):
    name = mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_GEOM, i)
    ct = env.model.geom_contype[i]
    ca = env.model.geom_conaffinity[i]
    # Collision happens when (ct1 & ca2) or (ct2 & ca1) is nonzero
    if (ct & btn_conaff) or (btn_contype & ca):
        print(f"  {name}: contype={ct}, conaffinity={ca}")

env.close()