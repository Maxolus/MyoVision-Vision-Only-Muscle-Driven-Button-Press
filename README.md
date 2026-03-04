# MyoVision — Vision-Only Muscle-Driven Button Press

A reinforcement learning system that trains a musculoskeletal arm model to press a button using **only camera images** as input. No proprioception, no coordinates — just pixels and muscles.

Built on [MyoSuite](https://sites.google.com/view/myosuite) and [MuJoCo](https://mujoco.org/), designed for comparison with human motor performance via Fitts' law.

---

## How It Works

A virtual camera renders 64×64 RGB frames of the workspace. Three consecutive frames are stacked and fed through a CNN, which outputs activation levels for 6 muscles (3× triceps, 2× biceps, 1× brachialis). The agent learns entirely from pixel input — it must discover where the button is, how to reach it, and how to coordinate muscle contractions to press it.

```
Camera Frame (64×64 RGB) × 3 stacked
        ↓
CNN → 512-d feature vector
        ↓
MLP → 6 muscle activations [0, 1]
        ↓
[TRIlong, TRIlat, TRImed, BIClong, BICshort, BRA]
```

---

## Prerequisites

- **Windows 10/11** with NVIDIA GPU (recommended)
- **Anaconda/Miniconda** installed
- ~4 GB disk space

---

## Installation

Open Anaconda Prompt (CMD, not PowerShell) and run each line:

```cmd
conda create -n myovision python=3.10 -y
conda activate myovision
```

Install PyTorch (GPU):

```cmd
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y
```

Or CPU-only:

```cmd
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

Install remaining dependencies:

```cmd
pip install mujoco myosuite stable-baselines3 gymnasium
pip install opencv-python-headless matplotlib pandas scipy tensorboard
pip install imageio imageio-ffmpeg tqdm
```

If you get numpy errors after installing:

```cmd
pip install numpy==1.26.4 --force-reinstall
```

---

## Project Structure

```
D:\MyoVision\
├── assets\elbow\
│   └── myoelbow_buttonpress.xml   ← MuJoCo model with button + camera
├── envs\
│   ├── __init__.py
│   └── button_press_env.py        ← Custom Gym environment
├── training\
│   ├── __init__.py
│   └── train_ppo_vision.py        ← PPO training script
├── scripts\
│   └── fitts_analysis.py          ← Fitts' law evaluation
├── results\                        ← Models, logs, videos (created automatically)
├── test_setup.py                   ← Verify installation
├── test_env.py                     ← Verify environment
├── visualize.py                    ← Watch trained agent
└── README.md                       ← You are here
```

---

## Important: XML Path

The button-press XML uses `<include>` tags that reference MyoSuite's internal assets. It must be placed inside the MyoSuite installation directory. Copy it there:

```cmd
copy "D:\MyoVision\assets\elbow\myoelbow_buttonpress.xml" "C:\Users\max-r\anaconda3\envs\myovision\lib\site-packages\myosuite\envs\myo\assets\elbow\myoelbow_buttonpress.xml"
```

Then always use this path when running scripts:

```
C:\Users\max-r\anaconda3\envs\myovision\lib\site-packages\myosuite\envs\myo\assets\elbow\myoelbow_buttonpress.xml
```

---

## Quick Start

### 1. Verify installation

```cmd
conda activate myovision
cd D:\MyoVision
python test_setup.py
```

Expected: `ALL TESTS PASSED`

### 2. Test the environment

```cmd
python test_env.py
```

Expected: `ALL ENV TESTS PASSED`

### 3. Train the agent

Short test run (10-20 minutes):

```cmd
python training\train_ppo_vision.py --n-envs 4 --total-timesteps 100000 --xml-path "C:\Users\max-r\anaconda3\envs\myovision\lib\site-packages\myosuite\envs\myo\assets\elbow\myoelbow_buttonpress.xml"
```

Full training run (several hours):

```cmd
python training\train_ppo_vision.py --n-envs 4 --total-timesteps 1000000 --xml-path "C:\Users\max-r\anaconda3\envs\myovision\lib\site-packages\myosuite\envs\myo\assets\elbow\myoelbow_buttonpress.xml"
```

Training saves checkpoints to `results\checkpoints\` and the best model to `results\best_model\`.

### 4. Watch the trained agent

```cmd
python visualize.py
```

Saves a video to `results\replay.mp4` and opens a live 3D viewer where you can rotate the camera with the mouse.

### 5. Monitor training progress

In a second terminal:

```cmd
conda activate myovision
tensorboard --logdir results\tb_logs
```

Open `http://localhost:6006` in your browser.

---

## Training Options

| Flag | Default | Description |
|---|---|---|
| `--n-envs` | 4 | Parallel environments (more = faster, more RAM) |
| `--total-timesteps` | 500000 | Total training steps |
| `--image-size` | 64 | Camera resolution (64 recommended) |
| `--frame-stack` | 3 | Number of stacked frames |
| `--max-steps` | 500 | Max steps per episode |
| `--camera-name` | static_cam | Camera to use (static_cam or button_cam) |
| `--randomization` | medium | Domain randomization: none, medium, high |
| `--device` | auto | cpu or cuda |
| `--seed` | 42 | Random seed |

---

## Fitts' Law Analysis

After training, evaluate the agent across different difficulty levels and compare with human motor performance:

```cmd
python scripts\fitts_analysis.py --model-path results\best_model\best_model.zip --xml-path "C:\Users\max-r\anaconda3\envs\myovision\lib\site-packages\myosuite\envs\myo\assets\elbow\myoelbow_buttonpress.xml"
```

Outputs to `results\fitts\`:
- `fitts_plot.png` — Fitts' law regression, accuracy by difficulty, throughput distribution
- `fitts_results.json` — slope, intercept, R², throughput
- `trial_data.json` — per-trial timing and accuracy data

### Key metrics for human comparison

| Metric | Human (typical) | What it means |
|---|---|---|
| Throughput | 4–10 bits/s | Information processing rate |
| Slope *b* | 100–200 ms/bit | Sensitivity to difficulty |
| R² | 0.90–0.99 | How well Fitts' law fits |
| Velocity symmetry | 0.4–0.5 | Bell-shaped velocity profile |
| Path efficiency | 0.8–0.95 | Straightness of reach |
| Submovements | 1–3 | Corrective adjustments |

---

## Troubleshooting

**"DLL load failed" for OpenCV**: Remove all opencv versions and reinstall headless:
```cmd
pip uninstall opencv-python opencv-contrib-python opencv-python-headless -y
pip install opencv-python-headless
```

**"numpy._core.multiarray failed to import"**: Version conflict. Fix with:
```cmd
pip install numpy==1.26.4 --force-reinstall
```

**Black images from renderer**: MuJoCo rendering backend issue. Don't set `MUJOCO_GL` on Windows — it uses WGL by default.

**Training is slow**: Reduce `--n-envs` or `--image-size`. On CPU, expect ~50-100 steps/second. On GPU, ~200-500 steps/second.

**XML "file not found" errors**: The XML must be in the MyoSuite installation directory. Re-run the copy command from the "Important: XML Path" section above.

---

## Architecture Details

- **Simulation**: MuJoCo 3.3 physics engine, 2ms timestep, 5 substeps per action
- **Musculoskeletal model**: MyoSuite MyoElbow — 1 DOF (elbow flexion), 6 Hill-type muscles
- **Button**: Spring-loaded slide joint (stiffness=40 N/m, 8mm travel), press detected via displacement + force threshold
- **Vision**: 64×64 RGB, 3-frame stack, NatureCNN encoder (3 conv layers → 512-d)
- **RL algorithm**: PPO (Proximal Policy Optimization) via Stable-Baselines3
- **Reward**: staged (reaching shaping + proximity bonus + press reward + effort penalty)

---

## License

The MyoSuite model files are under Apache License 2.0 (Vikash Kumar, Vittorio Caggiano, Huawei Wang). The environment wrapper, training scripts, and analysis code in this repository are provided as-is for research purposes.
