"""
Fitts' Law Analysis & Human Comparison
========================================
Evaluates trained agent across (D, W) conditions and produces
Fitts' law regression + comparison plots.

Usage:
    cd D:\MyoVision
    python scripts\fitts_analysis.py --model-path results\best_model\best_model.zip
"""

import os
import sys
import argparse
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from dataclasses import asdict

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from envs.button_press_env import ButtonPressEnv, TrialRecord


def compute_fitts_conditions():
    """Generate (distance, width) pairs spanning ID = 2 to 7."""
    conditions = []
    # Base button width
    widths = [0.020, 0.030, 0.040]  # meters (small, medium, large)

    for w in widths:
        for id_target in [2.0, 3.0, 4.0, 5.0, 6.0, 7.0]:
            # ID = log2(2D/W) => D = W * 2^(ID-1)
            d = w * (2 ** (id_target - 1))
            if d < 0.30:  # within reachable workspace
                conditions.append({
                    "distance": d,
                    "width": w,
                    "id": np.log2(2 * d / w),
                    "label": f"D={d:.3f}_W={w:.3f}"
                })

    return conditions


def run_evaluation(model_path, xml_path, n_trials_per_condition=10):
    """Run the agent across all Fitts conditions."""
    from stable_baselines3 import PPO

    conditions = compute_fitts_conditions()
    print(f"Testing {len(conditions)} conditions, {n_trials_per_condition} trials each")

    model = PPO.load(model_path)
    all_records = []

    for ci, cond in enumerate(conditions):
        print(f"\nCondition {ci+1}/{len(conditions)}: ID={cond['id']:.1f} "
              f"(D={cond['distance']:.3f}, W={cond['width']:.3f})")

        for trial in range(n_trials_per_condition):
            env = ButtonPressEnv(
                xml_path=xml_path,
                image_size=(64, 64),
                frame_stack=3,
                max_steps=500,
                randomization_level="none",
            )

            # TODO: override button position to match cond["distance"]
            # For now, use default position and record actual distance
            obs, info = env.reset(seed=ci * 1000 + trial)
            done = False

            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated

            record = info.get("trial_record", env.trial_record)
            record.condition = cond["label"]
            all_records.append(record)

            status = "HIT" if record.hit else "MISS"
            mt = record.movement_time * 1000 if record.hit else float("inf")
            print(f"  Trial {trial+1}: {status} | MT={mt:.0f}ms")

            env.close()

    return all_records, conditions


def analyze_fitts(records):
    """Perform Fitts' law regression on successful trials."""
    # Filter successful trials
    hits = [r for r in records if r.hit and r.movement_time > 0]

    if len(hits) < 3:
        print("Not enough successful trials for regression!")
        return None

    ids = np.array([r.index_of_difficulty for r in hits])
    mts = np.array([r.movement_time * 1000 for r in hits])  # convert to ms

    # Linear regression: MT = a + b * ID
    slope, intercept, r_value, p_value, std_err = stats.linregress(ids, mts)

    results = {
        "n_trials": len(hits),
        "n_total": len(records),
        "success_rate": len(hits) / len(records),
        "intercept_a_ms": intercept,
        "slope_b_ms_per_bit": slope,
        "r_squared": r_value ** 2,
        "p_value": p_value,
        "throughput_bits_per_s": 1000.0 / slope if slope > 0 else float("inf"),
        "mean_mt_ms": mts.mean(),
        "std_mt_ms": mts.std(),
    }

    return results


def compute_derived_metrics(records):
    """Compute path efficiency, velocity profile symmetry, etc."""
    metrics = []

    for r in records:
        if not r.hit or len(r.wrist_xyz) < 3:
            continue

        traj = np.array(r.wrist_xyz)

        # Path length
        diffs = np.diff(traj, axis=0)
        path_length = np.sum(np.linalg.norm(diffs, axis=1))
        straight_line = np.linalg.norm(traj[-1] - traj[0])
        path_efficiency = straight_line / path_length if path_length > 0 else 0.0

        # Velocity profile
        velocities = np.array(r.wrist_velocity)
        speeds = np.linalg.norm(velocities, axis=1)

        # Time to peak velocity / movement time
        if len(speeds) > 0 and speeds.max() > 0:
            peak_idx = np.argmax(speeds)
            symmetry = peak_idx / len(speeds)
        else:
            symmetry = 0.5

        # Submovements (zero-crossings in acceleration)
        if len(speeds) > 2:
            accel = np.diff(speeds)
            zero_crossings = np.sum(np.diff(np.sign(accel)) != 0)
        else:
            zero_crossings = 0

        metrics.append({
            "trial_id": r.trial_id,
            "path_efficiency": path_efficiency,
            "velocity_symmetry": symmetry,
            "submovements": zero_crossings,
            "movement_time_ms": r.movement_time * 1000,
            "index_of_difficulty": r.index_of_difficulty,
        })

    return metrics


def plot_fitts(records, results, output_path):
    """Generate Fitts' law plot."""
    hits = [r for r in records if r.hit and r.movement_time > 0]

    ids = np.array([r.index_of_difficulty for r in hits])
    mts = np.array([r.movement_time * 1000 for r in hits])

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Plot 1: Fitts' law
    ax = axes[0]
    ax.scatter(ids, mts, alpha=0.5, s=20, label="Trials")

    # Regression line
    id_range = np.linspace(ids.min(), ids.max(), 100)
    mt_pred = results["intercept_a_ms"] + results["slope_b_ms_per_bit"] * id_range
    ax.plot(id_range, mt_pred, "r-", linewidth=2,
            label=f"MT = {results['intercept_a_ms']:.0f} + {results['slope_b_ms_per_bit']:.0f} * ID\n"
                  f"R² = {results['r_squared']:.3f}")
    ax.set_xlabel("Index of Difficulty (bits)")
    ax.set_ylabel("Movement Time (ms)")
    ax.set_title("Fitts' Law")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Success rate by ID
    ax = axes[1]
    id_bins = np.arange(1.5, 8.0, 1.0)
    success_by_id = []
    id_centers = []
    for i in range(len(id_bins) - 1):
        in_bin = [r for r in records
                  if id_bins[i] <= r.index_of_difficulty < id_bins[i+1]]
        if in_bin:
            rate = sum(1 for r in in_bin if r.hit) / len(in_bin)
            success_by_id.append(rate * 100)
            id_centers.append((id_bins[i] + id_bins[i+1]) / 2)

    ax.bar(id_centers, success_by_id, width=0.8, alpha=0.7, color="steelblue")
    ax.set_xlabel("Index of Difficulty (bits)")
    ax.set_ylabel("Success Rate (%)")
    ax.set_title("Accuracy vs Difficulty")
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3, axis="y")

    # Plot 3: Throughput
    ax = axes[2]
    throughputs = []
    for r in hits:
        if r.movement_time > 0:
            tp = r.index_of_difficulty / r.movement_time
            throughputs.append(tp)
    ax.hist(throughputs, bins=20, alpha=0.7, color="coral")
    ax.axvline(np.mean(throughputs), color="red", linestyle="--",
               label=f"Mean: {np.mean(throughputs):.1f} bits/s")
    ax.set_xlabel("Throughput (bits/s)")
    ax.set_ylabel("Count")
    ax.set_title("Throughput Distribution")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved to: {output_path}")
    plt.close()


def main(args):
    records, conditions = run_evaluation(
        model_path=args.model_path,
        xml_path=args.xml_path,
        n_trials_per_condition=args.n_trials,
    )

    # Fitts' analysis
    results = analyze_fitts(records)
    if results:
        print("\n" + "=" * 50)
        print("FITTS' LAW RESULTS")
        print("=" * 50)
        print(f"  Successful trials: {results['n_trials']}/{results['n_total']}")
        print(f"  Success rate:      {results['success_rate']*100:.1f}%")
        print(f"  MT = {results['intercept_a_ms']:.1f} + {results['slope_b_ms_per_bit']:.1f} * ID  (ms)")
        print(f"  R² = {results['r_squared']:.4f}")
        print(f"  Throughput:        {results['throughput_bits_per_s']:.1f} bits/s")
        print(f"  Mean MT:           {results['mean_mt_ms']:.0f} ms")

        # Human comparison reference
        print("\n  --- Human Reference (typical) ---")
        print("  Throughput:        4-10 bits/s")
        print("  Slope b:           100-200 ms/bit")
        print("  R²:                0.90-0.99")

    # Derived metrics
    metrics = compute_derived_metrics(records)
    if metrics:
        pe = np.mean([m["path_efficiency"] for m in metrics])
        vs = np.mean([m["velocity_symmetry"] for m in metrics])
        sm = np.mean([m["submovements"] for m in metrics])
        print(f"\n  Path efficiency:   {pe:.3f} (1.0 = straight line)")
        print(f"  Vel. symmetry:     {vs:.3f} (human ~0.4-0.5)")
        print(f"  Submovements:      {sm:.1f} (human ~1-3)")

    # Save results
    os.makedirs(args.output_dir, exist_ok=True)

    if results:
        with open(os.path.join(args.output_dir, "fitts_results.json"), "w") as f:
            json.dump(results, f, indent=2)

        plot_fitts(records, results, os.path.join(args.output_dir, "fitts_plot.png"))

    # Save raw trial data
    trial_data = []
    for r in records:
        d = {
            "trial_id": r.trial_id,
            "condition": r.condition,
            "hit": r.hit,
            "target_distance": r.target_distance,
            "target_width": r.target_width,
            "index_of_difficulty": r.index_of_difficulty,
            "reaction_time": r.reaction_time,
            "movement_time": r.movement_time,
            "endpoint_error": r.endpoint_error,
        }
        trial_data.append(d)

    with open(os.path.join(args.output_dir, "trial_data.json"), "w") as f:
        json.dump(trial_data, f, indent=2)

    print(f"\nResults saved to: {args.output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--xml-path", type=str, default=None)
    parser.add_argument("--n-trials", type=int, default=10)
    parser.add_argument("--output-dir", type=str, default="results/fitts")

    args = parser.parse_args()

    if args.xml_path is None:
        args.xml_path = os.path.join(project_root, "assets", "elbow", "myoelbow_buttonpress.xml")

    main(args)
