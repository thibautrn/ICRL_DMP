#!/usr/bin/env python3
"""
ICL Optimizer - LOCAL with GEMINI LLM
Everything runs on YOUR COMPUTER. LLM queries go to Gemini via llm_interface.py
"""
import json
import re
import subprocess
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from prompt_builder import build_prompt, iteration_history
from llm_interface import query_gemini
import os

# ============================================================
# CONFIGURATION
# ============================================================
LOGS_DIR      = Path("./logs/robot_sideways2")
EPISODES_FILE = LOGS_DIR / "processed_episodes.json"
BASELINE_FILE = LOGS_DIR / "baseline.npz"

time_now      = datetime.now().strftime("%Y%m%d%H%M")
OUTPUT_DIR    = Path("/home/asurite.ad.asu.edu/troisin/Documents/robot/mujoco_bimanual/logs")
ICL_OUTPUT_DIR = OUTPUT_DIR / "real_icl_logs" / time_now
ICL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MAX_ITERATIONS = 30



HUMAN_REWARD_MODE = True

# ============================================================
# DATA LOADING
# ============================================================

def load_episodes():
    with open(EPISODES_FILE, 'r') as f:
        data = json.load(f)
    episodes = data['episodes']
    return episodes


def load_baseline():
    data = np.load(BASELINE_FILE, allow_pickle=True)
    return {
        'y0_star':  data.get('y0_star', np.array([0.6, 0.3, 1.2])),
        'g_star':   data.get('g_star',  np.array([0.8, 0.55, 1.65])),
        'c':        data['c'],
        'h':        data['h'],
        'K':        float(data['K']),
        'D':        float(data['D']),
        'alpha_s':  float(data['alpha_s']),
        'run_time': float(data['run_time']),
        'M':        len(data['c'])
    }


# ============================================================
# WEIGHT PARSING + SAVING
# ============================================================

def parse_response(response_text, expected_size):
    numbers = re.findall(r'-?\d+\.?\d*', response_text)

    if len(numbers) >= expected_size:
        weights_flat = [float(n) for n in numbers[:expected_size]]
        M       = expected_size // 3
        weights = np.array(weights_flat).reshape(3, M)
        return weights, "reasoning"

    print(f"    ❌ Only found {len(numbers)} numbers, need {expected_size}")
    return None, None


def save_weights(iteration, weights, baseline, iter_dir):
    weights_file = iter_dir / f"weights_iter{iteration:03d}.npz"
    np.savez_compressed(
        weights_file,
        w=weights,
        y0=baseline['y0_star'],
        g=baseline['g_star'],
        iteration=iteration
    )
    return weights_file


# ============================================================
# PLOTTING
# ============================================================

def plot_rewards_progress(iterations, rewards, save_path=None):
    if not iterations or not rewards or len(iterations) != len(rewards):
        return

    it_arr      = np.asarray(iterations, dtype=int)
    rewards_arr = np.asarray(rewards,    dtype=float)

    sort_idx       = np.argsort(it_arr)
    it_sorted      = it_arr[sort_idx]
    rewards_sorted = rewards_arr[sort_idx]

    plt.figure(figsize=(8, 4))
    plt.plot(it_sorted, rewards_sorted, linewidth=2, label="Reward")

    best_idx = np.argmax(rewards_sorted)
    plt.scatter(it_sorted[best_idx], rewards_sorted[best_idx],
                marker="*", s=200, zorder=5, label="Best")

    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.xlabel("Iteration")
    plt.ylabel("Total Reward")
    plt.title("Optimization Progress")
    plt.grid(True)
    plt.legend()
    plt.ylim(0, 2000)
    plt.xlim(0, MAX_ITERATIONS)

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=120, bbox_inches="tight")

    plt.close()

def apply_human_reward_override(reward: dict) -> dict:
    if not HUMAN_REWARD_MODE:
        return reward

    print(f"\n🎯 Human reward override (HUMAN_REWARD_MODE=True)")
    print(f"   Camera says ball_in_cup = {reward['ball_in_cup']}")
    print(f"   y = ball in cup (+1000) | n = not in cup | number = overwrite total | ENTER = keep camera")
    user_input = input("   >>> ").strip().lower()

    if user_input == '':
        print(f"   → Keeping camera result (total_reward={reward['total_reward']:.1f})")
    elif user_input == 'y':
        reward = dict(reward)
        reward['ball_in_cup']      = True
        reward['collision_reward'] = 1000.0
        reward['total_reward']     = (
            reward.get('distance_reward', 0.0)
            + reward.get('height_bonus',  0.0)
            + 1000.0
        )
        print(f"   ✓ Overridden → ball_in_cup=True, total_reward={reward['total_reward']:.1f}")
    elif user_input == 'n':
        reward = dict(reward)
        reward['ball_in_cup']      = False
        reward['collision_reward'] = 0.0
        reward['total_reward']     = (
            reward.get('distance_reward', 0.0)
            + reward.get('height_bonus',  0.0)
        )
        print(f"   ✓ Overridden → ball_in_cup=False, total_reward={reward['total_reward']:.1f}")
    else:
        try:
            override_value = float(user_input)
            reward = dict(reward)
            reward['total_reward'] = override_value
            print(f"   ✓ Total reward overwritten → {override_value:.1f}")
        except ValueError:
            print(f"   → Unrecognised input, keeping camera result (total_reward={reward['total_reward']:.1f})")

    return reward

# ============================================================
# ROS2 ENVIRONMENT
# ============================================================

def get_ros2_env():
    """
    Source ROS2 workspace setup in a clean shell (no conda)
    and capture the resulting environment variables.
    Called ONCE at startup.
    """
    print("🔧 Loading ROS2 environment...")
    ros2_ws = "/home/asurite.ad.asu.edu/troisin/Documents/robot/bimanual_mocap/bimanual_ws"
    ros2_setup = f"{ros2_ws}/install/setup.sh"
    home_dir = "/home/asurite.ad.asu.edu/troisin"

    seed_env = {
        'HOME': home_dir,
        'PATH': '/usr/bin:/bin:/usr/local/bin:/opt/ros/jazzy/bin',
        'USER': os.environ.get('USER', 'troisin'),
    }

    print("🔧 Loading ROS2 environment...")
    result = subprocess.run(
        f'source /opt/ros/jazzy/setup.bash && source {ros2_setup} && env',
        shell=True,
        executable='/bin/bash',
        capture_output=True,
        text=True,
        env=seed_env
    )

    if result.returncode != 0:
        print(f"⚠️  Warning: ROS2 env setup returned non-zero: {result.stderr}")

    ros_env = {}
    for line in result.stdout.splitlines():
        if '=' in line:
            key, _, val = line.partition('=')
            ros_env[key] = val

    if 'ROS_DISTRO' not in ros_env:
        print("⚠️  Warning: ROS_DISTRO not found in env — setup.sh may have failed")
    else:
        print(f"✓ ROS2 environment loaded (ROS_DISTRO={ros_env['ROS_DISTRO']})")

    return ros_env


# ============================================================
# SUBPROCESS CALLS
# ============================================================

def call_mujoco_sim(weights_file, iteration):
    """
    Run mujoco_sim.py locally — visualization only, no reward.
    Returns (success: bool, cup_collision: bool).
    Exit code 0 = clean, 2 = cup collision, anything else = error.
    """
    print(f"\n🎮 Running MuJoCo visualization...")
    result = subprocess.run(
        ['python', 'full_robot/mujoco_sim.py', str(weights_file), str(iteration)],
        capture_output=True,
        text=True
    )

    if result.returncode == 2:
        return True, True   # clean run, collision detected
    if result.returncode != 0:
        print(f"❌ MuJoCo failed: {result.stderr}")
        return False, False
    return True, False



def call_gazebo_exec(weights_file, iteration, ros_env):
    """
    Run gazebo_exec.py which handles:
      - launching move_right_arm_dmp.py via ros2 run
      - launching camera_reward.py simultaneously
      - returning the reward JSON
    """
    print(f"\n🌐 Running Gazebo + ZED...")
    print(f"   Weights: {weights_file}")

    result = subprocess.run(
        ['python3', 'full_robot/gazebo_exec.py', str(weights_file), str(iteration)],
        capture_output=True,
        text=True
    )


    # Parse JSON reward output from gazebo_exec.py
    for line in result.stdout.strip().split('\n'):
        if line.startswith('{'):
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                continue

    print(f"❌ No valid JSON reward found in output:\n{result.stdout}")
    return None


# ============================================================
# MAIN LOOP
# ============================================================

def main():
    print("=" * 70)
    print("ICL OPTIMIZER")
    print("LLM: Gemini (via llm_interface.py)")
    print("=" * 70)
    print(f"REWARD STRUCTURE:")
    print(f"  1. Distance reward (0-750 pts) - closer to cup = more points")
    print(f"  2. Height bonus (+200 pts)     - ball above cup rim at best moment")
    print(f"  3. Collision jackpot (+1000 pts) - ball lands in cup")
    print(f"  4. Self collision → total reward overwritten to -1")
    print("=" * 70)

    # Load data
    print(f"\n📂 Loading data...")
    episodes = load_episodes()
    baseline = load_baseline()
    print(f"✓ Baseline: M={baseline['M']} basis functions")
    print(f"✓ Episodes: {len(episodes)} loaded")
    print(f"✓ Output dir: {ICL_OUTPUT_DIR}")

    # Load ROS2 env ONCE — reused for every gazebo call
    ros_env = get_ros2_env()

    # Optimization state
    optimization_results = []
    history      = ""
    reward_iters = []
    reward_values = []

    bar_format = (
        "{desc} | {percentage:3.0f}%|{bar}| "
        "{n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}"
    )

    print("\n" + "=" * 60 + " STARTING OPTIMIZATION " + "=" * 60 + "\n")

    with tqdm(total=MAX_ITERATIONS, desc="Optimization", unit="iter",
              bar_format=bar_format) as pbar:
        pbar.set_postfix(last="0.0", best="0.0")

        for iteration in range(1, MAX_ITERATIONS + 1):

            # ── STEP 1: Build prompt and query Gemini ──────────────────────
            prompt = build_prompt(
                baseline, iteration, MAX_ITERATIONS,
                episodes, LOGS_DIR, history, n_show=20
            )

            response_text = query_gemini(prompt)
            if response_text is None:
                print(f"\n❌ Gemini failed at iteration {iteration}, skipping")
                pbar.update(1)
                continue

            # Save prompt + response
            iter_dir = ICL_OUTPUT_DIR / f"iter_{iteration:03d}"
            iter_dir.mkdir(parents=True, exist_ok=True)

            with open(iter_dir / "prompt.txt", "w", encoding="utf-8") as f:
                f.write(prompt)
            with open(iter_dir / "response.txt", "w", encoding="utf-8") as f:
                f.write(response_text)

            # ── STEP 2: Parse weights ──────────────────────────────────────
            expected_size = 3 * baseline['M']
            weights, reasoning = parse_response(response_text, expected_size)

            if weights is None:
                pbar.update(1)
                continue

            weights_file = save_weights(iteration, weights, baseline, iter_dir)

            # ── STEP 3: MuJoCo visualization (optional preview) ────────────
            mujoco_ok, cup_collision = call_mujoco_sim(weights_file, iteration)

            if not mujoco_ok:
                pbar.update(1)
                continue
            print(f"cup_collision = {cup_collision}")
            # ── STEP 3b: Auto-skip on cup collision ────────────────────────
            if cup_collision:
                print(f"\n💥 Collision detected in simulation — skipping real robot, reward = -1")
                collision_reward = {
                    'total_reward':        -1.0,
                    'ball_in_cup':         False,
                    'collision_reward':    0.0,
                    'distance_reward':     0.0,
                    'height_bonus':        0.0,
                    'min_distance_to_cup': float('inf'),
                    'cup_collision':       True,
                }
                history = iteration_history(iteration, weights, collision_reward, history)
                optimization_results.append({
                    'iteration':    iteration,
                    'weights_file': str(weights_file),
                    'reasoning':    reasoning,
                    'reward':       collision_reward,
                })
                reward_iters.append(iteration)
                reward_values.append(-1.0)
                pbar.set_postfix(last="-1.0 💥", best=f"{max(r['reward']['total_reward'] for r in optimization_results):.1f}")
                pbar.update(1)
                continue

            # ── STEP 4: User gate before real robot ────────────────────────
            print(f"\n{'─' * 70}")
            print(f"🤔 Send to Gazebo + Real Robot? (ENTER = yes, 's' = skip)")
            user_input = input(">>> ").strip().lower()

            if user_input == 's':
                print("⏭️  Skipped — moving to next iteration")
                collision_reward = {
                    'total_reward':        -1.0,
                    'ball_in_cup':         False,
                    'collision_reward':    0.0,
                    'distance_reward':     0.0,
                    'height_bonus':        0.0,
                    'min_distance_to_cup': float('inf'),
                    'cup_collision':       True,
                }
                history = iteration_history(iteration, weights, collision_reward, history)
                optimization_results.append({
                    'iteration':    iteration,
                    'weights_file': str(weights_file),
                    'reasoning':    reasoning,
                    'reward':       collision_reward,
                })
                reward_iters.append(iteration)
                reward_values.append(-1.0)
                pbar.set_postfix(last="-1.0 💥", best=f"{max(r['reward']['total_reward'] for r in optimization_results):.1f}")
                pbar.update(1)
                continue

            # ── STEP 5: Gazebo + ZED (real reward) ────────────────────────
            gazebo_reward = call_gazebo_exec(weights_file, iteration, ros_env)

            if gazebo_reward is None:
                print("❌ Gazebo failed, skipping iteration")
                pbar.update(1)
                continue
            print(gazebo_reward)
            gazebo_reward = apply_human_reward_override(gazebo_reward)
            with open(iter_dir / "reward.json", "w") as f:
                json.dump(gazebo_reward, f, indent=2)
            # ── STEP 6: Update history and results ────────────────────────
            history = iteration_history(iteration, weights, gazebo_reward, history)

            optimization_results.append({
                'iteration':    iteration,
                'weights_file': str(weights_file),
                'reasoning':    reasoning,
                'reward':       gazebo_reward
            })

            reward_iters.append(iteration)
            reward_values.append(gazebo_reward['total_reward'])

            # Plot every 5 iterations
            if iteration % 5 == 0:
                plot_rewards_progress(
                    reward_iters,
                    reward_values,
                    save_path=ICL_OUTPUT_DIR / f"reward_curve_iter_{iteration:03d}.pdf"
                )

            # Update progress bar
            last_reward  = f"{gazebo_reward['total_reward']:.1f}"
            self_col_tag = " 💥" if gazebo_reward.get('self_collision', False) else ""
            best_reward  = f"{max(r['reward']['total_reward'] for r in optimization_results):.1f}"

            pbar.set_postfix(last=f"{last_reward}{self_col_tag}", best=best_reward)
            pbar.update(1)

            print(f"\n{'─' * 70}")
            print(f"  Total:    {gazebo_reward['total_reward']:.1f} pts")
            print(f"  Distance: {gazebo_reward['min_distance_to_cup']:.3f} m")
            print(f"  In cup:   {gazebo_reward['ball_in_cup']}")
            print(f"  Best so far: {best_reward} pts")

    # ── SUMMARY ────────────────────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print("OPTIMIZATION COMPLETE")
    print(f"{'=' * 70}")

    if not optimization_results:
        print("\n❌ No successful iterations completed.")
        return

    best = max(optimization_results, key=lambda x: x['reward']['total_reward'])
    print(f"\n🏆 Best iteration: {best['iteration']}")
    print(f"   Reward:         {best['reward']['total_reward']:.1f} pts")
    print(f"   Weights file:   {best['weights_file']}")

    # Final reward curve
    plot_rewards_progress(
        reward_iters, reward_values,
        save_path=ICL_OUTPUT_DIR / "reward_curve_final.pdf"
    )

    # Save full results JSON
    results_file = ICL_OUTPUT_DIR / "optimization_results.json"
    with open(results_file, 'w') as f:
        json.dump({
            'model':          'gemini',
            'max_iterations': MAX_ITERATIONS,
            'reward_structure': {
                'distance_reward':  '0-750 pts (proximity to cup)',
                'height_bonus':     '+200 pts (ball above rim at best moment)',
                'collision_reward': '+1000 pts (ball in cup)',
                'self_collision':   'total reward overwritten to -1',
            },
            'results':        optimization_results,
            'best_iteration': best['iteration'],
            'best_reward':    best['reward']['total_reward'],
        }, f, indent=2)

    print(f"\n💾 Results: {results_file}")
    print(f"💾 All files: {ICL_OUTPUT_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()