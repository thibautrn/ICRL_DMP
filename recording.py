#!/usr/bin/env python3
"""
DMP Execution Launcher - Synchronized DMP Execution + Camera Reward
Launches move_right_arm_dmp.py (DMP execution) and camera_reward.py (ZED) simultaneously
"""
import sys
import json
import subprocess
import time
import threading
from pathlib import Path

# ============================================================
# CONFIGURATION
# ============================================================
CONDA_PYTHON  = "/home/asurite.ad.asu.edu/troisin/miniconda3/envs/mocap_env/bin/python"
CAMERA_SCRIPT = "full_robot/camera_reward.py"


# ============================================================
# SYNCHRONIZATION  (unchanged from recording_launcher.py)
# ============================================================

class RecordingSyncLauncher:
    """Launch DMP executor (ROS2) and camera reward (conda) simultaneously"""

    def __init__(self, episode_id):
        self.episode_id      = episode_id
        self.camera_process  = None
        self.start_signal    = threading.Event()
        self.camera_started  = False
        self.error           = None

    def start_camera(self):
        """Start camera immediately (no wait) — runs in background"""
        try:
            print(f"📷 Starting camera capture (background)...")
            self.camera_process = subprocess.Popen(
                [CONDA_PYTHON, CAMERA_SCRIPT, str(self.episode_id)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            self.camera_started = True
            print(f"   ✓ Camera running")
        except Exception as e:
            self.error = f"Camera start failed: {e}"

    def get_camera_result(self, timeout=15.0):
        """Wait for camera to finish and return reward dict"""
        if self.camera_process is None:
            return None
        
        print(f"\n📷 Waiting for camera to finish processing...")
        try:
            stdout, stderr = self.camera_process.communicate(timeout=timeout)
            if self.camera_process.returncode == 0:
                for line in stdout.strip().split('\n'):
                    if line.startswith('{'):
                        reward_data = json.loads(line)
                        print(f"   ✓ Reward: {reward_data['total_reward']:.1f}")
                        return reward_data
                print(f"   ❌ No valid JSON output from camera")
                return None
            else:
                print(f"   ❌ Camera failed: {stderr}")
                return None
        except subprocess.TimeoutExpired:
            print(f"   ❌ Camera timeout!")
            self.camera_process.kill()
            return None


# ============================================================
# ROS2 ENVIRONMENT  (unchanged from recording_launcher.py)
# ============================================================

def get_ros2_env():
    ros2_ws = "/home/asurite.ad.asu.edu/troisin/Documents/robot/bimanual_mocap/bimanual_ws"
    ros2_setup = f"{ros2_ws}/install/setup.sh"
    home_dir = "/home/asurite.ad.asu.edu/troisin"

    import os
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
        print("⚠️  Warning: ROS_DISTRO not found — setup.sh may have failed")
    else:
        print(f"✓ ROS2 environment loaded (ROS_DISTRO={ros_env['ROS_DISTRO']})")

    # Add DISPLAY for GUI
    ros_env['DISPLAY'] = ':0'

    return ros_env


# ============================================================
# MAIN
# ============================================================

def run_single_weights_file(weights_file, ros_env, ros2_ws, output_dir):
    """Execute DMP with a single weights file and return episode data"""
    import re

    weights_name = Path(weights_file).stem
    # Use the numeric ID already embedded in the filename (e.g. joints_20260218_175803_weights → 20260218_175803)
    # Strip trailing _weights/_recon suffix first, then collect all digit sequences
    stem_stripped = re.sub(r'_(weights|recon)$', '', weights_name)
    digits = re.findall(r'\d+', stem_stripped)
    episode_id = '_'.join(digits) if digits else weights_name

    print(f"\n{'='*70}")
    print(f"DMP EXECUTION + CAMERA REWARD")
    print(f"Weights: {weights_name}")
    print(f"Episode: {episode_id}")
    print(f"{'='*70}\n")

    # Start camera in background
    launcher = RecordingSyncLauncher(episode_id)
    camera_thread = threading.Thread(target=launcher.start_camera)
    camera_thread.start()
    time.sleep(1.0)

    while not launcher.camera_started:
        if launcher.error:
            print(f"❌ {launcher.error}")
            return None
        time.sleep(0.1)

    print("✓ Camera running in background\n")

    # Run DMP execution script (blocks until complete)
    print(f"🤖 Starting DMP execution...\n")

    print("PATH in ros_env:", ros_env.get('PATH', 'NOT FOUND')[:200])

    logger_result = subprocess.run(
        ['ros2', 'run', 'bimanualrobot_system_tests', 'move_right_arm_fast_dmp_ball.py', str(weights_file)],
        env=ros_env,
        cwd=ros2_ws
    )

    if logger_result.returncode != 0:
        print("\n❌ DMP execution failed!")
        if launcher.camera_process:
            launcher.camera_process.kill()
        return None

    print(f"\n✓ DMP execution complete")

    # Let camera finish its CAPTURE_DURATION naturally and compute the reward
    camera_reward = launcher.get_camera_result(timeout=30.0)

    if camera_reward is None:
        print("\n❌ Failed to get reward from camera!")
        return None

    print(f"\n{'='*70}")
    print("RESULTS")
    print(f"{'='*70}")
    print(f"Episode:      {episode_id}")
    print(f"Weights:      {weights_name}")
    print(f"Total reward: {camera_reward['total_reward']:.1f}")
    print(f"Ball in cup:  {camera_reward['ball_in_cup']}")
    print(f"Distance:     {camera_reward['min_distance_to_cup']:.3f} m")
    print(f"{'='*70}\n")



    # Save episode JSON
    episode_data = {
        'episode_id': episode_id,
        'timestamp': episode_id,
        'weights_file': str(weights_file),
        'weights_name': weights_name,
        'reward': camera_reward,
        'total_reward': camera_reward['total_reward'],
        'ball_in_cup': camera_reward['ball_in_cup'],
        'min_distance_to_cup': camera_reward['min_distance_to_cup'],
    }

    episode_file = output_dir / f"episode_{weights_name}.json"
    
    with open(episode_file, 'w') as f:
        json.dump(episode_data, f, indent=2)
    
    print(f"💾 Episode saved: {episode_file}\n")

    return episode_data


def main():
    ros2_ws = "/home/asurite.ad.asu.edu/troisin/Documents/robot/bimanual_mocap/bimanual_ws"

    print(f"\n{'='*70}")
    print(f"BATCH DMP EXECUTION + CAMERA REWARD")
    print(f"{'='*70}\n")

    # Find all weights files in logs/robot_episode
    weights_dir = Path("/home/asurite.ad.asu.edu/troisin/Documents/robot/mujoco_bimanual/logs/robot_sideways2")
    if not weights_dir.exists():
        print(f"❌ Directory not found: {weights_dir}")
        sys.exit(1)

    weights_files = sorted(weights_dir.glob("*weights*.npz"))
    
    if not weights_files:
        print(f"❌ No weights files found in {weights_dir}")
        sys.exit(1)

    print(f"Found {len(weights_files)} weights files:")
    for wf in weights_files:
        print(f"  - {wf.name}")
    print()

    # Get ROS2 environment once
    ros_env = get_ros2_env()

    # Process each weights file
    results = []
    for i, weights_file in enumerate(weights_files, 1):
        print(f"\n{'#'*70}")
        print(f"Processing {i}/{len(weights_files)}: {weights_file.name}")
        print(f"{'#'*70}")
        
        episode_data = run_single_weights_file(weights_file, ros_env, ros2_ws, weights_dir)
        
        if episode_data:
            results.append(episode_data)
            print(f"✓ Success for {weights_file.name}\n")
        else:
            print(f"❌ Failed for {weights_file.name}\n")
        
        # Wait between runs
        if i < len(weights_files):
            print(f"⏳ Waiting 30s before next iteration...")
            time.sleep(10.0)

    # Summary
    print(f"\n{'='*70}")
    print(f"BATCH PROCESSING COMPLETE")
    print(f"{'='*70}")
    print(f"Total processed: {len(results)}/{len(weights_files)}")
    print(f"Success rate: {len(results)/len(weights_files)*100:.1f}%")
    print(f"Results saved to: {weights_dir}")
    print(f"{'='*70}\n")

    # Output summary JSON for potential parent process
    summary = {
        'total_weights': len(weights_files),
        'successful': len(results),
        'failed': len(weights_files) - len(results),
        'results': results
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()