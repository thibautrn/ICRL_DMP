#!/usr/bin/env python3
"""
Gazebo Executor - Launches move_right_arm_dmp.py (ROS2) and camera_reward.py simultaneously.
Called by main_icl.py with: python gazebo_exec.py <weights_file> <iteration>
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
ROS2_WS       = "/home/asurite.ad.asu.edu/troisin/Documents/robot/bimanual_mocap/bimanual_ws"
ROS2_SETUP    = f"{ROS2_WS}/install/setup.sh"
HOME_DIR      = "/home/asurite.ad.asu.edu/troisin"


# ============================================================
# ROS2 ENVIRONMENT  (unchanged pattern)
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
# CAMERA LAUNCHER  (same pattern as recording_launcher.py)
# ============================================================

class CameraLauncher:
    def __init__(self, episode_id):
        self.episode_id     = episode_id
        self.camera_process = None
        self.camera_started = False
        self.error          = None

    def start_camera(self):
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

    def get_camera_result(self, timeout=30.0):
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
# MAIN
# ============================================================

def main():
    if len(sys.argv) < 3:
        print("Usage: python gazebo_exec.py <weights_file> <iteration>")
        sys.exit(1)

    weights_file = sys.argv[1]
    iteration    = sys.argv[2]

    print(f"\n{'='*70}")
    print(f"GAZEBO EXEC — Iteration {iteration}")
    print(f"Weights: {weights_file}")
    print(f"{'='*70}\n")

    ros_env = get_ros2_env()

    # Start camera in background
    launcher = CameraLauncher(iteration)
    camera_thread = threading.Thread(target=launcher.start_camera)
    camera_thread.start()
    time.sleep(1.0)

    while not launcher.camera_started:
        if launcher.error:
            print(f"❌ {launcher.error}")
            sys.exit(1)
        time.sleep(0.1)

    print("✓ Camera running in background\n")

    # Run DMP on real robot via ros2 run (blocks until complete)
    print(f"🤖 Starting DMP execution...\n")
    dmp_result = subprocess.run(
        ['ros2', 'run', 'bimanualrobot_system_tests', 'move_right_arm_fast_dmp_ball.py', weights_file],
        env=ros_env,
        cwd=ROS2_WS
    )

    # if dmp_result.returncode != 0:
    #     print("\n❌ DMP execution failed!")
    #     if launcher.camera_process:
    #         launcher.camera_process.kill()
    #     sys.exit(1)

    print(f"\n✓ DMP execution complete")

    # Signal camera to stop now that the motion is done
    if launcher.camera_process and launcher.camera_process.poll() is None:
        print("📷 Signaling camera to stop capture...")
        launcher.camera_process.send_signal(__import__('signal').SIGTERM)

    # Wait for camera to finish computing and output reward (short grace period)
    reward = launcher.get_camera_result(timeout=15.0)

    if reward is None:
        print("\n❌ Failed to get reward from camera!")
        sys.exit(1)

    print(f"\n{'='*70}")
    print(f"RESULTS — Iteration {iteration}")
    print(f"{'='*70}")
    print(f"Total reward: {reward['total_reward']:.1f}")
    print(f"Ball in cup:  {reward['ball_in_cup']}")
    print(f"Distance:     {reward['min_distance_to_cup']:.3f} m")
    print(f"{'='*70}\n")

    # Add weights file to reward so main_icl.py can reference it
    reward["weights_file"] = str(weights_file)

    # Print JSON for main_icl.py to parse
    print(json.dumps(reward))


if __name__ == "__main__":
    main()