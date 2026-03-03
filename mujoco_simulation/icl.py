#!/usr/bin/env python3
"""
Integrated ICL + DMP Evaluation Loop
Generates weights with LLM, tests them in MuJoCo, feeds back rewards, iterates.

UPDATED: Now tracks cup collisions (cup hitting arm/stand) instead of self-collisions,
         with optional RED highlighting when viewer is enabled.
"""

import json
import numpy as np
from pathlib import Path
import requests
import torch
import time
import mujoco
import mujoco.viewer
from datetime import datetime
import csv
import math
import matplotlib.pyplot as plt
from tqdm import tqdm
from matplotlib.ticker import MaxNLocator

from prompt_builder import build_prompt, iteration_history
from llm_interface import query_gemini

MAX_ITERS = 100

# Directories
time_now = datetime.now().strftime("%Y%m%d%H%M")
OUTPUT_DIR = Path("./logs")
ICL_OUTPUT_DIR = OUTPUT_DIR / "icl_logs" / time_now
ICL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Files
LOGS_DIR = Path("./logs/custom_ex")
EPISODES_FILE = LOGS_DIR / "processed_episodes.json"
BASELINE_FILE = LOGS_DIR / "baseline.npz"

# MuJoCo Simulation Parameters
MUJOCO_XML = "bimanualrobot.xml"
SHOW_VIEWER = False
CYCLE_SECONDS = 0.025
STARTUP_WAIT = 5.0
END_POSITION_MOVE_TIME = 3.0
END_POSITION_HOLD_TIME = 0.5
MIN_JOINT_STEP_RAD = np.deg2rad(0.1)

# Joint configurations
STARTUP_JOINT_POSITIONS = {
    "leftarm_shoulder_pan_joint": -1.57,
    "leftarm_shoulder_lift_joint": 1.01,
    "leftarm_elbow_joint": -0.113,
    "leftarm_wrist_1_joint": 0.0,
    "leftarm_wrist_2_joint": 0.0,
    "leftarm_wrist_3_joint": 0.0,
}

END_JOINT_POSITIONS = {
    "leftarm_shoulder_pan_joint": -1.6,
    "leftarm_shoulder_lift_joint": 0.85,
    "leftarm_elbow_joint": 0.0,
    "leftarm_wrist_1_joint": 0.0,
    "leftarm_wrist_2_joint": 0.0,
    "leftarm_wrist_3_joint": 0.0,
}

IK_JOINTS = [
    "leftarm_shoulder_pan_joint",
    "leftarm_shoulder_lift_joint",
    "leftarm_elbow_joint",
]

JOINTS = [
    "leftarm_shoulder_pan_joint",
    "leftarm_shoulder_lift_joint",
    "leftarm_elbow_joint",
    "leftarm_wrist_1_joint",
    "leftarm_wrist_2_joint",
    "leftarm_wrist_3_joint",
]

# ============================================================
# HELPER FUNCTIONS - ICL
# ============================================================

def load_episodes_simple(filepath):
    with open(filepath, 'r') as f:
        data = json.load(f)
    episodes = data['episodes']
    if episodes:
        cup_center = np.array([ep['cup_center'] for ep in episodes])
    else:
        cup_center = [0.73, 0.46, 1.56]
    cup_radius = 0.10
    return episodes, cup_center, cup_radius

def load_baseline(filepath):
    data = np.load(filepath, allow_pickle=True)
    return {
        'y0_star': data.get('y0_star', np.array([0.6, 0.3, 1.2])),
        'g_star': data.get('g_star', np.array([0.8, 0.55, 1.65])),
        'c': data['c'],
        'h': data['h'],
        'K': float(data['K']),
        'D': float(data['D']),
        'alpha_s': float(data['alpha_s']),
        'run_time': float(data['run_time']),
        'M': len(data['c'])
    }

def parse_response(response_text, expected_size):
    import re
    numbers = re.findall(r'-?\d+\.?\d*', response_text)
    if len(numbers) >= expected_size:
        weights_flat = [float(n) for n in numbers[:expected_size]]
        M = expected_size // 3
        weights = np.array(weights_flat).reshape(3, M)
        return weights, "reasoning"
    print(f"    ❌ Only found {len(numbers)} numbers, need {expected_size}")
    return None, None

# ============================================================
# CUP COLLISION
# ============================================================

def build_robot_geoms(model):
    """Build set of all robot geom IDs (cup + arm + stand)."""
    robot_geoms = set()
    cup_geoms = set()
    
    for i in range(model.ngeom):
        body_id = model.geom_bodyid[i]
        body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
        geom_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i)
        
        # All robot parts (cup, arm, stand)
        if body_name and ("leftarm" in body_name or "stand" in body_name):
            robot_geoms.add(i)
            
            # Also track which are cup geoms
            if geom_name and "left_cup" in geom_name:
                cup_geoms.add(i)
    
    return robot_geoms, cup_geoms

def check_cup_collision(model, data, robot_geoms, cup_geoms, highlight=False):
    """Check if cup hits any other robot parts (excluding stand — its collision mesh is misaligned)."""
    if data.ncon == 0:
        return False, None

    for i in range(data.ncon):
        contact = data.contact[i]
        geom1_id = contact.geom1
        geom2_id = contact.geom2
        
        # Skip contacts with stand (collision mesh is misaligned from visual)
        name1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom1_id) or ""
        name2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom2_id) or ""
        if name1 == "stand_collision_geom" or name2 == "stand_collision_geom":
            continue

        geom1_is_cup = geom1_id in cup_geoms
        geom2_is_cup = geom2_id in cup_geoms
        geom1_is_robot = geom1_id in robot_geoms
        geom2_is_robot = geom2_id in robot_geoms
        
        # One is cup, other is robot (but not both cup)
        if (geom1_is_cup and geom2_is_robot and not geom2_is_cup) or \
           (geom2_is_cup and geom1_is_robot and not geom1_is_cup):
            
            if highlight:
                model.geom_rgba[geom1_id] = [1.0, 0.0, 0.0, 1.0]
                model.geom_rgba[geom2_id] = [1.0, 0.0, 0.0, 1.0]
            
            return True, f"{name1 or f'geom{geom1_id}'} ↔ {name2 or f'geom{geom2_id}'}"
    
    return False, None

def store_original_colors(model, robot_geoms):
    """Store original colors of all robot geoms."""
    original_colors = {}
    for geom_id in robot_geoms:
        original_colors[geom_id] = model.geom_rgba[geom_id].copy()
    return original_colors

def restore_colors(model, original_colors):
    """Restore original colors."""
    for geom_id, color in original_colors.items():
        model.geom_rgba[geom_id] = color

# ============================================================
# DMP FUNCTIONS
# ============================================================

def canonical_by_steps(N, dt, tau, alpha_s):
    S = np.empty(int(N), float)
    s = 1.0
    for k in range(int(N)):
        S[k] = s
        s += dt * (-alpha_s * s / tau)
    return S

def design_matrix(S, c, h):
    Phi = np.empty((len(S), len(c)), float)
    for k, s in enumerate(S):
        psi = np.exp(-h * (s - c)**2)
        Phi[k, :] = (psi / (psi.sum() + 1e-12)) * s
    return Phi

def rollout_dmp_3d(y0, g, W, K, D, tau, dt, c, h, alpha_s):
    N   = int(np.round(tau / dt)) + 1
    S   = canonical_by_steps(N, dt, tau, alpha_s)
    Phi = design_matrix(S, c, h)
    Y  = np.zeros((N, 3), float)
    Yd = np.zeros_like(Y)
    Y[0] = y0
    for k in range(N-1):
        acc = np.zeros(3, float)
        for d in range(3):
            f = float(Phi[k].dot(W[d]))
            acc[d] = (K*(g[d] - Y[k,d]) - D*tau*Yd[k,d] + K * f * (g[d] - y0[d])) / (tau**2)
        Yd[k+1] = Yd[k] + acc * dt
        Y[k+1]  = Y[k]  + Yd[k+1] * dt
    return Y

class DMPJointRolloutSource:
    def __init__(self, baseline, weights, dt):
        c, h = baseline["c"], baseline["h"]
        K, D = float(baseline["K"]), float(baseline["D"])
        alpha_s = float(baseline["alpha_s"])
        tau = float(baseline["run_time"])
        y0 = baseline['y0_star'].copy()
        g = baseline['g_star'].copy()
        Y = rollout_dmp_3d(y0, g, weights, K, D, tau, dt, c, h, alpha_s)
        self.Y = Y
        self.dt = float(dt)
        self.N  = len(Y)
        self.k  = 0

    def next(self):
        if self.k >= self.N:
            return None
        Yk = self.Y[self.k].copy()
        self.k += 1
        return Yk


class SingleDMPRun:
    def __init__(self, model, data, dmp_source, ball_output_file, robot_geoms, cup_geoms, show_viewer):
        self.m = model
        self.d = data
        self.robot_geoms = robot_geoms
        self.cup_geoms = cup_geoms
        self.show_viewer = show_viewer
        self.collision_detected = False

        # Build mappings
        self.joint_to_qpos_idx = {}
        self.joint_to_ctrl_idx = {}

        for i in range(self.m.njnt):
            joint_name = mujoco.mj_id2name(self.m, mujoco.mjtObj.mjOBJ_JOINT, i)
            self.joint_to_qpos_idx[joint_name] = self.m.jnt_qposadr[i]

        for i in range(self.m.nu):
            actuator_name = mujoco.mj_id2name(self.m, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
            joint_name = actuator_name.replace("_motor", "_joint")
            self.joint_to_ctrl_idx[joint_name] = i

        # Set startup position
        for joint_name, position in STARTUP_JOINT_POSITIONS.items():
            if joint_name in self.joint_to_qpos_idx:
                self.d.qpos[self.joint_to_qpos_idx[joint_name]] = float(position)
            if joint_name in self.joint_to_ctrl_idx:
                self.d.ctrl[self.joint_to_ctrl_idx[joint_name]] = float(position)

        self.d.qvel[:] = 0.0
        mujoco.mj_forward(self.m, self.d)

        # Reset ball to initial position
        try:
            ball_body = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, "left_cup_ball")
            ball_jnt_id = None
            for i in range(self.m.njnt):
                if self.m.jnt_bodyid[i] == ball_body:
                    ball_jnt_id = i
                    break
            if ball_jnt_id is not None:
                qpos_adr = self.m.jnt_qposadr[ball_jnt_id]
                qvel_adr = self.m.jnt_dofadr[ball_jnt_id]
                self.d.qpos[qpos_adr:qpos_adr+3] = [0.80, 0.55, 1.25]
                self.d.qpos[qpos_adr+3:qpos_adr+7] = [1.0, 0.0, 0.0, 0.0]
                self.d.qvel[qvel_adr:qvel_adr+6] = 0.0
                mujoco.mj_forward(self.m, self.d)
        except:
            pass

        # Find ball body and cup center
        try:
            self.ball_body_id = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, "left_cup_ball")
        except:
            self.ball_body_id = None

        try:
            self.cup_center_site = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_SITE, "left_cup_center")
        except:
            self.cup_center_site = None

        # Ball in cup detection geoms
        try:
            self.ball_geom = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_GEOM, "left_cup_ball_geom")
            self.sensor_geom = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_GEOM, "left_cup_sensor_geom")
        except:
            self.ball_geom = None
            self.sensor_geom = None

        # State machine
        self.state = "WAITING"
        self.start_time = None
        self.dmp_start_time = None
        self.end_move_start_time = None

        # DMP
        self._dmp = dmp_source
        self._last_dmp_time = None
        self._pending_q = None
        self._last_q_cmd = None

        # Logging
        # Columns: t_sec  ball_x  ball_y  ball_z  above_cup  cup_x  cup_y  cup_z  ball_in_cup  cup_collision
        self.ball_log = open(ball_output_file, "w", buffering=1)
        self.ball_log.write("# t_sec  ball_x  ball_y  ball_z  above_cup  cup_x  cup_y  cup_z  ball_in_cup  cup_collision\n")

    def is_ball_above_cup(self, data):
        if self.ball_body_id is None:
            return False
        ball_z = data.xpos[self.ball_body_id][2]
        if self.cup_center_site is not None:
            cup_rim_z = data.site_xpos[self.cup_center_site][2]
            return ball_z > cup_rim_z
        return False

    def check_ball_in_cup(self):
        """Check if ball is currently touching the sensor."""
        if self.ball_geom is None or self.sensor_geom is None:
            return False
        for i in range(self.d.ncon):
            contact = self.d.contact[i]
            if (contact.geom1 == self.ball_geom and contact.geom2 == self.sensor_geom) or \
               (contact.geom2 == self.ball_geom and contact.geom1 == self.sensor_geom):
                return True
        return False

    def log_ball(self, now):
        if self.ball_log is None or self.ball_log.closed or self.ball_body_id is None:
            return

        ball_pos = self.d.xpos[self.ball_body_id]
        ball_x, ball_y, ball_z = map(float, ball_pos)

        above_cup = self.is_ball_above_cup(self.d)
        above_flag = 1 if above_cup else 0

        if self.cup_center_site is not None:
            cup_center = self.d.site_xpos[self.cup_center_site]
            cup_x, cup_y, cup_z = map(float, cup_center)
        else:
            cup_x, cup_y, cup_z = 0.0, 0.0, 0.0

        # Ball in cup (boolean)
        ball_in_cup_flag = 1 if self.check_ball_in_cup() else 0

        # Cup collision (boolean) - with highlighting if viewer is on
        cup_col, _ = check_cup_collision(self.m, self.d, self.robot_geoms, self.cup_geoms, 
                                         highlight=(self.show_viewer and not self.collision_detected))
        if cup_col and not self.collision_detected:
            self.collision_detected = True
        cup_collision_flag = 1 if cup_col else 0

        self.ball_log.write(
            f"{now:.6f} {ball_x:.6f} {ball_y:.6f} {ball_z:.6f} {above_flag} "
            f"{cup_x:.6f} {cup_y:.6f} {cup_z:.6f} {ball_in_cup_flag} {cup_collision_flag}\n"
        )

    def update_dmp(self, now):
        if self._dmp is None:
            return False
        if self._last_dmp_time is None:
            self._last_dmp_time = now
        finished = False
        while (now - self._last_dmp_time) >= self._dmp.dt:
            q_raw = self._dmp.next()
            if q_raw is None:
                finished = True
                break
            self._last_dmp_time += self._dmp.dt

            self._pending_q = q_raw
        if self._pending_q is None:
            return finished
        q = self._pending_q.copy()
        if self._last_q_cmd is not None:
            dq = np.abs(q - self._last_q_cmd)
            if float(np.max(dq)) < MIN_JOINT_STEP_RAD:
                return finished
        joint_targets = {}
        for i, joint_name in enumerate(IK_JOINTS):
            joint_targets[joint_name] = float(q[i])
        for joint_name in JOINTS:
            if joint_name not in joint_targets:
                joint_targets[joint_name] = STARTUP_JOINT_POSITIONS.get(joint_name, 0.0)
        for joint_name, target_pos in joint_targets.items():
            if joint_name in self.joint_to_ctrl_idx:
                ctrl_idx = self.joint_to_ctrl_idx[joint_name]
                self.d.ctrl[ctrl_idx] = float(target_pos)
        self._last_q_cmd = q.copy()
        return finished

    def update_end_movement(self, now):
        elapsed = now - self.end_move_start_time
        total_time = END_POSITION_MOVE_TIME + END_POSITION_HOLD_TIME
        if elapsed >= total_time:
            return True
        for joint_name, end_pos in END_JOINT_POSITIONS.items():
            if joint_name in self.joint_to_ctrl_idx:
                if elapsed < END_POSITION_MOVE_TIME:
                    qpos_idx = self.joint_to_qpos_idx.get(joint_name)
                    if qpos_idx is None:
                        continue
                    t = elapsed / END_POSITION_MOVE_TIME
                    t_smooth = 0.5 - 0.5 * np.cos(t * np.pi)
                    current_pos = self.d.qpos[qpos_idx]
                    target = current_pos + t_smooth * (end_pos - current_pos)
                    self.d.ctrl[self.joint_to_ctrl_idx[joint_name]] = float(target)
                else:
                    self.d.ctrl[self.joint_to_ctrl_idx[joint_name]] = float(end_pos)
        return False

    def update(self, now):
        if self.start_time is None:
            self.start_time = now
        if self.state == "WAITING":
            elapsed = now - self.start_time
            if elapsed >= STARTUP_WAIT:
                self.state = "DMP_PLAYING"
                self.dmp_start_time = now
        elif self.state == "DMP_PLAYING":
            finished = self.update_dmp(now)
            if finished:
                self.state = "MOVING_TO_END"
                self.end_move_start_time = now
        elif self.state == "MOVING_TO_END":
            done = self.update_end_movement(now)
            if done:
                self.state = "DONE"
                self.close_logs()
                return True
        elif self.state == "DONE":
            return True
        self.log_ball(now)
        return False

    def close_logs(self):
        if self.ball_log and not self.ball_log.closed:
            self.ball_log.close()

    def is_done(self):
        return self.state == "DONE"

# ============================================================
# REWARD
# ============================================================

def compute_trajectory_reward(ball_data):
    """
    Compute reward from ball_data only.
    Columns: t  ball_x  ball_y  ball_z  above_cup  cup_x  cup_y  cup_z  ball_in_cup  cup_collision
    """
    if len(ball_data) == 0:
        return {
            'total_reward': 0.0,
            'ball_in_cup': False,
            'min_distance_to_cup': float('inf'),
            'collision_reward': 0.0,
            'distance_reward': 0.0,
            'height_bonus': 0.0,
            'cup_collision': False,
        }

    ball_positions  = ball_data[:, 1:4]
    above_cup_flags = ball_data[:, 4]
    cup_centers     = ball_data[:, 5:8]
    ball_in_cup_col = ball_data[:, 8]
    cup_col_col     = ball_data[:, 9]

    cup_center   = np.mean(cup_centers, axis=0)
    distances_3d = np.linalg.norm(ball_positions - cup_center, axis=1)

    # Calculate reward at EVERY timestep
    MAX_DISTANCE_FOR_REWARD = 0.75
    distance_rewards = []
    height_bonuses   = []

    for i in range(len(distances_3d)):
        dist  = distances_3d[i]
        above = above_cup_flags[i] > 0.5
        dist_rew   = 750.0 * (1.0 - dist / MAX_DISTANCE_FOR_REWARD) if dist <= MAX_DISTANCE_FOR_REWARD else 0.0
        height_bon = 200.0 if above else 0.0
        distance_rewards.append(dist_rew)
        height_bonuses.append(height_bon)

    # Best moment (highest combined distance + height reward)
    combined = np.array(distance_rewards) + np.array(height_bonuses)
    best_idx = np.argmax(combined)

    distance_reward  = distance_rewards[best_idx]
    height_bonus     = height_bonuses[best_idx]
    best_distance    = distances_3d[best_idx]
    ball_above_best  = above_cup_flags[best_idx] > 0.5

    # Ball in cup: any timestep where ball_in_cup == 1
    ball_in_cup = bool(np.any(ball_in_cup_col > 0.5))
    collision_reward = 1000.0 if ball_in_cup else 0.0

    # Cup collision: any timestep where cup_collision == 1
    cup_collision = bool(np.any(cup_col_col > 0.5))

    # If cup collision → overwrite total reward to -1
    if cup_collision:
        total_reward = -1.0
    else:
        total_reward = distance_reward + height_bonus + collision_reward

    return {
        'total_reward':          float(total_reward),
        'ball_in_cup':           bool(ball_in_cup),
        'collision_reward':      float(collision_reward),
        'distance_reward':       float(distance_reward),
        'height_bonus':          float(height_bonus),
        'min_distance_to_cup':   float(best_distance),
        'ball_went_above_cup':   bool(ball_above_best),
        'cup_center':            cup_center.tolist(),
        'cup_collision':         bool(cup_collision),
    }

def plot_rewards_progress(iterations, rewards, save_path=None):
    if iterations is None or rewards is None:
        return
    if len(iterations) == 0 or len(rewards) == 0:
        return
    if len(iterations) != len(rewards):
        raise ValueError("iterations and rewards must have the same length")

    it_arr      = np.asarray(iterations, dtype=int)
    rewards_arr = np.asarray(rewards, dtype=float)

    sort_idx       = np.argsort(it_arr)
    it_sorted      = it_arr[sort_idx]
    rewards_sorted = rewards_arr[sort_idx]

    plt.figure(figsize=(12, 6))
    # plt.plot(it_sorted, rewards_sorted, linewidth=1, label="Reward", color = "blue")
    plt.plot(it_sorted, rewards_sorted, 'b-', linewidth=2, marker='o', markersize=5)
    best_sorted_idx = np.argmax(rewards_sorted)
    best_iter = it_sorted[best_sorted_idx]
    best_val  = rewards_sorted[best_sorted_idx]

    plt.axvline(best_iter, linestyle="--", color="green", linewidth=1)
    plt.axhline(best_val, linestyle="--", color="green", linewidth=1)
    plt.scatter(best_iter, best_val, marker="o", zorder=5, label="Best", color = "green")

    plt.annotate(
        f"Iteration: {best_iter}\nReward: {best_val:.2f}",
        xy=(best_iter, best_val),
        xytext=(-10, -50),  # offset from point
        textcoords="offset points",
        fontsize=11,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="green"),
    )

    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.xlabel("Iteration", fontsize=12)
    plt.ylabel("Total Reward", fontsize=12)
    plt.title("Optimization Progress", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=11)
    plt.ylim(0, 2000)
    plt.xlim(0, MAX_ITERS)

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=500)

    plt.close()

def evaluate_weights_in_mujoco(model, data, baseline, weights, iteration, output_dir, robot_geoms, cup_geoms, original_colors):
    """Evaluate DMP weights in MuJoCo and return reward."""
    iter_dir = output_dir / f"iter_{iteration:03d}"
    iter_dir.mkdir(parents=True, exist_ok=True)

    timestamp  = datetime.now().strftime("%Y%m%d_%H%M%S")
    identifier = f"{timestamp}_iter{iteration}"
    ball_file  = iter_dir / f"ball_{identifier}.txt"

    # Reset simulation
    mujoco.mj_resetData(model, data)
    
    # Restore colors at start
    restore_colors(model, original_colors)

    # Create DMP source
    dmp_src = DMPJointRolloutSource(baseline, weights, CYCLE_SECONDS)

    # Run simulation
    controller = SingleDMPRun(model, data, dmp_src, ball_file, robot_geoms, cup_geoms, SHOW_VIEWER)

    last_update_time = 0.0

    if SHOW_VIEWER:
        with mujoco.viewer.launch_passive(model, data) as viewer:
            viewer.cam.lookat[:] = [0.045, 0.293, 1.4]
            viewer.cam.distance = 4.0
            viewer.cam.elevation = 0
            viewer.cam.azimuth = 0
            viewer.opt.geomgroup[:] = 1
            viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = 0

            while viewer.is_running() and not controller.is_done():
                sim_time = data.time
                if sim_time - last_update_time >= CYCLE_SECONDS:
                    last_update_time = sim_time
                    done = controller.update(sim_time)
                    if done:
                        break
                mujoco.mj_step(model, data)
                viewer.sync()
    else:
        while not controller.is_done():
            sim_time = data.time
            if sim_time - last_update_time >= CYCLE_SECONDS:
                last_update_time = sim_time
                done = controller.update(sim_time)
                if done:
                    break
            mujoco.mj_step(model, data)

    # Load ball trajectory and compute reward (everything from ball.txt now)
    try:
        ball_data = np.loadtxt(ball_file, comments='#')
        if len(ball_data.shape) == 1:
            ball_data = ball_data.reshape(1, -1)
    except:
        ball_data = np.array([])

    reward_data = compute_trajectory_reward(ball_data)
    reward_data['iteration'] = iteration

    # Save reward details
    reward_file = iter_dir / "reward_details.txt"
    with open(reward_file, 'w') as f:
        f.write(f"ITERATION {iteration} REWARD BREAKDOWN\n")
        f.write("="*50 + "\n\n")
        f.write(f"Total Reward:          {reward_data['total_reward']:.2f} pts\n\n")
        f.write("COMPONENTS:\n")
        f.write(f"  Distance Reward:     {reward_data['distance_reward']:.2f} pts\n")
        f.write(f"  Height Bonus:        {reward_data['height_bonus']:.2f} pts\n")
        f.write(f"  Collision Reward:    {reward_data['collision_reward']:.2f} pts\n\n")
        f.write("DETAILS:\n")
        f.write(f"  Best Distance:       {reward_data['min_distance_to_cup']:.4f} m\n")
        f.write(f"  Ball Above Rim:      {'Yes' if reward_data['ball_went_above_cup'] else 'No'}\n")
        f.write(f"  Ball In Cup:         {'Yes' if reward_data['ball_in_cup'] else 'No'}\n")
        f.write(f"  Cup Collision:       {'Yes ⚠️' if reward_data['cup_collision'] else 'No'}\n")
        f.write(f"  Cup Center:          [{reward_data['cup_center'][0]:.3f}, {reward_data['cup_center'][1]:.3f}, {reward_data['cup_center'][2]:.3f}]\n")

    return reward_data

# ============================================================
# MAIN INTEGRATED LOOP
# ============================================================

def main():
    print("="*70)
    print("INTEGRATED ICL + DMP OPTIMIZATION LOOP")
    print("="*70)
    print(f"REWARD STRUCTURE:")
    print(f"  1. Distance reward (0-750 pts) - closer to cup = more points")
    print(f"  2. Height bonus (+200 pts) - if ball above rim at best moment")
    print(f"  3. Collision jackpot (+1000 pts) - ball lands in cup")
    print(f"  4. Cup collision (cup hits arm/stand) → total reward overwritten to -1")
    if SHOW_VIEWER:
        print(f"  → Colliding parts will turn RED")
    print("="*70)

    print(f"\n📂 Loading data...")
    baseline = load_baseline(BASELINE_FILE)
    episodes, cup_center, cup_radius = load_episodes_simple(EPISODES_FILE)
    print(f"✓ Baseline: M={baseline['M']} basis functions")
    print(f"✓ Episodes: {len(episodes)} loaded")

    print(f"\n🎮 Loading MuJoCo model...")
    model_mj = mujoco.MjModel.from_xml_path(MUJOCO_XML)
    data_mj  = mujoco.MjData(model_mj)
    print("✓ MuJoCo loaded!")

    # Build robot and cup geoms ONCE
    robot_geoms, cup_geoms = build_robot_geoms(model_mj)
    print(f"✓ Robot geoms: {len(robot_geoms)} found")
    print(f"✓ Cup geoms: {len(cup_geoms)} found")
    
    # Store original colors
    original_colors = store_original_colors(model_mj, robot_geoms)
    print(f"✓ Original colors stored\n")

    conversation_history = []
    optimization_results = []

    print("\n" + "="*60 + "STARTING OPTIMIZATION PROCESS" + "="*60 + "\n")

    history = ""

    bar_format = (
        "{desc} | {percentage:3.0f}%|{bar}| "
        "{n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}"
    )

    reward_iters  = []
    reward_values = []

    with tqdm(total=MAX_ITERS, desc="Optimization", unit="iter", bar_format=bar_format) as pbar:
        pbar.set_postfix(last="0.0", best="0.0")

        for iteration in range(1, MAX_ITERS + 1):

            prompt  = build_prompt(baseline, cup_center, iteration, MAX_ITERS, episodes, LOGS_DIR, history, n_show=20)
            # print(prompt)
            response_text = query_gemini(prompt)
            iter_dir = ICL_OUTPUT_DIR / f"iter_{iteration:03d}"
            iter_dir.mkdir(parents=True, exist_ok=True)

            with open(iter_dir / "prompt.txt", "w", encoding="utf-8") as f:
                f.write(prompt)
            with open(iter_dir / "response.txt", "w", encoding="utf-8") as f:
                f.write(response_text)

            expected_size    = 3 * baseline['M']
            weights, reasoning = parse_response(response_text, expected_size)

            weights_file = iter_dir / f"weights_iter{iteration:03d}.npz"
            np.savez_compressed(
                weights_file,
                w=weights,
                y0=baseline['y0_star'],
                g=baseline['g_star'],
                iteration=iteration,
                reasoning=reasoning
            )

            reward_data = evaluate_weights_in_mujoco(
                model_mj, data_mj, baseline, weights, iteration, ICL_OUTPUT_DIR, robot_geoms, cup_geoms, original_colors
            )

            history = iteration_history(iteration, weights, reward_data, history)

            optimization_results.append({
                'iteration':    iteration,
                'weights_file': str(weights_file),
                'reasoning':    reasoning,
                'reward':       reward_data
            })

            conversation_history.append({
                'user':      prompt,
                'assistant': response_text
            })

            last_reward  = f"{reward_data['total_reward']:.1f}"
            best_reward  = f"{max(r['reward']['total_reward'] for r in optimization_results):.1f}"
            cup_col_tag = " 💥" if reward_data['cup_collision'] else ""

            reward_iters.append(iteration)
            reward_values.append(reward_data['total_reward'])

            plot_rewards_progress(
                reward_iters,
                reward_values,
                save_path=ICL_OUTPUT_DIR / "reward_curve" / f"reward_curve_iter_{iteration:03d}.pdf"
                )

            pbar.set_postfix(last=f"{last_reward}{cup_col_tag}", best=f"{best_reward}")
            pbar.update(1)

    print("\n" + "="*60 + "OPTIMIZATION PROCESS COMPLETE" + "="*60 + "\n")

    # Restore colors at the end
    restore_colors(model_mj, original_colors)

    if not optimization_results:
        print("\n❌ No successful iterations completed!")
        return

    best_result = max(optimization_results, key=lambda x: x['reward']['total_reward'])

    print(f"\nBest iteration: {best_result['iteration']}")
    print(f"Best reward:    {best_result['reward']['total_reward']:.1f}")
    print(f"Weights file:   {best_result['weights_file']}")

    results_file = ICL_OUTPUT_DIR / "optimization_results.json"
    with open(results_file, 'w') as f:
        json.dump({
            'model': "gemini-3-flash-preview",
            'max_iterations': MAX_ITERS,
            'reward_structure': {
                'distance_reward':       '0-750 pts (proximity to cup)',
                'height_bonus':          '+200 pts (ball above rim at best moment)',
                'collision_reward':      '+1000 pts (ball in cup)',
                'cup_collision':         'cup hits arm/stand → total reward overwritten to -1',
            },
            'results':        optimization_results,
            'best_iteration': best_result['iteration'],
            'best_reward':    best_result['reward']['total_reward']
        }, f, indent=2)

    print(f"\n💾 Results saved to: {results_file}")
    print(f"💾 All files in:     {ICL_OUTPUT_DIR}")
    print("="*70)

if __name__ == "__main__":
    main()