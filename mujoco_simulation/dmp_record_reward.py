#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import time
import math
import numpy as np
import mujoco
import mujoco.viewer
import csv
import json
from pathlib import Path
from glob import glob

# ============================ CONFIG ============================

SHOW_VIEWER = True
MUJOCO_XML = "bimanualrobot.xml"

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

# ============================ POSITIONS ============================
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

# ============================ TIMING ============================
CYCLE_SECONDS = 0.025
STARTUP_WAIT = 2.0
END_POSITION_MOVE_TIME = 1.0
END_POSITION_HOLD_TIME = 0.5
LPF_CUTOFF_HZ = 3.5
MIN_JOINT_STEP_RAD = np.deg2rad(0.1)

# ============================ DMP CONFIG ============================
def parse_args():
    parser = argparse.ArgumentParser(
        description="visualizing DMP rollouts and recording ball trajectories"
    )
    parser.add_argument("--log_dir", type=str, required=True, help="Path to icl_log directory")
    return parser.parse_args()

args = parse_args()
LOG_DIR = Path(args.log_dir)
BASELINE_DIR = Path("logs")
BASELINE_PATH = str(BASELINE_DIR / "baseline.npz")
WEIGHTS_DIR = LOG_DIR

DMP_DT = CYCLE_SECONDS
TAU_OVERRIDE = None
USE_WEIGHTED_OFFSET_GOAL = True

BALL_OUTPUT_DIR = LOG_DIR / "logs"
BALL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================ CUP COLLISION ============================

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

# ============================ REWARD COMPUTATION ============================

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
            'max_height': 0.0,
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

    combined = np.array(distance_rewards) + np.array(height_bonuses)
    best_idx = np.argmax(combined)

    distance_reward = distance_rewards[best_idx]
    height_bonus    = height_bonuses[best_idx]
    best_distance   = distances_3d[best_idx]
    ball_above_best = above_cup_flags[best_idx] > 0.5

    # Ball in cup: any timestep where ball_in_cup == 1
    ball_in_cup      = bool(np.any(ball_in_cup_col > 0.5))
    collision_reward = 1000.0 if ball_in_cup else 0.0

    # Cup collision: any timestep where cup_collision == 1
    cup_collision = bool(np.any(cup_col_col > 0.5))

    # Cup collision → overwrite total reward to -1
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
        'max_height':            float(np.max(ball_positions[:, 2])),
        'ball_went_above_cup':   bool(ball_above_best),
        'cup_center':            cup_center.tolist(),
        'cup_collision':         bool(cup_collision),
    }

# ============================ DMP ROLLOUT ============================

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
    def __init__(self, baseline_path, weights_path, dt, tau_override=None, use_weighted_offset_goal=True):
        b = np.load(baseline_path)
        w = np.load(weights_path, allow_pickle=True)

        c, h = b["c"], b["h"]
        K, D = float(b["K"]), float(b["D"])
        alpha_s = float(b["alpha_s"])

        tau_saved = float(b["tau"])
        tau = float(tau_override) if tau_override is not None else tau_saved

        W    = w["w"]
        y0_w = w["y0"].astype(float)
        g_w  = w["g"].astype(float)

        y0 = y0_w.copy()
        g  = y0 + (g_w - y0_w) if use_weighted_offset_goal else g_w.copy()

        Y = rollout_dmp_3d(y0, g, W, K, D, tau, dt, c, h, alpha_s)

        self.Y  = Y
        self.dt = float(dt)
        self.N  = len(Y)
        self.k  = 0

    def next(self):
        if self.k >= self.N:
            return None
        Yk = self.Y[self.k].copy()
        self.k += 1
        return Yk

# ============================ SINGLE RUN CONTROLLER ============================

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

        # Reset ball
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
                self.d.qpos[qpos_adr:qpos_adr+3]   = [0.80, 0.55, 1.25]
                self.d.qpos[qpos_adr+3:qpos_adr+7] = [1.0, 0.0, 0.0, 0.0]
                self.d.qvel[qvel_adr:qvel_adr+6]   = 0.0
                mujoco.mj_forward(self.m, self.d)
        except:
            pass

        # Find ball body and cup center
        try:
            self.ball_body_id = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, "left_cup_ball")
            print(f"    ✓ Found ball body (ID: {self.ball_body_id})")
        except:
            self.ball_body_id = None

        try:
            self.cup_center_site = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_SITE, "left_cup_center")
            print(f"    ✓ Found cup center site (ID: {self.cup_center_site})")
        except:
            self.cup_center_site = None

        # Ball in cup detection geoms
        try:
            self.ball_geom   = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_GEOM, "left_cup_ball_geom")
            self.sensor_geom = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_GEOM, "left_cup_sensor_geom")
        except:
            self.ball_geom   = None
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

        above_cup  = self.is_ball_above_cup(self.d)
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
                self.d.ctrl[self.joint_to_ctrl_idx[joint_name]] = float(target_pos)
        self._last_q_cmd = q.copy()
        return finished

    def update_end_movement(self, now):
        elapsed    = now - self.end_move_start_time
        total_time = END_POSITION_MOVE_TIME + END_POSITION_HOLD_TIME
        if elapsed >= total_time:
            return True
        for joint_name, end_pos in END_JOINT_POSITIONS.items():
            if joint_name in self.joint_to_ctrl_idx:
                if elapsed < END_POSITION_MOVE_TIME:
                    qpos_idx = self.joint_to_qpos_idx.get(joint_name)
                    if qpos_idx is None:
                        continue
                    t        = elapsed / END_POSITION_MOVE_TIME
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

# ============================ BATCH PROCESSOR ============================

def run_single_weight(model, data, weights_path, baseline_path, ball_output_file, robot_geoms, cup_geoms, original_colors):
    """Run one DMP weight file and record ball trajectory."""
    mujoco.mj_resetData(model, data)
    
    # Restore colors at start
    restore_colors(model, original_colors)

    dmp_src = DMPJointRolloutSource(
        baseline_path=baseline_path,
        weights_path=str(weights_path),
        dt=DMP_DT,
        tau_override=TAU_OVERRIDE,
        use_weighted_offset_goal=USE_WEIGHTED_OFFSET_GOAL,
    )

    controller = SingleDMPRun(model, data, dmp_src, ball_output_file, robot_geoms, cup_geoms, SHOW_VIEWER)

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

def compute_episode_reward(ball_output_file):
    """Compute reward from ball.txt only."""
    try:
        ball_data = np.loadtxt(ball_output_file, comments='#')
        if len(ball_data.shape) == 1:
            ball_data = ball_data.reshape(1, -1)
        if len(ball_data) == 0 or ball_data.shape[1] < 10:
            print(f"      ✗ Invalid ball data (need 10 columns)")
            return None
    except Exception as e:
        print(f"      ✗ Error loading ball data: {e}")
        return None

    reward_data = compute_trajectory_reward(ball_data)

    status = "🎯" if reward_data['ball_in_cup'] else ("⬆" if reward_data['ball_went_above_cup'] else "○")
    cup_col_tag = " 💥" if reward_data['cup_collision'] else ""

    print(f"      {status} Reward: {reward_data['total_reward']:.1f}{cup_col_tag}")
    print(f"         - Ball in cup:      {reward_data['ball_in_cup']} (+{reward_data['collision_reward']:.0f})")
    print(f"         - Min distance:     {reward_data['min_distance_to_cup']:.3f}m")
    print(f"         - Distance reward:  {reward_data['distance_reward']:.1f}")
    print(f"         - Above cup:        {reward_data['ball_went_above_cup']} (+{reward_data['height_bonus']:.0f})")
    print(f"         - Max height:       {reward_data['max_height']:.3f}m")
    print(f"         - Cup collision:    {'Yes ⚠️' if reward_data['cup_collision'] else 'No'}")

    return reward_data

def main():
    print("="*70)
    print("BATCH DMP PLAYBACK WITH IMMEDIATE REWARD COMPUTATION")
    print("="*70)
    print(f"REWARD STRUCTURE:")
    print(f"  1. Distance reward (0-750 pts) - closer to cup = more points")
    print(f"  2. Height bonus (+200 pts) - if ball goes above cup rim")
    print(f"  3. Collision jackpot (+1000 pts) - ball lands in cup")
    print(f"  4. Cup collision (cup hits arm/stand) → total reward overwritten to -1")
    if SHOW_VIEWER:
        print(f"  → Colliding parts will turn RED")
    print("="*70)

    weight_pattern = str(WEIGHTS_DIR / "*weights*.npz")
    weight_files   = sorted(glob(weight_pattern))

    if not weight_files:
        print(f"ERROR: No weight files found matching: {weight_pattern}")
        return

    print(f"\nFound {len(weight_files)} weight files")
    print(f"Baseline: {BASELINE_PATH}")
    print(f"Output directory: {BALL_OUTPUT_DIR}")
    print()

    print("Loading MuJoCo model...")
    model = mujoco.MjModel.from_xml_path(MUJOCO_XML)
    data  = mujoco.MjData(model)
    print("✓ Model loaded")

    # Build robot and cup geoms ONCE
    robot_geoms, cup_geoms = build_robot_geoms(model)
    print(f"✓ Robot geoms: {len(robot_geoms)} found")
    print(f"✓ Cup geoms: {len(cup_geoms)} found")
    
    # Store original colors
    original_colors = store_original_colors(model, robot_geoms)
    print(f"✓ Original colors stored\n")
    episodes = []

    for i, weight_path in enumerate(weight_files, 1):
        weight_file = Path(weight_path)
        weight_name = weight_file.stem.replace("wrist_", "").replace("_weights", "")

        ball_output_file = BALL_OUTPUT_DIR / f"ball_{weight_name}.txt"

        print(f"[{i}/{len(weight_files)}] Processing: {weight_file.name}")
        print(f"    Ball output: {ball_output_file.name}")

        start_time = time.time()
        run_single_weight(model, data, weight_path, BASELINE_PATH, ball_output_file, robot_geoms, cup_geoms, original_colors)
        elapsed = time.time() - start_time

        print(f"    ✓ Simulation complete in {elapsed:.1f}s")
        print(f"    Computing reward...")

        reward_data = compute_episode_reward(ball_output_file)

        if reward_data is not None:
            episode = {
                'episode_id':      weight_name,
                'weight_file':     str(weight_path),
                'ball_file':       str(ball_output_file),
                'simulation_time': float(elapsed),
                **reward_data
            }
            episodes.append(episode)

        print()

    # Restore colors at the end
    restore_colors(model, original_colors)

    if not episodes:
        print("No valid episodes processed!")
        return

    episodes.sort(key=lambda x: x['total_reward'], reverse=True)

    output_file = BALL_OUTPUT_DIR.parent / "processed_episodes.json"
    with open(output_file, 'w') as f:
        json.dump({
            'note': 'Cup collision → reward = -1. Otherwise: distance + height_bonus + collision',
            'reward_structure': {
                'distance_reward':  '0-750 pts (proximity to cup)',
                'height_bonus':     '+200 pts (if above cup at best moment)',
                'collision_reward': '+1000 pts (ball in cup)',
                'cup_collision':    'cup hits arm/stand → total reward overwritten to -1',
            },
            'episodes': episodes
        }, f, indent=2)

    # Summary
    print("="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Total episodes processed: {len(episodes)}")

    num_in_cup    = sum(1 for ep in episodes if ep['ball_in_cup'])
    num_above_cup = sum(1 for ep in episodes if ep['ball_went_above_cup'])
    num_cup_col   = sum(1 for ep in episodes if ep['cup_collision'])
    avg_reward    = np.mean([ep['total_reward'] for ep in episodes])
    avg_distance  = np.mean([ep['min_distance_to_cup'] for ep in episodes])
    avg_height    = np.mean([ep['max_height'] for ep in episodes])

    print(f"\nBall in cup:       {num_in_cup}/{len(episodes)} ({100*num_in_cup/len(episodes):.1f}%)")
    print(f"Ball above cup:    {num_above_cup}/{len(episodes)} ({100*num_above_cup/len(episodes):.1f}%)")
    print(f"Cup collisions:    {num_cup_col}/{len(episodes)} ({100*num_cup_col/len(episodes):.1f}%)")
    print(f"Average reward:    {avg_reward:.1f}")
    print(f"Average distance:  {avg_distance:.3f}m ({avg_distance*100:.1f}cm)")
    print(f"Average height:    {avg_height:.3f}m")

    print(f"\nTop 5 episodes:")
    for i, ep in enumerate(episodes[:min(5, len(episodes))]):
        status  = "🎯" if ep['ball_in_cup'] else ("⬆" if ep['ball_went_above_cup'] else "○")
        cup_col = " 💥" if ep['cup_collision'] else ""
        print(f"  {i+1}. {status} {ep['episode_id']}: reward={ep['total_reward']:.1f}{cup_col}")
        print(f"     dist={ep['min_distance_to_cup']:.3f}m, height={ep['max_height']:.3f}m")

    if len(episodes) > 5:
        print(f"\nBottom 3 episodes:")
        for i, ep in enumerate(episodes[-3:]):
            idx     = len(episodes) - 3 + i + 1
            cup_col = " 💥" if ep['cup_collision'] else ""
            print(f"  {idx}. ○ {ep['episode_id']}: reward={ep['total_reward']:.1f}{cup_col}, "
                  f"dist={ep['min_distance_to_cup']:.3f}m")

    print(f"\n{'='*70}")
    print("REWARD BREAKDOWN (averages)")
    print(f"{'='*70}")
    avg_collision_reward = np.mean([ep['collision_reward'] for ep in episodes])
    avg_distance_reward  = np.mean([ep['distance_reward'] for ep in episodes])
    avg_height_bonus     = np.mean([ep['height_bonus'] for ep in episodes])

    print(f"Collision reward:   {avg_collision_reward:6.1f} points")
    print(f"Distance reward:    {avg_distance_reward:6.1f} points")
    print(f"Height bonus:       {avg_height_bonus:6.1f} points")
    print(f"{'─'*40}")
    print(f"Total:              {avg_collision_reward + avg_distance_reward + avg_height_bonus:6.1f} points")

    print(f"\n💾 Results saved to: {output_file}")
    print(f"💾 Ball trajectories: {BALL_OUTPUT_DIR}")
    print("="*70)

if __name__ == "__main__":
    main()