#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
import struct
import socket
import threading
import math
import numpy as np
import mujoco
import mujoco.viewer
import csv
from datetime import datetime

import pinocchio as pin
from pinocchio.robot_wrapper import RobotWrapper
from scipy.optimize import least_squares

# ============================ CONFIG ============================

# URDF & IK target frame
URDF_PATH = "bimanualrobot.urdf"
F_WRIST = "leftarm_wrist_2_link"  # IK target frame

# MuJoCo model
MUJOCO_XML = "bimanualrobot.xml"

# IK DOFs
IK_JOINTS = [
    "leftarm_shoulder_pan_joint",
    "leftarm_shoulder_lift_joint",
    "leftarm_elbow_joint",
]

# All controlled joints (must match MuJoCo actuator order)
JOINTS = [
    "leftarm_shoulder_pan_joint",
    "leftarm_shoulder_lift_joint",
    "leftarm_elbow_joint",
    "leftarm_wrist_1_joint",
    "leftarm_wrist_2_joint",
    "leftarm_wrist_3_joint",
]

# Wearable → robot mapping
SHOULDER_ANCHOR = np.array([0.045, 0.2925, 1.526], dtype=float)
L1 = 0.298511306318538     # shoulder→elbow
L2 = 0.23293990641364998   # elbow→wrist_2

# UDP
UDP_PORT = 50003
PACK_FMT = "ffff fff ffff fff ffff fff ffff"  # 25 floats

# Loop & timing
CYCLE_SECONDS = 0.025   # ~40 Hz
UPSAMPLE_FACTOR = 2     # mid + current

# Smoothing / guards
LPF_CUTOFF_HZ = 3.5
SPIKE_MAX_SPEED = 2.0       # m/s
MIN_JOINT_STEP_RAD = np.deg2rad(0.1)

# IK solver params
W_POS = 1.0
W_REG = 1e-3
MAX_ITERS = 300
XTOL = FTOL = GTOL = 1e-8
VERBOSE = 0

# ============================ UDP LISTENER ============================

# Shared UDP state
_udp_lock = threading.Lock()
_hand_pos = None
_larm_pos = None
_uarm_pos = None

def udp_listener(port=UDP_PORT):
    """Receive wearable packet and keep the latest sample (thread)."""
    global _hand_pos, _larm_pos, _uarm_pos
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("0.0.0.0", port))
    print(f"[UDP] Listening on udp://0.0.0.0:{port}")
    unpack = struct.Struct(PACK_FMT).unpack_from
    while True:
        try:
            data, _ = sock.recvfrom(1024)
            if len(data) < 100:
                continue
            (hw, hx, hy, hz,
             hpx, hpy, hpz,
             lw, lx, ly, lz,
             lpx, lpy, lpz,
             uw, ux, uy, uz,
             upx, upy, upz,
             qw, qx, qy, qz) = unpack(data)
            with _udp_lock:
                _hand_pos = (hpx, hpy, hpz)
                _larm_pos = (lpx, lpy, lpz)
                _uarm_pos = (upx, upy, upz)
        except Exception as e:
            print(f"[UDP ERROR] {e}")

# ============================ COLLISION TRACKER ============================

class CollisionTracker:
    """Tracks ball-in-cup collisions and records events to CSV"""
    
    def __init__(self, model, data, enabled=True):
        self.model = model
        self.data = data
        self.enabled = enabled
        
        if not self.enabled:
            return
        
        # Color palette for lid color changes
        self.colors = [
            [0, 1, 0, 0.3],    # Green
            [1, 0, 0, 0.3],    # Red
            [0, 0, 1, 0.3],    # Blue
            [1, 1, 0, 0.3],    # Yellow
            [1, 0, 1, 0.3],    # Magenta
            [0, 1, 1, 0.3],    # Cyan
            [1, 0.5, 0, 0.3],  # Orange
            [0.5, 0, 1, 0.3],  # Purple
        ]
        
        # State tracking
        self.color_idx = 0
        self.contact_count = 0
        self.prev_touching = False
        
        # Find geometries
        try:
            self.ball_geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "left_cup_ball_geom")
            print(f"[Collision] Found left_cup_ball_geom (ID: {self.ball_geom})")
        except:
            print("[Collision] ERROR: Could not find 'left_cup_ball_geom'")
            self.enabled = False
            return
        
        try:
            self.sensor_geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "left_cup_lid_geom")
            print(f"[Collision] Found left_cup_lid_geom (ID: {self.sensor_geom})")
        except:
            print("[Collision] ERROR: Could not find 'left_cup_lid_geom'")
            print("           Add the lid sensor to your XML!")
            self.enabled = False
            return
        
        # Find ball body for position
        try:
            self.ball_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "left_cup_ball")
            print(f"[Collision] Found left_cup_ball body (ID: {self.ball_body})")
        except:
            print("[Collision] WARNING: Could not find 'left_cup_ball' body")
            self.ball_body = None
        
        # Find lid site for color changing and position
        try:
            self.lid_site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "left_cup_sensor_site")
            print(f"[Collision] Found left_cup_sensor_site (ID: {self.lid_site})")
        except:
            print("[Collision] WARNING: Could not find 'left_cup_sensor_site'")
            self.lid_site = None
        
        # Create CSV file
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.csv_filename = f"ball_in_cup_events_{timestamp_str}.csv"
        
        with open(self.csv_filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['event_number', 'timestamp', 
                           'ball_x', 'ball_y', 'ball_z',
                           'lid_x', 'lid_y', 'lid_z'])
        
        print(f"[Collision] Recording events to {self.csv_filename}")
    
    def update_lid_color(self):
        """Update lid site color"""
        if self.lid_site is not None:
            self.model.site_rgba[self.lid_site] = self.colors[self.color_idx % len(self.colors)]
    
    def check_collision(self):
        """Check for ball-sensor collision and log event"""
        if not self.enabled:
            return
        
        # Check all contacts
        touching = False
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            geom1 = contact.geom1
            geom2 = contact.geom2
            
            # Check if ball and bottom sensor are colliding
            if (geom1 == self.ball_geom and geom2 == self.sensor_geom) or \
               (geom2 == self.ball_geom and geom1 == self.sensor_geom):
                touching = True
                break
        
        # Detect NEW collision (ball just entered cup!)
        if touching and not self.prev_touching:
            self.contact_count += 1
            self.color_idx += 1
            self.update_lid_color()
            
            # Get positions
            sim_time = self.data.time
            
            if self.ball_body is not None:
                ball_pos = self.data.xpos[self.ball_body].copy()
            else:
                ball_pos = np.array([0, 0, 0])
            
            if self.lid_site is not None:
                lid_pos = self.data.site_xpos[self.lid_site].copy()
            else:
                lid_pos = np.array([0, 0, 0])
            
            # Log to CSV
            with open(self.csv_filename, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    self.contact_count,
                    sim_time,
                    ball_pos[0], ball_pos[1], ball_pos[2],
                    lid_pos[0], lid_pos[1], lid_pos[2]
                ])
            
            color_rgb = self.colors[self.color_idx % len(self.colors)][:3]
            print(f"🎯 [BALL IN CUP #{self.contact_count}] t={sim_time:.2f}s | Color: RGB{np.round(color_rgb, 2)}")
            print(f"   Ball pos: [{ball_pos[0]:.4f}, {ball_pos[1]:.4f}, {ball_pos[2]:.4f}]")
            print(f"   Lid pos:  [{lid_pos[0]:.4f}, {lid_pos[1]:.4f}, {lid_pos[2]:.4f}]")
        
        self.prev_touching = touching
    
    def print_summary(self):
        """Print summary statistics"""
        if not self.enabled:
            return
        
        print("\n" + "="*60)
        print("COLLISION TRACKING SUMMARY")
        print("="*60)
        print(f"File: {self.csv_filename}")
        print(f"Total ball-in-cup events: {self.contact_count}")
        print("="*60 + "\n")

# ============================ MAPPING HELPERS ============================

def remap_watch_to_base(p):
    """(x,y,z)_watch -> (z, -x, y)_base"""
    x, y, z = map(float, p)
    return np.array([z, -x, y], dtype=float)

def unit(v):
    n = np.linalg.norm(v)
    return v / (n + 1e-12)

def scale_watch_to_left_robot(uarm_watch, larm_watch, hand_watch):
    """Map watch 3 pts (upper, lower, hand) to robot-base wrist (W)."""
    Sh = remap_watch_to_base(uarm_watch)
    El = remap_watch_to_base(larm_watch)
    Wr = remap_watch_to_base(hand_watch)

    # left-arm flip
    # Sh[1] = -Sh[1]
    # El[1] = -El[1]
    # Wr[1] = -Wr[1]

    u1 = unit(El - Sh)  # shoulder->elbow dir
    u2 = unit(Wr - El)  # elbow->wrist dir

    E_robot = SHOULDER_ANCHOR + L1 * u1
    W_robot = E_robot + L2 * u2
    return E_robot, W_robot

# ============================ LOW-PASS FILTER ============================

class LowPassEMA:
    def __init__(self, fc_hz=LPF_CUTOFF_HZ):
        self.fc = float(fc_hz)
        self.y = None
        self.t_last = None
    
    def update(self, x, t_now):
        x = np.asarray(x, float)
        if self.y is None or self.t_last is None:
            self.y = x.copy()
            self.t_last = float(t_now)
            return self.y
        dt = max(float(t_now - self.t_last), 1e-3)
        alpha = 1.0 - math.exp(-2.0 * math.pi * self.fc * dt)
        self.y = (1.0 - alpha) * self.y + alpha * x
        self.t_last = float(t_now)
        return self.y

# ============================ PINOCCHIO HELPERS ============================

def build_index_maps(model: pin.Model, joint_names):
    idx_q_vars = []
    for jn in joint_names:
        jid = model.getJointId(jn)
        if jid == 0:
            raise RuntimeError(f"Joint not found in model: {jn}")
        idx_q_vars.append(model.joints[jid].idx_q)

    lb_all = np.array(model.lowerPositionLimit, dtype=float)
    ub_all = np.array(model.upperPositionLimit, dtype=float)
    lb = lb_all[idx_q_vars].copy()
    ub = ub_all[idx_q_vars].copy()
    lb[~np.isfinite(lb)] = -1e9
    ub[~np.isfinite(ub)] = +1e9
    return np.array(idx_q_vars), lb, ub

def full_q_from_vars(model: pin.Model, idx_q_vars, q_vars):
    q = pin.neutral(model)
    for v, i in zip(q_vars, idx_q_vars):
        q[i] = float(v)
    return q

def residual_wrist_only(q_vars, model, data, fid_wrist, idx_q_vars, target_W,
                        w_pos=W_POS, w_reg=W_REG):
    q = full_q_from_vars(model, idx_q_vars, q_vars)
    pin.forwardKinematics(model, data, q)
    pin.updateFramePlacements(model, data)
    pW = data.oMf[fid_wrist].translation
    err_pos = (pW - target_W) * w_pos
    err_reg = w_reg * q_vars
    return np.hstack((err_pos, err_reg))

def solve_wrist_ik_least_squares(robot: RobotWrapper,
                                 fid_wrist: int,
                                 idx_q_vars,
                                 lb, ub,
                                 target_W,
                                 seed):
    model = robot.model
    data = model.createData()
    res = least_squares(
        residual_wrist_only,
        np.array(seed, float),
        bounds=(lb, ub),
        args=(model, data, fid_wrist, idx_q_vars, target_W, W_POS, W_REG),
        max_nfev=MAX_ITERS,
        xtol=XTOL, ftol=FTOL, gtol=GTOL,
        verbose=VERBOSE
    )
    q_vars_sol = res.x
    q_full = full_q_from_vars(model, idx_q_vars, q_vars_sol)
    pin.forwardKinematics(model, data, q_full)
    pin.updateFramePlacements(model, data)
    pW = data.oMf[fid_wrist].translation
    err = np.linalg.norm(pW - target_W)
    return res.success, q_vars_sol, q_full, pW, err

# ============================ MUJOCO CONTROLLER ============================

class MuJoCoTeleop:
    def __init__(self, xml_path=MUJOCO_XML):
        print("[MuJoCo] Loading model...")
        self.m = mujoco.MjModel.from_xml_path(xml_path)
        self.d = mujoco.MjData(self.m)
        
        # Build joint name to qpos index mapping
        self.joint_to_qpos_idx = {}
        self.joint_to_ctrl_idx = {}
        
        for i in range(self.m.njnt):
            joint_name = mujoco.mj_id2name(self.m, mujoco.mjtObj.mjOBJ_JOINT, i)
            qpos_addr = self.m.jnt_qposadr[i]
            self.joint_to_qpos_idx[joint_name] = qpos_addr
        
        for i in range(self.m.nu):
            actuator_name = mujoco.mj_id2name(self.m, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
            # Actuator names are like "leftarm_shoulder_pan_motor"
            # Joint names are like "leftarm_shoulder_pan_joint"
            joint_name = actuator_name.replace("_motor", "_joint")
            self.joint_to_ctrl_idx[joint_name] = i
        
        print(f"[MuJoCo] Found {self.m.njnt} joints, {self.m.nu} actuators")
        
        # Initialize collision tracker
        self.collision_tracker = CollisionTracker(self.m, self.d, enabled=True)
        
        # Pinocchio for IK
        print("[Pinocchio] Loading URDF...")
        self.robot = RobotWrapper.BuildFromURDF(URDF_PATH, [])
        self.pin_model = self.robot.model
        
        if not self.pin_model.existFrame(F_WRIST):
            raise RuntimeError(f"Frame not found: {F_WRIST}")
        self.fid_wrist = self.pin_model.getFrameId(F_WRIST)
        
        # IK setup
        self.idx_q_vars, self.lb, self.ub = build_index_maps(self.pin_model, IK_JOINTS)
        
        # State tracking
        self._w_lpf = LowPassEMA(fc_hz=LPF_CUTOFF_HZ)
        self._prev_W = None
        self._prev_W_t = None
        self._last_qvars = None
        self._last_q_cmd = None
        
        print("[MuJoCo] Teleop initialized!")
    
    def get_current_joint_positions(self):
        """Get current joint positions from MuJoCo."""
        positions = {}
        for joint_name, qpos_idx in self.joint_to_qpos_idx.items():
            positions[joint_name] = float(self.d.qpos[qpos_idx])
        return positions
    
    def set_joint_targets(self, joint_targets):
        """Set position targets for joints via MuJoCo actuators."""
        for joint_name, target_pos in joint_targets.items():
            if joint_name in self.joint_to_ctrl_idx:
                ctrl_idx = self.joint_to_ctrl_idx[joint_name]
                self.d.ctrl[ctrl_idx] = float(target_pos)
    
    def update(self, target_W, now):
        """
        Compute IK for target wrist position and update MuJoCo controls.
        Returns True if successful, False otherwise.
        """
        # Apply low-pass filter
        W = self._w_lpf.update(target_W, now)
        
        # Spike guard
        if self._prev_W is not None and self._prev_W_t is not None:
            dt = max(now - self._prev_W_t, 1e-3)
            if np.linalg.norm(W - self._prev_W) / dt > SPIKE_MAX_SPEED:
                return False
        
        # Get current joint positions from MuJoCo
        current_positions = self.get_current_joint_positions()
        js_now = [current_positions.get(jn, 0.0) for jn in JOINTS]
        
        # IK seed
        if self._last_qvars is not None:
            qvars_seed = self._last_qvars.copy()
        else:
            js_vec = np.array(js_now, dtype=float)
            qvars_seed = np.array(
                [js_vec[JOINTS.index(jn)] for jn in IK_JOINTS],
                dtype=float,
            )
        
        # Solve IK
        ok, qvars, qfull, pW, err = solve_wrist_ik_least_squares(
            self.robot, self.fid_wrist,
            self.idx_q_vars, self.lb, self.ub,
            W, seed=qvars_seed,
        )
        
        if not ok:
            self._prev_W = W.copy()
            self._prev_W_t = now
            return False
        
        # Build command for all joints
        q_cmd = list(js_now)
        for jn, val in zip(IK_JOINTS, qvars):
            if jn in JOINTS:
                q_cmd[JOINTS.index(jn)] = float(val)
        
        # Deadband check
        if self._last_q_cmd is not None:
            dq = np.abs(np.array(q_cmd) - np.array(self._last_q_cmd))
            if float(np.max(dq)) < MIN_JOINT_STEP_RAD:
                self._prev_W = W.copy()
                self._prev_W_t = now
                return False
        
        # Update MuJoCo controls
        joint_targets = dict(zip(JOINTS, q_cmd))
        self.set_joint_targets(joint_targets)
        
        # Remember state
        self._last_qvars = qvars
        self._last_q_cmd = q_cmd
        self._prev_W = W.copy()
        self._prev_W_t = now
        
        print(f"[IK] W_target={np.round(W, 3)} | err={err:.4f}m | qvars={np.round(np.degrees(qvars), 1)}°")
        return True

# ============================ MAIN ============================

def main():
    # Start UDP listener
    th = threading.Thread(target=udp_listener, daemon=True)
    th.start()
    
    print("Waiting for UDP data...")
    time.sleep(1.0)
    
    # Initialize teleop controller
    teleop = MuJoCoTeleop(MUJOCO_XML)
    
    # Initialize MuJoCo to mid-range positions
    print("[MuJoCo] Setting initial pose to mid-range...")
    for i in range(teleop.m.nu):
        ctrl_range = teleop.m.actuator_ctrlrange[i]
        mid_pos = (ctrl_range[0] + ctrl_range[1]) / 2.0
        teleop.d.ctrl[i] = mid_pos
    
    # Forward kinematics to update visualization
    mujoco.mj_forward(teleop.m, teleop.d)
    
    print("[MuJoCo] Launching viewer...")
    print("\nControls:")
    print("  - left-click + drag: Rotate view")
    print("  - Scroll: Zoom")
    print("  - Space: Pause/Resume simulation")
    print("  - Send UDP data on port", UDP_PORT, "to control the arm")
    print("  - Collision detection ACTIVE - lid changes color when ball enters cup!")
    print("\nTeleoperation active!\n")
    
    last_tick = time.time()
    
    try:
        with mujoco.viewer.launch_passive(teleop.m, teleop.d) as viewer:
            # Camera setup
            viewer.cam.lookat[:] = teleop.m.stat.center
            viewer.cam.distance = 3.0 * float(teleop.m.stat.extent)
            viewer.cam.elevation = -25
            viewer.cam.azimuth = 135
            
            # Visualization
            viewer.opt.geomgroup[:] = 1
            viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = 0
            
            while viewer.is_running():
                now = time.time()
                
                # Control loop timing
                if now - last_tick >= CYCLE_SECONDS:
                    last_tick = now
                    
                    # Read wearable data
                    with _udp_lock:
                        hp, lp, up = _hand_pos, _larm_pos, _uarm_pos
                    
                    if hp is not None and lp is not None and up is not None:
                        # Map to wrist target
                        _, W_raw = scale_watch_to_left_robot(up, lp, hp)
                        
                        # Update controller
                        teleop.update(W_raw, now)
                
                # Check for collisions every simulation step
                teleop.collision_tracker.check_collision()
                
                # Step simulation
                mujoco.mj_step(teleop.m, teleop.d)
                viewer.sync()
    
    finally:
        # Print summary when exiting
        teleop.collision_tracker.print_summary()
        print("\nTeleoperation stopped.")

if __name__ == "__main__":
    main()