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

# Ball reset configuration
BALL_RESET_POSITION = np.array([0.80, 0.55, 1.25], dtype=float)  # Initial ball position
BALL_RESET_VELOCITY = np.array([0.0, 0.0, 0.0], dtype=float)     # Zero velocity

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

# ============================ STARTUP POSITION CONFIG ============================
# Set the initial spawn joint angles here (radians)
STARTUP_JOINT_POSITIONS = {
    "leftarm_shoulder_pan_joint": -1.57,
    "leftarm_shoulder_lift_joint": 1.01,
    "leftarm_elbow_joint": -0.113,
    "leftarm_wrist_1_joint": 0.0,
    "leftarm_wrist_2_joint": 0.0,
    "leftarm_wrist_3_joint": 0.0,
}

# End position - robot moves here when stopping logging (similar to startup but different)
END_JOINT_POSITIONS = {
    "leftarm_shoulder_pan_joint": -1.6,     # Slightly different from -1.57
    "leftarm_shoulder_lift_joint": 0.85,     # Slightly different from 1.01
    "leftarm_elbow_joint": 0.0,              # More extended than -0.113
    "leftarm_wrist_1_joint": 0.0,
    "leftarm_wrist_2_joint": 0.0,
    "leftarm_wrist_3_joint": 0.0,
}

# Time to move to end position (seconds)
END_POSITION_MOVE_TIME = 1.0

# ============================ LOGGING & RESET CONFIG ============================
from pathlib import Path

LOG_DIR = Path("./logs")  # Directory for log files

# Reset position (mid-range for all joints)
RESET_TO_MID_RANGE = True  # If True, reset to mid-range; if False, use custom position below
CUSTOM_RESET_POSITION = {  # Only used if RESET_TO_MID_RANGE = False
    "leftarm_shoulder_pan_joint": 1.57,
    "leftarm_shoulder_lift_joint": 0.0,
    "leftarm_elbow_joint": 0.5,
    "leftarm_wrist_1_joint": 0.0,
    "leftarm_wrist_2_joint": 0.0,
    "leftarm_wrist_3_joint": 0.0,
}

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

# ============================ COLLISION DETECTION ============================

# Color palette
colors = [
    [0, 1, 0, 0.3],    # Green
    [1, 0, 0, 0.3],    # Red
    [0, 0, 1, 0.3],    # Blue
    [1, 1, 0, 0.3],    # Yellow
    [1, 0, 1, 0.3],    # Magenta
    [0, 1, 1, 0.3],    # Cyan
    [1, 0.5, 0, 0.3],  # Orange
    [0.5, 0, 1, 0.3],  # Purple
]

# State tracking (module level like original)
color_idx = 0
contact_count = 0
prev_touching = False

# Geom/site IDs (will be set in init)
ball_geom = None
sensor_geom = None
lid_site = None

# CSV file
csv_filename = None

def init_collision_detection(model):
    """Initialize collision detection - call once at startup"""
    global ball_geom, sensor_geom, lid_site, csv_filename
    
    # Find the geometries
    try:
        ball_geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "left_cup_ball_geom")
        print(f"✓ Found left_cup_ball_geom (ID: {ball_geom})")
    except:
        print("✗ ERROR: Could not find 'left_cup_ball_geom'")
        return False

    
    sensor_geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "left_cup_sensor_geom")
    print(f"✓ Found left_cup_sensor_geom (ID: {sensor_geom})")


    # Find the site for color changing
    try:
        lid_site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "left_cup_sensor_site")
        print(f"✓ Found left_cup_sensor_site (ID: {lid_site})")
    except:
        print("✗ WARNING: Could not find 'left_cup_sensor_site'")
        lid_site = None
    
    return True

def update_color(model):
    """Change the lid site color"""
    if lid_site is not None:
        model.site_rgba[lid_site] = colors[color_idx % len(colors)]

def check_collision(model, data):
    """Check if ball reached the cup bottom (ball in cup!)"""
    global color_idx, contact_count, prev_touching
    
    if ball_geom is None or sensor_geom is None:
        return
    
    # Check all contacts
    touching = False
    for i in range(data.ncon):
        contact = data.contact[i]
        geom1 = contact.geom1
        geom2 = contact.geom2
        
        # Check if ball and bottom sensor are colliding
        if (geom1 == ball_geom and geom2 == sensor_geom) or \
           (geom2 == ball_geom and geom1 == sensor_geom):
            touching = True
            break
    
    # Detect NEW collision (ball just entered cup!)
    if touching and not prev_touching:
        contact_count += 1
        color_idx += 1
        update_color(model)
        
        # Get positions
        sim_time = data.time
        
        # Get ball position
        try:
            ball_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "left_cup_ball")
            ball_pos = data.xpos[ball_body].copy()
        except:
            ball_pos = np.array([0, 0, 0])
        
        # Get lid position
        if lid_site is not None:
            lid_pos = data.site_xpos[lid_site].copy()
        else:
            lid_pos = np.array([0, 0, 0])
        
        # Log to CSV
        with open(csv_filename, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                contact_count,
                sim_time,
                ball_pos[0], ball_pos[1], ball_pos[2],
                lid_pos[0], lid_pos[1], lid_pos[2]
            ])
        
        color_rgb = colors[color_idx % len(colors)][:3]
        print(f"🎯 BALL IN CUP! #{contact_count} | Color: RGB{np.round(color_rgb, 2)}")
        print(f"   t={sim_time:.2f}s | Ball: [{ball_pos[0]:.3f}, {ball_pos[1]:.3f}, {ball_pos[2]:.3f}]")
    
    prev_touching = touching

def print_collision_summary():
    """Print summary statistics"""
    if csv_filename is not None:
        print("\n" + "="*60)
        print(f"Total collisions detected: {contact_count}")
        print(f"Events saved to: {csv_filename}")
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
            joint_name = actuator_name.replace("_motor", "_joint")
            self.joint_to_ctrl_idx[joint_name] = i
        
        print(f"[MuJoCo] Found {self.m.njnt} joints, {self.m.nu} actuators")
        
        # ============================ SET STARTUP POSITION ============================
        print("[MuJoCo] Setting startup position...")
        
        for joint_name, position in STARTUP_JOINT_POSITIONS.items():
            if joint_name in self.joint_to_qpos_idx:
                qpos_idx = self.joint_to_qpos_idx[joint_name]
                self.d.qpos[qpos_idx] = float(position)
                print(f"  {joint_name:28s} = {position:.6f} rad")
        
        self.d.qvel[:] = 0.0
        
        for joint_name, position in STARTUP_JOINT_POSITIONS.items():
            if joint_name in self.joint_to_ctrl_idx:
                ctrl_idx = self.joint_to_ctrl_idx[joint_name]
                self.d.ctrl[ctrl_idx] = float(position)
        
        mujoco.mj_forward(self.m, self.d)
        
        print("[MuJoCo] ✓ Startup position set")
        # ============================ END STARTUP POSITION ============================
        
        # Initialize collision detection
        init_collision_detection(self.m)
        
        # Logging state
        self._logging = False
        self._log_fh = None
        self._collision_log_fh = None
        self._ball_log_fh = None
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        
        # Find ball body for position tracking
        try:
            self.ball_body_id = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, "left_cup_ball")
            print(f"✓ Found left_cup_ball body for position tracking (ID: {self.ball_body_id})")
        except:
            print("✗ WARNING: Could not find 'left_cup_ball' body - ball logging disabled")
            self.ball_body_id = None
        
        # Command flags (thread-safe)
        self._cmd_reset = False
        self._cmd_reset_ball = False
        self._cmd_start_log = False
        self._cmd_stop_log = False
        
        # End position movement state
        self._moving_to_end = False
        self._end_move_start_time = None
        self._end_move_start_joints = None

        self._udp_enabled = False 
        
        # Pinocchio for IK
        print("[Pinocchio] Loading URDF...")
        self.robot = RobotWrapper.BuildFromURDF(URDF_PATH, [])
        self.pin_model = self.robot.model
        
        if not self.pin_model.existFrame(F_WRIST):
            raise RuntimeError(f"Frame not found: {F_WRIST}")
        self.fid_wrist = self.pin_model.getFrameId(F_WRIST)

        self.cup_center_site = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_SITE, "left_cup_center")
        
        # IK setup
        self.idx_q_vars, self.lb, self.ub = build_index_maps(self.pin_model, IK_JOINTS)
        
        # State tracking
        self._w_lpf = LowPassEMA(fc_hz=LPF_CUTOFF_HZ)
        self._prev_W = None
        self._prev_W_t = None
        self._last_qvars = None
        self._last_q_cmd = None

        self.rim_corner_ids = []
        for i in range(1, 5):
            try:
                corner_id = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_SITE, f"left_cup_rim_corner{i}")
                self.rim_corner_ids.append(corner_id)
                print(f"✓ Found left_cup_rim_corner{i} (ID: {corner_id})")
            except:
                print(f"✗ WARNING: Could not find left_cup_rim_corner{i}")
        
        if len(self.rim_corner_ids) == 4:
            print("✓ All 4 rim corners found - tilt-robust detection enabled")
        else:
            print("⚠ Missing rim corners - detection may be inaccurate")
        
        print("[MuJoCo] Teleop initialized!")
    
    def request_reset(self):
        """Request reset (called from GUI thread)"""
        self._cmd_reset = True
        print("[Command] Reset requested")
    
    def request_reset_ball(self):
        """Request ball-only reset (called from GUI thread)"""
        self._cmd_reset_ball = True
        print("[Command] Ball reset requested")
    
    def request_start_logging(self):
        """Request start logging (called from GUI thread)"""
        self._cmd_start_log = True
        print("[Command] Start logging requested")
    
    def request_stop_logging(self):
        """Request stop logging (called from GUI thread)"""
        self._cmd_stop_log = True
        print("[Command] Stop logging requested")
    
    def _do_reset_position(self):
        """Actually reset robot position (called from main thread)"""
        print("[Reset] Resetting to initial position...")
        
        if RESET_TO_MID_RANGE:
            for i in range(self.m.nu):
                ctrl_range = self.m.actuator_ctrlrange[i]
                mid_pos = (ctrl_range[0] + ctrl_range[1]) / 2.0
                self.d.ctrl[i] = mid_pos
            print("[Reset] Set targets to mid-range positions")
        else:
            for joint_name, position in CUSTOM_RESET_POSITION.items():
                if joint_name in self.joint_to_ctrl_idx:
                    ctrl_idx = self.joint_to_ctrl_idx[joint_name]
                    self.d.ctrl[ctrl_idx] = float(position)
            print("[Reset] Set targets to custom positions")
        
        print("[Reset] Robot will move to position")
    
    def _do_reset_ball(self):
        """Reset ball position and velocity independently (called from main thread)"""
        if self.ball_body_id is None:
            print("[Ball Reset] Ball body not found - cannot reset")
            return
        
        print("[Ball Reset] Resetting ball to initial position...")
        
        try:
            ball_jnt_id = None
            for i in range(self.m.njnt):
                if self.m.jnt_bodyid[i] == self.ball_body_id:
                    ball_jnt_id = i
                    break
            
            if ball_jnt_id is not None:
                qpos_adr = self.m.jnt_qposadr[ball_jnt_id]
                qvel_adr = self.m.jnt_dofadr[ball_jnt_id]
                
                self.d.qpos[qpos_adr:qpos_adr+3] = BALL_RESET_POSITION
                self.d.qpos[qpos_adr+3:qpos_adr+7] = [1.0, 0.0, 0.0, 0.0]
                self.d.qvel[qvel_adr:qvel_adr+6] = 0.0
                
                print(f"[Ball Reset] ✓ Via joint: position={BALL_RESET_POSITION}, velocity=zero")
            else:
                print("[Ball Reset] No joint found, trying direct body position...")
                self.d.xpos[self.ball_body_id] = BALL_RESET_POSITION
                print(f"[Ball Reset] ⚠ Set position via xpos (velocity reset may not work)")
            
            mujoco.mj_forward(self.m, self.d)
            print(f"[Ball Reset] ✓ Complete")
            
        except Exception as e:
            print(f"[Ball Reset] ✗ Error: {e}")
            import traceback
            traceback.print_exc()
    
    def _do_start_logging(self):
        """Actually start logging (called from main thread)"""
        if self._logging:
            print("[Logging] Already logging!")
            return False
        
        print("[1/1] Creating log files...")
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        wrist_path = LOG_DIR / f"wrist_{timestamp_str}.txt"
        collision_path = LOG_DIR / f"collisions_{timestamp_str}.txt"
        ball_path = LOG_DIR / f"ball_{timestamp_str}.txt"
        
        try:
            self._log_fh = open(wrist_path, "w", buffering=1)
            self._log_fh.write("# t_sec  q1(shoulder_pan)  q2(shoulder_lift)  q3(elbow)   (joint angles in radians)\n")
            
            self._collision_log_fh = open(collision_path, "w", buffering=1)
            self._collision_log_fh.write("# t_sec  event_num  ball_x  ball_y  ball_z  lid_x  lid_y  lid_z\n")
            
            if self.ball_body_id is not None:
                self._ball_log_fh = open(ball_path, "w", buffering=1)
                self._ball_log_fh.write("# t_sec  ball_x  ball_y  ball_z  above_cup(1/0)  cup_center_x  cup_center_y  cup_center_z\n")
                print(f"  Ball: {ball_path}")
            
            global csv_filename
            csv_filename = str(collision_path)
            
            self._logging = True
            self._udp_enabled = True 
            self._logging_start_time = time.time()

            print(f"\n[SUCCESS] Files created:")
            print(f"  Wrist: {wrist_path}")
            print(f"  Collisions: {collision_path}")
            print("="*60 + "\n")
            return True
            
        except Exception as e:
            if self._log_fh:
                self._log_fh.close()
            if self._collision_log_fh:
                self._collision_log_fh.close()
            if self._ball_log_fh:
                self._ball_log_fh.close()
            self._log_fh = self._collision_log_fh = self._ball_log_fh = None
            self._udp_enabled = False 
            print(f"[ERROR] Failed to start logging: {e}")
            print("="*60 + "\n")
            return False
    
    def _do_stop_logging(self):
        """Actually stop logging - initiates move to end position"""
        if not self._logging:
            print("[Logging] Not currently logging")
            return False
        
        print("\n[STOP LOGGING] Moving to end position while recording...")
        print(f"  Target end position:")
        for joint_name, pos in END_JOINT_POSITIONS.items():
            if joint_name in IK_JOINTS:  # Only print the 3 IK joints
                print(f"    {joint_name}: {np.degrees(pos):.1f}° ({pos:.4f} rad)")
        
        # Disable UDP teleop
        self._udp_enabled = False
        
        # Start end position movement
        self._moving_to_end = True
        self._end_move_start_time = time.time()
        
        # Store current joint positions as start
        self._end_move_start_joints = {}
        for joint_name in END_JOINT_POSITIONS.keys():
            if joint_name in self.joint_to_qpos_idx:
                qpos_idx = self.joint_to_qpos_idx[joint_name]
                self._end_move_start_joints[joint_name] = float(self.d.qpos[qpos_idx])
        
        print(f"  Moving over {END_POSITION_MOVE_TIME}s...")
        print(f"  (Still recording joint angles during movement)")
        return True
    
    def _update_end_position_movement(self, now):
        """Update robot position during end position movement"""
        if not self._moving_to_end:
            return False
        
        elapsed = now - self._end_move_start_time

        # Total time: 1 sec interpolate + 0.5 sec hold = 1.5 sec
        TOTAL_TIME = END_POSITION_MOVE_TIME + 0.5
        
        if elapsed < TOTAL_TIME:
            # Keep commanding end position throughout
            for joint_name, end_pos in END_JOINT_POSITIONS.items():
                if joint_name in self.joint_to_ctrl_idx:
                    ctrl_idx = self.joint_to_ctrl_idx[joint_name]
                    
                    if elapsed < END_POSITION_MOVE_TIME:
                        # Phase 1: Interpolate
                        t = elapsed / END_POSITION_MOVE_TIME
                        t_smooth = 0.5 - 0.5 * np.cos(t * np.pi)
                        start_pos = self._end_move_start_joints[joint_name]
                        current_target = start_pos + t_smooth * (end_pos - start_pos)
                        self.d.ctrl[ctrl_idx] = float(current_target)
                    else:
                        # Phase 2: Hold at end
                        self.d.ctrl[ctrl_idx] = float(end_pos)
            
            self.log_joint_angles(now)
            self.log_ball_position(now)
            return False
        
        # Time's up - close files
        print(f"\n[END POSITION] Complete after {elapsed:.1f}s")
        
        paths = []
        if self._log_fh:
            paths.append(self._log_fh.name)
            self._log_fh.close()
            self._log_fh = None
        if self._collision_log_fh:
            paths.append(self._collision_log_fh.name)
            self._collision_log_fh.close()
            self._collision_log_fh = None
        if self._ball_log_fh:
            paths.append(self._ball_log_fh.name)
            self._ball_log_fh.close()
            self._ball_log_fh = None
        
        self._logging = False
        self._moving_to_end = False
        
        print(f"\n[LOGGING STOPPED] Files saved:")
        for p in paths:
            print(f"  {p}")
        return True
    
    def get_cup_rim_center(self, data):
        """Get the center position of the 4 cup rim corners"""
        if len(self.rim_corner_ids) != 4:
            return None
        
        corner_positions = [data.site_xpos[corner_id] for corner_id in self.rim_corner_ids]
        cup_center = np.mean(corner_positions, axis=0)
        return cup_center
    
    def process_commands(self):
        """Process pending commands (called from main loop)"""
        if self._cmd_reset:
            self._cmd_reset = False
            self._do_reset_position()
        
        if self._cmd_reset_ball:
            self._cmd_reset_ball = False
            self._do_reset_ball()
        
        if self._cmd_start_log:
            self._cmd_start_log = False
            self._do_start_logging()
        
        if self._cmd_stop_log:
            self._cmd_stop_log = False
            self._do_stop_logging()
    
    def log_joint_angles(self, timestamp):
        """Log joint angles if logging is active"""
        if self._logging and self._log_fh:
            q1 = float(self.d.qpos[self.joint_to_qpos_idx["leftarm_shoulder_pan_joint"]])
            q2 = float(self.d.qpos[self.joint_to_qpos_idx["leftarm_shoulder_lift_joint"]])
            q3 = float(self.d.qpos[self.joint_to_qpos_idx["leftarm_elbow_joint"]])
            
            self._log_fh.write(f"{timestamp:.6f} {q1:.6f} {q2:.6f} {q3:.6f}\n")
    
    def is_ball_above_cup(self, data):
        """Check if ball is higher than the cup center site (using cup rim height)"""
        if self.ball_body_id is None:
            return False
        
        ball_z = data.xpos[self.ball_body_id][2]
        
        # Use the center site if available (faster and cleaner)
        if self.cup_center_site is not None:
            cup_rim_z = data.site_xpos[self.cup_center_site][2]
            return ball_z > cup_rim_z
        
        # Fallback to checking rim corners if center site not available
        if len(self.rim_corner_ids) == 4:
            corner_z_values = [data.site_xpos[corner_id][2] for corner_id in self.rim_corner_ids]
            max_corner_z = max(corner_z_values)
            return ball_z > max_corner_z
        
        # No way to determine cup height
        return False

    def log_ball_position(self, timestamp):
        """Log ball position and cup center if logging is active"""
        if self._logging and self._ball_log_fh and self.ball_body_id is not None:
            ball_pos = self.d.xpos[self.ball_body_id]
            ball_x, ball_y, ball_z = map(float, ball_pos)
            
            above_cup = self.is_ball_above_cup(self.d)
            above_flag = 1 if above_cup else 0
            
            cup_center = self.get_cup_rim_center(self.d)
            
            if cup_center is not None:
                cup_x, cup_y, cup_z = map(float, cup_center)
            else:
                cup_x, cup_y, cup_z = 0.0, 0.0, 0.0
            
            self._ball_log_fh.write(
                f"{timestamp:.6f} {ball_x:.6f} {ball_y:.6f} {ball_z:.6f} {above_flag} "
                f"{cup_x:.6f} {cup_y:.6f} {cup_z:.6f}\n"
            )
    
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

        # Log joint angles
        self.log_joint_angles(now)
        
        # Log ball position
        self.log_ball_position(now)
        
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
    
    # Initialize teleop controller (will set startup position)
    teleop = MuJoCoTeleop(MUJOCO_XML)
    
    # Start control GUI in separate thread
    print("[GUI] Starting control window...")
    gui_thread = threading.Thread(target=lambda: control_gui(teleop), daemon=True)
    gui_thread.start()
    time.sleep(0.5)  # Let GUI start
    
    print("[MuJoCo] Launching viewer...")
    print("\n" + "="*60)
    print("CONTROLS:")
    print("="*60)
    print("  VIEWER:")
    print("    - left-click + drag: Rotate view")
    print("    - Scroll: Zoom")
    print("    - Space: Pause/Resume simulation")
    print("    - ESC: Exit")
    print("")
    print("  CONTROL WINDOW (separate popup):")
    print("    - START LOGGING button")
    print("    - STOP LOGGING button (moves to end position while recording)")
    print("    - RESET POSITION button")
    print("    - RESET BALL button")
    print("")
    print(f"  UDP: Listening on port {UDP_PORT}")
    print("  Collision detection: ACTIVE")
    print("  Ball position tracking: ACTIVE")
    print(f"  End position move time: {END_POSITION_MOVE_TIME}s")
    print("="*60 + "\n")
    
    last_tick = time.time()
    
    try:
        with mujoco.viewer.launch_passive(teleop.m, teleop.d) as viewer:
            # Camera setup
            viewer.cam.lookat[0] = 0.045
            viewer.cam.lookat[1] = 0.293
            viewer.cam.lookat[2] = 1.4
            
            viewer.cam.distance = 4.0
            viewer.cam.elevation = 0
            viewer.cam.azimuth = 0
            
            # Visualization
            viewer.opt.geomgroup[:] = 1
            viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = 0
            
            print("[Ready] Use control window to start logging\n")
            
            while viewer.is_running():
                now = time.time()
                
                # Process any pending commands from GUI
                teleop.process_commands()
                
                # Handle end position movement (if active)
                if teleop._moving_to_end:
                    teleop._update_end_position_movement(now)
                
                # Control loop timing
                if now - last_tick >= CYCLE_SECONDS:
                    last_tick = now
                    if teleop._udp_enabled:
                        # Read wearable data
                        with _udp_lock:
                            hp, lp, up = _hand_pos, _larm_pos, _uarm_pos
                        
                        if hp is not None and lp is not None and up is not None:
                            # Map to wrist target
                            _, W_raw = scale_watch_to_left_robot(up, lp, hp)
                            
                            # Update controller (logs wrist and ball if logging active)
                            teleop.update(W_raw, now)
                
                # Check for collisions every simulation step
                check_collision(teleop.m, teleop.d)
                
                # Step simulation
                mujoco.mj_step(teleop.m, teleop.d)
                viewer.sync()
    
    except KeyboardInterrupt:
        print("\n[Interrupt] Shutting down...")
    
    finally:
        # Stop logging if active
        if teleop._logging:
            teleop._logging = False
            if teleop._log_fh:
                teleop._log_fh.close()
            if teleop._collision_log_fh:
                teleop._collision_log_fh.close()
            if teleop._ball_log_fh:
                teleop._ball_log_fh.close()
        
        # Print collision summary
        print_collision_summary()
        print("\nTeleoperation stopped.")


def control_gui(teleop):
    """Simple GUI control window using tkinter"""
    try:
        import tkinter as tk
        from tkinter import ttk
    except ImportError:
        print("[GUI] WARNING: tkinter not available, using print-based control instead")
        print("     Install tkinter with: sudo apt-get install python3-tk")
        return
    
    root = tk.Tk()
    root.title("MuJoCo Teleop Control")
    root.geometry("300x250")
    
    status_var = tk.StringVar(value="Ready")
    
    def start_log():
        status_var.set("Starting logging...")
        root.update()
        teleop.request_start_logging()
        status_var.set("LOGGING ACTIVE")
        start_btn.config(state='disabled')
        stop_btn.config(state='normal')
    
    def stop_log():
        status_var.set("Moving to end position...")
        root.update()
        teleop.request_stop_logging()
        status_var.set("Moving to end & recording...")
        # Don't re-enable start button yet - will happen when movement completes
    
    def reset_pos():
        status_var.set("Resetting position...")
        root.update()
        teleop.request_reset()
        status_var.set("Position reset requested")
    
    def reset_ball():
        status_var.set("Resetting ball...")
        root.update()
        teleop.request_reset_ball()
        status_var.set("Ball reset to initial position")
    
    # Status label
    status_label = ttk.Label(root, textvariable=status_var, font=('Arial', 12, 'bold'))
    status_label.pack(pady=20)
    
    # Buttons
    start_btn = ttk.Button(root, text="START LOGGING", command=start_log, width=20)
    start_btn.pack(pady=5)
    
    stop_btn = ttk.Button(root, text="STOP LOGGING", command=stop_log, width=20, state='disabled')
    stop_btn.pack(pady=5)
    
    reset_btn = ttk.Button(root, text="RESET POSITION", command=reset_pos, width=20)
    reset_btn.pack(pady=5)
    
    reset_ball_btn = ttk.Button(root, text="RESET BALL", command=reset_ball, width=20)
    reset_ball_btn.pack(pady=5)
    
    # Info
    info_label = ttk.Label(root, text="Logs saved to ./logs/", font=('Arial', 9))
    info_label.pack(pady=10)
    
    # Polling loop to check if movement is done and re-enable start button
    def check_movement_status():
        if not teleop._logging and not teleop._moving_to_end:
            if stop_btn['state'] == 'normal':  # Was logging
                start_btn.config(state='normal')
                stop_btn.config(state='disabled')
                status_var.set("Ready")
        root.after(100, check_movement_status)  # Check every 100ms
    
    check_movement_status()
    
    root.mainloop()

if __name__ == "__main__":
    main()