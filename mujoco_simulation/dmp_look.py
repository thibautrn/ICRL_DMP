#!/usr/bin/env python3
"""
Quick viewer to visualize saved DMP trajectories
Usage: 
  python view_dmp.py           # Uses ITERATION from config
  python view_dmp.py 5         # Views iteration 5
  python view_dmp.py 15        # Views iteration 15
"""

import numpy as np
import mujoco
import mujoco.viewer
from pathlib import Path
import sys

# ============ CONFIGURATION ============
# Set these to view specific runs
RUN_FOLDER = "202602241622"  # Folder name in logs/icl_logs/
ITERATION = 2               # Default iteration to view

# Files
MUJOCO_XML = "bimanualrobot.xml"
LOGS_BASE = Path("./logs/real_icl_logs")
# ========================================

# Copy constants from your main script
CYCLE_SECONDS = 0.025
STARTUP_WAIT = 2.0
END_POSITION_MOVE_TIME = 1.0
END_POSITION_HOLD_TIME = 0.5
LPF_CUTOFF_HZ = 3.5
MIN_JOINT_STEP_RAD = np.deg2rad(0.1)

STARTUP_JOINT_POSITIONS = {
    "rightarm_shoulder_pan_joint": 1.57,
    "rightarm_shoulder_lift_joint": 1.01,
    "rightarm_elbow_joint": -0.113,
    "rightarm_wrist_1_joint": 0.0,
    "rightarm_wrist_2_joint": 0.0,
    "rightarm_wrist_3_joint": 0.0,
}

END_JOINT_POSITIONS = {
    "rightarm_shoulder_pan_joint": 1.6,
    "rightarm_shoulder_lift_joint": 0.85,
    "rightarm_elbow_joint": 0.0,
    "rightarm_wrist_1_joint": 0.0,
    "rightarm_wrist_2_joint": 0.0,
    "rightarm_wrist_3_joint": 0.0,
}

IK_JOINTS = [
    "rightarm_shoulder_pan_joint",
    "rightarm_shoulder_lift_joint",
    "rightarm_elbow_joint",
]

JOINTS = [
    "rightarm_shoulder_pan_joint",
    "rightarm_shoulder_lift_joint",
    "rightarm_elbow_joint",
    "rightarm_wrist_1_joint",
    "rightarm_wrist_2_joint",
    "rightarm_wrist_3_joint",
]

# Copy DMP functions from your main script
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

import math

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

class SimpleController:
    def __init__(self, model, data, dmp_source):
        self.m = model
        self.d = data
        
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
            ball_body = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, "right_cup_ball")
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
        
        # State machine
        self.state = "WAITING"
        self.dmp_start_time = None
        self.end_move_start_time = None
        
        # DMP
        self._dmp = dmp_source
        self._joint_lpf = [LowPassEMA(fc_hz=LPF_CUTOFF_HZ) for _ in range(3)]
        self._last_dmp_time = None
        self._pending_q = None
        self._last_q_cmd = None
    
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
            
            q_filtered = np.zeros(3, float)
            for i in range(3):
                q_filtered[i] = self._joint_lpf[i].update(q_raw[i], self._last_dmp_time)
            
            self._pending_q = q_filtered
        
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
        if self.end_move_start_time is None:
            self.end_move_start_time = now
        
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
        if self.state == "WAITING":
            if now >= STARTUP_WAIT:
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
                return True
        
        elif self.state == "DONE":
            return True
        
        return False
    
    def is_done(self):
        return self.state == "DONE"

def main():
    # Check for command line argument
    iteration = ITERATION  # Default from config
    
    if len(sys.argv) > 1:
        try:
            iteration = int(sys.argv[1])
            print(f"📌 Using iteration from command line: {iteration}")
        except ValueError:
            print(f"⚠️  Invalid iteration number '{sys.argv[1]}', using default: {ITERATION}")
            iteration = ITERATION
    
    # Load weights
    run_dir = LOGS_BASE / RUN_FOLDER
    iter_dir = run_dir / f"iter_{iteration:03d}"
    weights_file = iter_dir / f"weights_iter{iteration:03d}.npz"
    
    if not weights_file.exists():
        print(f"❌ Weights file not found: {weights_file}")
        print(f"Looking in: {iter_dir}")
        return
    
    print(f"📂 Loading: {weights_file}")
    data = np.load(weights_file, allow_pickle=True)
    weights = data['w']
    
    # Load baseline
    baseline_file = Path("./logs/robot_logs/baseline.npz")
    baseline_data = np.load(baseline_file, allow_pickle=True)
    baseline = {
        'y0_star': baseline_data.get('y0_star', np.array([0.6, 0.3, 1.2])),
        'g_star': baseline_data.get('g_star', np.array([0.8, 0.55, 1.65])),
        'c': baseline_data['c'],
        'h': baseline_data['h'],
        'K': float(baseline_data['K']),
        'D': float(baseline_data['D']),
        'alpha_s': float(baseline_data['alpha_s']),
        'run_time': float(baseline_data['run_time']),
        'M': len(baseline_data['c'])
    }
    
    print(f"✓ Weights shape: {weights.shape}")
    print(f"✓ Weight range: [{weights.min():.1f}, {weights.max():.1f}]")
    
    # Load MuJoCo
    model = mujoco.MjModel.from_xml_path(MUJOCO_XML)
    data_mj = mujoco.MjData(model)
    
    # Create DMP source
    dmp_src = DMPJointRolloutSource(baseline, weights, CYCLE_SECONDS)
    
    # Create controller
    controller = SimpleController(model, data_mj, dmp_src)
    
    # Run with viewer
    print(f"\n▶️  Playing iteration {iteration} from run {RUN_FOLDER}...")
    print("Press ESC to close viewer")
    
    last_update_time = 0.0
    
    with mujoco.viewer.launch_passive(model, data_mj) as viewer:
        viewer.cam.lookat[:] = [0.045, 0.293, 1.4]
        viewer.cam.distance = 4.0
        viewer.cam.elevation = 0
        viewer.cam.azimuth = 0
        
        viewer.opt.geomgroup[:] = 1
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = 0
        
        while viewer.is_running() and not controller.is_done():
            sim_time = data_mj.time
            
            if sim_time - last_update_time >= CYCLE_SECONDS:
                last_update_time = sim_time
                done = controller.update(sim_time)
                if done:
                    break
            
            mujoco.mj_step(model, data_mj)
            viewer.sync()
    
    print("✓ Done!")

if __name__ == "__main__":
    main()