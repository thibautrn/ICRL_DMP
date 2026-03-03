#!/usr/bin/env python3
"""
MuJoCo Sim - Visualize DMP weights in simulation
Called by main_icl.py with: python mujoco_sim.py <weights_file> <iteration>
Pure visualization only — reward is computed later by Gazebo + ZED.

UPDATED: Now checks cup collisions (cup hitting arm/stand) with RED highlighting.
"""
import sys
import json
import math
import numpy as np
import mujoco
import mujoco.viewer
from pathlib import Path


# ============================================================
# CONFIGURATION
# ============================================================
MUJOCO_XML = "bimanualrobot.xml"
CYCLE_SECONDS = 0.025
STARTUP_WAIT = 2.0
END_POSITION_MOVE_TIME = 1.0
END_POSITION_HOLD_TIME = 0.5
LPF_CUTOFF_HZ = 3.5
MIN_JOINT_STEP_RAD = np.deg2rad(0.1)
SPEED_MULTIPLIER = 3  # 1 = real-time, 2 = 2x faster, 3 = 3x faster, etc.

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


# ============================================================
# CUP COLLISION
# ============================================================

def build_robot_geoms(model):
    """Build set of all robot geom IDs (cup + arm + stand)."""
    robot_geoms = set()
    cup_geoms   = set()
    stand_geoms = set()

    for i in range(model.ngeom):
        body_id   = model.geom_bodyid[i]
        body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
        geom_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i)

        if body_name and ("rightarm" in body_name or "stand" in body_name):
            robot_geoms.add(i)

            if geom_name and "right_cup" in geom_name:
                cup_geoms.add(i)

            if geom_name and "stand" in geom_name:
                stand_geoms.add(i)

    return robot_geoms, cup_geoms, stand_geoms


def store_original_colors(model, robot_geoms):
    """Store original RGBA colors of all robot geoms."""
    return {geom_id: model.geom_rgba[geom_id].copy() for geom_id in robot_geoms}


def restore_colors(model, original_colors):
    """Restore original RGBA colors."""
    for geom_id, color in original_colors.items():
        model.geom_rgba[geom_id] = color


def check_cup_collision(model, data, robot_geoms, cup_geoms, original_colors, collision_detected):
    """
    Check if cup hits any other robot part. Highlights colliding geoms RED
    on first detection and prints a warning. Returns updated collision_detected flag.
    """
    if data.ncon == 0:
        return collision_detected

    for i in range(data.ncon):
        contact   = data.contact[i]
        geom1_id  = contact.geom1
        geom2_id  = contact.geom2

        geom1_is_cup   = geom1_id in cup_geoms
        geom2_is_cup   = geom2_id in cup_geoms
        geom1_is_robot = geom1_id in robot_geoms
        geom2_is_robot = geom2_id in robot_geoms

        # Skip contacts with the stand (its collision mesh is misaligned from visual)
        name1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom1_id) or ""
        name2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom2_id) or ""
        if name1 == "stand_collision_geom" or name2 == "stand_collision_geom":
            continue

        if (geom1_is_cup and geom2_is_robot and not geom2_is_cup) or \
           (geom2_is_cup and geom1_is_robot and not geom1_is_cup):

            # Highlight RED on first detection
            if not collision_detected:
                label1 = name1 or f"geom{geom1_id}"
                label2 = name2 or f"geom{geom2_id}"
                print(f"  ⚠️  Cup collision detected: {label1} ↔ {label2}")
                model.geom_rgba[geom1_id] = [1.0, 0.0, 0.0, 1.0]
                model.geom_rgba[geom2_id] = [1.0, 0.0, 0.0, 1.0]

            return True  # collision_detected = True

    return collision_detected


# ============================================================
# DMP
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
    Y   = np.zeros((N, 3), float)
    Yd  = np.zeros_like(Y)
    Y[0] = y0
    for k in range(N - 1):
        acc = np.zeros(3, float)
        for d in range(3):
            f = float(Phi[k].dot(W[d]))
            acc[d] = (K * (g[d] - Y[k, d]) - D * tau * Yd[k, d] + K * f * (g[d] - y0[d])) / (tau ** 2)
        Yd[k + 1] = Yd[k] + acc * dt
        Y[k + 1]  = Y[k]  + Yd[k + 1] * dt
    return Y


class DMPJointRolloutSource:
    def __init__(self, baseline, weights, dt):
        c, h    = baseline["c"], baseline["h"]
        K, D    = float(baseline["K"]), float(baseline["D"])
        alpha_s = float(baseline["alpha_s"])
        tau     = float(baseline["run_time"])
        y0      = baseline['y0_star'].copy()
        g       = baseline['g_star'].copy()

        Y = rollout_dmp_3d(y0, g, weights, K, D, tau, dt, c, h, alpha_s)
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


# ============================================================
# CONTROLLER
# ============================================================

class SimpleController:
    def __init__(self, model, data, dmp_source):
        self.m = model
        self.d = data

        # Build joint mappings
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
            ball_body = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, "right_cup_ball")
            ball_jnt_id = None
            for i in range(self.m.njnt):
                if self.m.jnt_bodyid[i] == ball_body:
                    ball_jnt_id = i
                    break
            if ball_jnt_id is not None:
                qpos_adr = self.m.jnt_qposadr[ball_jnt_id]
                qvel_adr = self.m.jnt_dofadr[ball_jnt_id]
                self.d.qpos[qpos_adr:qpos_adr + 3]     = [0.80, 0.55, 1.25]
                self.d.qpos[qpos_adr + 3:qpos_adr + 7] = [1.0, 0.0, 0.0, 0.0]
                self.d.qvel[qvel_adr:qvel_adr + 6]      = 0.0
                mujoco.mj_forward(self.m, self.d)
        except Exception:
            pass

        # State machine
        self.state               = "WAITING"
        self.dmp_start_time      = None
        self.end_move_start_time = None

        # DMP
        self._dmp           = dmp_source
        self._last_dmp_time = None
        self._pending_q     = None
        self._last_q_cmd    = None

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
        if self.end_move_start_time is None:
            self.end_move_start_time = now

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
                    current  = self.d.qpos[qpos_idx]
                    self.d.ctrl[self.joint_to_ctrl_idx[joint_name]] = float(
                        current + t_smooth * (end_pos - current)
                    )
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


# ============================================================
# MAIN
# ============================================================

def main():
    if len(sys.argv) != 3:
        print("Usage: python mujoco_sim.py <weights_file> <iteration>")
        sys.exit(1)

    weights_file = Path(sys.argv[1])
    iteration    = int(sys.argv[2])

    if not weights_file.exists():
        print(f"❌ Weights file not found: {weights_file}")
        sys.exit(1)

    # Load weights
    weights_data = np.load(weights_file, allow_pickle=True)
    weights = weights_data['w']

    # Load baseline
    baseline_file = Path("./logs/robot_sideways2/baseline.npz")
    baseline_data = np.load(baseline_file, allow_pickle=True)
    baseline = {
        'y0_star':  baseline_data.get('y0_star', np.array([0.6, 0.3, 1.2])),
        'g_star':   baseline_data.get('g_star',  np.array([0.8, 0.55, 1.65])),
        'c':        baseline_data['c'],
        'h':        baseline_data['h'],
        'K':        float(baseline_data['K']),
        'D':        float(baseline_data['D']),
        'alpha_s':  float(baseline_data['alpha_s']),
        'run_time': float(baseline_data['run_time']),
        'M':        len(baseline_data['c'])
    }

    print(f"🎮 MuJoCo visualization — iteration {iteration}")
    print(f"   Weights: {weights.shape}, range=[{weights.min():.1f}, {weights.max():.1f}]")
    print("   Press ESC to close viewer")

    # Load model
    model   = mujoco.MjModel.from_xml_path(MUJOCO_XML)
    data_mj = mujoco.MjData(model)

    # Build geom sets and store original colors
    robot_geoms, cup_geoms, stand_geoms = build_robot_geoms(model)
    original_colors        = store_original_colors(model, robot_geoms)
    print(f"   Robot geoms: {len(robot_geoms)} | Cup geoms: {len(cup_geoms)}")

    # Create DMP source and controller
    dmp_src    = DMPJointRolloutSource(baseline, weights, CYCLE_SECONDS)
    controller = SimpleController(model, data_mj, dmp_src)

    last_update_time   = 0.0
    collision_detected = False

    try:
        with mujoco.viewer.launch_passive(model, data_mj) as viewer:
            viewer.cam.lookat[:]  = [0.045, 0.293, 1.4]
            viewer.cam.distance   = 4.0
            viewer.cam.elevation  = 0
            viewer.cam.azimuth    = 0
            viewer.opt.geomgroup[:] = 1
            viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = 0

            while viewer.is_running() and not controller.is_done():
                for _ in range(SPEED_MULTIPLIER):
                    if controller.is_done():
                        break
                    sim_time = data_mj.time
                    if sim_time - last_update_time >= CYCLE_SECONDS:
                        last_update_time = sim_time
                        done = controller.update(sim_time)
                        collision_detected = check_cup_collision(
                            model, data_mj, robot_geoms, cup_geoms,
                            original_colors, collision_detected
                        )
                        if done:
                            break
                    mujoco.mj_step(model, data_mj)
                viewer.sync()

    except Exception as e:
        pass  # viewer closed, that's fine

    # Restore colors before exit
    restore_colors(model, original_colors)

    if collision_detected:
        print("⚠️  Cup collision occurred during this run.")
    else:
        print("✓  No cup collisions detected.")

    print("✓ Simulation complete")
    print(json.dumps({"success": True, "message": "Simulation shown", "cup_collision": collision_detected}))
    sys.exit(2 if collision_detected else 0)


if __name__ == "__main__":
    main()