#!/usr/bin/env python3
"""
ZED Reward Calculator - Capture with ZED i2 camera and compute reward
"""
import sys
import json
import time
import numpy as np
from pathlib import Path


# ============================================================
# CONFIGURATION
# ============================================================
CAPTURE_DURATION = 15.0  # seconds - typical demo is ~5-10s
CAPTURE_FPS      = 30    # captures per second
SHOW_CAMERA      = True  # set False to disable live preview

# Color ranges (HSV)
LOWER_PINK = np.array([155, 50, 40], dtype=np.uint8)   # darker pinks
UPPER_PINK = np.array([179, 255, 231], dtype=np.uint8)

# Blue cup — covers hsl(192–209) at all lighting (dim, normal, strong, very dark).
# Hue is the tight discriminator (H_cv 93–108 = 186°–216°).
# S/V wide to handle all lighting conditions seen in samples.
LOWER_BLUE = np.array([ 91,  60,  20], dtype=np.uint8)
UPPER_BLUE = np.array([110, 255, 165], dtype=np.uint8)

MIN_BALL_AREA = 100
MIN_CUP_AREA  = 3000

# Ball-in-cup tolerances (WORLD frame, floor as origin)
X_TOLERANCE = 0.08
Y_TOLERANCE = 0.30
Z_TOLERANCE = 0.15

TIME_THRESHOLD = 2.0  # seconds ball must stay in cup to count


# ============================================================
# ZED CAMERA
# ============================================================

def initialize_zed_camera():
    try:
        import pyzed.sl as sl

        print("📷 Initializing ZED i2 camera...")

        zed  = sl.Camera()
        init = sl.InitParameters()
        init.camera_resolution = sl.RESOLUTION.HD720
        init.camera_fps        = 30
        init.depth_mode        = sl.DEPTH_MODE.ULTRA
        init.coordinate_units  = sl.UNIT.METER
        init.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Z_UP
        init.sdk_verbose       = False

        for i in range(5):
            err = zed.open(init)
            if err == sl.ERROR_CODE.SUCCESS:
                print("✓ ZED opened")
                zed.set_camera_settings(sl.VIDEO_SETTINGS.AEC_AGC,   0)
                zed.set_camera_settings(sl.VIDEO_SETTINGS.EXPOSURE,  70)
                zed.set_camera_settings(sl.VIDEO_SETTINGS.GAIN,      40)
                break
            print(f"⚠️  ZED open failed ({err}), retry {i+1}/5...")
            time.sleep(1)

        if not zed.is_opened():
            print("❌ Failed to open ZED after retries")
            return None

        tracking_params = sl.PositionalTrackingParameters()
        tracking_params.set_floor_as_origin = True
        err = zed.enable_positional_tracking(tracking_params)
        if err != sl.ERROR_CODE.SUCCESS:
            print(f"❌ enable_positional_tracking failed: {err}")
            return None

        print("✓ Positional tracking enabled (floor as origin)")
        return zed

    except ImportError:
        print("⚠️  ZED SDK not available, using mock mode")
        return "MOCK"
    except Exception as e:
        print(f"❌ ZED initialization error: {e}")
        return None


def close_zed_camera(zed):
    if zed is not None and zed != "MOCK":
        try:
            zed.disable_positional_tracking()
        except Exception:
            pass
        try:
            zed.close()
            print("✓ ZED camera closed")
        except Exception:
            pass


# ============================================================
# DETECTION HELPERS
# ============================================================

def get_largest_blob(mask, min_area=300):
    import cv2
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    c    = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(c)
    if area < min_area:
        return None
    M = cv2.moments(c)
    if M["m00"] == 0:
        return None
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    return cx, cy, area, c


def get_world_position(point_cloud, px, py):
    err, point = point_cloud.get_value(px, py)
    X, Y, Z = float(point[0]), float(point[1]), float(point[2])
    if np.isfinite(X) and np.isfinite(Y) and np.isfinite(Z):
        return (X, Y, Z)
    return None


def get_rim_world_position(point_cloud, contour, frame_shape, n_top_rows=12, neighborhood=11):
    """
    Find the world position of the cup rim by sampling the topmost pixels.
    Takes all contour points within `n_top_rows` of the highest pixel,
    samples a `neighborhood`x`neighborhood` depth patch around each,
    and returns the median XYZ of all valid samples.
    """
    pts = contour[:, 0, :]          # (N, 2): each row is [img_x, img_y]
    min_y = int(pts[:, 1].min())    # smallest image-y = highest on screen
    top_band = pts[pts[:, 1] <= min_y + n_top_rows]

    h, w = frame_shape[:2]
    half = neighborhood // 2
    samples = []
    for px, py in top_band:
        for dy in range(-half, half + 1):
            for dx in range(-half, half + 1):
                nx, ny = int(px) + dx, int(py) + dy
                if 0 <= nx < w and 0 <= ny < h:
                    pos = get_world_position(point_cloud, nx, ny)
                    if pos is not None:
                        samples.append(pos)

    if not samples:
        return None
    s = np.array(samples)
    return (float(np.median(s[:, 0])),
            float(np.median(s[:, 1])),
            float(np.median(s[:, 2])))


def get_blob_world_position(point_cloud, px, py, frame_shape, neighborhood=9):
    """
    Robust world position for a blob centroid.
    Samples a `neighborhood`x`neighborhood` patch and returns median XYZ
    of all valid depth hits — avoids single-pixel stereo failures.
    """
    h, w = frame_shape[:2]
    half = neighborhood // 2
    samples = []
    for dy in range(-half, half + 1):
        for dx in range(-half, half + 1):
            nx, ny = int(px) + dx, int(py) + dy
            if 0 <= nx < w and 0 <= ny < h:
                pos = get_world_position(point_cloud, nx, ny)
                if pos is not None:
                    samples.append(pos)
    if not samples:
        return None
    s = np.array(samples)
    return (float(np.median(s[:, 0])),
            float(np.median(s[:, 1])),
            float(np.median(s[:, 2])))


def check_ball_in_cup(ball_pos, cup_pos):
    if ball_pos is None or cup_pos is None:
        return False
    bx, by, bz = ball_pos  # x=left/right, y=depth, z=height
    cx, cy, cz = cup_pos
    return (abs(bx - cx) < X_TOLERANCE and
            abs(by - cy) < Y_TOLERANCE and
            abs(bz - cz) < Z_TOLERANCE)


# ============================================================
# TRAJECTORY CAPTURE
# ============================================================

def capture_trajectory(zed, duration, fps):
    import cv2
    import pyzed.sl as sl

    print(f"📷 Capturing trajectory for {duration}s...")

    runtime = sl.RuntimeParameters()
    runtime.measure3D_reference_frame = sl.REFERENCE_FRAME.WORLD

    left        = sl.Mat()
    point_cloud = sl.Mat()
    kernel      = np.ones((5, 5), np.uint8)

    trajectory   = []
    start_time   = time.time()
    last_capture = start_time
    interval     = 1.0 / fps

    in_cup_start_time     = None
    ball_confirmed_in_cup = False

    while time.time() - start_time < duration:
        now = time.time()

        if zed.grab(runtime) != sl.ERROR_CODE.SUCCESS:
            time.sleep(0.01)
            continue

        if now - last_capture < interval:
            continue
        last_capture = now

        zed.retrieve_image(left, sl.VIEW.LEFT)
        zed.retrieve_measure(point_cloud, sl.MEASURE.XYZRGBA)

        img_bgra = left.get_data()
        frame    = cv2.cvtColor(img_bgra, cv2.COLOR_BGRA2BGR)
        hsv      = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        ball_mask = cv2.inRange(hsv, LOWER_PINK, UPPER_PINK)
        ball_mask = cv2.morphologyEx(ball_mask, cv2.MORPH_OPEN,  kernel, iterations=1)
        ball_mask = cv2.morphologyEx(ball_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        ball_blob = get_largest_blob(ball_mask, MIN_BALL_AREA)

        cup_mask  = cv2.inRange(hsv, LOWER_BLUE, UPPER_BLUE)
        cup_mask  = cv2.morphologyEx(cup_mask, cv2.MORPH_OPEN,  kernel, iterations=1)
        cup_mask  = cv2.morphologyEx(cup_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        cup_blob  = get_largest_blob(cup_mask, MIN_CUP_AREA)

        ball_pos       = None
        cup_pos        = None
        rim_pos        = None
        cup_top_height = None

        if ball_blob is not None:
            bx, by, _, _ = ball_blob
            ball_pos = get_blob_world_position(point_cloud, bx, by, frame.shape, neighborhood=9)

        if cup_blob is not None:
            cx, cy, _, cc_for_rim = cup_blob
            cup_pos = get_blob_world_position(point_cloud, cx, cy, frame.shape, neighborhood=9)
            # Sample real rim depth from topmost contour pixels
            rim_pos = get_rim_world_position(point_cloud, cc_for_rim, frame.shape)
            if rim_pos is not None:
                cup_top_height = rim_pos[2]           # real measured rim Z (height)
            elif cup_pos is not None:
                cup_top_height = cup_pos[2] + 0.05    # fallback: centroid + 5cm

        is_in_cup = check_ball_in_cup(ball_pos, cup_pos)

        above_cup = False
        if ball_pos is not None and cup_top_height is not None:
            above_cup = ball_pos[2] > cup_top_height  # Z is height

        # ── Live preview ──────────────────────────────────────────
        if SHOW_CAMERA:
            disp = frame.copy()
            if ball_blob is not None:
                bx, by, _, bc = ball_blob
                cv2.drawContours(disp, [bc], -1, (0, 255, 0), 2)
                cv2.circle(disp, (bx, by), 6, (0, 255, 0), -1)
                if ball_pos:
                    cv2.putText(disp, f"ball x={ball_pos[0]:.3f}", (bx + 8, by - 8),  cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
                    cv2.putText(disp, f"     y={ball_pos[1]:.3f}", (bx + 8, by + 8),  cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
                    cv2.putText(disp, f"     z={ball_pos[2]:.3f}", (bx + 8, by + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
                else:
                    cv2.putText(disp, "ball (no depth)", (bx + 8, by), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
            if cup_blob is not None:
                cx2, cy2, _, cc = cup_blob
                cv2.drawContours(disp, [cc], -1, (255, 80, 0), 2)
                cv2.circle(disp, (cx2, cy2), 6, (255, 80, 0), -1)
                if cup_pos:
                    cv2.putText(disp, f"cup  x={cup_pos[0]:.3f}", (cx2 + 8, cy2 - 8),  cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 80, 0), 1)
                    cv2.putText(disp, f"     y={cup_pos[1]:.3f}", (cx2 + 8, cy2 + 8),  cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 80, 0), 1)
                    cv2.putText(disp, f"     z={cup_pos[2]:.3f}", (cx2 + 8, cy2 + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 80, 0), 1)
                else:
                    cv2.putText(disp, "cup (no depth)", (cx2 + 8, cy2), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 80, 0), 1)
                # Rim point: topmost contour pixel (image space) + sampled world XYZ
                rim_pt = tuple(cc[cc[:, :, 1].argmin()][0])
                cv2.circle(disp, rim_pt, 6, (0, 255, 255), -1)
                cv2.line(disp, (rim_pt[0] - 10, rim_pt[1]), (rim_pt[0] + 10, rim_pt[1]), (0, 255, 255), 1)
                if rim_pos is not None:
                    src = "rim"
                    rim_xyz = rim_pos
                elif cup_pos and cup_top_height is not None:
                    src = "rim~"
                    rim_xyz = (cup_pos[0], cup_pos[1], cup_top_height)
                else:
                    rim_xyz = None
                if rim_xyz is not None:
                    cv2.putText(disp, f"{src}  x={rim_xyz[0]:.3f}", (rim_pt[0] + 8, rim_pt[1] - 8),  cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
                    cv2.putText(disp, f"      y={rim_xyz[1]:.3f}", (rim_pt[0] + 8, rim_pt[1] + 8),  cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
                    cv2.putText(disp, f"      z={rim_xyz[2]:.3f}", (rim_pt[0] + 8, rim_pt[1] + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
            # Distance + status overlay
            elapsed = now - start_time
            status  = "IN CUP!" if ball_confirmed_in_cup else ("above rim" if above_cup else "")
            cv2.putText(disp, f"t={elapsed:.1f}s  {status}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2)
            if ball_pos is not None and cup_pos is not None:
                ref_pos = rim_pos if rim_pos is not None else cup_pos
                dist = float(np.linalg.norm(np.array(ball_pos) - np.array(ref_pos)))
                label = "dist(rim)" if rim_pos is not None else "dist(cup)"
                cv2.putText(disp, f"{label}={dist:.3f}m", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2)
            # Big banner when ball is above the rim or in the cup
            if ball_confirmed_in_cup:
                cv2.putText(disp, "*** BALL IN CUP! ***", (10, 130),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0, 255, 0), 4)
            elif above_cup:
                cv2.putText(disp, "ABOVE RIM", (10, 130),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0, 255, 255), 4)
            cv2.imshow("ZED Camera Reward", disp)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        # ──────────────────────────────────────────────────────────

        if is_in_cup and not ball_confirmed_in_cup:
            if in_cup_start_time is None:
                in_cup_start_time = now
            elif now - in_cup_start_time >= TIME_THRESHOLD:
                ball_confirmed_in_cup = True
                print("🎯 Ball confirmed IN CUP!")
        elif not is_in_cup and not ball_confirmed_in_cup:
            in_cup_start_time = None

        if ball_pos is not None and cup_pos is not None:
            trajectory.append({
                'time':           now - start_time,
                'ball_pos':       list(ball_pos),
                'cup_pos':        list(cup_pos),
                'cup_top_height': cup_top_height,
                'above_cup':      above_cup,
                'ball_in_cup':    ball_confirmed_in_cup,
            })

    total_time = trajectory[-1]['time'] if trajectory else 0.0
    print(f"✓ Captured {len(trajectory)} frames ({total_time:.1f}s)")
    if SHOW_CAMERA:
        cv2.destroyAllWindows()
    return trajectory


def capture_trajectory_mock(duration, fps):
    print(f"📷 [MOCK] Generating mock trajectory...")
    trajectory = []
    n_frames   = int(duration * fps)
    for i in range(n_frames):
        t        = i / fps
        ball_pos = [0.75 + 0.05 * np.sin(t * 2), 0.5 + 0.3 * np.sin(t), 1.30]
        cup_pos  = [0.73, 0.46, 1.30]
        trajectory.append({
            'time':           t,
            'ball_pos':       ball_pos,
            'cup_pos':        cup_pos,
            'cup_top_height': cup_pos[1] + 0.05,
            'above_cup':      ball_pos[1] > cup_pos[1] + 0.05,
            'ball_in_cup':    False,
        })
    print(f"✓ [MOCK] Generated {len(trajectory)} frames")
    return trajectory


# ============================================================
# REWARD CALCULATION
# ============================================================

def compute_reward_from_trajectory(trajectory):
    if len(trajectory) == 0:
        return {
            'total_reward':          0.0,
            'ball_in_cup':           False,
            'min_distance_to_cup':   float('inf'),
            'ball_went_above_cup':   False,
            'distance_reward':       0.0,
            'height_bonus':          0.0,
            'collision_reward':      0.0,
            'num_frames':            0,
        }

    ball_positions = np.array([t['ball_pos'] for t in trajectory])
    cup_positions  = np.array([t['cup_pos']  for t in trajectory])
    above_flags    = np.array([t['above_cup']    for t in trajectory])
    in_cup_flags   = np.array([t['ball_in_cup']  for t in trajectory])

    distances_3d = np.linalg.norm(ball_positions - cup_positions, axis=1)
    min_distance = float(np.min(distances_3d))

    ball_went_above = bool(np.any(above_flags))
    ball_in_cup     = bool(np.any(in_cup_flags))

    MAX_DISTANCE = 0.75
    distance_reward  = 750.0 * (1.0 - min_distance / MAX_DISTANCE) if min_distance <= MAX_DISTANCE else 0.0
    height_bonus     = 200.0 if ball_went_above else 0.0
    collision_reward = 1000.0 if ball_in_cup else 0.0
    total_reward     = distance_reward + height_bonus + collision_reward

    return {
        'total_reward':          float(total_reward),
        'ball_in_cup':           bool(ball_in_cup),
        'min_distance_to_cup':   float(min_distance),
        'ball_went_above_cup':   bool(ball_went_above),
        'distance_reward':       float(distance_reward),
        'height_bonus':          float(height_bonus),
        'collision_reward':      float(collision_reward),
        'cup_center':            cup_positions.tolist(),
        'num_frames':            len(trajectory),
    }


# ============================================================
# MAIN
# ============================================================

def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('iteration', type=str)
    args = parser.parse_args()

    iteration = args.iteration

    print(f"\n{'='*70}")
    print(f"ZED REWARD CALCULATOR - Episode {iteration}")
    print(f"{'='*70}\n")

    zed = initialize_zed_camera()

    if zed is None:
        print("❌ Failed to initialize ZED camera")
        sys.exit(1)

    print("✓ Camera ready, starting capture...\n")

    try:
        if zed == "MOCK":
            trajectory = capture_trajectory_mock(CAPTURE_DURATION, CAPTURE_FPS)
        else:
            trajectory = capture_trajectory(zed, CAPTURE_DURATION, CAPTURE_FPS)

        output_dir = Path("logs/robot_episodes") / f"episode_{iteration}"
        output_dir.mkdir(parents=True, exist_ok=True)
        trajectory_file = output_dir / f"zed_trajectory_{iteration}.json"
        with open(trajectory_file, 'w') as f:
            json.dump(trajectory, f, indent=2)
        print(f"💾 Saved trajectory: {trajectory_file}")

        reward_data = compute_reward_from_trajectory(trajectory)

        print(f"\n{'='*70}")
        print(f"REWARD RESULTS")
        print(f"{'='*70}")
        print(f"Total reward:  {reward_data['total_reward']:.1f}")
        print(f"  Distance:    {reward_data['distance_reward']:.1f} pts  (min: {reward_data['min_distance_to_cup']*100:.1f}cm)")
        print(f"  Height:      {reward_data['height_bonus']:.1f} pts  (above cup: {reward_data['ball_went_above_cup']})")
        print(f"  Collision:   {reward_data['collision_reward']:.1f} pts  (in cup: {reward_data['ball_in_cup']})")
        print(f"  Frames:      {reward_data['num_frames']}")
        print(f"{'='*70}\n")

        print(json.dumps(reward_data))

    finally:
        close_zed_camera(zed)


if __name__ == "__main__":
    main()