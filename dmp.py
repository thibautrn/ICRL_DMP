import os, glob, math, time
import numpy as np

# ======================= USER SETTINGS =======================
INPUT_DIR   = "logs/robot_sideways2"       # folder with *.txt (timestamp x y z)
OUTPUT_DIR  = "logs/robot_sideways2"         # we'll save baseline + weights here
FILES       = sorted(glob.glob(os.path.join(INPUT_DIR, "joints_*.txt")))  # or make a hardcoded list

# DMP hyperparameters (robust defaults)
M_BASIS     = 30          # number of RBFs per axis (25–50 is common)
ALPHA_S     = 5.0         # canonical decay
K_SPRING    = 1000.0      # stiffness
D_DAMP      = 2.0 * math.sqrt(K_SPRING)  # near-critical damping
LAMBDA_RIDGE= 1e-3        # ridge on weights
GOAL_INVARIANT = True     # True => canonicalize all demos to shared (y0*, g*)
RESAMPLE_HZ = 200.0       # uniform resample for fitting
SAVE_ROLLOUTS = True      # also save reconstructed paths for quick checks
# =============================================================

DT = 1.0 / RESAMPLE_HZ

# ---------------- I/O & preprocessing ----------------

def load_txt_xyz(path):
    """Load arrays (t, Y) from 'timestamp x y z' lines; ignores lines with '#'."""
    ts, xs, ys, zs = [], [], [], []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 4:
                continue
            t, x, y, z = map(float, parts[:4])
            ts.append(t); xs.append(x); ys.append(y); zs.append(z)
    if not ts:
        return np.array([]), np.zeros((0,3))
    t = np.asarray(ts, float)
    Y = np.vstack([xs, ys, zs]).T.astype(float)
    return t, Y

def uniform_resample(t, Y, dt=DT):
    """Resample Y(t) onto uniform grid [t0, t1] with step dt (half-open)."""
    t0, t1 = float(t[0]), float(t[-1])
    if t1 <= t0 + 1e-12:
        return np.array([t0]), Y[:1]
    t_uni = np.arange(t0, t1, dt)
    if len(t_uni) < 2:
        t_uni = np.array([t0, t1], float)
    Y_uni = np.empty((len(t_uni), Y.shape[1]), float)
    for d in range(Y.shape[1]):
        Y_uni[:, d] = np.interp(t_uni, t, Y[:, d])
    return t_uni, Y_uni

def smooth_derivatives(Y, dt=DT):
    """Central-difference derivatives (simple, no SciPy dependency)."""
    yd  = np.gradient(Y, dt, axis=0)
    ydd = np.gradient(yd, dt, axis=0)
    return yd, ydd

# ---------------- DMP core (discrete, goal-invariant) ----------------

def canonical_by_steps(N, dt, tau, alpha_s):
    """Canonical phase s with exactly N samples."""
    S = np.empty(int(N), float)
    s = 1.0
    for k in range(int(N)):
        S[k] = s
        s += dt * (-alpha_s * s / tau)
    return S

def rbf_params_linear(M):
    """Centers linearly in [0,1]; widths from neighbor spacing."""
    c = np.linspace(0.0, 1.0, M)
    d = np.diff(c)
    d = np.r_[d[:1], d, d[-1:]]  # pad
    h = 1.0 / ((0.65 * d[:-1])**2 + 1e-12)
    return c, h

def design_matrix(S, c, h):
    """Φ with len(S) rows; row_k = (psi/Σpsi)*s_k."""
    M = len(c)
    Phi = np.empty((len(S), M), float)
    for k, s in enumerate(S):
        psi = np.exp(-h * (s - c)**2)
        Phi[k, :] = (psi / (psi.sum() + 1e-12)) * s
    return Phi

def fit_w_one_axis(y, yd, ydd, y0, g, Kspr, Ddmp, tau, Phi, lam=LAMBDA_RIDGE):
    """Ridge regression to get w for one axis (goal-invariant form)."""
    den = Kspr * ((g - y0) + 1e-9)  # protect from tiny span
    f_tgt = ((tau**2) * ydd - Kspr * (g - y) + Ddmp * tau * yd) / den
    A = Phi.T @ Phi + lam * np.eye(Phi.shape[1])
    b = Phi.T @ f_tgt
    return np.linalg.solve(A, b)

def rollout_dmp_1d(y0, g, w, Kspr, Ddmp, tau, S, Phi, dt=DT):
    """Roll out 1D DMP given precomputed S, Φ."""
    N = len(S)
    y  = np.zeros(N, float)
    yd = np.zeros(N, float)
    ydd = np.zeros(N, float)
    y[0] = y0
    for k in range(N-1):
        f = float(Phi[k].dot(w))
        acc = (Kspr*(g - y[k]) - Ddmp*tau*yd[k] + Kspr * f * (g - y0)) / (tau**2)
        ydd[k]  = acc
        yd[k+1] = yd[k] + acc * dt
        y[k+1]  = y[k] + yd[k+1] * dt
    f_last = float(Phi[-1].dot(w))
    ydd[-1] = (Kspr*(g - y[-1]) - Ddmp*tau*yd[-1] + Kspr * f_last * (g - y0)) / (tau**2)
    return y, yd, ydd

# ---------------- goal-invariant retargeting ----------------

def best_fit_similarity(y0_demo, g_demo, y0_star, g_star):
    """Return (scale, R) mapping demo direction to star direction."""
    v_demo = g_demo - y0_demo
    v_star = g_star - y0_star
    nd = np.linalg.norm(v_demo) + 1e-12
    ns = np.linalg.norm(v_star) + 1e-12
    scale = ns / nd
    a = v_demo / nd; b = v_star / ns
    v = np.cross(a, b); s = np.linalg.norm(v); c = float(np.clip(a.dot(b), -1.0, 1.0))
    if s < 1e-8:
        if c > 0:
            R = np.eye(3)
        else:
            # 180°: choose any orthogonal axis
            u = np.array([1.0,0.0,0.0])
            if abs(a.dot(u)) > 0.9:
                u = np.array([0.0,1.0,0.0])
            v = np.cross(a, u); v /= (np.linalg.norm(v)+1e-12)
            K = np.array([[0,-v[2],v[1]],[v[2],0,-v[0]],[-v[1],v[0],0]])
            R = -np.eye(3) + 2*np.outer(v,v)
    else:
        vx = np.array([[0,-v[2],v[1]],[v[2],0,-v[0]],[-v[1],v[0],0]])
        R = np.eye(3) + vx + vx@vx * ((1-c)/(s**2))
    return scale, R

def retarget_demo(Y, y0_demo, g_demo, y0_star, g_star):
    """Similarity transform mapping demo line (y0->g) to star (y0*->g*)."""
    scale, R = best_fit_similarity(y0_demo, g_demo, y0_star, g_star)
    Y2 = (Y - y0_demo) * scale
    Y2 = (R @ Y2.T).T
    return y0_star + Y2

# ---------------- main ----------------

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    if not FILES:
        print(f"No .txt files found in {INPUT_DIR}")
        return

    # ---- load & resample all demos ----
    demos = []
    for p in FILES:
        t_raw, Y_raw = load_txt_xyz(p)
        if len(t_raw) < 5:
            print(f"[skip] too few samples: {os.path.basename(p)}")
            continue
        t, Y = uniform_resample(t_raw, Y_raw, dt=DT)
        demos.append((p, t, Y))
        print(f"[load] {os.path.basename(p)}  K={len(t)}  T={t[-1]-t[0]:.3f}s")
    if not demos:
        print("No valid demos after resampling.")
        return

    # ---- nominal timing for baseline (mean of actual durations) ----
    durations = [ (len(t)-1) * DT for (_, t, _) in demos ]
    run_time_nom = float(np.mean(durations))

    # ---- choose shared y0*, g* (goal-invariant) ----
    if GOAL_INVARIANT:
        y0s = np.vstack([Y[0]   for (_,_,Y) in demos])
        gs  = np.vstack([Y[-1]  for (_,_,Y) in demos])
        y0_star = np.mean(y0s, axis=0)
        g_star  = np.mean(gs,  axis=0)
        print(f"[canon] shared y0*={np.round(y0_star,3)}  g*={np.round(g_star,3)}")
    else:
        y0_star = None; g_star = None

    # ---- basis (shared) & baseline save ----
    c, h = rbf_params_linear(M_BASIS)
    baseline = dict(
        c=c, h=h,
        K=K_SPRING, D=D_DAMP,
        tau=run_time_nom, alpha_s=ALPHA_S,
        run_time=run_time_nom,
        goal_invariant=GOAL_INVARIANT,
        y0_star=y0_star if GOAL_INVARIANT else np.array([0.0, 0.0, 0.0]),  # ADD THIS
        g_star=g_star if GOAL_INVARIANT else np.array([0.0, 0.0, 0.0])     # ADD THIS
    )
    np.savez_compressed(os.path.join(OUTPUT_DIR, "baseline.npz"), **baseline)
    print("[baseline]")
    print("  M =", len(c))
    print("  K =", K_SPRING, " D =", D_DAMP, " alpha_s =", ALPHA_S, " run_time =", run_time_nom)

    # ---- fit per-demo weights ----
    for (path, t, Y) in demos:
        N   = len(t)
        T   = (N - 1) * DT
        tau = T  # per-demo tau to match its resampled length
        S   = canonical_by_steps(N, DT, tau, ALPHA_S)
        Phi = design_matrix(S, c, h)

        y0_demo = Y[0].copy()
        g_demo  = Y[-1].copy()
        if GOAL_INVARIANT:
            Y_fit = retarget_demo(Y, y0_demo, g_demo, y0_star, g_star)
            y0 = y0_star.copy(); g = g_star.copy()
        else:
            Y_fit = Y.copy()
            y0 = y0_demo.copy(); g = g_demo.copy()

        Yd, Ydd = smooth_derivatives(Y_fit, dt=DT)

        # Fit per axis
        W = np.zeros((3, M_BASIS), float)
        for d in range(3):
            W[d] = fit_w_one_axis(
                Y_fit[:, d], Yd[:, d], Ydd[:, d],
                y0[d], g[d], K_SPRING, D_DAMP, tau, Phi, lam=LAMBDA_RIDGE
            )

        # Optional: rollout to verify
        if SAVE_ROLLOUTS:
            Yrec  = np.zeros_like(Y_fit)
            Ydrec = np.zeros_like(Y_fit)
            Yddrec= np.zeros_like(Y_fit)
            for d in range(3):
                y, yd, ydd = rollout_dmp_1d(y0[d], g[d], W[d], K_SPRING, D_DAMP, tau, S, Phi, dt=DT)
                Yrec[:, d], Ydrec[:, d], Yddrec[:, d] = y, yd, ydd
            out_rec = os.path.join(
                OUTPUT_DIR, os.path.splitext(os.path.basename(path))[0] + "_recon.npz"
            )
            np.savez_compressed(out_rec, t=t, y=Yrec, yd=Ydrec, ydd=Yddrec, y_fit=Y_fit, y_raw=Y)

        # Save weights
        meta = dict(file=os.path.basename(path), tau=tau, used_goal_invariant=GOAL_INVARIANT)
        out_w = os.path.join(
            OUTPUT_DIR, os.path.splitext(os.path.basename(path))[0] + "_weights.npz"
        )
        np.savez_compressed(out_w, w=W, y0=y0, g=g, meta=np.array(meta, dtype=object))

        # Print quick stats
        print(f"[fit] {os.path.basename(path)} -> {os.path.basename(out_w)}")
        for d, axis in enumerate("xyz"):
            print(f"      w[{axis}]: mean={W[d].mean():+.3e}  std={W[d].std():.3e}  "
                  f"min={W[d].min():+.3e}  max={W[d].max():+.3e}")

    print(f"\nSaved baseline to: {os.path.join(OUTPUT_DIR, 'baseline.npz')}")
    print("Weights (and recon if enabled) saved next to it. Done.")

if __name__ == "__main__":
    main()