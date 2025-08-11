#!/usr/bin/env python3
"""
Full-grid benchmark.py

- ALWAYS post-process to the *full* grid (length = raw bitstring length L).
- Embed analytic targets onto that same grid with zeros between logical bins.
- Parity offset (0 or 1) for embedding is picked by best match to the CLEAN run.

Targets:
  classical   -> binomial over n+1 bins
  exponential -> discrete exp over n+1 bins
  hadamard    -> n must be 4; fixed 5-bin analytic distribution

Metrics (full-grid):
  - TVD(noisy,clean)
  - TVD(clean,target_full), TVD(noisy,target_full)
  - KL(clean||target_full), KL(noisy||target_full)
  - Hellinger(noisy,clean)
  - Integrity (fraction of one-hot strings in noisy counts)
"""

import argparse, os, json
import numpy as np
import matplotlib.pyplot as plt

from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error, thermal_relaxation_error, ReadoutError

from galton import GaltonBoard  # make sure galton.py sits next to this file


# ---------------- utilities ----------------
def tvd(p, q): return 0.5 * np.abs(np.asarray(p) - np.asarray(q)).sum()

def kl_div(p, q):
    p = np.asarray(p, float); q = np.asarray(q, float)
    m = (p > 0) & (q > 0)
    return float(np.sum(p[m] * np.log(p[m] / q[m])))

def hellinger(p, q):
    p = np.sqrt(np.asarray(p, float)); q = np.sqrt(np.asarray(q, float))
    return float(np.sqrt(0.5 * ((p - q) ** 2).sum()))

def aggregate(vals):
    a = np.asarray(vals, float)
    return float(a.mean()), float(a.std(ddof=1)) if len(a) > 1 else (float(a.mean()), 0.0)

def onehot_integrity(counts_dict):
    tot = sum(counts_dict.values()) or 1
    good = sum(v for s, v in counts_dict.items() if s.count('1') == 1)
    return good / tot


# ---- full-grid post-processing (no compression) ----
def onehot_counts_to_fullgrid(counts_dict, endianness='as_is'):
    """Keep all L slots so off-support bins show leakage."""
    if not counts_dict:
        return np.array([]), np.array([])
    L = len(next(iter(counts_dict)))
    counts = np.zeros(L, dtype=float)
    for raw_s, c in counts_dict.items():
        s = raw_s[::-1] if endianness == 'qiskit' else raw_s
        if s.count('1') != 1:
            continue
        counts[s.index('1')] += c
    probs = counts / counts.sum() if counts.sum() else counts
    positions = np.array([2*i - (L - 1) for i in range(L)], dtype=int)
    return positions, probs


# ---- analytic targets (bins) ----
def target_binomial_bins(n):
    from math import comb
    p = np.array([comb(n, k) for k in range(n + 1)], dtype=float)
    return p / p.sum()

def target_exponential_bins(n, lam):
    x = np.arange(n + 1)
    p = np.exp(-lam * x)
    return p / p.sum()

def target_hadamard_n4_bins():
    # symmetric 4-step Hadamard walk over 5 bins: [-4,-2,0,2,4]
    return np.array([1/16, 3/8, 1/8, 3/8, 1/16], dtype=float)


# ---- embed bins onto full grid ----
def embed_every_other(p_bins, L, offset):
    """Place p_bins at indices offset, offset+2, ... (zeros elsewhere)."""
    t = np.zeros(L, dtype=float)
    idxs = offset + 2 * np.arange(len(p_bins))
    idxs = idxs[idxs < L]           # guard if L shorter than expected
    t[idxs] = p_bins[:len(idxs)]
    return t, idxs

def choose_offset_by_clean(p_bins, p_clean_full):
    """Pick parity offset (0 or 1) that best matches CLEAN full-grid TVD."""
    L = len(p_clean_full)
    t0, idx0 = embed_every_other(p_bins, L, offset=0)
    t1, idx1 = embed_every_other(p_bins, L, offset=1)
    tvd0 = tvd(p_clean_full, t0)
    tvd1 = tvd(p_clean_full, t1)
    return (t0, idx0, 0) if tvd0 <= tvd1 else (t1, idx1, 1)

def leakage_mass(p_full, target_full):
    """Probability mass in slots where target_full is zero."""
    mask = target_full == 0.0
    return float(np.sum(p_full[mask]))


# ---------------- Noise model ----------------
def build_synthetic_noise_model():
    """Simple backend-agnostic noise model (tweak numbers if you like)."""
    # gate durations (s)
    t_sx, t_x, t_cx = 35e-9, 70e-9, 300e-9
    T1, T2 = 50e-6, 70e-6

    dep1 = depolarizing_error(0.001, 1)
    dep2 = depolarizing_error(0.01,  2)

    rel_sx = thermal_relaxation_error(T1, T2, t_sx)
    rel_x  = thermal_relaxation_error(T1, T2, t_x)
    rel_cx = thermal_relaxation_error(T1, T2, t_cx)

    err_sx = dep1.compose(rel_sx)
    err_x  = dep1.compose(rel_x)
    err_cx = dep2.compose(rel_cx)

    p01, p10 = 0.02, 0.03
    meas_err = ReadoutError([[1-p01, p01],
                             [p10,  1-p10]])

    nm = NoiseModel()
    nm.add_all_qubit_quantum_error(err_sx, ['sx'])
    nm.add_all_qubit_quantum_error(err_x,  ['x'])
    nm.add_all_qubit_quantum_error(err_cx, ['cx'])
    nm.add_all_qubit_readout_error(meas_err)
    return nm


# ---------------- Circuit builder ----------------
def build_circuit(board: GaltonBoard, target: str, lam: float | None):
    if target == 'classical':
        return board.classical_gb()
    elif target == 'hadamard':
        return board.quantum_walk_gb()
    elif target == 'exponential':
        if lam is None:
            raise ValueError("--lambda is required for target=exponential")
        return board.exponential_gb(lam)
    else:
        raise ValueError(f"Unknown target '{target}'")


# ---------------- Plot ----------------
def plot_fullgrid_overlay(positions, p_noisy, p_clean, p_target_full, title, out_path):
    xs = positions.astype(float)
    w = 0.36
    plt.figure(figsize=(8.2, 5))
    plt.bar(xs - w/2, p_noisy, width=w, label='Noisy', edgecolor='black', alpha=0.9)
    plt.bar(xs + w/2, p_clean, width=w, label='Clean', edgecolor='black', alpha=0.85)
    plt.plot(xs, p_target_full, 'o', label='Target (embedded)')
    plt.xlabel('Position'); plt.ylabel('Probability')
    plt.title(title); plt.grid(axis='y', ls='--', alpha=0.45)
    plt.legend(); plt.tight_layout()
    plt.savefig(out_path, dpi=200); plt.close()


# ---------------- CLI ----------------
def main():
    ap = argparse.ArgumentParser(description="QGB full-grid benchmark")
    ap.add_argument('--target', choices=['classical','exponential','hadamard'], required=True)
    ap.add_argument('--lambda', dest='lam', type=float, default=None, help='lambda for exponential')
    ap.add_argument('--steps', type=int, required=True, help='n levels (n+1 bins)')
    ap.add_argument('--shots', type=int, default=2048)
    ap.add_argument('--seeds', type=int, default=8)
    ap.add_argument('--opt', type=int, default=2, help='transpile optimization level (0-3)')
    ap.add_argument('--out_dir', type=str, default='results_fullgrid')
    ap.add_argument('--save_json', type=str, default=None)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # build board & circuit
    board = GaltonBoard(args.steps)
    qc = build_circuit(board, args.target, args.lam)

    # simulators
    sim_clean = AerSimulator()
    sim_noisy = AerSimulator(noise_model=build_synthetic_noise_model())

    # choose bins-level target
    if args.target == 'classical':
        p_bins = target_binomial_bins(board.n)
    elif args.target == 'exponential':
        p_bins = target_exponential_bins(board.n, args.lam)
    elif args.target == 'hadamard':
        if board.n != 4:
            raise ValueError("For hadamard target we fix n=4 for analytic reference.")
        p_bins = target_hadamard_n4_bins()
    else:
        p_bins = None

    per_seed = []
    probs_noisy_full_all, probs_clean_full_all = [], []
    positions_ref = None

    for seed in range(args.seeds):
        sim_clean.set_options(seed_simulator=seed)
        sim_noisy.set_options(seed_simulator=seed)

        # run (GaltonBoard.run_circuit already transpiles for backend)
        counts_clean = board.run_circuit(qc, sim_clean, shots=args.shots)
        counts_noisy = board.run_circuit(qc, sim_noisy, shots=args.shots)

        # full-grid post-processing
        positions, p_clean_full = onehot_counts_to_fullgrid(counts_clean, endianness='as_is')
        _,         p_noisy_full = onehot_counts_to_fullgrid(counts_noisy, endianness='as_is')
        if positions_ref is None:
            positions_ref = positions
        else:
            assert len(positions) == len(positions_ref), "position length changed across seeds"

        # embed target onto full grid; pick parity by clean alignment
        t_full, idxs, used_off = choose_offset_by_clean(p_bins, p_clean_full)

        # metrics
        m = {}
        m["seed"] = seed
        m["offset"] = used_off
        m["tvd_noisy_clean_full"] = tvd(p_noisy_full, p_clean_full)
        m["hellinger_noisy_clean"] = hellinger(p_noisy_full, p_clean_full)
        m["tvd_clean_target_full"] = tvd(p_clean_full, t_full)
        m["tvd_noisy_target_full"] = tvd(p_noisy_full, t_full)
        m["kl_clean_target_full"]  = kl_div(p_clean_full, t_full)
        m["kl_noisy_target_full"]  = kl_div(p_noisy_full, t_full)
        m["leakage_clean"] = leakage_mass(p_clean_full, t_full)
        m["leakage_noisy"] = leakage_mass(p_noisy_full, t_full)
        m["integrity_noisy"] = onehot_integrity(counts_noisy)
        per_seed.append(m)

        probs_clean_full_all.append(p_clean_full)
        probs_noisy_full_all.append(p_noisy_full)

    # aggregate
    def agg(key): return aggregate([x[key] for x in per_seed])
    tvd_nc_m, tvd_nc_s   = agg("tvd_noisy_clean_full")
    hel_nc_m, hel_nc_s   = agg("hellinger_noisy_clean")
    tvd_ct_m, tvd_ct_s   = agg("tvd_clean_target_full")
    tvd_nt_m, tvd_nt_s   = agg("tvd_noisy_target_full")
    kl_ct_m,  kl_ct_s    = agg("kl_clean_target_full")
    kl_nt_m,  kl_nt_s    = agg("kl_noisy_target_full")
    leak_c_m, leak_c_s   = agg("leakage_clean")
    leak_n_m, leak_n_s   = agg("leakage_noisy")
    integ_m,  integ_s    = agg("integrity_noisy")

    print("\n=== Metrics (mean ± std over seeds, full-grid) ===")
    print(f"TVD(noisy,clean)     : {tvd_nc_m:.6f} ± {tvd_nc_s:.6f}")
    print(f"Hellinger(noisy,clean): {hel_nc_m:.6f} ± {hel_nc_s:.6f}")
    print(f"TVD(clean,target)    : {tvd_ct_m:.6f} ± {tvd_ct_s:.6f}")
    print(f"TVD(noisy,target)    : {tvd_nt_m:.6f} ± {tvd_nt_s:.6f}")
    print(f"KL(clean||target)    : {kl_ct_m:.6f} ± {kl_ct_s:.6f}")
    print(f"KL(noisy||target)    : {kl_nt_m:.6f} ± {kl_nt_s:.6f}")
    print(f"Leakage(clean)       : {leak_c_m:.4f} ± {leak_c_s:.4f}")
    print(f"Leakage(noisy)       : {leak_n_m:.4f} ± {leak_n_s:.4f}")
    print(f"Integrity(noisy)     : {integ_m:.3f} ± {integ_s:.3f}")

    # mean probabilities for plot
    p_clean_mean = np.mean(np.stack(probs_clean_full_all, axis=0), axis=0)
    p_noisy_mean = np.mean(np.stack(probs_noisy_full_all, axis=0), axis=0)

    # choose a final target embedding using the clean-mean
    t_full_mean, _, used_off_final = choose_offset_by_clean(p_bins, p_clean_mean)

    title = f"{args.target.capitalize()} n={args.steps} (shots={args.shots}, seeds={args.seeds})"
    out_png = os.path.join(args.out_dir, f"{args.target}_n{args.steps}_fullgrid.png")
    plot_fullgrid_overlay(positions_ref, p_noisy_mean, p_clean_mean, t_full_mean, title, out_png)
    print(f"[saved plot] {out_png}")

    summary = {
        "config": {
            "target": args.target,
            "steps": args.steps,
            "shots": args.shots,
            "seeds": args.seeds,
            "opt_level": args.opt,
            "noise": "synthetic",
            "endianness": "as_is",
            "lambda": args.lam,
            "target_offset_final": used_off_final,
        },
        "metrics_mean_std": {
            "tvd_noisy_clean_full": [tvd_nc_m, tvd_nc_s],
            "hellinger_noisy_clean": [hel_nc_m, hel_nc_s],
            "tvd_clean_target_full": [tvd_ct_m, tvd_ct_s],
            "tvd_noisy_target_full": [tvd_nt_m, tvd_nt_s],
            "kl_clean_target_full": [kl_ct_m, kl_ct_s],
            "kl_noisy_target_full": [kl_nt_m, kl_nt_s],
            "leakage_clean": [leak_c_m, leak_c_s],
            "leakage_noisy": [leak_n_m, leak_n_s],
            "integrity_noisy": [integ_m, integ_s],
        },
        "positions": positions_ref.tolist(),
        "probs_mean": {
            "clean_full": p_clean_mean.tolist(),
            "noisy_full": p_noisy_mean.tolist(),
            "target_full": t_full_mean.tolist(),
        },
        "per_seed": per_seed,
    }

    if args.save_json:
        with open(args.save_json, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"[saved json] {args.save_json}")


if __name__ == "__main__":
    main()
