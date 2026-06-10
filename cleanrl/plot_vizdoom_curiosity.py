"""Generate paper-style VizDoom Curiosity-Critic plots from runs/*/metrics.jsonl.

This script is deliberately dependency-light: it implements IQM-style curves and
bootstrap confidence intervals with NumPy instead of requiring `rliable`.

Usage from the repo root:
    python cleanrl/plot_vizdoom_curiosity.py --runs-dir runs --out paper_figures/vizdoom
"""
import argparse
import csv
import json
import math
import os
from collections import defaultdict

import numpy as np


METHOD_ORDER = ["cc", "c_v2", "rnd", "c_v1", "ppo", "random"]
METHOD_LABEL = {
    "cc": "Curiosity-Critic",
    "c_v2": "Curiosity V2",
    "rnd": "RND",
    "c_v1": "Curiosity V1",
    "ppo": "PPO",
    "random": "Random",
}
DEFAULT_METRICS = [
    "charts/avg_episodic_return_periodic",
    "charts/goal_reached_rate_100ep",
    "eval/wm_holdout_l2",
    "mechanism/tv_zone_fraction",
    "mechanism/intrinsic_tv_mean_raw",
    "mechanism/intrinsic_non_tv_mean_raw",
    "charts/SPS",
]


def _safe_name(s):
    return (
        s.replace("/", "_")
        .replace(" ", "_")
        .replace(".", "p")
        .replace("=", "")
        .replace("__", "_")
        .strip("_")
    )


def _condition(args):
    if not bool(args.get("noisy_tv", False)):
        return "plain"
    alpha = float(args.get("noise_alpha", 1.0))
    if abs(alpha - 1.0) < 1e-9:
        return "noisytv"
    return f"noisytv_alpha{alpha:g}"


def _read_jsonl(path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def load_runs(runs_dir):
    runs = []
    if not os.path.isdir(runs_dir):
        return runs
    for name in sorted(os.listdir(runs_dir)):
        run_dir = os.path.join(runs_dir, name)
        meta_path = os.path.join(run_dir, "run_meta.json")
        metrics_path = os.path.join(run_dir, "metrics.jsonl")
        if not (os.path.isfile(meta_path) and os.path.isfile(metrics_path)):
            continue
        try:
            with open(meta_path) as f:
                meta = json.load(f)
            args = meta.get("args", {})
            metrics = _read_jsonl(metrics_path)
        except Exception as exc:
            print(f"[warn] skipping {run_dir}: {exc}")
            continue
        if not metrics:
            continue
        runs.append(
            {
                "run_dir": run_dir,
                "name": name,
                "args": args,
                "metrics": metrics,
                "scenario": args.get("scenario", "unknown"),
                "condition": _condition(args),
                "method": args.get("method", "unknown"),
                "seed": int(args.get("seed", -1)),
                "num_actions": int(meta.get("num_actions") or 3),
            }
        )
    return runs


def _max_global_step(run):
    steps = []
    for row in run["metrics"]:
        try:
            step = float(row.get("global_step", np.nan))
        except (TypeError, ValueError):
            continue
        if math.isfinite(step):
            steps.append(step)
    return max(steps) if steps else 0.0


def filter_runs(runs, min_planned_frac):
    """Drop smoke/duplicate runs before making paper plots.

    The runbook leaves smoke tests under `runs/`, while final jobs use the same
    scenario/method names. Once any full-planned run exists, short-planned smoke
    jobs should not be treated as independent seeds. For repeated attempts with
    the same scenario/condition/method/seed, keep the run with the most logged
    environment steps.
    """
    if not runs:
        return runs
    max_planned = max(float(run["args"].get("total_timesteps", 0) or 0) for run in runs)
    if max_planned > 0 and min_planned_frac > 0:
        cutoff = max_planned * min_planned_frac
        before = len(runs)
        runs = [run for run in runs if float(run["args"].get("total_timesteps", 0) or 0) >= cutoff]
        skipped = before - len(runs)
        if skipped:
            print(f"[filter] skipped {skipped} short-planned run(s) below {cutoff:.0f} total_timesteps")

    best = {}
    for run in runs:
        key = (run["scenario"], run["condition"], run["method"], run["seed"])
        step = _max_global_step(run)
        if key not in best or step > best[key][0]:
            best[key] = (step, run)
    deduped = [item[1] for item in sorted(best.values(), key=lambda x: x[1]["name"])]
    if len(deduped) != len(runs):
        print(f"[filter] kept longest run per scenario/condition/method/seed ({len(deduped)}/{len(runs)})")
    return deduped


def _series(run, metric):
    steps, vals = [], []
    for row in run["metrics"]:
        if metric not in row:
            continue
        x = float(row.get("global_step", np.nan))
        y = float(row[metric])
        if math.isfinite(x) and math.isfinite(y):
            steps.append(x)
            vals.append(y)
    if not steps:
        return None
    order = np.argsort(steps)
    return np.asarray(steps, dtype=np.float64)[order], np.asarray(vals, dtype=np.float64)[order]


def _iqm(values):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.nan
    # Interquartile mean as the average of the empirical quantile function over [0.25, 0.75].
    qs = np.linspace(0.25, 0.75, 101)
    return float(np.mean(np.quantile(values, qs)))


def _bootstrap_iqm(values, rng, n_boot):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.nan, np.nan, np.nan
    center = _iqm(values)
    if values.size == 1 or n_boot <= 0:
        return center, center, center
    boots = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        sample = values[rng.integers(0, values.size, size=values.size)]
        boots[i] = _iqm(sample)
    lo, hi = np.quantile(boots, [0.025, 0.975])
    return center, float(lo), float(hi)


def _aggregate_curves(runs, metric, points, n_boot, rng):
    series = []
    for run in runs:
        s = _series(run, metric)
        if s is not None:
            series.append(s)
    if not series:
        return None
    max_step = min(float(steps[-1]) for steps, _ in series)
    if max_step <= 0:
        return None
    grid = np.linspace(0.0, max_step, points)
    values = []
    for steps, vals in series:
        values.append(np.interp(grid, steps, vals, left=vals[0], right=vals[-1]))
    values = np.asarray(values, dtype=np.float64)
    center, lo, hi = [], [], []
    for t in range(values.shape[1]):
        c, l, h = _bootstrap_iqm(values[:, t], rng, n_boot)
        center.append(c)
        lo.append(l)
        hi.append(h)
    return grid, np.asarray(center), np.asarray(lo), np.asarray(hi), values


def plot_metric_groups(runs, metric, out_dir, points, n_boot, rng):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    grouped = defaultdict(list)
    for run in runs:
        if _series(run, metric) is None:
            continue
        grouped[(run["scenario"], run["condition"], run["method"])].append(run)

    scenarios_conditions = sorted({(s, c) for s, c, _ in grouped})
    made = []
    for scenario, condition in scenarios_conditions:
        fig, ax = plt.subplots(figsize=(7.2, 4.6))
        any_line = False
        for method in METHOD_ORDER:
            group_runs = grouped.get((scenario, condition, method), [])
            if not group_runs:
                continue
            agg = _aggregate_curves(group_runs, metric, points, n_boot, rng)
            if agg is None:
                continue
            grid, center, lo, hi, _ = agg
            ax.plot(grid, center, label=f"{METHOD_LABEL.get(method, method)} (n={len(group_runs)})", linewidth=2)
            ax.fill_between(grid, lo, hi, alpha=0.18)
            any_line = True
        if not any_line:
            plt.close(fig)
            continue
        ax.set_title(f"{scenario} / {condition}: {metric}")
        ax.set_xlabel("agent-environment steps")
        ax.set_ylabel(metric)
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8)
        fig.tight_layout()
        path = os.path.join(out_dir, f"{_safe_name(scenario)}__{_safe_name(condition)}__{_safe_name(metric)}.png")
        fig.savefig(path, dpi=160)
        plt.close(fig)
        made.append(path)
    return made


def _condition_alpha(condition):
    if condition == "plain":
        return 0.0
    if condition == "noisytv":
        return 1.0
    if condition.startswith("noisytv_alpha"):
        try:
            return float(condition.replace("noisytv_alpha", ""))
        except ValueError:
            return None
    return None


def plot_noise_sweeps(runs, metric, out_dir, n_boot, rng):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    grouped = defaultdict(list)
    for run in runs:
        alpha = _condition_alpha(run["condition"])
        if alpha is None or run["method"] not in {"cc", "c_v2", "rnd"}:
            continue
        if _series(run, metric) is None:
            continue
        grouped[(run["scenario"], run["method"], alpha)].append(run)

    made = []
    scenarios = sorted({scenario for scenario, _, _ in grouped})
    for scenario in scenarios:
        fig, ax = plt.subplots(figsize=(6.4, 4.2))
        any_line = False
        for method in ("cc", "c_v2", "rnd"):
            xs, centers, lows, highs = [], [], [], []
            alphas = sorted(alpha for s, m, alpha in grouped if s == scenario and m == method)
            for alpha in alphas:
                finals = []
                for run in grouped[(scenario, method, alpha)]:
                    s = _series(run, metric)
                    if s is not None:
                        finals.append(float(s[1][-1]))
                if not finals:
                    continue
                c, lo, hi = _bootstrap_iqm(np.asarray(finals), rng, n_boot)
                xs.append(alpha)
                centers.append(c)
                lows.append(lo)
                highs.append(hi)
            if not xs:
                continue
            xs, centers, lows, highs = map(np.asarray, (xs, centers, lows, highs))
            ax.plot(xs, centers, marker="o", linewidth=2, label=METHOD_LABEL.get(method, method))
            ax.fill_between(xs, lows, highs, alpha=0.18)
            any_line = True
        if not any_line:
            plt.close(fig)
            continue
        ax.set_title(f"{scenario}: final {metric} vs noise alpha")
        ax.set_xlabel("noise alpha (0=plain, 1=full noisy-TV)")
        ax.set_ylabel(metric)
        ax.set_xlim(-0.04, 1.04)
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8)
        fig.tight_layout()
        path = os.path.join(out_dir, f"{_safe_name(scenario)}__noise_sweep__{_safe_name(metric)}.png")
        fig.savefig(path, dpi=160)
        plt.close(fig)
        made.append(path)
    return made


def write_summary(runs, metrics, out_path, n_boot, rng):
    rows = []
    grouped = defaultdict(list)
    for run in runs:
        grouped[(run["scenario"], run["condition"], run["method"])].append(run)
    for (scenario, condition, method), group_runs in sorted(grouped.items()):
        for metric in metrics:
            finals = []
            steps = []
            for run in group_runs:
                s = _series(run, metric)
                if s is None:
                    continue
                x, y = s
                steps.append(float(x[-1]))
                finals.append(float(y[-1]))
            if not finals:
                continue
            center, lo, hi = _bootstrap_iqm(np.asarray(finals), rng, n_boot)
            rows.append(
                {
                    "scenario": scenario,
                    "condition": condition,
                    "method": method,
                    "metric": metric,
                    "n_runs": len(finals),
                    "max_step_min": min(steps),
                    "final_iqm": center,
                    "final_ci_low": lo,
                    "final_ci_high": hi,
                    "final_mean": float(np.mean(finals)),
                    "final_std": float(np.std(finals)),
                }
            )
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["scenario"])
        writer.writeheader()
        writer.writerows(rows)
    return rows


# --------------------------------------------------------------------------- #
#  Neural-FLOPs accounting (hardware-independent compute cost per method)
#
#  FLOPs here are fully determined by architecture + config (independent of the
#  trained weights, hardware, and run-to-run noise), so we compute them post-hoc:
#  measure one fwd (rollout) and one fwd+bwd (update) of each net per sample with
#  torch's FlopCounterMode, then scale by the deterministic per-update call counts.
#  NOTE: this captures the *neural* compute only -- env simulation (VizDoom, CPU)
#  has no meaningful FLOP count, so the rollout bucket reflects policy/aux forwards,
#  not the env-stepping that dominates rollout wall-clock. That is intentional: it
#  is a clean, hardware-independent measure of each method's neural overhead.
# --------------------------------------------------------------------------- #
FLOP_BUCKETS = ["rollout_policy", "rollout_aux", "wm_update", "aux_update", "policy_update"]
FLOPS_X_METRICS = {"charts/avg_episodic_return_periodic", "eval/wm_holdout_l2"}
_PERCALL_CACHE = {}


def _measure_per_call_flops(method, wm_arch, num_actions):
    """FLOPs of one fwd (and fwd+bwd) of each net `method` uses, per single sample.
    Cached by (method, wm_arch, num_actions). Returns None if torch/model code is unavailable."""
    key = (method, wm_arch, num_actions)
    if key in _PERCALL_CACHE:
        return _PERCALL_CACHE[key]
    try:
        import torch
        import torch.nn.functional as F
        from torch.utils.flop_counter import FlopCounterMode

        from ppo_curiosity_critic_vizdoom import Agent, Args, RNDModel, build_critic, build_world_model
    except Exception as exc:
        print(f"[flops] disabled (cannot import torch / training module): {exc}")
        _PERCALL_CACHE[key] = None
        return None

    args_obj = Args(wm_arch=wm_arch)
    obs1 = torch.zeros(1, 4, 84, 84)
    act1 = F.one_hot(torch.zeros(1, dtype=torch.long), num_classes=num_actions).float()
    nxt1 = torch.zeros(1, 1, 84, 84)

    def fwd(fn):
        with torch.no_grad(), FlopCounterMode(display=False) as fc:
            fn()
        return float(fc.get_total_flops())

    def fwdbwd(loss_fn):
        with FlopCounterMode(display=False) as fc:
            loss_fn().backward()
        return float(fc.get_total_flops())

    try:
        agent = Agent(num_actions)
        wm = build_world_model(args_obj, num_actions)
        pc = {
            "agent_value_fwd": fwd(lambda: agent.get_value(obs1)),
            "agent_av_fwd": fwd(lambda: agent.get_action_and_value(obs1)),
            "wm_fwd": fwd(lambda: wm(obs1, act1)),
            "wm_fwdbwd": fwdbwd(lambda: wm(obs1, act1).sum()),
        }

        def _agent_loss():
            _, logp, ent, ve, vi = agent.get_action_and_value(obs1)
            return logp.sum() + ent.sum() + ve.sum() + vi.sum()

        pc["agent_av_fwdbwd"] = fwdbwd(_agent_loss)
        if method == "cc":
            critic = build_critic(args_obj, num_actions)
            pc["critic_fwd"] = fwd(lambda: critic(obs1, act1))
            pc["critic_fwdbwd"] = fwdbwd(lambda: critic(obs1, act1).sum())
        if method == "rnd":
            rnd = RNDModel()
            pc["rnd_fwd"] = fwd(lambda: rnd(nxt1))
            pc["rnd_fwdbwd"] = fwdbwd(lambda: rnd(nxt1)[0].sum())  # only the predictor is trained
    except Exception as exc:
        print(f"[flops] measurement failed for {key}: {exc}")
        pc = None
    _PERCALL_CACHE[key] = pc
    return pc


def compute_flops_profile(run):
    """Per-update neural-FLOP breakdown + cumulative totals for one run (None if unavailable)."""
    args = run["args"]
    method = run["method"]
    pc = _measure_per_call_flops(method, args.get("wm_arch", "downsample"), run.get("num_actions") or 3)
    if not pc:
        return None
    num_envs = int(args.get("num_envs", 32))
    num_steps = int(args.get("num_steps", 128))
    num_mb = int(args.get("num_minibatches", 4))
    epochs = int(args.get("update_epochs", 4))
    up = float(args.get("update_proportion", 0.25))
    holdout = int(args.get("holdout_size", 10000))
    batch = num_envs * num_steps
    mb = batch // max(num_mb, 1)
    sub = mb * up  # expected size of the 25% world-model/aux subset
    n_roll = num_steps * num_envs
    is_random = method == "random"

    b = {k: 0.0 for k in FLOP_BUCKETS}
    if not is_random:
        b["rollout_policy"] = n_roll * (pc["agent_value_fwd"] + pc["agent_av_fwd"])
        b["policy_update"] = epochs * num_mb * mb * pc["agent_av_fwdbwd"]
    if method == "cc":
        b["rollout_aux"] = n_roll * (pc["wm_fwd"] + pc["critic_fwd"])
        b["aux_update"] = epochs * num_mb * sub * (pc["critic_fwdbwd"] + pc["wm_fwd"])  # critic + post-update WM fwd
    elif method == "c_v1":
        b["rollout_aux"] = n_roll * pc["wm_fwd"]
    elif method == "c_v2":
        b["rollout_aux"] = n_roll * 2.0 * pc["wm_fwd"]  # current + previous world model
    elif method == "rnd":
        b["rollout_aux"] = n_roll * pc["rnd_fwd"]
        b["aux_update"] = epochs * num_mb * sub * pc["rnd_fwdbwd"]
    b["wm_update"] = epochs * num_mb * sub * pc["wm_fwdbwd"]  # passive WM trained for every method

    per_update_train = sum(b.values())
    updates = [r.get("update") for r in run["metrics"] if isinstance(r.get("update"), int)]
    n_updates = max(updates) if updates else 0
    n_evals = sum(1 for r in run["metrics"] if "eval/wm_holdout_l2" in r)
    return {
        "buckets_per_update": b,
        "per_update_train": per_update_train,
        "flops_per_env_step": per_update_train / batch if batch else 0.0,
        "eval_per_eval": holdout * pc["wm_fwd"],
        "n_updates": n_updates,
        "n_evals": n_evals,
        "cumulative_train": per_update_train * n_updates,
        "cumulative_eval": holdout * pc["wm_fwd"] * n_evals,
    }


def attach_flops(runs):
    ok = False
    for run in runs:
        run["flops"] = compute_flops_profile(run)
        ok = ok or run["flops"] is not None
    return ok


def write_flops_breakdown(runs, out_path):
    grouped = defaultdict(list)
    for run in runs:
        if run.get("flops"):
            grouped[(run["scenario"], run["condition"], run["method"])].append(run)
    rows = []
    for (scenario, condition, method), grp in sorted(grouped.items()):
        f0 = grp[0]["flops"]  # per-update buckets are identical across seeds (same config)
        row = {"scenario": scenario, "condition": condition, "method": method, "n_runs": len(grp)}
        for k in FLOP_BUCKETS:
            row[f"per_update_{k}"] = f0["buckets_per_update"][k]
        row["per_update_train_total"] = f0["per_update_train"]
        row["flops_per_env_step"] = f0["flops_per_env_step"]
        row["cumulative_train_mean"] = float(np.mean([g["flops"]["cumulative_train"] for g in grp]))
        row["cumulative_eval_mean"] = float(np.mean([g["flops"]["cumulative_eval"] for g in grp]))
        rows.append(row)
    if rows:
        with open(out_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    return rows


def plot_flops_breakdown(runs, out_dir):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    grouped = defaultdict(dict)  # (scenario, condition) -> method -> per-update buckets
    for run in runs:
        if run.get("flops"):
            grouped[(run["scenario"], run["condition"])][run["method"]] = run["flops"]["buckets_per_update"]
    made = []
    for (scenario, condition), by_method in sorted(grouped.items()):
        methods = [m for m in METHOD_ORDER if m in by_method]
        if not methods:
            continue
        fig, ax = plt.subplots(figsize=(7.2, 4.6))
        x = np.arange(len(methods))
        bottoms = np.zeros(len(methods))
        for bucket in FLOP_BUCKETS:
            vals = np.array([by_method[m][bucket] for m in methods]) / 1e9  # GFLOPs/update
            ax.bar(x, vals, bottom=bottoms, label=bucket)
            bottoms += vals
        ax.set_xticks(x)
        ax.set_xticklabels([METHOD_LABEL.get(m, m) for m in methods], rotation=20, ha="right")
        ax.set_ylabel("neural GFLOPs per update")
        ax.set_title(f"{scenario} / {condition}: neural compute per update by method")
        ax.grid(True, axis="y", alpha=0.25)
        ax.legend(fontsize=8)
        fig.tight_layout()
        path = os.path.join(out_dir, f"{_safe_name(scenario)}__{_safe_name(condition)}__flops_per_update.png")
        fig.savefig(path, dpi=160)
        plt.close(fig)
        made.append(path)
    return made


def plot_metric_vs_flops(runs, metric, out_dir, points, n_boot, rng):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    grouped = defaultdict(list)
    for run in runs:
        if run.get("flops") and run["flops"]["flops_per_env_step"] > 0 and _series(run, metric) is not None:
            grouped[(run["scenario"], run["condition"], run["method"])].append(run)

    made = []
    for scenario, condition in sorted({(s, c) for s, c, _ in grouped}):
        fig, ax = plt.subplots(figsize=(7.2, 4.6))
        any_line = False
        for method in METHOD_ORDER:
            grp = grouped.get((scenario, condition, method), [])
            series = []
            for run in grp:
                s = _series(run, metric)
                if s is None:
                    continue
                steps, vals = s
                series.append((steps * run["flops"]["flops_per_env_step"], vals))  # x -> cumulative training FLOPs
            if not series:
                continue
            max_x = min(float(x[-1]) for x, _ in series)
            if max_x <= 0:
                continue
            grid = np.linspace(0.0, max_x, points)
            vv = np.asarray([np.interp(grid, x, y, left=y[0], right=y[-1]) for x, y in series])
            center, lo, hi = [], [], []
            for t in range(vv.shape[1]):
                c, l, h = _bootstrap_iqm(vv[:, t], rng, n_boot)
                center.append(c)
                lo.append(l)
                hi.append(h)
            center, lo, hi = map(np.asarray, (center, lo, hi))
            ax.plot(grid / 1e12, center, label=f"{METHOD_LABEL.get(method, method)} (n={len(series)})", linewidth=2)
            ax.fill_between(grid / 1e12, lo, hi, alpha=0.18)
            any_line = True
        if not any_line:
            plt.close(fig)
            continue
        ax.set_title(f"{scenario} / {condition}: {metric} vs neural compute")
        ax.set_xlabel("cumulative training neural FLOPs (TFLOPs)")
        ax.set_ylabel(metric)
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8)
        fig.tight_layout()
        path = os.path.join(out_dir, f"{_safe_name(scenario)}__{_safe_name(condition)}__{_safe_name(metric)}__vs_flops.png")
        fig.savefig(path, dpi=160)
        plt.close(fig)
        made.append(path)
    return made


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-dir", default="runs")
    parser.add_argument("--out", default="paper_figures/vizdoom")
    parser.add_argument("--metrics", nargs="+", default=DEFAULT_METRICS)
    parser.add_argument("--points", type=int, default=200)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--min-planned-frac",
        type=float,
        default=0.5,
        help="skip runs whose configured total_timesteps is below this fraction of the largest discovered run; use 0 to disable",
    )
    parser.add_argument(
        "--no-flops",
        action="store_true",
        help="skip the neural-FLOPs accounting (needs torch + the training module importable)",
    )
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    runs = load_runs(args.runs_dir)
    if not runs:
        raise SystemExit(f"no runs with run_meta.json + metrics.jsonl found under {args.runs_dir}")
    runs = filter_runs(runs, args.min_planned_frac)
    if not runs:
        raise SystemExit("all discovered runs were filtered out; lower --min-planned-frac")
    rng = np.random.default_rng(args.seed)
    print(f"loaded {len(runs)} runs from {args.runs_dir}")

    do_flops = (not args.no_flops) and attach_flops(runs)
    if do_flops:
        print("[flops] computed neural-FLOPs profiles")

    made = []
    for metric in args.metrics:
        paths = plot_metric_groups(runs, metric, args.out, args.points, args.bootstrap, rng)
        made.extend(paths)
        sweep_paths = plot_noise_sweeps(runs, metric, args.out, args.bootstrap, rng)
        made.extend(sweep_paths)
        msg = f"{metric}: wrote {len(paths)} curve plots + {len(sweep_paths)} noise-sweep plots"
        if do_flops and metric in FLOPS_X_METRICS:
            vf = plot_metric_vs_flops(runs, metric, args.out, args.points, args.bootstrap, rng)
            made.extend(vf)
            msg += f" + {len(vf)} vs-FLOPs plots"
        print(msg)

    summary_path = os.path.join(args.out, "summary_final_metrics.csv")
    rows = write_summary(runs, args.metrics, summary_path, args.bootstrap, rng)
    print(f"wrote {summary_path} ({len(rows)} rows)")

    if do_flops:
        fb_path = os.path.join(args.out, "flops_breakdown.csv")
        fb_rows = write_flops_breakdown(runs, fb_path)
        fb_plots = plot_flops_breakdown(runs, args.out)
        made.extend(fb_plots)
        print(f"wrote {fb_path} ({len(fb_rows)} rows) + {len(fb_plots)} FLOPs breakdown plots")

    print(f"wrote {len(made)} plot files under {args.out}")


if __name__ == "__main__":
    main()
