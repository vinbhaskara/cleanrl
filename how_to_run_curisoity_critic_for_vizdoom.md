# How To Run Curiosity-Critic on VizDoom (MyWayHome)

This guide runs `cleanrl/ppo_curiosity_critic_vizdoom.py` for the paper's
real-environment experiments and collects the figures/videos for the company
presentation. Everything runs on the Linux box with the RTX 3090 Ti.

The script is a single file that implements **six methods** behind one
`--method` flag, all sharing the same World-Model / Neural-Critic / PPO code:

| `--method` | What it is | Aux nets |
|---|---|---|
| `random` | uniform random actions, no learning (exploration floor) | none |
| `ppo`    | PPO on extrinsic reward only | none |
| `c_v1`   | Curiosity V1: `r = e(s,a|θ_t)` (zero baseline) | World Model |
| `c_v2`   | Curiosity V2: one-step error improvement | World Model (+prev snapshot) |
| `rnd`    | Random Network Distillation | RND predictor/target |
| `cc`     | **Curiosity-Critic (ours)**: `r = e(s,a|θ_t) − φ(s,a)` | World Model + Neural Critic |

Two conditions, selected with `--noisy-tv`:

- **plain** (omit the flag): deterministic MyWayHome maze.
- **noisy-tv** (`--noisy-tv`): same maze + a visitable noise panel painted into the
  observation while the agent is within `--tv-radius` of its episode start. This is
  the noisy-TV trap that raw prediction-error methods fall into and Curiosity-Critic
  is designed to escape. The default `--tv-radius 150` was calibrated for MyWayHome
  sparse via `--probe-maze` (maze max reach ~516): a localized start-region zone the
  agent starts inside but can leave, keeping the far rooms / goal noise-free.

---

## 1. One-Time Setup

Install ViZDoom system dependencies and the Python packages:

```bash
sudo apt update
sudo apt install -y cmake git libboost-all-dev libsdl2-dev libopenal-dev tmux

# in your cleanrl venv
pip install vizdoom opencv-python matplotlib
```

The training stack (`torch`, `gymnasium>=1.0`, `tyro`, `tensorboard`, `wandb`) is
already installed for the existing cleanrl scripts.

Confirm the scenario wads are present (they are bundled in the repo at the
default `--wad-dir ./vizdoom_scenarios`):

```bash
ls vizdoom_scenarios/
# my_way_home_sparse.wad  my_way_home_verySparse.wad  my_way_home_dense.wad
```

(Optional) log in to Weights & Biases:

```bash
wandb login
```

---

## 1b. Build the held-out WM-eval sets (run once)

Build the fixed held-out transition sets used to score every method's world-model accuracy
(`eval/wm_holdout_l2`) — one set per seed, shared by all methods of that seed:

```bash
python cleanrl/build_holdout.py --scenario sparse --seeds 1 2 3 --size 2048
python cleanrl/build_holdout.py --scenario very_sparse --seeds 1 2 3 --size 2048
```

This writes `vizdoom_holdout/holdout_{sparse,very_sparse}_seed{1,2,3}.npz` plus a **coverage
heatmap overlaid on the top-down maze map** per seed/scenario. **Open the six heatmaps and confirm
the samples cover rooms and corridors, not just spawn / wall-bump loops** — that's the visual gate
that the WM-accuracy metric is measuring global quality, not just the spawn room. If any look thin,
raise `--size` or `--p-forward`. (Training auto-loads the matching-seed/scenario file; if missing it
collects one inline with a warning — but running this once gives deliberate, inspected coverage.)

## 2. Phase 0 — Smoke Test (DO THIS FIRST)

A short run that verifies VizDoom launches, the wad/map load, throughput is
acceptable, and the noisy-TV trap is set up correctly. **Everything downstream
assumes this passes.**

```bash
python cleanrl/ppo_curiosity_critic_vizdoom.py \
    --method cc --scenario sparse --noisy-tv \
    --total-timesteps 200000 --num-envs 8 \
    --capture-video --video-every 1 --wm-panel-every 5 --heatmap-every 5 \
    --seed 1
```

Check the console and `runs/<run_name>/`:

- **`SPS:` printed** and steady. Multiply by 86,400 to estimate agent steps/day; a 30M-step
  run should fit in well under a day. If SPS is low, raise `--num-envs` toward your
  CPU core count.
- **`runs/<run_name>/viz/wm_panel_*.png`** shows input / WM prediction / true next / abs error.
- **`runs/<run_name>/viz/heatmap_*.png`** shows visitation and mean intrinsic reward overlaid on
  the top-down maze map, with the printed `TV-zone time fraction`.
- **`runs/<run_name>/videos/*.mp4`** plays the agent in the maze.
- **`runs/<run_name>/map_vids/*.mp4`** shows the same rollout as a top-down moving dot/trail on
  the fixed 2D maze map.
- **`runs/<run_name>/plots/*.png`** is generated automatically at normal completion from that
  run's own `metrics.jsonl`, for quick single-run inspection.

Also test the post-run plotting loop:

```bash
python cleanrl/plot_vizdoom_curiosity.py --runs-dir runs --out paper_figures/vizdoom_smoke
```

Confirm it writes plot PNGs plus `paper_figures/vizdoom_smoke/summary_final_metrics.csv`.

Map-name check: if VizDoom errors on the map, list it with
`python -c "import vizdoom,os; g=vizdoom.DoomGame(); g.set_doom_scenario_path('vizdoom_scenarios/my_way_home_sparse.wad'); print('ok')"`
and pass the correct `--doom-map` (default `map01`).

Noisy-TV tuning: open a heatmap/video. The TV zone should cover roughly the
starting room. If it is too small/large, adjust `--tv-radius` (game units; default
`150`) and re-run the smoke test.

---

## 3. The Experiment Matrix

Per `next-steps-for-paper-plan.md`:

- Scenario: `sparse` is the primary TMLR matrix. `very_sparse` is a reduced stress matrix
  (`cc`/`rnd`/`c_v2`, plain + full noisy-TV only).
- Seeds: **all methods/scenarios → seeds 1–3** (uniform; report IQM-style curves + bootstrap CIs). `c_v2` runs in the early wave alongside `cc`/`rnd`. Mini noise-α sweep (α=0.33, 0.66) for `cc`/`c_v2`/`rnd` only on sparse — see Phase 4; endpoints α=0 (plain, Phase 2) and α=1 (full noise, Phase 1) already covered.
- `--total-timesteps 30000000` (30M) per run.

Use `tmux` so jobs survive disconnects:

```bash
tmux new -s cc_vizdoom
```

### Phase 1 — Headline: noisy-TV (run this first)

The core result + presentation footage: `cc` vs `rnd` vs `c_v2` on noisy-TV
(`c_v2` promoted into the early wave — it's the closest competitor, so this is your
earliest read on whether the learned baseline beats the one-step baseline).

```bash
PROJECT=curiosity-critic-vizdoom-JUN2026

# Headline noisy-TV (full static, alpha=1 default): CC, RND, C_V2 -- seeds 1-3
for SEED in 1 2 3; do
  for METHOD in cc rnd c_v2; do
    python cleanrl/ppo_curiosity_critic_vizdoom.py \
      --method $METHOD --scenario sparse --noisy-tv \
      --total-timesteps 30000000 --seed $SEED \
      --capture-video --save-model \
      --track --wandb-project-name $PROJECT
  done
done
```

### Phase 2 — Deterministic: plain MyWayHome

Full method set, no `--noisy-tv`.

```bash
PROJECT=curiosity-critic-vizdoom-JUN2026

for SEED in 1 2 3; do
  for METHOD in cc rnd; do
    python cleanrl/ppo_curiosity_critic_vizdoom.py \
      --method $METHOD --scenario sparse \
      --total-timesteps 30000000 --seed $SEED \
      --capture-video --save-model \
      --track --wandb-project-name $PROJECT
  done
done

for SEED in 1 2 3; do
  for METHOD in c_v1 c_v2 ppo random; do
    python cleanrl/ppo_curiosity_critic_vizdoom.py \
      --method $METHOD --scenario sparse \
      --total-timesteps 30000000 --seed $SEED \
      --capture-video --save-model \
      --track --wandb-project-name $PROJECT
  done
done
```

### Phase 3 — Completion: finish noisy-TV baselines

```bash
PROJECT=curiosity-critic-vizdoom-JUN2026

for SEED in 1 2 3; do
  for METHOD in c_v1 ppo random; do
    python cleanrl/ppo_curiosity_critic_vizdoom.py \
      --method $METHOD --scenario sparse --noisy-tv \
      --total-timesteps 30000000 --seed $SEED \
      --capture-video --save-model \
      --track --wandb-project-name $PROJECT
  done
done
```

### Phase 4 — Noise-level sweep (intermediate α, for the noise curve)

`cc`, `c_v2`, `rnd` at intermediate noise blends. Endpoints are already covered:
**α=0 = Phase 2 (plain)**, **α=1 = Phase 1 (full noise)**. `--noise-alpha` blends the
patch: `obs = (1−α)·clean + α·noise` (α=1 = full static, the default).

```bash
PROJECT=curiosity-critic-vizdoom-JUN2026

for SEED in 1 2 3; do
  for ALPHA in 0.33 0.66; do
    for METHOD in cc c_v2 rnd; do
      python cleanrl/ppo_curiosity_critic_vizdoom.py \
        --method $METHOD --scenario sparse --noisy-tv --noise-alpha $ALPHA \
        --total-timesteps 30000000 --seed $SEED \
        --capture-video --save-model \
        --track --wandb-project-name $PROJECT
    done
  done
done
```

### Phase 5 — Very-sparse stress matrix

Reduced TMLR stress test: `cc`, `rnd`, and `c_v2` only, on `very_sparse`, in both plain and
full noisy-TV conditions. No intermediate α sweep here; full noise uses the default `--noise-alpha 1`.

```bash
PROJECT=curiosity-critic-vizdoom-JUN2026

# very_sparse plain
for SEED in 1 2 3; do
  for METHOD in cc rnd c_v2; do
    python cleanrl/ppo_curiosity_critic_vizdoom.py \
      --method $METHOD --scenario very_sparse \
      --total-timesteps 30000000 --seed $SEED \
      --capture-video --save-model \
      --track --wandb-project-name $PROJECT
  done
done

# very_sparse noisy-TV, full static alpha=1 default
for SEED in 1 2 3; do
  for METHOD in cc rnd c_v2; do
    python cleanrl/ppo_curiosity_critic_vizdoom.py \
      --method $METHOD --scenario very_sparse --noisy-tv \
      --total-timesteps 30000000 --seed $SEED \
      --capture-video --save-model \
      --track --wandb-project-name $PROJECT
  done
done
```

Detach from tmux with `Ctrl-b` then `d`; reattach with `tmux attach -t cc_vizdoom`.

---

## 4. What To Watch During Training

Primary:

- `charts/avg_episodic_return` — main learning curve. Compare at equal `global_step`.
- `charts/episodic_length` — shorter means the agent is finding the vest.

**W&B x-axis:** runs log scalars directly to W&B with `global_step` declared as the step metric.
`global_step` is total vectorized agent-environment interactions (`num_envs × env steps`), so a
30M run should span roughly 0 → 30,000,000 on the x-axis. If a custom W&B panel shows its internal
`Step` selector, switch the x-axis to `global_step`.

Curiosity diagnostics:

- `charts/curiosity_reward_mean` — normalized intrinsic reward.
- `losses/fwd_loss` — WM prediction MSE (`cc`,`c_v1`,`c_v2`) or RND predictor MSE (`rnd`).
- `losses/error_before` (WM methods) — raw WM error before update.
- `losses/error_after`, `losses/critic_loss`, `charts/critic_pred_mean` (`cc` only) —
  the critic estimate should track `error_after`, not collapse to 0 or explode.
- `charts/SPS` — report this since the curiosity methods carry extra networks.

Red flags:

- Return flat while `charts/curiosity_reward_mean` collapses to ~0 immediately.
- `losses/critic_loss` explodes / NaN.
- Only one seed succeeds while others regress.

---

## 5. Collecting Visuals For The Presentation

Every run writes to `runs/<run_name>/`:

- `viz/wm_panel_*.png` — WM predicted vs. true frame + error map (and critic baseline for `cc`).
- `viz/heatmap_*.png` — maze visitation + mean intrinsic-reward heatmaps with the
  TV-zone time fraction in the title.
- `videos/*.mp4` — gameplay clips.
- `map_vids/*.mp4` — top-down 2D trajectory videos for the same periodic policy rollouts.

The **headline visual** is the noisy-TV trap, built from Phase 1 runs:

1. Take the latest `videos/*.mp4` from a `rnd` (or `c_v1`) noisy-TV run and a `cc`
   noisy-TV run at the same `global_step` and place them side by side: RND/V1 fixates
   on the TV; Curiosity-Critic ignores it and reaches the vest.
2. Pair with the `heatmap_*.png`: RND/V1 concentrate visitation in the TV zone (high
   TV-zone fraction); Curiosity-Critic spreads toward the goal (low TV-zone fraction).
3. Add the `wm_panel_*.png` to show the mechanism: the WM error stays high in the noise
   panel, and the `cc` critic baseline `φ` learns to subtract exactly that region.

If `--track` is on, all three are also logged to W&B under `viz/` (and videos under
`viz/video`), so you can pull them straight from the run page.

### Noise-overlaid videos (post-hoc, for the talk)

The training-time `videos/*.mp4` do **not** show the noise patch: the noise is
overlaid on the agent's grayscale *observation* (which drives training), not on the
RGB buffer used to render the video. To produce presentation videos with the noisy
TV visible, regenerate them from the saved `--save-model` checkpoints with the
standalone tool — this does not touch the training code and needs no re-runs:

```bash
# one run:
python cleanrl/regenerate_vizdoom_video.py \
    --checkpoint runs/<run_name>/ppo_curiosity_critic_vizdoom.cleanrl_model

# batch over every noisy Phase-1 run:
for f in runs/vizdoom_sparse_noisytv__*/ppo_curiosity_critic_vizdoom.cleanrl_model; do
    python cleanrl/regenerate_vizdoom_video.py --checkpoint "$f"
done
```

Each invocation writes **two** mp4s next to the checkpoint:
- `<checkpoint>_noisyTV.mp4` — pretty full-res RGB with the noise patch re-overlaid (a
  faithful *reconstruction* of what the agent experienced).
- `<checkpoint>_obs.mp4` — the agent's **actual observation**: the exact grayscale pixels
  it saw (including the real noise it acted on), nearest-neighbor upscaled (`--obs-scale`).

Useful flags: `--greedy` (argmax actions for a cleaner deterministic clip), `--steps`
(rollout length), `--seed`, `--out`, `--obs-scale`. The script replays the run's own
saved config and prints the TV-zone fraction (≈0.1 for CC, ≈1.0 for RND) as a sanity
check. These replay the final policy, so they are the end-of-training "money shot" — CC
reaching the vest vs. RND stuck on the static, now with the static on screen.

---

## 5b. What Gets Saved Each Run (so you never have to rerun)

Every run writes to `runs/<run_name>/`:
- `metrics.jsonl` — every logged scalar per update (losses, SPS, eval, timing, episodic), for easy post-hoc plotting without parsing tfevents.
- `run_meta.json` — args + git commit + device (reproducibility).
- `checkpoints/ckpt_update*.cleanrl_model` — full models (policy + world model + critic/RND + obs/reward stats) every `--ckpt-every` updates, so **any training instant is reconstructable**.
- `viz/wm_panel_*.png`, `viz/heatmap_*.png` (maze-overlay visitation / intrinsic maps), and
  `viz/positions_*.npz` (raw x/y/intrinsic/TV-zone so heatmaps/coverage can be re-rendered later).
- `videos/update*.mp4` (RGB) **and** `update*_obs.mp4` (the agent's exact grayscale observation, noise included).
- `map_vids/update*.mp4` — top-down WAD-map trajectory videos generated from the same rollout as
  `videos/update*.mp4`.
- `plots/*.png` + `plots/summary_final_metrics.csv` — automatic per-run plots written at normal
  training exit. Disable with `--no-post-plot`; set `--post-plot-dir` to redirect them.

New metrics (logged for **all** methods):
- **`eval/wm_holdout_l2`** — held-out world-model accuracy on a fixed deterministic transition set (cached per seed in `./vizdoom_holdout/`, identical across methods of a seed). **Every method trains a world model** (passive for `rnd`/`ppo`/`random`), so this number is directly comparable — it's the measurement of the "better / faster world model" claim. Cadence: `--eval-every`.
- **`mechanism/tv_zone_fraction`** — fraction of rollout samples inside the noisy-TV zone.
- **`mechanism/intrinsic_tv_mean_raw`**, **`mechanism/intrinsic_non_tv_mean_raw`** — raw intrinsic
  reward inside vs. outside the TV zone, before reward normalization.
- **`charts/goal_hits_update`**, **`charts/goal_reached_rate_update`**,
  **`charts/goal_reached_rate_100ep`** — vest/goal success signals. In sparse / very-sparse
  MyWayHome, positive reward is treated as vest collection.
- **`time/*`** — per-update timing breakdown: `rollout_s`, `update_s`, `reward_aux_s` (aux model during reward), `wm_update_s`, `aux_update_s` (critic backprop for `cc`), `aux_total_s`, `eval_s`. `--profile-timing` (default on) uses cuda syncs for accurate GPU component timing.
- **`charts/*_periodic`** — episodic return/length logged every update (dense, regular), so methods that rarely finish episodes still span the full x-axis (fixes the wandb "short line" look). The original per-episode `charts/*` are untouched.

**Disk note:** periodic full checkpoints are ~25–30 MB each; at `--ckpt-every 200` that's ~1 GB per 30M run (~50 GB across the full matrix). Raise `--ckpt-every` (e.g. 500) or set it to 0 if disk-constrained.

## 5c. Generate Paper Figures

Each training job writes quick single-run plots to `runs/<run_name>/plots/` automatically at normal
exit. Run this aggregate pass after any job or phase finishes; it auto-discovers completed
`runs/*/metrics.jsonl` files:

```bash
python cleanrl/plot_vizdoom_curiosity.py --runs-dir runs --out paper_figures/vizdoom
```

The script groups runs by scenario / condition / method / seed and writes:
- IQM-style learning curves with bootstrap CIs for return, goal rate, held-out WM error, TV-zone
  fraction, intrinsic reward in/out of TV, and SPS.
- `paper_figures/vizdoom/summary_final_metrics.csv` with final IQM / CI / mean / std per group.

It has no `rliable` dependency; IQM and bootstrap CIs are implemented directly with NumPy so this
can run immediately on the training box.

By default it filters out obvious short-planned smoke/diagnostic jobs once full-planned runs are
present, and keeps only the longest run for each scenario / condition / method / seed. If you are
intentionally plotting only short diagnostics, use a separate output folder as in the smoke step; if
you need to disable this filter, pass `--min-planned-frac 0`.

## 6. How To Judge The Winner

- **Noisy-TV (Phase 1):** Curiosity-Critic should keep a low TV-zone visitation
  fraction and higher return/faster vest-finding than RND and C_V2; V1 is checked in
  the completion wave as the zero-baseline ablation.
  This is the paper's claim-1 result.
- **Plain (Phase 2):** Curiosity-Critic should match or beat RND/V1/V2 on return and
  episode length. This is the deterministic case.
- Report mean ± std over seeds; the win must not hinge on a single seed. Report
  `charts/SPS` / wall-clock since the curiosity methods train extra networks.

---

## 7. Useful One-Offs

Single quick local run without W&B:

```bash
python cleanrl/ppo_curiosity_critic_vizdoom.py --method cc --scenario sparse --noisy-tv \
    --total-timesteps 2000000 --num-envs 16 --seed 1
```

Very-sparse one-off sanity check:

```bash
python cleanrl/ppo_curiosity_critic_vizdoom.py --method cc --scenario very_sparse \
    --total-timesteps 30000000 --seed 1 --track --wandb-project-name curiosity-critic-vizdoom-JUN2026
```

Notes:

- `--num-envs` defaults to 32; set it near your CPU core count for best throughput.
- ICM and Disagreement are de-scoped for the TMLR version; this script intentionally
  focuses on RND, PPO/random floors, and the V1/V2 ablations.
