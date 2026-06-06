# Steps For Vin — VizDoom Curiosity-Critic Runbook

Your single source of truth for getting from "code on a branch" to "results +
presentation visuals." Work entirely on branch **`vin/cc-vizdoom-preflight`**.

## Which to run first: preflight or master's smoke test?

**Run THIS branch's `--preflight` first.** Do not bother with master's smoke test.

Reasons:
- This branch contains *everything master has* (the training script + the
  how-to) **plus** the `--preflight` self-test. There is nothing on master you
  need that isn't here.
- Preflight is ~30 seconds and isolates the env-integration unknowns (map name,
  screen format, reward, tv-radius, API drift, deps) **without** spinning up 32
  processes or the training loop. If something is wrong, you get a clean, fast
  error instead of a confusing failure deep inside training.
- The smoke test (minutes, full training loop + viz) is the *second* gate, run
  only after preflight is green.

Order: **preflight → smoke test → sparse Phases 1–4 → very-sparse stress Phase 5.**

---

## Step 0 — Get the branch onto the box

From your **Mac** (where the preflight edits live, uncommitted). Stage only the
VizDoom work explicitly — `git add -A` would also pull in `paper/` (LaTeX +
build artifacts) and is avoided on purpose:

```bash
git add .gitignore \
        cleanrl/build_holdout.py \
        cleanrl/plot_vizdoom_curiosity.py \
        cleanrl/ppo_curiosity_critic_vizdoom.py \
        cleanrl/regenerate_vizdoom_video.py \
        how_to_run_curisoity_critic_for_vizdoom.md \
        next-steps-for-paper-plan.md \
        preflight_readme.md \
        steps-for-vin.md \
        vizdoom_scenarios/
git commit -m "Add VizDoom preflight self-test, bundle MyWayHome scenarios, docs"
git push -u origin vin/cc-vizdoom-preflight
```

The upstream clones (`noreward-rl/`, `large-scale-curiosity/`) are gitignored so
they won't be committed. `paper/` is intentionally left out — handle it
separately if you want it tracked here.

On the **Linux 3090 box**:

```bash
git fetch origin
git checkout vin/cc-vizdoom-preflight
git pull
```

## Step 1 — Install dependencies (Linux box, once)

```bash
sudo apt update
sudo apt install -y cmake git libboost-all-dev libsdl2-dev libopenal-dev tmux
# in your cleanrl venv:
pip install vizdoom opencv-python matplotlib
ls vizdoom_scenarios/   # expect the 3 my_way_home_*.wad files (bundled in the repo)
```

## Step 1.5 — Build the held-out WM-eval sets (run once)

```bash
python cleanrl/build_holdout.py --scenario sparse --seeds 1 2 3 --size 10000
python cleanrl/build_holdout.py --scenario very_sparse --seeds 1 2 3 --size 10000
```

Writes `vizdoom_holdout/holdout_{sparse,very_sparse}_seed{1,2,3}.npz` + a **coverage heatmap
overlaid on the top-down maze map** per seed/scenario. Open the 6 `*_coverage.png` files and confirm
samples span rooms and corridors (not just spawn / wall-bump loops). All methods of a seed score
their world model on the matching-seed/scenario set, so it must be built before the runs.

## Step 2 — PREFLIGHT (the fast gate)  ← run this first

```bash
python cleanrl/ppo_curiosity_critic_vizdoom.py --preflight --method cc --scenario sparse --noisy-tv
python cleanrl/ppo_curiosity_critic_vizdoom.py --preflight --method rnd --scenario sparse
```

- Both must end with `PREFLIGHT PASSED`.
- If anything fails, see `preflight_readme.md` (map name, tv-radius, deps fixes).
- `--tv-radius` default is **150**, calibrated for MyWayHome sparse via
  `python cleanrl/ppo_curiosity_critic_vizdoom.py --probe-maze --scenario sparse`
  (maze max reach ~516). Verify the trap behaviorally from the first noisy-TV
  runs' heatmaps (V1/RND should dwell in the TV zone; CC should leave) and adjust
  only if needed.

**Gate: do not proceed until both preflights pass.**

## Step 3 — Phase-0 smoke test (the second gate)

```bash
python cleanrl/ppo_curiosity_critic_vizdoom.py \
    --method cc --scenario sparse --noisy-tv \
    --total-timesteps 200000 --num-envs 8 \
    --capture-video --video-every 1 --wm-panel-every 5 --heatmap-every 5 \
    --save-model --ckpt-every 50 --seed 1
```

Confirm in `runs/<run_name>/`:
- a healthy, steady `SPS:`. This visual smoke test uses `--video-every 1`, so SPS will be
  much lower than full-run training throughput; use it mainly to catch hangs and artifact bugs.
- `viz/wm_panel_*.png`, maze-overlaid `viz/heatmap_*.png`, `videos/*.mp4`, and
  `map_vids/*.mp4` are produced and look sane. The top-down `map_vids/` video should show
  a red `vest` cross in the fixed goal room.
- `checkpoints/ckpt_update000050.cleanrl_model` exists, plus the final
  `ppo_curiosity_critic_vizdoom.cleanrl_model` when the smoke run exits.
- if `--track` is on, W&B charts should use `global_step` as the x-axis. That is total
  vectorized env interactions (`num_envs × env steps`), so a 30M run should span roughly
  0 → 30,000,000 rather than update/logging count.
- at normal completion, `runs/<run_name>/plots/` is created automatically from that run's
  `metrics.jsonl` for quick per-run inspection.

Then test the post-run analysis loop too:

```bash
python cleanrl/plot_vizdoom_curiosity.py --runs-dir runs --out paper_figures/vizdoom_smoke
```

Confirm it writes PNG plots plus `paper_figures/vizdoom_smoke/summary_final_metrics.csv`.

If you want a cleaner throughput estimate after the visual smoke passes, run a short no-viz probe
with `--video-every 0 --wm-panel-every 0 --heatmap-every 0`; then `SPS × 86,400` is the rough
agent-steps/day estimate. If that SPS is low, raise `--num-envs` toward your CPU core count and
re-check.

**Gate: do not launch the full matrix until the smoke test looks right.**

## Step 4 — Full runs (Phases 1–5)

Follow `how_to_run_curisoity_critic_for_vizdoom.md` §3, in `tmux`:
- **Phase 1 (headline):** noisy-TV — `cc`, `rnd`, `c_v2` (3 seeds). `c_v2` is promoted here (closest competitor → early signal on whether the learned baseline is the real win).
- **Phase 2:** plain MyWayHome — full method set.
- **Phase 3:** finish noisy-TV baselines (`c_v1`, `ppo`, `random`).
- **Phase 4:** sparse noisy-TV mini noise-α sweep (`cc`, `c_v2`, `rnd` only; α=0.33, 0.66).
- **Phase 5:** very-sparse stress matrix — `cc`, `rnd`, `c_v2` only, plain + full noisy-TV.

Seeds: all methods/scenarios → 1–3 (uniform; report IQM-style curves + bootstrap CIs). All at
`--total-timesteps 30000000`, with `--track --save-model --capture-video`. Sparse remains the
primary TMLR matrix; very-sparse is the stress-test appendix/secondary result.

## Step 5 — Generate paper figures (rerun after each phase)

```bash
python cleanrl/plot_vizdoom_curiosity.py --runs-dir runs --out paper_figures/vizdoom
```

Each individual job also writes quick single-run plots to `runs/<run_name>/plots/` at normal exit.
This command is the aggregate paper-figure pass: it reads every completed `runs/*/metrics.jsonl`,
groups by scenario / condition / method / seed, and writes IQM-style curves with bootstrap CIs plus
`summary_final_metrics.csv`. Rerun it whenever a job or phase finishes; it is incremental and will
use whatever completed runs are present. It also filters obvious short-planned smoke runs once
full-planned runs exist, and keeps only the longest run for each scenario / condition / method / seed.

Key plots to inspect:
- sparse noisy-TV: return, goal-rate, TV-zone fraction, held-out WM error,
- sparse plain: return and held-out WM error,
- sparse noise-α sweep: the α-specific noisy-TV curves,
- very-sparse stress: `cc`/`rnd`/`c_v2` plain + full noisy-TV.

## Step 6 — Collect the presentation visuals

From the Phase-1 noisy-TV runs (see how-to §5):
1. side-by-side video: `rnd`/`c_v1` fixating on the TV vs. `cc` reaching the vest,
2. visitation heatmaps (high vs. low TV-zone fraction),
3. WM-prediction panel showing the critic baseline subtracting the noise region.

All also appear in W&B under `viz/` when `--track` is on.

**Every run is now fully instrumented (so you never have to rerun):** `metrics.jsonl`
(all scalars/update), periodic full `checkpoints/`, held-out world-model accuracy
(`eval/wm_holdout_l2`, comparable across all methods since every method trains a WM),
`time/*` compute breakdowns, `charts/*_periodic` dense return plots, `viz/positions_*.npz`
raw heatmap data, RGB + exact-observation videos, and top-down `map_vids/` trajectory
videos, plus automatic per-run `plots/`. See how-to §5b. Disk: periodic checkpoints ≈ 1 GB / 30M run; raise `--ckpt-every`
if tight.

**Note — training-time `videos/*.mp4` do NOT show the noise patch.** The noise is
overlaid on the agent's grayscale *observation* (which drives training), not on the
RGB buffer used for the video. To get presentation videos with the noisy TV visible,
regenerate them post-hoc from the saved checkpoints (no re-runs, training untouched):

```bash
# one run:
python cleanrl/regenerate_vizdoom_video.py \
  --checkpoint runs/<run_name>/ppo_curiosity_critic_vizdoom.cleanrl_model
# batch every noisy Phase-1 run:
for f in runs/vizdoom_sparse_noisytv__*/ppo_curiosity_critic_vizdoom.cleanrl_model; do
  python cleanrl/regenerate_vizdoom_video.py --checkpoint "$f"; done
```

Each writes two mp4s: `<checkpoint>_noisyTV.mp4` (pretty RGB reconstruction) and
`<checkpoint>_obs.mp4` (the agent's exact observation — the real grayscale pixels it
saw, upscaled). Add `--greedy` for a cleaner deterministic clip. The script prints the
TV-zone fraction (≈0.1 for CC, ≈1.0 for RND) as a check.

---

## Decision gates at a glance

```
preflight (cc+rnd) PASS ──► smoke + plot loop sane ──► sparse Phases 1–4 ──► very-sparse Phase 5 ──► figures/visuals
        │                        │
        └─ fix per preflight_readme.md
                                 └─ raise --num-envs / fix viz, re-run
```
