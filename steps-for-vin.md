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

Order: **preflight → smoke test → Phases 1–3.**

---

## Step 0 — Get the branch onto the box

From your **Mac** (where the preflight edits live, uncommitted):

```bash
git add -A
git commit -m "Add VizDoom preflight self-test + docs"
git push -u origin vin/cc-vizdoom-preflight
```

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
pip install vizdoom opencv-python "imageio[ffmpeg]" matplotlib
ls noreward-rl/doomFiles/wads/   # expect the 3 my_way_home_*.wad files
```

## Step 2 — PREFLIGHT (the fast gate)  ← run this first

```bash
python cleanrl/ppo_curiosity_critic_vizdoom.py --preflight --method cc --scenario sparse --noisy-tv
python cleanrl/ppo_curiosity_critic_vizdoom.py --preflight --method rnd --scenario sparse
```

- Both must end with `PREFLIGHT PASSED`.
- If anything fails, see `preflight_readme.md` (map name, tv-radius, deps fixes).
- Tune `--tv-radius` here if the `[tv]` line warns; whatever value you settle on,
  reuse it in every later command.

**Gate: do not proceed until both preflights pass.**

## Step 3 — Phase-0 smoke test (the second gate)

```bash
python cleanrl/ppo_curiosity_critic_vizdoom.py \
    --method cc --scenario sparse --noisy-tv \
    --total-timesteps 200000 --num-envs 8 \
    --capture-video --video-every 1 --wm-panel-every 5 --heatmap-every 5 --seed 1
```

Confirm in `runs/<run_name>/`:
- a healthy, steady `SPS:` (× 86,400 ≈ frames/day; a 30M run should fit under a day),
- `viz/wm_panel_*.png`, `viz/heatmap_*.png`, and `videos/*.mp4` are produced and look sane.

If SPS is low, raise `--num-envs` toward your CPU core count and re-check.

**Gate: do not launch the full matrix until the smoke test looks right.**

## Step 4 — Full runs (Phases 1–3)

Follow `how_to_run_curisoity_critic_for_vizdoom.md` §3, in `tmux`:
- **Phase 1 (headline):** noisy-TV — `cc`, `rnd` (seeds 1–5) and `c_v1` (seeds 1–3).
- **Phase 2:** plain MyWayHome — full method set.
- **Phase 3:** finish noisy-TV baselines (`c_v2`, `ppo`, `random`).

Seeds: `cc`/`rnd` → 1–5; `c_v1`/`c_v2`/`ppo`/`random` → 1–3. All at
`--total-timesteps 30000000`, with `--track --save-model --capture-video`.

## Step 5 — Collect the presentation visuals

From the Phase-1 noisy-TV runs (see how-to §5):
1. side-by-side video: `rnd`/`c_v1` fixating on the TV vs. `cc` reaching the vest,
2. visitation heatmaps (high vs. low TV-zone fraction),
3. WM-prediction panel showing the critic baseline subtracting the noise region.

All also appear in W&B under `viz/` when `--track` is on.

---

## Decision gates at a glance

```
preflight (cc+rnd) PASS ──► smoke test sane ──► Phase 1 ──► Phases 2–3 ──► visuals
        │                        │
        └─ fix per preflight_readme.md
                                 └─ raise --num-envs / fix viz, re-run
```
