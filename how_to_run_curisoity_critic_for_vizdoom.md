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
  is designed to escape.

---

## 1. One-Time Setup

Install ViZDoom system dependencies and the Python packages:

```bash
sudo apt update
sudo apt install -y cmake git libboost-all-dev libsdl2-dev libopenal-dev tmux

# in your cleanrl venv
pip install vizdoom opencv-python "imageio[ffmpeg]" matplotlib
```

The training stack (`torch`, `gymnasium>=1.0`, `tyro`, `tensorboard`, `wandb`) is
already installed for the existing cleanrl scripts.

Confirm the scenario wads are present (they ship in the cloned `noreward-rl` repo,
which is the default `--wad-dir`):

```bash
ls noreward-rl/doomFiles/wads/
# my_way_home_sparse.wad  my_way_home_verySparse.wad  my_way_home_dense.wad
```

(Optional) log in to Weights & Biases:

```bash
wandb login
```

---

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

- **`SPS:` printed** and steady. Multiply by 86,400 to estimate frames/day; a 30M-frame
  run should fit in well under a day. If SPS is low, raise `--num-envs` toward your
  CPU core count.
- **`runs/<run_name>/viz/wm_panel_*.png`** shows input / WM prediction / true next / abs error.
- **`runs/<run_name>/viz/heatmap_*.png`** shows visitation and the printed `TV-zone time fraction`.
- **`runs/<run_name>/videos/*.mp4`** plays the agent in the maze.

Map-name check: if VizDoom errors on the map, list it with
`python -c "import vizdoom,os; g=vizdoom.DoomGame(); g.set_doom_scenario_path('noreward-rl/doomFiles/wads/my_way_home_sparse.wad'); print('ok')"`
and pass the correct `--doom-map` (default `map01`).

Noisy-TV tuning: open a heatmap/video. The TV zone should cover roughly the
starting room. If it is too small/large, adjust `--tv-radius` (game units; default
`100`) and re-run the smoke test.

---

## 3. The Experiment Matrix

Per `next-steps-for-paper-plan.md`:

- Scenario: `sparse` (primary). `very_sparse` optional for extra exploration stress.
- Seeds: **`cc` and `rnd` → seeds 1–5**; **`c_v1`, `c_v2`, `ppo`, `random` → seeds 1–3**.
- `--total-timesteps 30000000` (30M) per run.

Use `tmux` so jobs survive disconnects:

```bash
tmux new -s cc_vizdoom
```

### Phase 1 — Headline: noisy-TV (run this first)

The core result + presentation footage: `cc` vs `rnd` vs `c_v1` on noisy-TV.

```bash
PROJECT=curiosity-critic-vizdoom

# Curiosity-Critic and RND: seeds 1-5
for SEED in 1 2 3 4 5; do
  for METHOD in cc rnd; do
    python cleanrl/ppo_curiosity_critic_vizdoom.py \
      --method $METHOD --scenario sparse --noisy-tv \
      --total-timesteps 30000000 --seed $SEED \
      --capture-video --save-model \
      --track --wandb-project-name $PROJECT
  done
done

# Curiosity V1 (special case, zero baseline): seeds 1-3
for SEED in 1 2 3; do
  python cleanrl/ppo_curiosity_critic_vizdoom.py \
    --method c_v1 --scenario sparse --noisy-tv \
    --total-timesteps 30000000 --seed $SEED \
    --capture-video --save-model \
    --track --wandb-project-name $PROJECT
done
```

### Phase 2 — Deterministic: plain MyWayHome

Full method set, no `--noisy-tv`.

```bash
PROJECT=curiosity-critic-vizdoom

for SEED in 1 2 3 4 5; do
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
PROJECT=curiosity-critic-vizdoom

for SEED in 1 2 3; do
  for METHOD in c_v2 ppo random; do
    python cleanrl/ppo_curiosity_critic_vizdoom.py \
      --method $METHOD --scenario sparse --noisy-tv \
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

---

## 6. How To Judge The Winner

- **Noisy-TV (Phase 1):** Curiosity-Critic should keep a low TV-zone visitation
  fraction and higher return/faster vest-finding than RND and V1, which get trapped.
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

Very-sparse stress test:

```bash
python cleanrl/ppo_curiosity_critic_vizdoom.py --method cc --scenario very_sparse \
    --total-timesteps 30000000 --seed 1 --track --wandb-project-name curiosity-critic-vizdoom
```

Notes:

- `--num-envs` defaults to 32; set it near your CPU core count for best throughput.
- ICM and Disagreement baselines are intentionally not in this script yet; they are
  the planned later additions.
