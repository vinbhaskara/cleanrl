# Curiosity-Critic — Next Steps for Paper

**Target venue:** TMLR.

**Goal of this phase:** extend the Curiosity-Critic paper from the controlled grid world to one real pixel-based environment, demonstrating (1) robustness to irreducible aleatoric noise and (2) effective exploration in the deterministic case.

---

## 1. Paper structure

- **Controlled section (kept):** the existing 30×30 noisy-TV grid-world study, including the oracle critic and noise-floor / critic-convergence plots. This carries the mechanism evidence.
- **Real-environment section (new):** VizDoom `MyWayHome`, run in a plain (deterministic) condition and a noisy-TV condition.
- **Limitations:** state that Curiosity-Critic requires recurring transitions. Environments that regenerate every episode (e.g. Procgen) conflate irreducible noise with informative novelty and are out of scope.

---

## 2. Environment and conditions

Base task: **VizDoom `MyWayHome`** (sparse-reward 3D navigation maze), served through the official Farama-Foundation ViZDoom Gymnasium API, using the `my_way_home` scenario files reused from Pathak's `noreward-rl` repo.

Two conditions on the same maze:

1. **Plain** — fully deterministic maze (no sticky actions), no added noise.
2. **Noisy-TV** — the same deterministic maze with an added on-screen panel of i.i.d. random noise re-sampled every step (a visitable region, Burda-style). All aleatoric noise lives in this panel. The noise-region mask is recorded for the mechanism metric.

Observation format matches the existing harness: 84×84 grayscale, 4-frame stack, frameskip 4. The world-model and neural-critic architectures transfer unchanged.

---

## 3. Methods

| Method | Plain | Noisy-TV |
|---|---|---|
| PPO (no intrinsic reward) | ✓ | ✓ |
| Curiosity V1 (raw WM error) | ✓ | ✓ |
| Curiosity V2 (one-step improvement) | ✓ | ✓ |
| RND | ✓ | ✓ |
| **Curiosity-Critic (ours)** | ✓ | ✓ |

V1 and V2 reuse the Curiosity-Critic world model and serve as the zero-baseline and one-step-baseline ablations of our framework.
ICM and Disagreement are de-scoped for the TMLR version; the VizDoom section focuses on RND,
PPO/random floors, and the V1/V2 ablations that isolate the Curiosity-Critic baseline.

---

## 4. Seeds

- **All methods: 3 seeds** (uniform; de-scoped from 5 to keep the matrix tractable). Report IQM-style curves + bootstrap CIs so 3 seeds stays credible.
- **C_V2 is in the early/main wave alongside CC and RND** (it's the closest competitor — the one-step-baseline special case — so an early read on it tells us whether the learned baseline is the real win).
- **Mini noise-α sweep:** intermediate noise levels **α ∈ {0.33, 0.66}** for **CC, C_V2, RND only**, 3 seeds. Endpoints are already covered: α=0 = the plain/deterministic runs, α=1 = the full-noise runs. Noise model = convex blend `obs[patch] = (1−α)·clean + α·noise` (resampled each step; α=1 reproduces the current full-noise patch).
- **Very-sparse stress matrix:** `very_sparse` scenario for **CC, C_V2, RND only**, 3 seeds,
  in plain and full noisy-TV (α=1) conditions. No intermediate α sweep on very-sparse.

---

## 5. Execution order

**Phase 0 — Harness and smoke test.**
Build the VizDoom training harness, run a short job, and record steps-per-second and the per-run step budget.

**Phase 1 — Headline (noisy-TV).**
Run {Curiosity-Critic, RND, Curiosity V2} on the noisy-TV condition first (C_V2 promoted — closest competitor, early signal), 3 seeds each. This produces the core result and the presentation footage.

**Phase 2 — Deterministic (plain).**
Run the full method set on the plain condition.

**Phase 3 — Completion.**
Finish the sparse noisy-TV baselines (`c_v1`, PPO, random). Finalize all seed counts per Section 4.

**Phase 4 — Sparse noise-level sweep.**
Run intermediate noise levels α ∈ {0.33, 0.66} for CC, C_V2, and RND only.

**Phase 5 — Very-sparse stress test.**
Run CC, C_V2, and RND on `very_sparse`, in plain and full noisy-TV conditions.

---

## 6. Compute

- Each run: **30M agent-environment steps** (~120M repeated Doom tics with `frame_skip=4`), ~8–12 hours on the 3090 Ti.
- Sparse primary matrix: 54 runs. Very-sparse stress matrix: 18 additional runs.
- Total TMLR matrix: 72 runs, ~3.5-5 weeks of serial GPU time depending on measured SPS.
- Each run stays at or under the 1-day-per-seed-per-method budget.

---

## 7. Metrics and deliverables

Produced for every run, used in both the paper and the company presentation:

- Episodic return vs. agent-environment steps (primary).
- Maze goal-reached rate (`charts/goal_reached_rate_100ep`; positive reward = vest collection).
- TV-zone visitation fraction (`mechanism/tv_zone_fraction`).
- Intrinsic reward on the noise region vs. the rest of the maze
  (`mechanism/intrinsic_tv_mean_raw` vs. `mechanism/intrinsic_non_tv_mean_raw`).
- Held-out clean world-model error (`eval/wm_holdout_l2`) on fixed deterministic transition sets.
- Curiosity-Critic baseline estimate over training.
- Side-by-side gameplay video: RND agent fixating on the noisy TV vs. Curiosity-Critic agent ignoring it and reaching the goal.
- World-model panel: predicted next frame, actual next frame, error map, critic baseline.
- Paper figures generated from `metrics.jsonl` via `cleanrl/plot_vizdoom_curiosity.py`
  (IQM-style curves + bootstrap CIs, plus final-metric CSV summaries).

---

## 8. Implementation stack and engineering tasks

**Stack:**

- Algorithm and training: PyTorch, forked from cleanRL `ppo_curiosity_critic_envpool.py` and `ppo_rnd_envpool.py`.
- Environment backend: official Farama-Foundation ViZDoom via its Gymnasium API.
- Scenarios: `my_way_home` sparse and very-sparse `.wad`/`.cfg` files reused from Pathak's `noreward-rl` repo.
- Vectorization: Gymnasium `AsyncVectorEnv`, 16–32 worker processes.
- Preprocessing: 84×84 grayscale, 4-frame stack, frameskip 4 — identical to the Atari harness, so the world-model and neural-critic architectures transfer unchanged.
- The TensorFlow curiosity codebases (`large-scale-curiosity`, `noreward-rl`) are not ported; only their scenario files are reused.

**Tasks, in order:**

1. Build the ViZDoom Gymnasium env wrapper: load `my_way_home`, enable `POSITION_X/Y/ANGLE` game variables, discrete action set (turn left, turn right, move forward), frameskip 4, 84×84 grayscale output, 4-frame stack. Wrap in `AsyncVectorEnv`.
2. Add the very-sparse scenario as a selectable variant.
3. Implement the noisy-TV observation wrapper: a fixed rectangular panel re-sampling i.i.d. noise every step, with fixed documented geometry and intensity, exposing the noise-region mask for logging.
4. Fork `cleanrl/ppo_curiosity_critic_vizdoom.py` from the existing Curiosity-Critic script; swap in the vectorized ViZDoom env; keep the world model, neural critic, PPO, GAE, and reward pipeline unchanged.
5. Implement the baselines behind a single `--method` flag: PPO (no intrinsic), Curiosity V1, Curiosity V2, RND, Curiosity-Critic, plus random exploration floor.
6. Implement logging and presentation hooks: TensorBoard + wandb scalars; episode video capture; world-model image panels (predicted / actual / error / baseline); maze visitation and intrinsic-reward heatmaps from the POSITION variables; per-region intrinsic-reward curves (TV panel vs. rest).
7. Phase 0 smoke test on the Linux box: confirm steps-per-second and the 30M-step run budget.
8. Run Phases 1–5 per Section 5.

---

## 9. Setup

- Development on the Mac.
- All training on the Linux box with the RTX 3090 Ti.
- Dependencies: the Farama-Foundation `vizdoom` package (Gymnasium API) plus the `my_way_home` sparse and very-sparse scenario files from `noreward-rl`.
