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
| ICM | ✓ | ✓ |
| **Curiosity-Critic (ours)** | ✓ | ✓ |
| Disagreement | — | ✓ |

V1 and V2 reuse the Curiosity-Critic world model and serve as the zero-baseline and one-step-baseline ablations of our framework.

---

## 4. Seeds

- **All methods: 5 seeds** (uniform — cleaner for TMLR, no awkward per-method seed counts to justify).

---

## 5. Execution order

**Phase 0 — Harness and smoke test.**
Build the VizDoom training harness, run a short job, and record steps-per-second and the per-run frame budget.

**Phase 1 — Headline (noisy-TV).**
Run {Curiosity-Critic, RND, ICM, V1} on the noisy-TV condition, 5 seeds each. This produces the core result and the presentation footage.

**Phase 2 — Deterministic (plain).**
Run the full method set on the plain condition.

**Phase 3 — Completion.**
Run V2 (both conditions) and Disagreement (noisy-TV). Finalize all seed counts per Section 4.

---

## 6. Compute

- Each run: **30M frames**, ~8–12 hours on the 3090 Ti.
- Total matrix: ~51 runs, ~3.5 weeks of serial GPU time.
- Each run stays at or under the 1-day-per-seed-per-method budget.

---

## 7. Metrics and deliverables

Produced for every run, used in both the paper and the company presentation:

- Episodic return vs. environment frames (primary).
- Maze goal-reached rate.
- Intrinsic reward on the noise region vs. the rest of the maze (mechanism plot).
- Curiosity-Critic baseline estimate over training.
- Side-by-side gameplay video: RND/ICM agent fixating on the noisy TV vs. Curiosity-Critic agent ignoring it and reaching the goal.
- World-model panel: predicted next frame, actual next frame, error map, critic baseline.

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
5. Implement the baselines behind a single `--method` flag: PPO (no intrinsic), Curiosity V1, Curiosity V2, RND, ICM, Curiosity-Critic. Disagreement is added last, after the others are validated.
6. Implement logging and presentation hooks: TensorBoard + wandb scalars; episode video capture; world-model image panels (predicted / actual / error / baseline); maze visitation and intrinsic-reward heatmaps from the POSITION variables; per-region intrinsic-reward curves (TV panel vs. rest).
7. Phase 0 smoke test on the Linux box: confirm steps-per-second and the 30M-frame run budget.
8. Run Phases 1–3 per Section 5.

---

## 9. Setup

- Development on the Mac.
- All training on the Linux box with the RTX 3090 Ti.
- Dependencies: the Farama-Foundation `vizdoom` package (Gymnasium API) plus the `my_way_home` sparse and very-sparse scenario files from `noreward-rl`.
