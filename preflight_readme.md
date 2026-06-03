# Preflight Self-Test

`--preflight` is a ~30-second check that validates the VizDoom integration on
**real data** before you spend GPU-days on the full training matrix. It builds
ONE real VizDoom env (no multiprocessing, no training loop), steps random
actions, and runs a forward pass of the chosen method's networks on the
collected observations.

It exists because the env layer cannot be tested on the Mac (VizDoom is
Linux-only), so the first real validation happens here on the box.

## Run it

```bash
# Curiosity-Critic, noisy-TV condition (covers World Model + Neural Critic + the TV zone)
python cleanrl/ppo_curiosity_critic_vizdoom.py --preflight --method cc --scenario sparse --noisy-tv

# RND (covers the RND predictor/target path)
python cleanrl/ppo_curiosity_critic_vizdoom.py --preflight --method rnd --scenario sparse
```

Running both covers every auxiliary network and both conditions. `c_v1`/`c_v2`
reuse the same World Model that the `cc` preflight already exercises.

## What a PASS looks like

```
========================================================================
PREFLIGHT  method=cc  scenario=sparse  noisy_tv=True
========================================================================
[ok] torch device = cuda
[ok] import vizdoom
[ok] import cv2
[ok] import matplotlib (visualization)
[ok] import imageio (visualization)
[env] building VizDoomEnv (map=map01, wad_dir=./noreward-rl/doomFiles/wads) ...
[ok] obs (4, 84, 84) uint8; n_actions=3; rgb (120, 160, 3)
[ok] start position=(...)
[ok] stepped 20 actions; reward range=[...]; episode-end seen=False
[tv] in_tv_zone fraction = 0.65 (tv_radius=100.0, tv_panel=42)
[nets] forward pass for method=cc on real obs ...
[ok] agent: action=2 value_ext=... value_int=...
[ok] world model: output (1, 1, 84, 84)
[ok] neural critic: output (1, 1) baseline=...
========================================================================
PREFLIGHT PASSED  -  env, obs pipeline, and nets all work on real VizDoom data.
========================================================================
```

If you see `PREFLIGHT PASSED`, the integration is sound — proceed to the Phase-0
smoke test.

## Common failures and exact fixes

| Symptom | Cause | Fix |
|---|---|---|
| `ModuleNotFoundError: vizdoom` / `cv2` | deps not installed | `pip install vizdoom opencv-python` (see how-to §1) |
| Error during "building VizDoomEnv" mentioning the map | the map lump in the wad is not `map01` | inspect the wad / try `--doom-map map00` etc.; pass the right `--doom-map` |
| `FileNotFoundError: scenario wad not found` | wrong `--wad-dir` | point `--wad-dir` at `noreward-rl/doomFiles/wads` |
| `AssertionError: expected obs (4,84,84)` | screen buffer shape mismatch | report it — the resize/convert path needs adjusting for your VizDoom version |
| `[warn] never inside TV zone` | `--tv-radius` too small | raise `--tv-radius` (game units) until the start room registers |
| `[warn] always inside TV zone` | `--tv-radius` too large | lower `--tv-radius` |
| Any `AttributeError` on a `vizdoom` enum/method | VizDoom API/version drift | note the exact line; the enum/method name needs updating for your version |

Once both preflights pass (and you've tuned `--tv-radius` if needed), the same
flags carry straight into the smoke test and the full runs — nothing else
changes.
