# VizDoom Scenarios (MyWayHome)

These `.wad` files are the MyWayHome navigation scenarios used by
`cleanrl/ppo_curiosity_critic_vizdoom.py` (default `--wad-dir ./vizdoom_scenarios`):

- `my_way_home_sparse.wad` — sparse-reward MyWayHome (fixed farthest spawn)
- `my_way_home_verySparse.wad` — very-sparse variant (farther spawn)
- `my_way_home_dense.wad` — dense variant (for reference)

## Attribution

The wad files are redistributed from Deepak Pathak's `noreward-rl` repository
(ICM, "Curiosity-driven Exploration by Self-supervised Prediction", Pathak et
al., 2017), https://github.com/pathak22/noreward-rl, which is BSD-licensed.
The full license is in `SCENARIOS_LICENSE.txt`; the copyright notice is retained
as required.
