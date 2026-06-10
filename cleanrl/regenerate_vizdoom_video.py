# Post-hoc RGB + top-down-map video regenerator for VizDoom Curiosity-Critic runs.
#
# For each stored checkpoint, it replays the trained policy ONCE in the (noisy-TV) env and
# writes two videos from the SAME sampled transitions, so they stay perfectly in sync:
#   1) a full-res RGB video with the noise patch re-overlaid -- using the SAME
#      original-resolution patch as training (tv_panel x tv_panel), nearest-neighbor
#      scaled up onto the color frame so the static has the exact granularity the agent saw;
#   2) a top-down trajectory ("map") video of those same steps.
#
# Outputs (paired by update number; the map uses a _regen suffix so it never clobbers the
# training-time map_vids/update<NNNNNN>.mp4):
#   runs/<run>/videos/update<NNNNNN>_rgbvideo_w_noisepatch.mp4
#   runs/<run>/map_vids/update<NNNNNN>_mapvideo_regen.mp4
#
# The agent's exact grayscale observation video is already saved at training time, so
# this script no longer produces an _obs.mp4.
#
# Usage (from repo root):
#   # every checkpoint in a run:
#   python cleanrl/regenerate_vizdoom_video.py --run runs/vizdoom_sparse_noisytv__cc__1__...
#   # a single checkpoint:
#   python cleanrl/regenerate_vizdoom_video.py --checkpoint runs/<run>/checkpoints/ckpt_update007000.cleanrl_model
#   # batch over many runs:
#   for d in runs/vizdoom_sparse_noisytv__*/; do python cleanrl/regenerate_vizdoom_video.py --run "$d"; done
import argparse
import glob
import os
import re

import cv2
import numpy as np
import torch

from ppo_curiosity_critic_vizdoom import (
    Agent,
    Args,
    SCENARIO_WADS,
    load_wad_map_lines,
    load_wad_vest_positions,
    make_env,
    save_map_video,
)

OBS_SIZE = 84  # the grayscale observation side length used in training


def load_checkpoint(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # older torch without the weights_only kwarg
        return torch.load(path, map_location="cpu")


def _write_video(path, bgr_frames, fps):
    """Write a list of BGR uint8 frames to an mp4 via OpenCV."""
    height, width = bgr_frames[0].shape[:2]
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"cv2.VideoWriter failed to open for {path} (mp4v codec unavailable)")
    for frame in bgr_frames:
        writer.write(frame)
    writer.release()


def _update_label(ckpt_path, ckpt=None):
    """Pull the training update index from the checkpoint filename, falling back to its 'update' field."""
    m = re.search(r"update0*(\d+)", os.path.basename(ckpt_path))
    if m:
        return int(m.group(1))
    if ckpt is not None and isinstance(ckpt.get("update"), int):
        return ckpt["update"]
    return None


def _run_dir_for(ckpt_path):
    """The run directory for a checkpoint (parent of checkpoints/ if it lives there)."""
    d = os.path.dirname(os.path.abspath(ckpt_path))
    return os.path.dirname(d) if os.path.basename(d) == "checkpoints" else d


def render_rgb_and_map(ckpt_path, rgb_path, map_path, steps, seed, fps, greedy):
    """Roll out the checkpoint's policy ONCE and write two in-sync videos from the same transitions:
    (1) the RGB video with the training-size noise patch upscaled, and (2) the top-down map video."""
    ckpt = load_checkpoint(ckpt_path)
    saved = ckpt.get("args", {})
    args = Args(**{k: v for k, v in saved.items() if k in set(Args.__dataclass_fields__.keys())})
    method = saved.get("method", args.method)
    p = int(args.tv_panel)  # original training patch size (in the 84x84 obs)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = make_env(args, idx=0, seed=seed, expose_rgb=True)()
    n_actions = int(env.action_space.n)
    agent = Agent(n_actions).to(device)
    agent.load_state_dict(ckpt["policy_model"])
    agent.eval()

    rng = np.random.default_rng(seed)
    obs, info = env.reset(seed=seed)
    rgb_frames, xs, ys, in_tv, dones, tv_steps = [], [], [], [], [], 0
    with torch.no_grad():
        for _ in range(steps):
            obs_arr = np.asarray(obs)
            if method == "random":
                action_i = int(rng.integers(0, n_actions))
            else:
                obs_t = torch.tensor(obs_arr, dtype=torch.float32, device=device).unsqueeze(0)
                if greedy:
                    hidden = agent.network(obs_t / 255.0)
                    action_i = int(torch.argmax(agent.actor(hidden), dim=1).item())
                else:
                    action, _, _, _, _ = agent.get_action_and_value(obs_t)
                    action_i = int(action.item())
            obs, _, term, trunc, info = env.step(action_i)

            frame = np.array(info["rgb"], dtype=np.uint8)  # copy; don't mutate env buffer
            if info.get("in_tv_zone"):
                tv_steps += 1
                rh = max(1, int(frame.shape[0] * p / OBS_SIZE))
                rw = max(1, int(frame.shape[1] * p / OBS_SIZE))
                # original training-size patch (p x p), then nearest-neighbor scaled up to the RGB panel
                small = rng.integers(0, 256, size=(p, p), dtype=np.uint8)
                big = cv2.resize(small, (rw, rh), interpolation=cv2.INTER_NEAREST)[..., None]
                if args.noise_alpha >= 1.0:
                    frame[:rh, :rw] = big
                else:  # match the observation's blend so sweep videos show the right intensity
                    blended = (1.0 - args.noise_alpha) * frame[:rh, :rw].astype(np.float32) + args.noise_alpha * big
                    frame[:rh, :rw] = np.rint(blended).astype(np.uint8)
            rgb_frames.append(frame)
            xs.append(float(info.get("position_x", np.nan)))
            ys.append(float(info.get("position_y", np.nan)))
            in_tv.append(bool(info.get("in_tv_zone", False)))
            dones.append(bool(term or trunc))
            if term or trunc:
                obs, info = env.reset()
    env.close()

    # (1) RGB with the upscaled training-size noise patch
    _write_video(rgb_path, [cv2.cvtColor(f, cv2.COLOR_RGB2BGR) for f in rgb_frames], fps)
    # (2) Top-down map from the SAME sampled transitions (kept in sync with the RGB above)
    map_ok = False
    try:
        wad_path = os.path.join(args.wad_dir, SCENARIO_WADS[args.scenario])
        map_lines = load_wad_map_lines(wad_path, args.doom_map)
        vest = load_wad_vest_positions(wad_path, args.doom_map)
        save_map_video(map_path, map_lines, xs, ys, in_tv=in_tv, vest_positions=vest, dones=dones, fps=fps)
        map_ok = os.path.exists(map_path)
    except Exception as exc:  # never let a map failure lose the RGB video
        print(f"[map] skipped: {exc}")
    return len(rgb_frames), tv_steps, map_ok


def main():
    ap = argparse.ArgumentParser(description="Regenerate RGB videos with the training-size noise patch overlaid.")
    ap.add_argument("--run", help="run directory: process every checkpoints/ckpt_update*.cleanrl_model in it")
    ap.add_argument("--checkpoint", help="a single *.cleanrl_model to process instead of a whole run")
    ap.add_argument("--steps", type=int, default=525, help="number of agent steps to roll out per video")
    ap.add_argument("--seed", type=int, default=12345, help="rollout seed")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--greedy", action="store_true", help="take argmax actions instead of sampling")
    ap.add_argument("--overwrite", action="store_true", help="re-render even if the output already exists")
    cli = ap.parse_args()

    if cli.run:
        ckpts = sorted(glob.glob(os.path.join(cli.run, "checkpoints", "ckpt_update*.cleanrl_model")))
        if not ckpts:
            raise SystemExit(f"no checkpoints found under {cli.run}/checkpoints/")
    elif cli.checkpoint:
        ckpts = [cli.checkpoint]
    else:
        raise SystemExit("pass --run <run_dir> (all checkpoints) or --checkpoint <file> (one).")

    for ckpt_path in ckpts:
        run_dir = _run_dir_for(ckpt_path)
        vids_dir = os.path.join(run_dir, "videos")
        mapvids_dir = os.path.join(run_dir, "map_vids")
        os.makedirs(vids_dir, exist_ok=True)
        os.makedirs(mapvids_dir, exist_ok=True)
        upd = _update_label(ckpt_path)
        label = f"update{upd:06d}" if upd is not None else "final"
        rgb_path = os.path.join(vids_dir, f"{label}_rgbvideo_w_noisepatch.mp4")
        map_path = os.path.join(mapvids_dir, f"{label}_mapvideo_regen.mp4")
        if os.path.exists(rgb_path) and os.path.exists(map_path) and not cli.overwrite:
            print(f"skip (exists): {rgb_path} + {map_path}")
            continue
        n, tv, map_ok = render_rgb_and_map(ckpt_path, rgb_path, map_path, cli.steps, cli.seed, cli.fps, cli.greedy)
        print(f"wrote {rgb_path}" + (f" + {map_path}" if map_ok else "  (map skipped)")
              + f"  ({n} frames, {tv} in TV zone = {tv / max(n, 1):.1%})")


if __name__ == "__main__":
    main()
