# Standalone, post-hoc video regenerator for VizDoom Curiosity-Critic runs.
#
# It loads a saved `--save-model` checkpoint, replays the trained policy in the
# (noisy-TV) env, and writes two mp4s:
#   <name>_noisyTV.mp4 -- pretty full-res RGB with the noise patch re-overlaid (reconstruction)
#   <name>_obs.mp4     -- the agent's ACTUAL observation (exact grayscale pixels it saw,
#                         including the real noise), nearest-neighbor upscaled
# so the "noisy TV" is visible in the video.
#
# It does NOT modify or affect the training code/results: training-time videos
# stay clean and consistent across all runs; presentation videos are generated
# here, uniformly, from the checkpoints you already have.
#
# Usage (from repo root):
#   python cleanrl/regenerate_vizdoom_video.py \
#       --checkpoint runs/vizdoom_sparse_noisytv__cc__1__.../ppo_curiosity_critic_vizdoom.cleanrl_model
#
# Batch over every Phase-1 run:
#   for f in runs/vizdoom_sparse_noisytv__*/ppo_curiosity_critic_vizdoom.cleanrl_model; do
#       python cleanrl/regenerate_vizdoom_video.py --checkpoint "$f"; done
import argparse
import os

import cv2
import numpy as np
import torch

from ppo_curiosity_critic_vizdoom import Agent, Args, make_env

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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, help="path to a *.cleanrl_model file")
    ap.add_argument("--out", default=None, help="output mp4 path (default: alongside checkpoint)")
    ap.add_argument("--steps", type=int, default=525, help="number of agent steps to roll out")
    ap.add_argument("--seed", type=int, default=12345, help="rollout seed")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--greedy", action="store_true", help="take argmax actions instead of sampling")
    ap.add_argument("--obs-scale", type=int, default=4, help="upscale factor for the agent-observation video")
    cli = ap.parse_args()

    ckpt = load_checkpoint(cli.checkpoint)
    saved = ckpt.get("args", {})
    fields = set(Args.__dataclass_fields__.keys())
    args = Args(**{k: v for k, v in saved.items() if k in fields})
    print(f"loaded checkpoint: method={saved.get('method')} scenario={args.scenario} "
          f"noisy_tv={args.noisy_tv} tv_radius={args.tv_radius} tv_panel={args.tv_panel} "
          f"global_step={ckpt.get('global_step')}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = make_env(args, idx=0, seed=cli.seed, expose_rgb=True)()
    n_actions = int(env.action_space.n)
    agent = Agent(n_actions).to(device)
    agent.load_state_dict(ckpt["policy_model"])
    agent.eval()

    rng = np.random.default_rng(cli.seed)
    obs, info = env.reset(seed=cli.seed)
    rgb_frames, obs_frames, tv_steps = [], [], 0
    with torch.no_grad():
        for _ in range(cli.steps):
            obs_arr = np.asarray(obs)
            # Newest grayscale frame = exactly the view the agent acts on, including the real
            # noise the NoisyTVWrapper drew into the observation.
            obs_frames.append(obs_arr[-1].copy())
            obs_t = torch.tensor(obs_arr, dtype=torch.float32, device=device).unsqueeze(0)
            if cli.greedy:
                hidden = agent.network(obs_t / 255.0)
                action = torch.argmax(agent.actor(hidden), dim=1)
            else:
                action, _, _, _, _ = agent.get_action_and_value(obs_t)
            obs, _, term, trunc, info = env.step(int(action.item()))

            frame = np.array(info["rgb"], dtype=np.uint8)  # copy; don't mutate env buffer
            if info.get("in_tv_zone"):
                tv_steps += 1
                rh = max(1, int(frame.shape[0] * args.tv_panel / OBS_SIZE))
                rw = max(1, int(frame.shape[1] * args.tv_panel / OBS_SIZE))
                frame[:rh, :rw] = rng.integers(0, 256, size=(rh, rw, 1), dtype=np.uint8)
            rgb_frames.append(frame)
            if term or trunc:
                obs, info = env.reset()
    env.close()

    base = os.path.splitext(cli.out)[0] if cli.out else os.path.splitext(cli.checkpoint)[0]

    # (1) Pretty RGB reconstruction (noise re-overlaid onto the full-res color frame).
    rgb_path = cli.out or (base + "_noisyTV.mp4")
    _write_video(rgb_path, [cv2.cvtColor(f, cv2.COLOR_RGB2BGR) for f in rgb_frames], cli.fps)

    # (2) The agent's ACTUAL observation (exact pixels it saw: grayscale, nearest-neighbor
    # upscaled so the static stays crisp and true to the downsampled view).
    obs_path = base + "_obs.mp4"
    s = max(1, cli.obs_scale)
    obs_bgr = [
        cv2.cvtColor(
            cv2.resize(f, (OBS_SIZE * s, OBS_SIZE * s), interpolation=cv2.INTER_NEAREST),
            cv2.COLOR_GRAY2BGR,
        )
        for f in obs_frames
    ]
    _write_video(obs_path, obs_bgr, cli.fps)

    print(f"wrote {rgb_path} and {obs_path}  ({len(rgb_frames)} frames, "
          f"{tv_steps} in TV zone = {tv_steps / max(len(rgb_frames), 1):.1%})")


if __name__ == "__main__":
    main()
