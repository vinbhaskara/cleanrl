# PPO + Curiosity-Critic on VizDoom (MyWayHome), single-file cleanRL-style script.
#
# This is the VizDoom counterpart of `ppo_curiosity_critic_envpool.py`. The PPO /
# GAE / clip / value / coefficient machinery and the World-Model (WM) and
# Neural-Critic (NC) architectures are carried over unchanged from the Atari
# scripts (`ppo_rnd_envpool.py`, `ppo_curiosity_critic_envpool.py`). Only the
# environment layer changes: envpool Atari -> Farama-Foundation VizDoom served
# through the Gymnasium API and vectorized with `gymnasium.vector.AsyncVectorEnv`.
#
# A single `--method` flag selects one of the six paper baselines, all sharing
# the same WM/critic/PPO code so any performance gap traces to the reward signal:
#
#   method=random : uniform random actions, no learning (exploration floor)
#   method=ppo    : PPO on extrinsic reward only, no intrinsic reward
#   method=c_v1   : Curiosity V1   r = e(s,a|theta_t)                       (zero baseline)
#   method=c_v2   : Curiosity V2   r = e(s,a|theta_{t-1}) - e(s,a|theta_t)  (one-step improvement)
#   method=rnd    : Random Network Distillation (predictor error on next frame)
#   method=cc     : Curiosity-Critic (ours)  r = e(s,a|theta_t) - phi(s,a)  (learned baseline)
#
# For c_v2 the "one-step improvement" is rendered at rollout-batch granularity:
# the reward is the drop in WM error on the visited transition between the
# previous-iteration WM snapshot (theta_{t-1}) and the current frozen WM
# (theta_t), both evaluated at rollout time. This is the batched analogue of the
# per-step V2 reward and stays inside the single frozen-models-at-rollout loop.
#
# The two paper conditions are selected with `--noisy-tv`:
#   plain    : deterministic MyWayHome maze, no added noise
#   noisy-tv : same maze + a visitable noise panel (Burda-style noisy TV) that is
#              painted into the observation whenever the agent is within
#              `--tv-radius` of its episode start position. All aleatoric noise
#              lives in this panel; the WM target then contains an irreducibly
#              unpredictable region, trapping raw-prediction-error curiosity.
#
# Visualization hooks (for the company presentation) are best-effort and guarded:
# episode-return / curiosity scalars, WM-prediction image panels, position
# visitation + intrinsic-reward heatmaps (from VizDoom POSITION game variables),
# and periodic gameplay videos.
import functools
import json
import os
import random
import re
import struct
import sys
import time
from collections import deque
from dataclasses import dataclass
from typing import Optional

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "curiosity-critic-vizdoom-JUN2026"
    """the wandb's project name"""
    wandb_entity: Optional[str] = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """if toggled, periodically record gameplay videos to runs/<run_name>/videos"""
    save_model: bool = False
    """if toggled, saves the final policy, World Model, Neural Critic / RND weights"""

    # Algorithm specific arguments
    method: str = "cc"
    """which intrinsic-reward method to run: one of {random, ppo, c_v1, c_v2, rnd, cc}"""
    total_timesteps: int = 30000000
    """total timesteps of the experiments (paper default: 30M)"""
    learning_rate: float = 1e-4
    """the learning rate of the optimizer"""
    num_envs: int = 32
    """the number of parallel VizDoom processes (AsyncVectorEnv workers)"""
    num_steps: int = 128
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.999
    """the discount factor gamma (extrinsic)"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 4
    """the number of mini-batches"""
    update_epochs: int = 4
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.1
    """the surrogate clipping coefficient"""
    clip_vloss: bool = True
    """Toggles whether or not to use a clipped loss for the value function"""
    ent_coef: float = 0.001
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: Optional[float] = None
    """the target KL divergence threshold"""

    # Curiosity / intrinsic-reward arguments (kept identical to the Atari scripts)
    update_proportion: float = 0.25
    """proportion of experience used for World-Model / RND / Neural-Critic update"""
    int_coef: float = 1.0
    """coefficient of the intrinsic reward"""
    ext_coef: float = 2.0
    """coefficient of the extrinsic reward"""
    int_gamma: float = 0.99
    """Intrinsic reward discount rate"""
    num_iterations_obs_norm_init: int = 50
    """number of rollouts of random data used to initialize observation normalization"""

    # VizDoom environment arguments
    scenario: str = "sparse"
    """MyWayHome scenario: one of {sparse, very_sparse, dense}"""
    wad_dir: str = "./vizdoom_scenarios"
    """directory holding the my_way_home_*.wad scenario files (bundled in the repo)"""
    doom_map: str = "map01"
    """the map name inside the scenario wad"""
    frame_skip: int = 4
    """number of tics each action is repeated for"""
    living_reward: float = -0.0001
    """per-tic living reward (matches the ICM MyWayHome shaping); set 0 to rely on the wad"""
    episode_timeout: int = 2100
    """episode timeout in tics"""
    env_init_stagger: float = 0.3
    """seconds to stagger each worker's VizDoom launch (avoids simultaneous-launch hangs)"""

    # Noisy-TV condition arguments
    noisy_tv: bool = False
    """if toggled, paint a noise panel into the observation inside the TV zone"""
    tv_radius: float = 150.0
    """TV-zone radius (game units) around episode start; calibrated via --probe-maze for MyWayHome sparse (max reach ~516)"""
    tv_panel: int = 42
    """side length (pixels, in the 84x84 frame) of the square noise panel"""
    noise_alpha: float = 1.0
    """noisy-TV blend in [0,1]: obs[patch] = (1-alpha)*clean + alpha*noise. alpha=1 = full static (default); use <1 for the noise-level sweep"""

    # Visualization cadence (in PPO updates)
    video_every: int = 200
    """record a gameplay video every N updates (0 disables)"""
    wm_panel_every: int = 100
    """dump a WM prediction panel every N updates (0 disables)"""
    heatmap_every: int = 100
    """dump visitation / intrinsic-reward heatmaps every N updates (0 disables)"""
    video_steps: int = 525
    """number of agent steps to roll out when recording a video"""

    # Preflight self-test
    preflight: bool = False
    """if toggled, build ONE real VizDoom env, validate the obs/reward/nets path, and exit (no training)"""
    preflight_steps: int = 20
    """number of random steps to take during the preflight check"""
    probe_maze: bool = False
    """if toggled, drive a forward-biased policy to estimate maze extent, recommend --tv-radius, and exit"""
    probe_maze_steps: int = 3000
    """number of steps for the maze-extent probe"""

    # Instrumentation (thorough logging so runs never need repeating)
    eval_every: int = 100
    """held-out world-model accuracy eval cadence (updates)"""
    ckpt_every: int = 200
    """periodic full-model checkpoint cadence (updates); 0 disables"""
    holdout_size: int = 10000
    """number of held-out deterministic transitions for the WM-accuracy eval (cached per seed)"""
    holdout_dir: str = "./vizdoom_holdout"
    """directory for the cached per-seed held-out transition sets"""
    profile_timing: bool = True
    """if toggled, time each component (reward/aux/WM/policy) with cuda syncs for accurate breakdowns"""
    post_plot: bool = True
    """if toggled, generate per-run plots from this run's metrics.jsonl at normal training exit"""
    post_plot_dir: str = ""
    """directory for per-run post-training plots; empty means runs/<run_name>/plots"""

    # to be filled in at runtime
    batch_size: int = 0
    minibatch_size: int = 0
    num_iterations: int = 0


# ----------------------------------------------------------------------------- #
#  VizDoom Gymnasium environment + wrappers
# ----------------------------------------------------------------------------- #
SCENARIO_WADS = {
    "sparse": "my_way_home_sparse.wad",
    "very_sparse": "my_way_home_verySparse.wad",
    "dense": "my_way_home_dense.wad",
}


class VizDoomEnv(gym.Env):
    """Single VizDoom MyWayHome environment exposed through the Gymnasium API.

    Observation: 84x84 uint8 grayscale single frame (frame stacking is added by a
    separate wrapper). The RGB frame is also exposed via ``render()`` and the
    raw position via ``info`` for video capture and heatmaps.
    """

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, args: "Args", idx: int = 0, seed: int = 0, expose_rgb: bool = False):
        super().__init__()
        import vizdoom as vzd

        self._vzd = vzd
        self.idx = idx
        self.expose_rgb = expose_rgb
        self.frame_skip = args.frame_skip
        self.episode_timeout = args.episode_timeout
        wad_path = os.path.join(args.wad_dir, SCENARIO_WADS[args.scenario])
        if not os.path.isfile(wad_path):
            raise FileNotFoundError(f"scenario wad not found: {wad_path}")

        # Stagger launches: simultaneously spawning many VizDoom processes can hang.
        time.sleep(idx * args.env_init_stagger)

        game = vzd.DoomGame()
        game.set_doom_scenario_path(wad_path)
        game.set_doom_map(args.doom_map)
        game.set_screen_resolution(vzd.ScreenResolution.RES_160X120)
        game.set_screen_format(vzd.ScreenFormat.RGB24)
        game.set_render_hud(False)
        game.set_render_crosshair(False)
        game.set_render_weapon(False)
        game.set_render_decals(False)
        game.set_render_particles(False)
        game.clear_available_buttons()
        for button in (vzd.Button.TURN_LEFT, vzd.Button.TURN_RIGHT, vzd.Button.MOVE_FORWARD):
            game.add_available_button(button)
        game.clear_available_game_variables()
        for gv in (vzd.GameVariable.POSITION_X, vzd.GameVariable.POSITION_Y, vzd.GameVariable.ANGLE):
            game.add_available_game_variable(gv)
        game.set_episode_timeout(args.episode_timeout)
        if args.living_reward != 0.0:
            game.set_living_reward(args.living_reward)
        game.set_window_visible(False)
        game.set_mode(vzd.Mode.PLAYER)
        game.set_seed(int(seed))
        game.init()
        self.game = game

        n = len(game.get_available_buttons())
        self._actions = [[int(i == j) for i in range(n)] for j in range(n)]  # one-hot button presses
        self.action_space = gym.spaces.Discrete(n)
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(84, 84), dtype=np.uint8)
        self._last_rgb = np.zeros((120, 160, 3), dtype=np.uint8)
        self._last_pos = (0.0, 0.0, 0.0)

    def _read_state(self):
        state = self.game.get_state()
        if state is None:
            return self._last_rgb, self._last_pos
        buf = state.screen_buffer
        if buf.ndim == 3 and buf.shape[0] == 3:  # CHW -> HWC defensive
            buf = np.transpose(buf, (1, 2, 0))
        gv = state.game_variables
        pos = (float(gv[0]), float(gv[1]), float(gv[2])) if gv is not None and len(gv) >= 3 else (0.0, 0.0, 0.0)
        self._last_rgb = buf
        self._last_pos = pos
        return buf, pos

    @staticmethod
    def _to_obs(rgb):
        import cv2

        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        return cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA).astype(np.uint8)

    def reset(self, *, seed: Optional[int] = None, options=None):
        if seed is not None:
            self.game.set_seed(int(seed))
        self.game.new_episode()
        rgb, pos = self._read_state()
        info = {"position_x": pos[0], "position_y": pos[1], "position_angle": pos[2]}
        if self.expose_rgb:
            info["rgb"] = rgb
        return self._to_obs(rgb), info

    def step(self, action):
        reward = self.game.make_action(self._actions[int(action)], self.frame_skip)
        terminated = self.game.is_episode_finished()
        rgb, pos = self._read_state()
        # In sparse / very-sparse MyWayHome, positive reward corresponds to collecting the vest.
        info = {"position_x": pos[0], "position_y": pos[1], "position_angle": pos[2], "goal_reached": bool(reward > 0.0)}
        if self.expose_rgb:
            info["rgb"] = rgb
        return self._to_obs(rgb), float(reward), bool(terminated), False, info

    def render(self):
        return self._last_rgb

    def close(self):
        try:
            self.game.close()
        except Exception:
            pass


class NoisyTVWrapper(gym.Wrapper):
    """Paint a re-sampled i.i.d. noise panel into the frame while in the TV zone.

    The TV zone is a disk of radius ``tv_radius`` around the agent's episode-start
    position; the panel is a ``tv_panel`` x ``tv_panel`` square at the top-left of
    the 84x84 frame, re-sampled every step. This makes the WM target irreducibly
    unpredictable inside the zone (a visitable noisy TV), without needing any
    maze-specific coordinates. ``info['in_tv_zone']`` is exposed for logging.
    """

    def __init__(self, env, tv_radius: float, tv_panel: int, noise_alpha: float = 1.0, rng_seed: int = 0):
        super().__init__(env)
        self.tv_radius = tv_radius
        self.tv_panel = tv_panel
        self.noise_alpha = float(noise_alpha)
        self._rng = np.random.default_rng(rng_seed)
        self._start = None

    def _maybe_paint(self, obs, info):
        in_zone = False
        if self._start is not None:
            dx = info.get("position_x", 0.0) - self._start[0]
            dy = info.get("position_y", 0.0) - self._start[1]
            in_zone = (dx * dx + dy * dy) <= (self.tv_radius * self.tv_radius)
        if in_zone:
            p = self.tv_panel
            obs = obs.copy()
            noise = self._rng.integers(0, 256, size=(p, p), dtype=np.uint8)
            if self.noise_alpha >= 1.0:
                obs[:p, :p] = noise  # full static (alpha=1, default) -- identical to prior behavior
            else:
                blended = (1.0 - self.noise_alpha) * obs[:p, :p].astype(np.float32) + self.noise_alpha * noise
                obs[:p, :p] = np.rint(blended).astype(np.uint8)
        info["in_tv_zone"] = bool(in_zone)
        return obs, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._start = (info.get("position_x", 0.0), info.get("position_y", 0.0))
        return self._maybe_paint(obs, info)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        obs, info = self._maybe_paint(obs, info)
        return obs, reward, terminated, truncated, info


def _frame_stack(env, k=4):
    # gymnasium >=1.0 renamed FrameStack -> FrameStackObservation (and the kwarg).
    if hasattr(gym.wrappers, "FrameStackObservation"):
        return gym.wrappers.FrameStackObservation(env, stack_size=k)
    return gym.wrappers.FrameStack(env, num_stack=k)


def _build_single_env(args: "Args", idx: int, seed: int, expose_rgb: bool = False):
    env = VizDoomEnv(args, idx=idx, seed=seed, expose_rgb=expose_rgb)
    if args.noisy_tv:
        env = NoisyTVWrapper(
            env, tv_radius=args.tv_radius, tv_panel=args.tv_panel, noise_alpha=args.noise_alpha, rng_seed=seed
        )
    env = _frame_stack(env, 4)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    return env


def make_env(args: "Args", idx: int, seed: int, expose_rgb: bool = False):
    # functools.partial of a module-level builder is picklable for AsyncVectorEnv "spawn".
    return functools.partial(_build_single_env, args, idx, seed, expose_rgb)


# ----------------------------------------------------------------------------- #
#  Networks (carried over unchanged from the Atari scripts)
# ----------------------------------------------------------------------------- #
def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, num_actions):
        super().__init__()
        self.network = nn.Sequential(
            layer_init(nn.Conv2d(4, 32, 8, stride=4)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, 4, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, 3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(64 * 7 * 7, 256)),
            nn.ReLU(),
            layer_init(nn.Linear(256, 448)),
            nn.ReLU(),
        )
        self.extra_layer = nn.Sequential(layer_init(nn.Linear(448, 448), std=0.1), nn.ReLU())
        self.actor = nn.Sequential(
            layer_init(nn.Linear(448, 448), std=0.01),
            nn.ReLU(),
            layer_init(nn.Linear(448, num_actions), std=0.01),
        )
        self.critic_ext = layer_init(nn.Linear(448, 1), std=0.01)
        self.critic_int = layer_init(nn.Linear(448, 1), std=0.01)

    def get_action_and_value(self, x, action=None):
        hidden = self.network(x / 255.0)
        logits = self.actor(hidden)
        probs = Categorical(logits=logits)
        features = self.extra_layer(hidden)
        if action is None:
            action = probs.sample()
        return (
            action,
            probs.log_prob(action),
            probs.entropy(),
            self.critic_ext(features + hidden),
            self.critic_int(features + hidden),
        )

    def get_value(self, x):
        hidden = self.network(x / 255.0)
        features = self.extra_layer(hidden)
        return self.critic_ext(features + hidden), self.critic_int(features + hidden)


def _action_planes(action_onehot, height, width):
    return action_onehot[:, :, None, None].expand(-1, -1, height, width)


class ForwardCNN(nn.Module):
    """Action-conditioned U-Net World Model (predicts the next grayscale frame)."""

    def __init__(self, num_actions, frame_stack=4, channels=(64, 128, 256)):
        super().__init__()
        c1, c2, c3 = channels
        self.num_actions = num_actions
        self.down1 = nn.Sequential(
            layer_init(nn.Conv2d(frame_stack + num_actions, c1, kernel_size=8, stride=4)),
            nn.LeakyReLU(),
        )
        self.down2 = nn.Sequential(
            layer_init(nn.Conv2d(c1, c2, kernel_size=4, stride=2)),
            nn.LeakyReLU(),
        )
        self.bottleneck = nn.Sequential(
            layer_init(nn.Conv2d(c2, c3, kernel_size=3, stride=1)),
            nn.LeakyReLU(),
        )
        self.up2 = nn.Sequential(
            layer_init(nn.ConvTranspose2d(c3, c2, kernel_size=3, stride=1)),
            nn.LeakyReLU(),
        )
        self.fuse2 = nn.Sequential(
            layer_init(nn.Conv2d(c2 * 2, c2, kernel_size=3, stride=1, padding=1)),
            nn.LeakyReLU(),
        )
        self.up1 = nn.Sequential(
            layer_init(nn.ConvTranspose2d(c2, c1, kernel_size=4, stride=2)),
            nn.LeakyReLU(),
        )
        self.fuse1 = nn.Sequential(
            layer_init(nn.Conv2d(c1 * 2, c1, kernel_size=3, stride=1, padding=1)),
            nn.LeakyReLU(),
        )
        self.out = nn.Sequential(
            layer_init(nn.ConvTranspose2d(c1, 1, kernel_size=8, stride=4)),
        )

    def forward(self, obs_stack, action_onehot):
        _, _, height, width = obs_stack.shape
        action_map = _action_planes(action_onehot, height, width)
        h = torch.cat([obs_stack, action_map], dim=1)
        skip1 = self.down1(h)
        skip2 = self.down2(skip1)
        bottleneck = self.bottleneck(skip2)
        up2 = self.up2(bottleneck)
        up2 = self.fuse2(torch.cat([up2, skip2], dim=1))
        up1 = self.up1(up2)
        up1 = self.fuse1(torch.cat([up1, skip1], dim=1))
        return self.out(up1)


class CuriosityCriticCNN(nn.Module):
    """Action-conditioned Neural Critic predicting the scalar post-update WM error."""

    def __init__(self, num_actions, frame_stack=4):
        super().__init__()
        feature_output = 7 * 7 * 64
        self.encoder = nn.Sequential(
            layer_init(nn.Conv2d(frame_stack + num_actions, 32, kernel_size=8, stride=4)),
            nn.LeakyReLU(),
            layer_init(nn.Conv2d(32, 64, kernel_size=4, stride=2)),
            nn.LeakyReLU(),
            layer_init(nn.Conv2d(64, 64, kernel_size=3, stride=1)),
            nn.LeakyReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(feature_output, 512)),
            nn.ReLU(),
            layer_init(nn.Linear(512, 512)),
            nn.ReLU(),
            layer_init(nn.Linear(512, 1)),
        )

    def forward(self, obs_stack, action_onehot):
        _, _, height, width = obs_stack.shape
        action_map = _action_planes(action_onehot, height, width)
        h = torch.cat([obs_stack, action_map], dim=1)
        return self.encoder(h)


class RNDModel(nn.Module):
    """RND predictor/target operating on the single (normalized) next frame."""

    def __init__(self):
        super().__init__()
        feature_output = 7 * 7 * 64
        self.predictor = nn.Sequential(
            layer_init(nn.Conv2d(1, 32, kernel_size=8, stride=4)),
            nn.LeakyReLU(),
            layer_init(nn.Conv2d(32, 64, kernel_size=4, stride=2)),
            nn.LeakyReLU(),
            layer_init(nn.Conv2d(64, 64, kernel_size=3, stride=1)),
            nn.LeakyReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(feature_output, 512)),
            nn.ReLU(),
            layer_init(nn.Linear(512, 512)),
            nn.ReLU(),
            layer_init(nn.Linear(512, 512)),
        )
        self.target = nn.Sequential(
            layer_init(nn.Conv2d(1, 32, kernel_size=8, stride=4)),
            nn.LeakyReLU(),
            layer_init(nn.Conv2d(32, 64, kernel_size=4, stride=2)),
            nn.LeakyReLU(),
            layer_init(nn.Conv2d(64, 64, kernel_size=3, stride=1)),
            nn.LeakyReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(feature_output, 512)),
        )
        for param in self.target.parameters():
            param.requires_grad = False

    def forward(self, next_obs):
        return self.predictor(next_obs), self.target(next_obs)


class RunningMeanStd:
    """Standard Welford running mean/std (avoids gym/gymnasium version coupling)."""

    def __init__(self, epsilon=1e-4, shape=()):
        self.mean = np.zeros(shape, "float64")
        self.var = np.ones(shape, "float64")
        self.count = epsilon

    def update(self, x):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        tot = self.count + batch_count
        self.mean = self.mean + delta * batch_count / tot
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + np.square(delta) * self.count * batch_count / tot
        self.var = m2 / tot
        self.count = tot


class RewardForwardFilter:
    def __init__(self, gamma):
        self.rewems = None
        self.gamma = gamma

    def update(self, rews):
        self.rewems = rews if self.rewems is None else self.rewems * self.gamma + rews
        return self.rewems


def _normalize_stack(stack, mean_t, std_t):
    return ((stack - mean_t) / std_t).clip(-5, 5).float()


def _reward_error_per_sample(pred, target):
    """Raw curiosity error: 0.5 * squared L2 per sample (summed over pixels)."""
    return 0.5 * (pred - target).flatten(1).pow(2).sum(1)


def _update_loss_per_sample(pred, target):
    """Per-sample MSE used for the WM / RND-predictor gradient update."""
    return F.mse_loss(pred, target, reduction="none").flatten(1).mean(1)


def _cpu_state_dict(module):
    return {key: value.detach().cpu() for key, value in module.state_dict().items()}


class _Timer:
    """Accumulate wall-time into a dict; optional cuda sync for accurate GPU-component timing."""

    def __init__(self, store, key, sync, device):
        self.store, self.key, self.sync, self.device = store, key, sync, device

    def __enter__(self):
        if self.sync and self.device.type == "cuda":
            torch.cuda.synchronize()
        self._t = time.perf_counter()
        return self

    def __exit__(self, *exc):
        if self.sync and self.device.type == "cuda":
            torch.cuda.synchronize()
        self.store[self.key] = self.store.get(self.key, 0.0) + (time.perf_counter() - self._t)


def _git_commit():
    try:
        import subprocess

        return subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


def _read_wad_textmap(wad_path, doom_map="map01"):
    """Read the UDMF TEXTMAP lump for a MAPxx marker from a WAD."""
    doom_map = doom_map.upper()
    with open(wad_path, "rb") as f:
        magic = f.read(4)
        if magic not in {b"IWAD", b"PWAD"}:
            raise ValueError(f"{wad_path} is not a Doom WAD")
        num_lumps, directory_offset = struct.unpack("<ii", f.read(8))
        f.seek(directory_offset)
        lumps = []
        for _ in range(num_lumps):
            offset, size, name = struct.unpack("<ii8s", f.read(16))
            lumps.append((name.rstrip(b"\0").decode("ascii", "replace").upper(), offset, size))

        textmap_idx = None
        for i, (name, _, _) in enumerate(lumps):
            if name == doom_map:
                for j in range(i + 1, len(lumps)):
                    if lumps[j][0] == "TEXTMAP":
                        textmap_idx = j
                        break
                    if lumps[j][0] == "ENDMAP":
                        break
                break
        if textmap_idx is None:
            textmap_idx = next((i for i, lump in enumerate(lumps) if lump[0] == "TEXTMAP"), None)
        if textmap_idx is None:
            raise ValueError(f"no TEXTMAP lump found in {wad_path}")

        _, offset, size = lumps[textmap_idx]
        f.seek(offset)
        return f.read(size).decode("utf-8", "replace")


def _parse_udmf_blocks(text, kind):
    """Parse simple UDMF blocks like `vertex // 0 { x = ...; }`."""
    blocks = []
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        header = lines[i].strip().split("//", 1)[0].strip()
        if header == kind:
            j = i + 1
            while j < len(lines) and not lines[j].strip():
                j += 1
            if j < len(lines) and lines[j].strip() == "{":
                values = {}
                j += 1
                while j < len(lines) and lines[j].strip() != "}":
                    match = re.match(r"\s*(\w+)\s*=\s*(.*?)\s*;\s*$", lines[j])
                    if match:
                        key, raw = match.groups()
                        raw = raw.strip().strip('"')
                        if raw.lower() in {"true", "false"}:
                            values[key] = raw.lower() == "true"
                        else:
                            try:
                                values[key] = int(raw)
                            except ValueError:
                                try:
                                    values[key] = float(raw)
                                except ValueError:
                                    values[key] = raw
                    j += 1
                blocks.append(values)
                i = j
        i += 1
    return blocks


def load_wad_map_lines(wad_path, doom_map="map01"):
    """Return top-down WAD line segments in the same coordinate system as VizDoom POSITION_X/Y."""
    text = _read_wad_textmap(wad_path, doom_map)
    vertices = _parse_udmf_blocks(text, "vertex")
    linedefs = _parse_udmf_blocks(text, "linedef")
    lines = []
    for linedef in linedefs:
        v1, v2 = linedef.get("v1"), linedef.get("v2")
        if not isinstance(v1, int) or not isinstance(v2, int) or v1 >= len(vertices) or v2 >= len(vertices):
            continue
        a, b = vertices[v1], vertices[v2]
        if "x" not in a or "y" not in a or "x" not in b or "y" not in b:
            continue
        lines.append((float(a["x"]), float(a["y"]), float(b["x"]), float(b["y"]), bool(linedef.get("blocking", False))))
    return lines


def load_wad_vest_positions(wad_path, doom_map="map01"):
    """Return vest/goal thing positions from a UDMF WAD.

    The MyWayHome scenarios use Doom thing type 2018 for the collectable vest
    that terminates the sparse task.
    """
    text = _read_wad_textmap(wad_path, doom_map)
    positions = []
    for thing in _parse_udmf_blocks(text, "thing"):
        if thing.get("type") == 2018 and "x" in thing and "y" in thing:
            positions.append((float(thing["x"]), float(thing["y"])))
    return positions


def _map_extent(map_lines, xs=None, ys=None, pad_frac=0.04):
    x_vals, y_vals = [], []
    if map_lines:
        x_vals.append(np.asarray([[line[0], line[2]] for line in map_lines], dtype=np.float32).reshape(-1))
        y_vals.append(np.asarray([[line[1], line[3]] for line in map_lines], dtype=np.float32).reshape(-1))
    if xs is not None and len(xs):
        x_vals.append(np.asarray(xs, dtype=np.float32).reshape(-1))
    if ys is not None and len(ys):
        y_vals.append(np.asarray(ys, dtype=np.float32).reshape(-1))
    if not x_vals or not y_vals:
        return None
    all_x, all_y = np.concatenate(x_vals), np.concatenate(y_vals)
    x_pad = max(1.0, pad_frac * float(all_x.max() - all_x.min()))
    y_pad = max(1.0, pad_frac * float(all_y.max() - all_y.min()))
    return [float(all_x.min() - x_pad), float(all_x.max() + x_pad), float(all_y.min() - y_pad), float(all_y.max() + y_pad)]


def _plot_map_lines(ax, map_lines, alpha=0.45, zorder=4):
    for blocking in (False, True):
        for x1, y1, x2, y2, is_blocking in map_lines:
            if is_blocking != blocking:
                continue
            ax.plot(
                [x1, x2],
                [y1, y2],
                color="0.20" if blocking else "0.55",
                linewidth=1.0 if blocking else 0.6,
                alpha=alpha if blocking else alpha * 0.65,
                zorder=zorder,
            )


def collect_holdout_transitions(args, seed, size, p_forward=0.55, max_tics=100000):
    """Forward-biased random walk through the DETERMINISTIC (no-noise) maze, collecting
    (obs_stack, action, next_frame) transitions plus the (x, y) where each was sampled.

    Forward-bias + a long episode (raised timeout) spread coverage across the maze, instead of
    canceling out near the spawn the way a uniform turn-left/right/forward walk does. Action order
    is [TURN_LEFT, TURN_RIGHT, MOVE_FORWARD], so MOVE_FORWARD is the last index."""
    import dataclasses

    plain = dataclasses.replace(args, noisy_tv=False, episode_timeout=max_tics)
    env = make_env(plain, idx=0, seed=seed)()
    n = int(env.action_space.n)
    forward = n - 1
    rng = np.random.default_rng(seed)
    obs, info = env.reset(seed=seed)
    obs_l, act_l, nxt_l, xs, ys = [], [], [], [], []
    while len(obs_l) < size:
        a = forward if rng.random() < p_forward else int(rng.integers(0, n - 1))
        obs_l.append(np.asarray(obs).copy())
        nobs, _, term, trunc, info = env.step(a)
        nobs = np.asarray(nobs)
        act_l.append(a)
        nxt_l.append(nobs[-1].copy())
        xs.append(float(info.get("position_x", 0.0)))
        ys.append(float(info.get("position_y", 0.0)))
        obs = nobs
        if term or trunc:
            obs, info = env.reset()
    env.close()
    return (
        np.asarray(obs_l, dtype=np.uint8),
        np.asarray(act_l, dtype=np.int64),
        np.asarray(nxt_l, dtype=np.uint8),
        np.asarray(xs, dtype=np.float32),
        np.asarray(ys, dtype=np.float32),
    )


def collect_or_load_holdout(args):
    """Load the cached per-seed held-out set (built by build_holdout.py) or collect it inline.
    Cached per seed+scenario so all methods of a given seed are scored on identical transitions."""
    os.makedirs(args.holdout_dir, exist_ok=True)
    path = os.path.join(args.holdout_dir, f"holdout_{args.scenario}_seed{args.seed}.npz")
    if os.path.isfile(path):
        d = np.load(path)
        return d["obs"], d["act"], d["next_frame"]
    print(
        f"[holdout] {path} not found -> collecting inline. For deliberate 3-seed coverage + "
        f"coverage heatmaps, run: python cleanrl/build_holdout.py --scenario {args.scenario}"
    )
    obs, act, nxt, xs, ys = collect_holdout_transitions(args, args.seed, args.holdout_size)
    np.savez_compressed(path, obs=obs, act=act, next_frame=nxt, x=xs, y=ys)
    return obs, act, nxt


@torch.no_grad()
def eval_wm_holdout(world_model, holdout, obs_rms, num_actions, device, batch=256):
    """Mean raw-pixel L2 WM error on the held-out deterministic set. Predictions are de-normalized
    with the run's own obs_rms, so the metric lives in a common raw-pixel space comparable across
    methods regardless of their observation normalization."""
    obs_arr, act_arr, next_arr = holdout
    mean_t = torch.from_numpy(obs_rms.mean).to(device).float()
    std_t = torch.sqrt(torch.from_numpy(obs_rms.var).to(device)).float()
    total, n = 0.0, 0
    for i in range(0, len(obs_arr), batch):
        obs_b = torch.from_numpy(obs_arr[i : i + batch]).float().to(device)
        act_b = torch.from_numpy(act_arr[i : i + batch]).long().to(device)
        nxt_b = torch.from_numpy(next_arr[i : i + batch]).float().to(device).unsqueeze(1)
        obs_norm = ((obs_b - mean_t) / std_t).clip(-5, 5)
        ah = F.one_hot(act_b, num_classes=num_actions).float()
        pred_raw = world_model(obs_norm, ah) * std_t + mean_t
        err = (pred_raw - nxt_b).flatten(1).pow(2).sum(1).sqrt()
        total += err.sum().item()
        n += err.numel()
    return total / max(n, 1)


def save_full_checkpoint(
    path, args, global_step, update, agent, world_model, neural_critic, rnd_model, obs_rms, reward_rms, world_model_prev=None
):
    ckpt = {
        "args": vars(args),
        "global_step": global_step,
        "update": update,
        "policy_model": _cpu_state_dict(agent),
        "world_model": _cpu_state_dict(world_model),
        "obs_rms_mean": obs_rms.mean,
        "obs_rms_var": obs_rms.var,
        "reward_rms_mean": reward_rms.mean,
        "reward_rms_var": reward_rms.var,
    }
    if neural_critic is not None:
        ckpt["neural_critic"] = _cpu_state_dict(neural_critic)
    if rnd_model is not None:
        ckpt["rnd_predictor"] = _cpu_state_dict(rnd_model.predictor)
        ckpt["rnd_target"] = _cpu_state_dict(rnd_model.target)
    if world_model_prev is not None:  # c_v2's previous-iteration WM snapshot
        ckpt["world_model_prev"] = _cpu_state_dict(world_model_prev)
    torch.save(ckpt, path)


def run_post_training_plots(run_dir, args):
    """Generate quick per-run plots from this run's metrics.jsonl.

    The aggregate paper plots still come from `plot_vizdoom_curiosity.py --runs-dir runs`.
    This hook is intentionally single-run and cheap, so every completed job leaves
    immediate curves next to its raw metrics.
    """
    if not args.post_plot:
        return
    try:
        from plot_vizdoom_curiosity import DEFAULT_METRICS, load_runs, plot_metric_groups, write_summary

        out_dir = args.post_plot_dir or os.path.join(run_dir, "plots")
        os.makedirs(out_dir, exist_ok=True)
        run_dir_abs = os.path.abspath(run_dir)
        runs = [run for run in load_runs(os.path.dirname(run_dir_abs)) if os.path.abspath(run["run_dir"]) == run_dir_abs]
        if not runs:
            print(f"[plot] skipped: no run_meta.json + metrics.jsonl found for {run_dir}")
            return
        rng = np.random.default_rng(0)
        made = []
        for metric in DEFAULT_METRICS:
            made.extend(plot_metric_groups(runs, metric, out_dir, points=200, n_boot=0, rng=rng))
        summary_path = os.path.join(out_dir, "summary_final_metrics.csv")
        rows = write_summary(runs, DEFAULT_METRICS, summary_path, n_boot=0, rng=rng)
        print(f"[plot] wrote {len(made)} per-run plot(s) + {summary_path} ({len(rows)} rows)")
    except Exception as exc:  # pragma: no cover
        print(f"[plot] skipped post-training plots: {exc}")


# ----------------------------------------------------------------------------- #
#  Visualization helpers (best-effort; never crash training)
# ----------------------------------------------------------------------------- #
def save_wm_panel(path, obs_last, pred_next, true_next, critic_value=None):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        err = np.abs(pred_next - true_next)
        cols = [("input frame", obs_last), ("WM prediction", pred_next), ("true next", true_next), ("abs error", err)]
        fig, axes = plt.subplots(1, len(cols), figsize=(3 * len(cols), 3.2))
        for ax, (title, img) in zip(axes, cols):
            ax.imshow(img, cmap="gray")
            ax.set_title(title, fontsize=9)
            ax.axis("off")
        if critic_value is not None:
            fig.suptitle(f"critic baseline phi(s,a) = {critic_value:.3f}", fontsize=10)
        fig.tight_layout()
        fig.savefig(path, dpi=110)
        plt.close(fig)
        return path
    except Exception as exc:  # pragma: no cover - visualization is best-effort
        print(f"[viz] WM panel skipped: {exc}")
        return None


def save_heatmaps(path, xs, ys, intr, in_tv, bins=40, map_lines=None):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        xs, ys, intr = np.asarray(xs), np.asarray(ys), np.asarray(intr)
        if xs.size == 0:
            return None
        map_lines = map_lines or []
        extent = _map_extent(map_lines, xs, ys)
        hist_range = [[extent[0], extent[1]], [extent[2], extent[3]]] if extent is not None else None
        visit, xe, ye = np.histogram2d(xs, ys, bins=bins, range=hist_range)
        rsum, _, _ = np.histogram2d(xs, ys, bins=[xe, ye], weights=intr)
        with np.errstate(invalid="ignore", divide="ignore"):
            rmean = np.where(visit > 0, rsum / visit, 0.0)
        fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.8))
        img_extent = [xe[0], xe[-1], ye[0], ye[-1]]
        cmap_visit = plt.cm.viridis.copy()
        cmap_visit.set_bad(alpha=0.0)
        panels = (
            (axes[0], np.ma.masked_where(visit.T == 0, visit.T), "visitation"),
            (axes[1], np.ma.masked_where(visit.T == 0, rmean.T), "mean intrinsic reward"),
        )
        for ax, data, title in panels:
            ax.set_facecolor("#f7f7f4")
            if map_lines:
                _plot_map_lines(ax, map_lines, alpha=0.28, zorder=1)
            im = ax.imshow(
                data,
                origin="lower",
                aspect="equal",
                cmap=cmap_visit,
                extent=img_extent,
                interpolation="nearest",
                alpha=0.78 if map_lines else 1.0,
                zorder=2,
            )
            if map_lines:
                ax.scatter(xs, ys, s=1.3, c="black", alpha=0.16, linewidths=0, zorder=3)
                _plot_map_lines(ax, map_lines, alpha=0.45, zorder=4)
            ax.set_title(title, fontsize=10)
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_xlim(img_extent[0], img_extent[1])
            ax.set_ylim(img_extent[2], img_extent[3])
            ax.set_aspect("equal", adjustable="box")
            fig.colorbar(im, ax=ax, fraction=0.046)
        frac = float(np.mean(in_tv)) if len(in_tv) else 0.0
        fig.suptitle(f"TV-zone time fraction = {frac:.3f}", fontsize=10)
        fig.tight_layout()
        fig.savefig(path, dpi=110)
        plt.close(fig)
        return path
    except Exception as exc:  # pragma: no cover
        print(f"[viz] heatmap skipped: {exc}")
        return None


def _map_canvas_transform(extent, width, height):
    """World-coordinate to image-pixel transform with y-up map coordinates."""
    x_min, x_max, y_min, y_max = extent
    left, right, top, bottom = 72, width - 28, 44, height - 58
    plot_w, plot_h = right - left, bottom - top
    scale = min(plot_w / max(x_max - x_min, 1e-6), plot_h / max(y_max - y_min, 1e-6))
    used_w, used_h = (x_max - x_min) * scale, (y_max - y_min) * scale
    x0 = left + 0.5 * (plot_w - used_w)
    y0 = top + 0.5 * (plot_h - used_h)

    def to_px(x, y):
        px = x0 + (float(x) - x_min) * scale
        py = y0 + used_h - (float(y) - y_min) * scale
        return int(round(px)), int(round(py))

    return to_px, (int(round(x0)), int(round(y0)), int(round(used_w)), int(round(used_h)))


def _draw_vest_markers(frame, vest_positions, to_px):
    import cv2

    for vx, vy in vest_positions or []:
        cx, cy = to_px(vx, vy)
        size = 12
        color = (40, 40, 230)
        cv2.line(frame, (cx - size, cy - size), (cx + size, cy + size), color, 3, lineType=cv2.LINE_AA)
        cv2.line(frame, (cx - size, cy + size), (cx + size, cy - size), color, 3, lineType=cv2.LINE_AA)
        cv2.circle(frame, (cx, cy), size + 5, color, 1, lineType=cv2.LINE_AA)
        cv2.putText(frame, "vest", (cx + 14, cy - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, lineType=cv2.LINE_AA)


def save_map_video(path, map_lines, xs, ys, in_tv=None, vest_positions=None, fps=30, width=960, height=720):
    """Render a top-down trajectory video on the fixed WAD map.

    The transform intentionally matches the static heatmap convention: Doom x grows
    right, Doom y grows upward, while image pixels grow downward. Positions and WAD
    line vertices are both transformed by the same function to avoid orientation
    drift between overlays and videos.
    """
    try:
        import cv2

        xs = np.asarray(xs, dtype=np.float32)
        ys = np.asarray(ys, dtype=np.float32)
        if xs.size == 0 or ys.size == 0 or not map_lines:
            return None
        valid = np.isfinite(xs) & np.isfinite(ys)
        xs, ys = xs[valid], ys[valid]
        if in_tv is not None and len(in_tv):
            in_tv = np.asarray(in_tv, dtype=bool)[valid]
        else:
            in_tv = np.zeros(len(xs), dtype=bool)
        if xs.size == 0:
            return None

        extent = _map_extent(map_lines, xs, ys)
        to_px, (plot_x, plot_y, plot_w, plot_h) = _map_canvas_transform(extent, width, height)
        base = np.full((height, width, 3), 250, dtype=np.uint8)
        cv2.rectangle(base, (plot_x, plot_y), (plot_x + plot_w, plot_y + plot_h), (225, 225, 225), 1)
        cv2.putText(base, "top-down agent trajectory", (24, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (35, 35, 35), 2)
        cv2.putText(base, "x", (width - 45, height - 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (80, 80, 80), 1)
        cv2.putText(base, "y", (24, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (80, 80, 80), 1)

        for blocking in (False, True):
            color = (150, 150, 150) if not blocking else (45, 45, 45)
            thickness = 1 if not blocking else 2
            for x1, y1, x2, y2, is_blocking in map_lines:
                if bool(is_blocking) != blocking:
                    continue
                cv2.line(base, to_px(x1, y1), to_px(x2, y2), color, thickness, lineType=cv2.LINE_AA)
        _draw_vest_markers(base, vest_positions, to_px)

        writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
        if not writer.isOpened():
            print("[viz] map video skipped: cv2.VideoWriter failed to open (mp4v codec unavailable)")
            return None

        pts = [to_px(x, y) for x, y in zip(xs, ys)]
        for i, pt in enumerate(pts):
            frame = base.copy()
            if i >= 1:
                trail = np.asarray(pts[: i + 1], dtype=np.int32).reshape((-1, 1, 2))
                overlay = frame.copy()
                cv2.polylines(overlay, [trail], False, (210, 105, 30), 3, lineType=cv2.LINE_AA)
                frame = cv2.addWeighted(overlay, 0.72, frame, 0.28, 0)
            _draw_vest_markers(frame, vest_positions, to_px)
            color = (0, 150, 255) if in_tv[i] else (40, 70, 230)
            cv2.circle(frame, pt, 8, color, -1, lineType=cv2.LINE_AA)
            cv2.circle(frame, pt, 9, (20, 20, 20), 1, lineType=cv2.LINE_AA)
            cv2.putText(
                frame,
                f"step {i + 1}/{len(pts)}   x={xs[i]:.0f} y={ys[i]:.0f}",
                (24, height - 22),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (35, 35, 35),
                1,
                lineType=cv2.LINE_AA,
            )
            if in_tv[i]:
                cv2.putText(frame, "TV zone", (width - 112, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 120, 220), 2)
            writer.write(frame)
        writer.release()
        return path
    except Exception as exc:  # pragma: no cover
        print(f"[viz] map video skipped: {exc}")
        return None


@torch.no_grad()
def capture_video(path, args, agent, device, seed, map_path=None, map_lines=None, vest_positions=None):
    """Roll out the current policy in a fresh single env and write an mp4 via OpenCV.

    Uses cv2.VideoWriter (cv2 is already a hard dependency) instead of imageio to
    avoid imageio/imageio-ffmpeg version mismatches.
    """
    try:
        import cv2

        env = make_env(args, idx=0, seed=seed + 999, expose_rgb=True)()
        rng = np.random.default_rng(seed + 999)
        obs, _ = env.reset()
        rgb_frames, obs_frames, xs, ys, in_tv = [], [], [], [], []
        for _ in range(args.video_steps):
            obs_arr = np.asarray(obs)
            obs_frames.append(obs_arr[-1].copy())  # newest grayscale frame = exactly what the policy sees (real noise)
            if args.method == "random":
                action_i = int(rng.integers(0, int(env.action_space.n)))
            else:
                obs_t = torch.tensor(obs_arr, dtype=torch.float32, device=device).unsqueeze(0)
                action, _, _, _, _ = agent.get_action_and_value(obs_t)
                action_i = int(action.item())
            obs, _, term, trunc, info = env.step(action_i)
            rgb_frames.append(np.asarray(info.get("rgb")))
            xs.append(float(info.get("position_x", np.nan)))
            ys.append(float(info.get("position_y", np.nan)))
            in_tv.append(bool(info.get("in_tv_zone", False)))
            if term or trunc:
                obs, _ = env.reset()
        env.close()
        if not rgb_frames:
            return None
        height, width = rgb_frames[0].shape[:2]
        writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 30, (width, height))
        if not writer.isOpened():
            print("[viz] video skipped: cv2.VideoWriter failed to open (mp4v codec unavailable)")
            return None
        for frame in rgb_frames:
            writer.write(cv2.cvtColor(np.asarray(frame, dtype=np.uint8), cv2.COLOR_RGB2BGR))
        writer.release()
        # Also write the agent's exact observation (grayscale, nearest-neighbor upscaled, real noise included).
        obs_path = os.path.splitext(path)[0] + "_obs.mp4"
        s = 4
        ow, oh = 84 * s, 84 * s
        obs_writer = cv2.VideoWriter(obs_path, cv2.VideoWriter_fourcc(*"mp4v"), 30, (ow, oh))
        if obs_writer.isOpened():
            for frame in obs_frames:
                up = cv2.resize(np.asarray(frame, dtype=np.uint8), (ow, oh), interpolation=cv2.INTER_NEAREST)
                obs_writer.write(cv2.cvtColor(up, cv2.COLOR_GRAY2BGR))
            obs_writer.release()
        if map_path is not None and map_lines:
            save_map_video(map_path, map_lines, xs, ys, in_tv=in_tv, vest_positions=vest_positions, fps=30)
        return path
    except Exception as exc:  # pragma: no cover
        print(f"[viz] video skipped: {exc}")
        return None


# ----------------------------------------------------------------------------- #
#  Preflight self-test (fast, single env, no training)
# ----------------------------------------------------------------------------- #
def run_preflight(args: "Args"):
    """Validate the VizDoom integration on real data in ~30 seconds.

    Builds one real env (full wrapper stack), steps random actions, and runs a
    forward pass of the chosen method's networks on the collected observations.
    Surfaces the first-run unknowns (map name, screen format, reward, tv-radius,
    API/version drift, missing deps) with a clear PASS/FAIL before any GPU-days
    are spent on the full training matrix.
    """
    line = "=" * 72
    print(line)
    print(f"PREFLIGHT  method={args.method}  scenario={args.scenario}  noisy_tv={args.noisy_tv}")
    print(line)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    print(f"[ok] torch device = {device}")

    import importlib

    for mod in ("vizdoom", "cv2"):
        importlib.import_module(mod)  # hard requirement: let ImportError surface loudly
        print(f"[ok] import {mod}")
    try:
        importlib.import_module("matplotlib")
        print("[ok] import matplotlib (visualization)")
    except Exception as exc:
        print(f"[warn] matplotlib missing -> heatmaps/WM panels will be skipped: {exc}")

    print(f"\n[env] building VizDoomEnv (map={args.doom_map}, wad_dir={args.wad_dir}) ...")
    env = make_env(args, idx=0, seed=args.seed, expose_rgb=True)()
    obs, info = env.reset(seed=args.seed)
    obs = np.asarray(obs)
    assert obs.shape == (4, 84, 84), f"expected obs (4,84,84), got {obs.shape}"
    assert obs.dtype == np.uint8, f"expected uint8 obs, got {obs.dtype}"
    n_actions = int(env.action_space.n)
    rgb = np.asarray(info.get("rgb"))
    print(f"[ok] obs {obs.shape} {obs.dtype}; n_actions={n_actions}; rgb {rgb.shape}")
    print(f"[ok] start position=({info.get('position_x', 0.0):.1f}, {info.get('position_y', 0.0):.1f})")

    rewards, tv_hits, term_seen = [], 0, False
    last_obs, last_next, last_action = obs, obs[3], 0
    for _ in range(args.preflight_steps):
        a = int(np.random.randint(0, n_actions))
        nobs, r, term, trunc, info = env.step(a)
        nobs = np.asarray(nobs)
        rewards.append(float(r))
        last_obs, last_next, last_action = obs, nobs[3], a
        obs = nobs
        if info.get("in_tv_zone"):
            tv_hits += 1
        if term or trunc:
            term_seen = True
            obs, info = env.reset()
            obs = np.asarray(obs)
    env.close()
    print(f"[ok] stepped {args.preflight_steps} actions; reward range=[{min(rewards):.5f}, {max(rewards):.5f}]; episode-end seen={term_seen}")
    if args.noisy_tv:
        frac = tv_hits / args.preflight_steps
        print(f"[tv] in_tv_zone fraction = {frac:.2f} (tv_radius={args.tv_radius}, tv_panel={args.tv_panel})")
        if frac == 0.0:
            print("[warn] never inside TV zone -> increase --tv-radius (the start room should register as the zone)")
        elif frac == 1.0:
            print("[warn] always inside TV zone -> --tv-radius may be too large for random-walk steps")

    print(f"\n[nets] forward pass for method={args.method} on real obs ...")
    o0 = torch.tensor(last_obs[None], dtype=torch.float32, device=device)
    ah = F.one_hot(torch.tensor([last_action], device=device), n_actions).float()
    agent = Agent(n_actions).to(device)
    with torch.no_grad():
        act, _, _, ve, vi = agent.get_action_and_value(o0)
    print(f"[ok] agent: action={int(act.item())} value_ext={float(ve.item()):.3f} value_int={float(vi.item()):.3f}")
    if args.method in {"c_v1", "c_v2", "cc"}:
        wm = ForwardCNN(n_actions).to(device)
        with torch.no_grad():
            pred = wm(o0, ah)
        assert pred.shape == (1, 1, 84, 84), f"WM output {pred.shape}"
        print(f"[ok] world model: output {tuple(pred.shape)}")
        if args.method == "cc":
            nc = CuriosityCriticCNN(n_actions).to(device)
            with torch.no_grad():
                base = nc(o0, ah)
            print(f"[ok] neural critic: output {tuple(base.shape)} baseline={float(base.item()):.3f}")
    if args.method == "rnd":
        rnd = RNDModel().to(device)
        nf = torch.tensor(last_next[None, None], dtype=torch.float32, device=device)
        with torch.no_grad():
            pf, tf = rnd(nf)
        print(f"[ok] RND: predictor {tuple(pf.shape)} target {tuple(tf.shape)}")

    print("\n" + line)
    print("PREFLIGHT PASSED  -  env, obs pipeline, and nets all work on real VizDoom data.")
    print("Next: Phase-0 smoke test (see how_to_run_curisoity_critic_for_vizdoom.md).")
    print(line)


def run_probe_maze(args: "Args"):
    """Drive a forward-biased policy to estimate the maze extent and recommend --tv-radius.

    The noisy-TV zone is a disk of radius --tv-radius around the episode start. For a
    meaningful trap the zone must be localized (the agent starts inside it but can
    escape toward a noise-free path to the vest). This probe measures how far the
    agent travels from start so the radius can be sized to ~the starting region
    instead of guessed.
    """
    line = "=" * 72
    print(line)
    print(f"PROBE-MAZE  scenario={args.scenario}  (sizing --tv-radius)")
    print(line)
    rng = np.random.default_rng(args.seed)
    env = make_env(args, idx=0, seed=args.seed, expose_rgb=False)()
    _, info = env.reset(seed=args.seed)
    sx, sy = float(info.get("position_x", 0.0)), float(info.get("position_y", 0.0))
    xs, ys = [], []
    for t in range(args.probe_maze_steps):
        a = 2 if (t % 4 != 0) else int(rng.integers(0, 3))  # mostly MOVE_FORWARD (idx 2), occasional turn
        _, _, term, trunc, info = env.step(a)
        xs.append(float(info.get("position_x", sx)))
        ys.append(float(info.get("position_y", sy)))
        if term or trunc:
            env.reset()
    env.close()
    xs, ys = np.array(xs), np.array(ys)
    d = np.sqrt((xs - sx) ** 2 + (ys - sy) ** 2)
    in_zone = float(np.mean(d <= args.tv_radius))
    suggested = 0.3 * float(d.max())
    print(f"start = ({sx:.0f}, {sy:.0f})")
    print(f"x range = [{xs.min():.0f}, {xs.max():.0f}]   y range = [{ys.min():.0f}, {ys.max():.0f}]")
    print(f"distance from start over walk: max={d.max():.0f}  median={np.median(d):.0f}")
    print(f"current --tv-radius={args.tv_radius:.0f}  ->  in-zone fraction over this walk = {in_zone:.2f}")
    print(
        f"SUGGESTED --tv-radius ~ {suggested:.0f}  (~30% of max reach {d.max():.0f}: a localized start-region "
        f"TV the agent starts inside but can leave, keeping the far rooms / goal noise-free)"
    )
    print(line)


# ----------------------------------------------------------------------------- #
#  Main
# ----------------------------------------------------------------------------- #
if __name__ == "__main__":
    args = tyro.cli(Args)
    assert args.method in {"random", "ppo", "c_v1", "c_v2", "rnd", "cc"}, f"unknown method {args.method}"
    assert args.scenario in SCENARIO_WADS, f"unknown scenario {args.scenario}"
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size

    if args.preflight:
        run_preflight(args)
        sys.exit(0)
    if args.probe_maze:
        run_probe_maze(args)
        sys.exit(0)

    # Capability switches derived from the chosen method.
    use_intrinsic = args.method in {"c_v1", "c_v2", "rnd", "cc"}
    uses_wm = args.method in {"c_v1", "c_v2", "cc"}
    uses_critic = args.method == "cc"
    uses_rnd = args.method == "rnd"
    is_random = args.method == "random"
    int_coef = args.int_coef if use_intrinsic else 0.0

    run_name = f"vizdoom_{args.scenario}{'_noisytv' if args.noisy_tv else ''}__{args.method}__{args.seed}__{int(time.time())}"
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=False,
            config=vars(args),
            name=run_name,
            monitor_gym=False,
            save_code=True,
        )
        # Make W&B charts use true environment interactions on the x-axis. Relying
        # on TensorBoard sync can leave the UI showing W&B's internal logging step.
        wandb.define_metric("global_step")
        wandb.define_metric("*", step_metric="global_step")
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{k}|{v}|" for k, v in vars(args).items()])),
    )
    viz_dir = f"runs/{run_name}/viz"
    video_dir = f"runs/{run_name}/videos"
    map_video_dir = f"runs/{run_name}/map_vids"
    os.makedirs(viz_dir, exist_ok=True)
    if args.capture_video:
        os.makedirs(video_dir, exist_ok=True)
        os.makedirs(map_video_dir, exist_ok=True)
    viz_map_lines = []
    viz_vest_positions = []
    if args.heatmap_every or args.capture_video:
        wad_path = os.path.join(args.wad_dir, SCENARIO_WADS[args.scenario])
        try:
            viz_map_lines = load_wad_map_lines(wad_path, args.doom_map)
            viz_vest_positions = load_wad_vest_positions(wad_path, args.doom_map)
            print(f"[viz] loaded {len(viz_map_lines)} WAD map lines for heatmap/map-video overlays")
            if viz_vest_positions:
                pretty = ", ".join(f"({x:.0f},{y:.0f})" for x, y in viz_vest_positions)
                print(f"[viz] vest marker(s): {pretty}")
        except Exception as exc:
            print(f"[viz] maze overlay unavailable: {exc}")

    # seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    envs = gym.vector.AsyncVectorEnv(
        [make_env(args, idx=i, seed=args.seed + i) for i in range(args.num_envs)],
        context="spawn",
    )
    num_actions = int(envs.single_action_space.n)
    obs_shape = envs.single_observation_space.shape  # (4, 84, 84)

    agent = Agent(num_actions).to(device)
    # A world model is trained for EVERY method (identical architecture + training) so WM accuracy is
    # comparable across methods. For cc/c_v1/c_v2 it also produces the intrinsic reward; for
    # rnd/ppo/random it is a passive evaluation WM (trained on collected data, never feeds the reward).
    world_model = ForwardCNN(num_actions=num_actions).to(device)
    world_model_prev = ForwardCNN(num_actions=num_actions).to(device) if args.method == "c_v2" else None
    neural_critic = CuriosityCriticCNN(num_actions=num_actions).to(device) if uses_critic else None
    rnd_model = RNDModel().to(device) if uses_rnd else None
    if world_model_prev is not None:
        world_model_prev.load_state_dict(world_model.state_dict())

    wm_parameters = list(world_model.parameters())  # WM is always trained
    policy_aux_parameters = []
    if not is_random:
        policy_aux_parameters += list(agent.parameters())  # random has no learned policy
    if uses_rnd:
        policy_aux_parameters += list(rnd_model.predictor.parameters())
    wm_optimizer = optim.Adam(wm_parameters, lr=args.learning_rate, eps=1e-5)
    policy_optimizer = (
        optim.Adam(policy_aux_parameters, lr=args.learning_rate, eps=1e-5) if policy_aux_parameters else None
    )
    critic_optimizer = (
        optim.Adam(neural_critic.parameters(), lr=args.learning_rate, eps=1e-5) if uses_critic else None
    )

    reward_rms = RunningMeanStd()
    obs_rms = RunningMeanStd(shape=(1, 1, 84, 84))
    discounted_reward = RewardForwardFilter(args.int_gamma)

    # storage
    obs = torch.zeros((args.num_steps, args.num_envs) + obs_shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs)).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    curiosity_rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    ext_values = torch.zeros((args.num_steps, args.num_envs)).to(device)
    int_values = torch.zeros((args.num_steps, args.num_envs)).to(device)
    next_frames = torch.zeros((args.num_steps, args.num_envs, 84, 84)).to(device)
    avg_returns = deque(maxlen=20)
    recent_goal_successes = deque(maxlen=100)

    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.Tensor(np.asarray(next_obs)).to(device)
    next_done = torch.zeros(args.num_envs).to(device)

    # Initialize observation normalization for every method. RND uses it for reward, and the passive
    # WM-eval model uses it for all methods, so the normalization protocol should be identical.
    print("Initializing observation normalization...")
    next_ob = []
    for _ in range(args.num_steps * args.num_iterations_obs_norm_init):
        acs = np.random.randint(0, num_actions, size=(args.num_envs,))
        s, _, _, _, _ = envs.step(acs)
        s = np.asarray(s)
        next_ob += s[:, 3, :, :].reshape([-1, 1, 84, 84]).tolist()
        if len(next_ob) % (args.num_steps * args.num_envs) == 0:
            obs_rms.update(np.stack(next_ob))
            next_ob = []
    print("Done.")
    # resync next_obs with the env state after the warmup steps
    next_obs, _ = envs.reset()
    next_obs = torch.Tensor(np.asarray(next_obs)).to(device)

    # Held-out deterministic transitions for the WM-accuracy eval (cached per seed; identical across methods).
    print("Preparing held-out WM-eval set...")
    holdout = collect_or_load_holdout(args)
    ckpt_dir = f"runs/{run_name}/checkpoints"
    if args.ckpt_every:
        os.makedirs(ckpt_dir, exist_ok=True)
    metrics_file = open(f"runs/{run_name}/metrics.jsonl", "a")
    with open(f"runs/{run_name}/run_meta.json", "w") as _meta:
        json.dump(
            {
                "args": vars(args),
                "git_commit": _git_commit(),
                "num_actions": num_actions,
                "device": str(device),
                "created": time.time(),
            },
            _meta,
            indent=2,
        )
    last_ep = {"avg": float("nan"), "ret": float("nan"), "len": float("nan")}  # latest episodic stats (for per-update plots)

    def log_episode_infos(infos, curiosity_step_mean):
        # gymnasium 1.x: infos["episode"] (dict of arrays) + "_episode" mask
        # gymnasium 0.29: infos["final_info"] (array of per-env dicts/None), episode r/l are arrays
        pairs = []
        if "episode" in infos:
            ep = infos["episode"]
            mask = infos.get("_episode", ep.get("_r"))
            rs, ls = np.asarray(ep["r"]).reshape(-1), np.asarray(ep["l"]).reshape(-1)
            idxs = range(len(rs)) if mask is None else np.where(np.asarray(mask))[0]
            pairs = [(float(rs[i]), float(ls[i])) for i in idxs]
        elif "final_info" in infos:
            for fi in infos["final_info"]:
                if fi is not None and "episode" in fi:
                    pairs.append(
                        (
                            float(np.asarray(fi["episode"]["r"]).reshape(-1)[0]),
                            float(np.asarray(fi["episode"]["l"]).reshape(-1)[0]),
                        )
                    )
        for r, length in pairs:
            avg_returns.append(r)
            last_ep["avg"], last_ep["ret"], last_ep["len"] = float(np.mean(avg_returns)), r, length
            episode_row = {
                "global_step": global_step,
                "charts/avg_episodic_return": float(np.mean(avg_returns)),
                "charts/episodic_return": r,
                "charts/episodic_length": length,
                "charts/episode_curiosity_reward": curiosity_step_mean,
            }
            writer.add_scalar("charts/avg_episodic_return", episode_row["charts/avg_episodic_return"], global_step)
            writer.add_scalar("charts/episodic_return", r, global_step)
            writer.add_scalar("charts/episodic_length", length, global_step)
            writer.add_scalar("charts/episode_curiosity_reward", curiosity_step_mean, global_step)
            if args.track:
                wandb.log(episode_row)
            print(f"global_step={global_step}, episodic_return={r:.3f}")

    for update in range(1, args.num_iterations + 1):
        if args.anneal_lr:
            frac = 1.0 - (update - 1.0) / args.num_iterations
            if policy_optimizer is not None:
                policy_optimizer.param_groups[0]["lr"] = frac * args.learning_rate

        rollout_obs_mean = torch.from_numpy(obs_rms.mean).to(device)
        rollout_obs_std = torch.sqrt(torch.from_numpy(obs_rms.var).to(device))
        ep_xs, ep_ys, ep_intr, ep_tv = [], [], [], []
        goal_hits_update = 0
        episode_ends_update = 0
        tt = {}  # per-update timing accumulators (seconds)
        if args.profile_timing and device.type == "cuda":
            torch.cuda.synchronize()
        _t_rollout0 = time.perf_counter()

        for step in range(0, args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            with torch.no_grad():
                if is_random:
                    action = torch.randint(0, num_actions, (args.num_envs,), device=device)
                    logprob = torch.zeros(args.num_envs, device=device)
                else:
                    value_ext, value_int = agent.get_value(next_obs)
                    ext_values[step], int_values[step] = value_ext.flatten(), value_int.flatten()
                    action, logprob, _, _, _ = agent.get_action_and_value(next_obs)
            actions[step] = action
            logprobs[step] = logprob

            next_obs_np, reward, terminated, truncated, infos = envs.step(action.cpu().numpy())
            reward_np = np.asarray(reward, dtype=np.float32).reshape(-1)
            done = np.logical_or(terminated, truncated)
            next_obs_np = np.asarray(next_obs_np)
            rewards[step] = torch.tensor(reward_np, dtype=torch.float32, device=device).view(-1)
            next_obs = torch.Tensor(next_obs_np).to(device)
            next_done = torch.Tensor(done.astype(np.float32)).to(device)
            next_frames[step] = next_obs[:, 3, :, :]
            goal_mask = reward_np > 0.0
            goal_hits_update += int(goal_mask.sum())
            episode_ends_update += int(np.asarray(done).sum())
            for hit, ended in zip(goal_mask, np.asarray(done, dtype=bool)):
                if ended:
                    recent_goal_successes.append(bool(hit))

            # intrinsic reward with frozen auxiliary models (RND timing)
            if use_intrinsic:
                obs_norm = _normalize_stack(obs[step], rollout_obs_mean, rollout_obs_std)
                target_next = _normalize_stack(
                    next_obs[:, 3, :, :].reshape(args.num_envs, 1, 84, 84), rollout_obs_mean, rollout_obs_std
                )
                action_onehot = F.one_hot(action.long(), num_classes=num_actions).float()
                with torch.no_grad():
                    if args.method == "rnd":
                        with _Timer(tt, "reward_aux", args.profile_timing, device):
                            pred_f, targ_f = rnd_model(target_next)
                        intr = (targ_f - pred_f).pow(2).sum(1) / 2
                    else:
                        wm_pred = world_model(obs_norm, action_onehot)
                        err_before = _reward_error_per_sample(wm_pred, target_next)
                        if args.method == "c_v1":
                            intr = err_before
                        elif args.method == "c_v2":
                            wm_pred_prev = world_model_prev(obs_norm, action_onehot)
                            err_prev = _reward_error_per_sample(wm_pred_prev, target_next)
                            intr = (err_prev - err_before).clamp(min=0)
                        else:  # cc
                            with _Timer(tt, "reward_aux", args.profile_timing, device):
                                critic_pred = neural_critic(obs_norm, action_onehot).squeeze(-1).clamp(min=0)
                            intr = (err_before - critic_pred).clamp(min=0)
                    curiosity_rewards[step] = intr.detach()

            # accumulate positions / intrinsic for heatmaps
            px, py = infos.get("position_x"), infos.get("position_y")
            if px is not None and py is not None:
                ep_xs.extend(np.asarray(px).tolist())
                ep_ys.extend(np.asarray(py).tolist())
                ep_intr.extend(curiosity_rewards[step].cpu().numpy().tolist())
                tvz = infos.get("in_tv_zone")
                ep_tv.extend(np.asarray(tvz).tolist() if tvz is not None else [0] * args.num_envs)

            log_episode_infos(infos, float(curiosity_rewards[step].mean().item()))

        if args.profile_timing and device.type == "cuda":
            torch.cuda.synchronize()
        tt["rollout"] = time.perf_counter() - _t_rollout0

        # ---- normalize intrinsic rewards ----
        if use_intrinsic:
            curiosity_reward_per_env = np.array(
                [discounted_reward.update(r) for r in curiosity_rewards.cpu().data.numpy().T]
            )
            mean, std, count = (
                np.mean(curiosity_reward_per_env),
                np.std(curiosity_reward_per_env),
                len(curiosity_reward_per_env),
            )
            reward_rms.update_from_moments(mean, std**2, count)
            curiosity_rewards /= np.sqrt(reward_rms.var)

        # ---- GAE (policy methods only) ----
        if not is_random:
            with torch.no_grad():
                next_value_ext, next_value_int = agent.get_value(next_obs)
                next_value_ext, next_value_int = next_value_ext.reshape(1, -1), next_value_int.reshape(1, -1)
                ext_advantages = torch.zeros_like(rewards, device=device)
                int_advantages = torch.zeros_like(curiosity_rewards, device=device)
                ext_lastgaelam = 0
                int_lastgaelam = 0
                for t in reversed(range(args.num_steps)):
                    if t == args.num_steps - 1:
                        ext_nextnonterminal = 1.0 - next_done
                        ext_nextvalues = next_value_ext
                        int_nextvalues = next_value_int
                    else:
                        ext_nextnonterminal = 1.0 - dones[t + 1]
                        ext_nextvalues = ext_values[t + 1]
                        int_nextvalues = int_values[t + 1]
                    int_nextnonterminal = 1.0
                    ext_delta = rewards[t] + args.gamma * ext_nextvalues * ext_nextnonterminal - ext_values[t]
                    int_delta = curiosity_rewards[t] + args.int_gamma * int_nextvalues * int_nextnonterminal - int_values[t]
                    ext_advantages[t] = ext_lastgaelam = (
                        ext_delta + args.gamma * args.gae_lambda * ext_nextnonterminal * ext_lastgaelam
                    )
                    int_advantages[t] = int_lastgaelam = (
                        int_delta + args.int_gamma * args.gae_lambda * int_nextnonterminal * int_lastgaelam
                    )
                ext_returns = ext_advantages + ext_values
                int_returns = int_advantages + int_values

        # ---- flatten ----
        b_obs = obs.reshape((-1,) + obs_shape)
        b_actions = actions.reshape(-1)
        b_next_frames = next_frames.reshape(-1, 84, 84)
        if not is_random:
            b_logprobs = logprobs.reshape(-1)
            b_ext_advantages = ext_advantages.reshape(-1)
            b_int_advantages = int_advantages.reshape(-1)
            b_ext_returns = ext_returns.reshape(-1)
            b_int_returns = int_returns.reshape(-1)
            b_ext_values = ext_values.reshape(-1)
            b_advantages = b_int_advantages * int_coef + b_ext_advantages * args.ext_coef

        obs_rms.update(b_obs[:, 3, :, :].reshape(-1, 1, 84, 84).cpu().numpy())
        update_obs_mean = torch.from_numpy(obs_rms.mean).to(device)
        update_obs_std = torch.sqrt(torch.from_numpy(obs_rms.var).to(device))

        # snapshot current WM as theta_{t-1} for the next rollout's V2 reward
        if args.method == "c_v2":
            world_model_prev.load_state_dict(world_model.state_dict())

        # ---- optimize ----
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        critic_loss_sum = err_before_sum = err_after_sum = critic_pred_sum = 0.0
        wm_loss_sum = fwd_loss_log = 0.0
        v_loss = pg_loss = entropy_loss = approx_kl = old_approx_kl = torch.tensor(0.0, device=device)
        n_mb = 0
        last_panel = None  # (obs_last, pred, true, critic_value) for WM viz
        if args.profile_timing and device.type == "cuda":
            torch.cuda.synchronize()
        _t_update0 = time.perf_counter()
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]
                mb_actions_long = b_actions.long()[mb_inds]

                mb_obs_norm = _normalize_stack(b_obs[mb_inds], update_obs_mean, update_obs_std)
                mb_target = _normalize_stack(
                    b_next_frames[mb_inds].reshape(-1, 1, 84, 84), update_obs_mean, update_obs_std
                )
                mb_actions_onehot = F.one_hot(mb_actions_long, num_classes=num_actions).float()
                mask = (torch.rand(len(mb_inds), device=device) < args.update_proportion).float()
                denom = torch.max(mask.sum(), torch.tensor(1.0, device=device))

                # world model -- trained for EVERY method (passive for rnd/ppo/random)
                with _Timer(tt, "wm_update", args.profile_timing, device):
                    wm_pred = world_model(mb_obs_norm, mb_actions_onehot)
                    wm_loss = (_update_loss_per_sample(wm_pred, mb_target.detach()) * mask).sum() / denom
                err_before_sum += _reward_error_per_sample(wm_pred.detach(), mb_target).mean().item()
                wm_loss_sum += wm_loss.item()

                # RND predictor (aux) -- trained in the combined loss for rnd only
                rnd_loss = torch.tensor(0.0, device=device)
                if uses_rnd:
                    with _Timer(tt, "aux_update", args.profile_timing, device):
                        pred_f, targ_f = rnd_model(mb_target)
                        rnd_loss = (F.mse_loss(pred_f, targ_f.detach(), reduction="none").mean(-1) * mask).sum() / denom
                    fwd_loss_log = rnd_loss.item()
                else:
                    fwd_loss_log = wm_loss.item()

                if is_random:
                    loss = wm_loss  # no learned policy; just train the passive world model
                else:
                    _, newlogprob, entropy, new_ext_values, new_int_values = agent.get_action_and_value(
                        b_obs[mb_inds], mb_actions_long
                    )
                    logratio = newlogprob - b_logprobs[mb_inds]
                    ratio = logratio.exp()
                    with torch.no_grad():
                        old_approx_kl = (-logratio).mean()
                        approx_kl = ((ratio - 1) - logratio).mean()
                        clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                    mb_advantages = b_advantages[mb_inds]
                    if args.norm_adv:
                        mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                    pg_loss1 = -mb_advantages * ratio
                    pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                    new_ext_values, new_int_values = new_ext_values.view(-1), new_int_values.view(-1)
                    if args.clip_vloss:
                        ext_v_loss_unclipped = (new_ext_values - b_ext_returns[mb_inds]) ** 2
                        ext_v_clipped = b_ext_values[mb_inds] + torch.clamp(
                            new_ext_values - b_ext_values[mb_inds], -args.clip_coef, args.clip_coef
                        )
                        ext_v_loss_clipped = (ext_v_clipped - b_ext_returns[mb_inds]) ** 2
                        ext_v_loss = 0.5 * torch.max(ext_v_loss_unclipped, ext_v_loss_clipped).mean()
                    else:
                        ext_v_loss = 0.5 * ((new_ext_values - b_ext_returns[mb_inds]) ** 2).mean()
                    int_v_loss = 0.5 * ((new_int_values - b_int_returns[mb_inds]) ** 2).mean() if use_intrinsic else 0.0
                    v_loss = ext_v_loss + int_v_loss
                    entropy_loss = entropy.mean()
                    loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef + wm_loss + rnd_loss

                wm_optimizer.zero_grad()
                if policy_optimizer is not None:
                    policy_optimizer.zero_grad()
                loss.backward()
                if args.max_grad_norm:
                    if policy_aux_parameters:
                        nn.utils.clip_grad_norm_(policy_aux_parameters, args.max_grad_norm)
                if policy_optimizer is not None:
                    policy_optimizer.step()
                wm_optimizer.step()

                # neural critic (aux) regresses the post-WM-update error on the same samples
                if uses_critic:
                    with _Timer(tt, "aux_update", args.profile_timing, device):
                        with torch.no_grad():
                            post_pred = world_model(mb_obs_norm, mb_actions_onehot)
                            err_after = _reward_error_per_sample(post_pred, mb_target).detach()
                        critic_pred_train = neural_critic(mb_obs_norm, mb_actions_onehot).squeeze(-1)
                        critic_loss = (F.mse_loss(critic_pred_train, err_after, reduction="none") * mask).sum() / denom
                        critic_optimizer.zero_grad()
                        critic_loss.backward()
                        critic_optimizer.step()
                    critic_loss_sum += critic_loss.item()
                    err_after_sum += err_after.mean().item()
                    critic_pred_sum += critic_pred_train.detach().clamp(min=0).mean().item()

                # stash one minibatch sample for the WM-prediction panel (every method trains a WM now)
                if last_panel is None and args.wm_panel_every and update % args.wm_panel_every == 0:
                    with torch.no_grad():
                        cv = float(neural_critic(mb_obs_norm[:1], mb_actions_onehot[:1]).item()) if uses_critic else None
                        last_panel = (
                            b_obs[mb_inds][0, 3].cpu().numpy(),
                            world_model(mb_obs_norm[:1], mb_actions_onehot[:1])[0, 0].cpu().numpy(),
                            mb_target[0, 0].cpu().numpy(),
                            cv,
                        )
                n_mb += 1

            if args.target_kl is not None and not is_random and approx_kl > args.target_kl:
                break
        if args.profile_timing and device.type == "cuda":
            torch.cuda.synchronize()
        tt["update"] = time.perf_counter() - _t_update0

        # ---- held-out world-model accuracy (every method trains a WM) ----
        wm_holdout = float("nan")
        if args.eval_every and update % args.eval_every == 0:
            _t_eval0 = time.perf_counter()
            wm_holdout = eval_wm_holdout(world_model, holdout, obs_rms, num_actions, device)
            tt["eval"] = time.perf_counter() - _t_eval0

        # ---- scalar logging (mirrored to metrics.jsonl) ----
        sps = int(global_step / (time.time() - start_time))
        tt["total"] = tt.get("rollout", 0.0) + tt.get("update", 0.0)
        tv_arr = np.asarray(ep_tv, dtype=bool)
        intr_arr = np.asarray(ep_intr, dtype=np.float32)
        tv_count = int(tv_arr.sum()) if tv_arr.size else 0
        non_tv_count = int((~tv_arr).sum()) if tv_arr.size else 0
        policy_lr = policy_optimizer.param_groups[0]["lr"] if policy_optimizer is not None else wm_optimizer.param_groups[0]["lr"]
        critic_lr = critic_optimizer.param_groups[0]["lr"] if critic_optimizer is not None else 0.0
        row = {
            "global_step": global_step,
            "update": update,
            "charts/learning_rate": policy_lr,
            "charts/wm_learning_rate": wm_optimizer.param_groups[0]["lr"],
            "charts/critic_learning_rate": critic_lr,
            "charts/SPS": sps,
            "charts/goal_hits_update": goal_hits_update,
            "charts/episodes_update": episode_ends_update,
            "charts/goal_reached_rate_update": goal_hits_update / max(episode_ends_update, 1),
            "mechanism/tv_zone_fraction": float(tv_arr.mean()) if tv_arr.size else 0.0,
            "mechanism/intrinsic_tv_mean_raw": float(intr_arr[tv_arr].mean()) if tv_count else 0.0,
            "mechanism/intrinsic_non_tv_mean_raw": float(intr_arr[~tv_arr].mean()) if non_tv_count else 0.0,
            "mechanism/tv_sample_count": tv_count,
            "mechanism/non_tv_sample_count": non_tv_count,
            "losses/wm_loss": wm_loss_sum / max(n_mb, 1),
            "losses/error_before": err_before_sum / max(n_mb, 1),
            "time/rollout_s": tt.get("rollout", 0.0),
            "time/update_s": tt.get("update", 0.0),
            "time/total_s": tt.get("total", 0.0),
            "time/reward_aux_s": tt.get("reward_aux", 0.0),
            "time/wm_update_s": tt.get("wm_update", 0.0),
            "time/aux_update_s": tt.get("aux_update", 0.0),
            "time/aux_total_s": tt.get("reward_aux", 0.0) + tt.get("aux_update", 0.0),
            "time/eval_s": tt.get("eval", 0.0),
        }
        if not is_random:
            row["losses/value_loss"] = float(v_loss.item())
            row["losses/policy_loss"] = float(pg_loss.item())
            row["losses/entropy"] = float(entropy_loss.item())
            row["losses/approx_kl"] = float(approx_kl.item())
            row["losses/old_approx_kl"] = float(old_approx_kl.item())
        if use_intrinsic:
            row["losses/fwd_loss"] = fwd_loss_log
            row["charts/curiosity_reward_mean"] = float(curiosity_rewards.mean().item())
        if uses_critic:
            row["losses/critic_loss"] = critic_loss_sum / max(n_mb, 1)
            row["losses/error_after"] = err_after_sum / max(n_mb, 1)
            row["charts/critic_pred_mean"] = critic_pred_sum / max(n_mb, 1)
        if not np.isnan(wm_holdout):
            row["eval/wm_holdout_l2"] = wm_holdout
        # dense per-update episodic plots (new _periodic keys; non-destructive) so methods that solve
        # rarely still span the full x-axis in wandb instead of looking truncated
        if not np.isnan(last_ep["avg"]):
            row["charts/avg_episodic_return_periodic"] = last_ep["avg"]
            row["charts/episodic_return_periodic"] = last_ep["ret"]
            row["charts/episodic_length_periodic"] = last_ep["len"]
        if recent_goal_successes:
            row["charts/goal_reached_rate_100ep"] = float(np.mean(recent_goal_successes))
        for key, val in row.items():
            if key not in ("global_step", "update"):
                writer.add_scalar(key, val, global_step)
        if args.track:
            wandb.log(row)
        metrics_file.write(json.dumps(row) + "\n")
        metrics_file.flush()
        print(f"update={update} SPS={sps} wm_holdout={wm_holdout:.2f}")

        # ---- periodic full checkpoint (so any instant can be reconstructed without rerunning) ----
        if args.ckpt_every and update % args.ckpt_every == 0:
            save_full_checkpoint(
                f"{ckpt_dir}/ckpt_update{update:06d}.cleanrl_model", args, global_step, update,
                agent, world_model, neural_critic, rnd_model, obs_rms, reward_rms, world_model_prev=world_model_prev,
            )

        # ---- visualization dumps ----
        if last_panel is not None:
            p = save_wm_panel(f"{viz_dir}/wm_panel_update{update:06d}.png", *last_panel)
            if p and args.track:
                wandb.log({"global_step": global_step, "viz/wm_panel": wandb.Image(p)})
        if args.heatmap_every and update % args.heatmap_every == 0:
            p = save_heatmaps(
                f"{viz_dir}/heatmap_update{update:06d}.png",
                ep_xs,
                ep_ys,
                ep_intr,
                ep_tv,
                map_lines=viz_map_lines,
            )
            if p and args.track:
                wandb.log({"global_step": global_step, "viz/heatmap": wandb.Image(p)})
            # raw position/intrinsic data so any heatmap/coverage analysis can be re-rendered later
            np.savez_compressed(
                f"{viz_dir}/positions_update{update:06d}.npz",
                x=np.asarray(ep_xs, dtype=np.float32),
                y=np.asarray(ep_ys, dtype=np.float32),
                intr=np.asarray(ep_intr, dtype=np.float32),
                in_tv=np.asarray(ep_tv, dtype=np.float32),
            )
        if args.capture_video and args.video_every and update % args.video_every == 0:
            map_video_path = f"{map_video_dir}/update{update:06d}.mp4"
            p = capture_video(
                f"{video_dir}/update{update:06d}.mp4",
                args,
                agent,
                device,
                args.seed,
                map_path=map_video_path,
                map_lines=viz_map_lines,
                vest_positions=viz_vest_positions,
            )
            if p and args.track:
                wandb.log({"global_step": global_step, "viz/video": wandb.Video(p)})
                obs_mp4 = os.path.splitext(p)[0] + "_obs.mp4"
                if os.path.isfile(obs_mp4):
                    wandb.log({"global_step": global_step, "viz/video_obs": wandb.Video(obs_mp4)})
                if os.path.isfile(map_video_path):
                    wandb.log({"global_step": global_step, "viz/map_video": wandb.Video(map_video_path)})

    # ---- final save ----
    if args.save_model:
        path = f"runs/{run_name}/{args.exp_name}.cleanrl_model"
        save_full_checkpoint(
            path, args, global_step, args.num_iterations, agent, world_model,
            neural_critic, rnd_model, obs_rms, reward_rms, world_model_prev=world_model_prev,
        )
        print(f"model saved to {path}")
        if args.track:
            wandb.save(path)

    metrics_file.close()
    envs.close()
    writer.close()
    run_post_training_plots(f"runs/{run_name}", args)
    if args.track:
        wandb.finish()
