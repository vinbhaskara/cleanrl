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
import os
import random
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
    wandb_project_name: str = "curiosity-critic-vizdoom"
    """the wandb's project name"""
    wandb_entity: str = None
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
    target_kl: float = None
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
    num_iterations_obs_norm_init: int = 10
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
    tv_radius: float = 100.0
    """the agent is in the TV zone when within this distance (game units) of its episode start"""
    tv_panel: int = 42
    """side length (pixels, in the 84x84 frame) of the square noise panel"""

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

    def _read_state(self):
        state = self.game.get_state()
        if state is None:
            return self._last_rgb, (0.0, 0.0, 0.0)
        buf = state.screen_buffer
        if buf.ndim == 3 and buf.shape[0] == 3:  # CHW -> HWC defensive
            buf = np.transpose(buf, (1, 2, 0))
        gv = state.game_variables
        pos = (float(gv[0]), float(gv[1]), float(gv[2])) if gv is not None and len(gv) >= 3 else (0.0, 0.0, 0.0)
        self._last_rgb = buf
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
        info = {"position_x": pos[0], "position_y": pos[1], "position_angle": pos[2]}
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

    def __init__(self, env, tv_radius: float, tv_panel: int, rng_seed: int = 0):
        super().__init__(env)
        self.tv_radius = tv_radius
        self.tv_panel = tv_panel
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
            obs[:p, :p] = self._rng.integers(0, 256, size=(p, p), dtype=np.uint8)
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
        env = NoisyTVWrapper(env, tv_radius=args.tv_radius, tv_panel=args.tv_panel, rng_seed=seed)
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


def save_heatmaps(path, xs, ys, intr, in_tv, bins=40):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        xs, ys, intr = np.asarray(xs), np.asarray(ys), np.asarray(intr)
        if xs.size == 0:
            return None
        visit, xe, ye = np.histogram2d(xs, ys, bins=bins)
        rsum, _, _ = np.histogram2d(xs, ys, bins=[xe, ye], weights=intr)
        with np.errstate(invalid="ignore", divide="ignore"):
            rmean = np.where(visit > 0, rsum / visit, 0.0)
        fig, axes = plt.subplots(1, 2, figsize=(9, 4))
        for ax, data, title in ((axes[0], visit.T, "visitation"), (axes[1], rmean.T, "mean intrinsic reward")):
            im = ax.imshow(data, origin="lower", aspect="auto", cmap="viridis")
            ax.set_title(title, fontsize=10)
            ax.set_xlabel("x")
            ax.set_ylabel("y")
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


@torch.no_grad()
def capture_video(path, args, agent, device, seed):
    """Roll out the current policy in a fresh single env and write an mp4."""
    try:
        import imageio

        env = make_env(args, idx=0, seed=seed + 999, expose_rgb=True)()
        obs, _ = env.reset()
        frames = []
        for _ in range(args.video_steps):
            obs_t = torch.tensor(np.asarray(obs), dtype=torch.float32, device=device).unsqueeze(0)
            action, _, _, _, _ = agent.get_action_and_value(obs_t)
            obs, _, term, trunc, info = env.step(int(action.item()))
            frames.append(np.asarray(info.get("rgb")))
            if term or trunc:
                obs, _ = env.reset()
        env.close()
        imageio.mimsave(path, frames, fps=30)
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
    for mod in ("matplotlib", "imageio"):
        try:
            importlib.import_module(mod)
            print(f"[ok] import {mod} (visualization)")
        except Exception as exc:
            print(f"[warn] {mod} missing -> visualization will be skipped: {exc}")

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
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=False,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{k}|{v}|" for k, v in vars(args).items()])),
    )
    viz_dir = f"runs/{run_name}/viz"
    video_dir = f"runs/{run_name}/videos"
    os.makedirs(viz_dir, exist_ok=True)
    if args.capture_video:
        os.makedirs(video_dir, exist_ok=True)

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
    world_model = ForwardCNN(num_actions=num_actions).to(device) if uses_wm else None
    world_model_prev = ForwardCNN(num_actions=num_actions).to(device) if args.method == "c_v2" else None
    neural_critic = CuriosityCriticCNN(num_actions=num_actions).to(device) if uses_critic else None
    rnd_model = RNDModel().to(device) if uses_rnd else None
    if world_model_prev is not None:
        world_model_prev.load_state_dict(world_model.state_dict())

    combined_parameters = list(agent.parameters())
    if uses_wm:
        combined_parameters += list(world_model.parameters())
    if uses_rnd:
        combined_parameters += list(rnd_model.predictor.parameters())
    optimizer = optim.Adam(combined_parameters, lr=args.learning_rate, eps=1e-5)
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

    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.Tensor(np.asarray(next_obs)).to(device)
    next_done = torch.zeros(args.num_envs).to(device)

    # Initialize observation normalization with random data (WM/RND methods only).
    if use_intrinsic:
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
            writer.add_scalar("charts/avg_episodic_return", float(np.mean(avg_returns)), global_step)
            writer.add_scalar("charts/episodic_return", r, global_step)
            writer.add_scalar("charts/episodic_length", length, global_step)
            writer.add_scalar("charts/episode_curiosity_reward", curiosity_step_mean, global_step)
            print(f"global_step={global_step}, episodic_return={r:.3f}")

    for update in range(1, args.num_iterations + 1):
        if args.anneal_lr:
            frac = 1.0 - (update - 1.0) / args.num_iterations
            optimizer.param_groups[0]["lr"] = frac * args.learning_rate
            if critic_optimizer is not None:
                critic_optimizer.param_groups[0]["lr"] = frac * args.learning_rate

        rollout_obs_mean = torch.from_numpy(obs_rms.mean).to(device)
        rollout_obs_std = torch.sqrt(torch.from_numpy(obs_rms.var).to(device))
        ep_xs, ep_ys, ep_intr, ep_tv = [], [], [], []

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
            done = np.logical_or(terminated, truncated)
            next_obs_np = np.asarray(next_obs_np)
            rewards[step] = torch.tensor(np.asarray(reward), dtype=torch.float32, device=device).view(-1)
            next_obs = torch.Tensor(next_obs_np).to(device)
            next_done = torch.Tensor(done.astype(np.float32)).to(device)
            next_frames[step] = next_obs[:, 3, :, :]

            # intrinsic reward with frozen auxiliary models (RND timing)
            if use_intrinsic:
                obs_norm = _normalize_stack(obs[step], rollout_obs_mean, rollout_obs_std)
                target_next = _normalize_stack(
                    next_obs[:, 3, :, :].reshape(args.num_envs, 1, 84, 84), rollout_obs_mean, rollout_obs_std
                )
                action_onehot = F.one_hot(action.long(), num_classes=num_actions).float()
                with torch.no_grad():
                    if args.method == "rnd":
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

        # ---- random baseline: no learning, just logging ----
        if is_random:
            writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
            if args.heatmap_every and update % args.heatmap_every == 0:
                p = save_heatmaps(f"{viz_dir}/heatmap_update{update:06d}.png", ep_xs, ep_ys, ep_intr, ep_tv)
                if p and args.track:
                    wandb.log({"viz/heatmap": wandb.Image(p)}, step=global_step)
            continue

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

        # ---- GAE ----
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
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape(-1)
        b_ext_advantages = ext_advantages.reshape(-1)
        b_int_advantages = int_advantages.reshape(-1)
        b_ext_returns = ext_returns.reshape(-1)
        b_int_returns = int_returns.reshape(-1)
        b_ext_values = ext_values.reshape(-1)
        b_next_frames = next_frames.reshape(-1, 84, 84)
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
        critic_loss_sum = err_before_sum = err_after_sum = critic_pred_sum = fwd_loss_log = 0.0
        n_mb = 0
        last_panel = None  # (obs_last, pred, true, critic_value) for WM viz
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]
                mb_actions_long = b_actions.long()[mb_inds]

                forward_loss = torch.tensor(0.0, device=device)
                if uses_wm or uses_rnd:
                    mb_obs_norm = _normalize_stack(b_obs[mb_inds], update_obs_mean, update_obs_std)
                    mb_target = _normalize_stack(
                        b_next_frames[mb_inds].reshape(-1, 1, 84, 84), update_obs_mean, update_obs_std
                    )
                    mb_actions_onehot = F.one_hot(mb_actions_long, num_classes=num_actions).float()
                    mask = (torch.rand(len(mb_inds), device=device) < args.update_proportion).float()
                    denom = torch.max(mask.sum(), torch.tensor(1.0, device=device))
                    if uses_rnd:
                        pred_f, targ_f = rnd_model(mb_target)
                        fl = F.mse_loss(pred_f, targ_f.detach(), reduction="none").mean(-1)
                        forward_loss = (fl * mask).sum() / denom
                    else:
                        wm_pred = world_model(mb_obs_norm, mb_actions_onehot)
                        fl = _update_loss_per_sample(wm_pred, mb_target.detach())
                        forward_loss = (fl * mask).sum() / denom
                        err_before_sum += _reward_error_per_sample(wm_pred.detach(), mb_target).mean().item()
                    fwd_loss_log = forward_loss.item()

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
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef + forward_loss

                optimizer.zero_grad()
                loss.backward()
                if args.max_grad_norm:
                    nn.utils.clip_grad_norm_(combined_parameters, args.max_grad_norm)
                optimizer.step()

                # neural critic regresses the post-WM-update error on the same samples
                if uses_critic:
                    with torch.no_grad():
                        post_pred = world_model(mb_obs_norm, mb_actions_onehot)
                        err_after = _reward_error_per_sample(post_pred, mb_target).detach()
                    critic_pred_train = neural_critic(mb_obs_norm, mb_actions_onehot).squeeze(-1)
                    critic_loss_per = F.mse_loss(critic_pred_train, err_after, reduction="none")
                    critic_loss = (critic_loss_per * mask).sum() / denom
                    critic_optimizer.zero_grad()
                    critic_loss.backward()
                    if args.max_grad_norm:
                        nn.utils.clip_grad_norm_(neural_critic.parameters(), args.max_grad_norm)
                    critic_optimizer.step()
                    critic_loss_sum += critic_loss.item()
                    err_after_sum += err_after.mean().item()
                    critic_pred_sum += critic_pred_train.detach().clamp(min=0).mean().item()

                # stash one minibatch sample for the WM-prediction panel
                if uses_wm and last_panel is None and args.wm_panel_every and update % args.wm_panel_every == 0:
                    with torch.no_grad():
                        cv = None
                        if uses_critic:
                            cv = float(neural_critic(mb_obs_norm[:1], mb_actions_onehot[:1]).item())
                        last_panel = (
                            b_obs[mb_inds][0, 3].cpu().numpy(),
                            world_model(mb_obs_norm[:1], mb_actions_onehot[:1])[0, 0].cpu().numpy(),
                            mb_target[0, 0].cpu().numpy(),
                            cv,
                        )
                n_mb += 1

            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        # ---- logging ----
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        if use_intrinsic:
            writer.add_scalar("losses/fwd_loss", fwd_loss_log, global_step)
            writer.add_scalar("charts/curiosity_reward_mean", float(curiosity_rewards.mean().item()), global_step)
        if uses_wm:
            writer.add_scalar("losses/error_before", err_before_sum / max(n_mb, 1), global_step)
        if uses_critic:
            writer.add_scalar("losses/critic_loss", critic_loss_sum / max(n_mb, 1), global_step)
            writer.add_scalar("losses/error_after", err_after_sum / max(n_mb, 1), global_step)
            writer.add_scalar("charts/critic_pred_mean", critic_pred_sum / max(n_mb, 1), global_step)
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))

        # ---- visualization dumps ----
        if uses_wm and last_panel is not None:
            p = save_wm_panel(f"{viz_dir}/wm_panel_update{update:06d}.png", *last_panel)
            if p and args.track:
                wandb.log({"viz/wm_panel": wandb.Image(p)}, step=global_step)
        if args.heatmap_every and update % args.heatmap_every == 0:
            p = save_heatmaps(f"{viz_dir}/heatmap_update{update:06d}.png", ep_xs, ep_ys, ep_intr, ep_tv)
            if p and args.track:
                wandb.log({"viz/heatmap": wandb.Image(p)}, step=global_step)
        if args.capture_video and args.video_every and update % args.video_every == 0:
            p = capture_video(f"{video_dir}/update{update:06d}.mp4", args, agent, device, args.seed)
            if p and args.track:
                wandb.log({"viz/video": wandb.Video(p)}, step=global_step)

    # ---- save ----
    if args.save_model:
        model_dir = f"runs/{run_name}"
        checkpoint = {
            "args": vars(args),
            "global_step": global_step,
            "policy_model": _cpu_state_dict(agent),
            "obs_rms_mean": obs_rms.mean,
            "obs_rms_var": obs_rms.var,
            "reward_rms_mean": reward_rms.mean,
            "reward_rms_var": reward_rms.var,
        }
        if uses_wm:
            checkpoint["world_model"] = _cpu_state_dict(world_model)
        if uses_critic:
            checkpoint["neural_critic"] = _cpu_state_dict(neural_critic)
        if uses_rnd:
            checkpoint["rnd_predictor"] = _cpu_state_dict(rnd_model.predictor)
            checkpoint["rnd_target"] = _cpu_state_dict(rnd_model.target)
        path = f"{model_dir}/{args.exp_name}.cleanrl_model"
        torch.save(checkpoint, path)
        print(f"model saved to {path}")
        if args.track:
            wandb.save(path)

    envs.close()
    writer.close()
