# Build the held-out WM-evaluation sets once, up front, and store them per seed.
#
# Each set is a forward-biased random walk through the DETERMINISTIC (no-noise) MyWayHome maze,
# giving (obs_stack, action, next_frame) transitions that span the maze (not just the spawn room).
# The training script (ppo_curiosity_critic_vizdoom.py) loads the matching-seed file and scores
# EVERY method's world model on it, so WM accuracy is comparable across methods of the same seed.
#
# It also writes a coverage HEATMAP per seed, overlaid on the WAD's top-down map
# geometry, so you can visually confirm the samples are spread across rooms and
# corridors before trusting the metric.
#
# Usage (from repo root):
#   python cleanrl/build_holdout.py --scenario sparse --seeds 1 2 3 --size 10000
#
# Outputs (default ./vizdoom_holdout/):
#   holdout_<scenario>_seed<k>.npz            (obs, act, next_frame, x, y)
#   holdout_<scenario>_seed<k>_coverage.png   (visitation heatmap + maze overlay)
import argparse
import os
import re
import struct

import numpy as np

from ppo_curiosity_critic_vizdoom import Args, SCENARIO_WADS, collect_holdout_transitions


def _read_textmap(wad_path, doom_map="map01"):
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
    """Return top-down line segments from a UDMF WAD for plotting.

    The bundled MyWayHome WADs use UDMF (`TEXTMAP`) instead of classic LINEDEFS /
    VERTEXES lumps, so a tiny text parser is enough for a visual map overlay.
    """
    text = _read_textmap(wad_path, doom_map)
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


def _plot_map_lines(ax, map_lines, alpha=0.35, zorder=1):
    for blocking in (False, True):
        segments = [line for line in map_lines if line[4] == blocking]
        if not segments:
            continue
        color = "0.25" if blocking else "0.55"
        linewidth = 1.0 if blocking else 0.6
        line_alpha = alpha if blocking else alpha * 0.65
        for x1, y1, x2, y2, _ in segments:
            ax.plot([x1, x2], [y1, y2], color=color, linewidth=linewidth, alpha=line_alpha, zorder=zorder)


def save_coverage_heatmap(xs, ys, path, title, bins=40, wad_path=None, doom_map="map01"):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    xs, ys = np.asarray(xs), np.asarray(ys)
    map_lines = []
    if wad_path:
        try:
            map_lines = load_wad_map_lines(wad_path, doom_map)
        except Exception as exc:
            print(f"[warn] maze overlay unavailable for {wad_path}: {exc}")

    x_vals, y_vals = [xs], [ys]
    if map_lines:
        x_vals.append(np.asarray([[line[0], line[2]] for line in map_lines]).reshape(-1))
        y_vals.append(np.asarray([[line[1], line[3]] for line in map_lines]).reshape(-1))
    all_x, all_y = np.concatenate(x_vals), np.concatenate(y_vals)
    x_pad = max(1.0, 0.04 * float(all_x.max() - all_x.min()))
    y_pad = max(1.0, 0.04 * float(all_y.max() - all_y.min()))
    extent = [float(all_x.min() - x_pad), float(all_x.max() + x_pad), float(all_y.min() - y_pad), float(all_y.max() + y_pad)]

    hist, xe, ye = np.histogram2d(xs, ys, bins=bins, range=[[extent[0], extent[1]], [extent[2], extent[3]]])
    covered = int((hist > 0).sum())
    fig, ax = plt.subplots(figsize=(6.0, 5.2))
    ax.set_facecolor("#f5f5f2")
    if map_lines:
        _plot_map_lines(ax, map_lines, alpha=0.32, zorder=1)
    cmap = plt.cm.viridis.copy()
    cmap.set_bad(alpha=0.0)
    heat = np.ma.masked_where(hist.T == 0, hist.T)
    im = ax.imshow(
        heat,
        origin="lower",
        aspect="equal",
        cmap=cmap,
        extent=[xe[0], xe[-1], ye[0], ye[-1]],
        interpolation="nearest",
        alpha=0.72 if map_lines else 1.0,
        zorder=2,
    )
    ax.scatter(xs, ys, s=2.0, c="black", alpha=0.20, linewidths=0, zorder=3)
    if map_lines:
        _plot_map_lines(ax, map_lines, alpha=0.42, zorder=4)
    ax.set_title(
        f"{title}\ncovered cells = {covered}/{bins*bins}   "
        f"x[{xs.min():.0f},{xs.max():.0f}]  y[{ys.min():.0f},{ys.max():.0f}]",
        fontsize=9,
    )
    ax.set_xlabel("x (game units)")
    ax.set_ylabel("y (game units)")
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_aspect("equal", adjustable="box")
    fig.colorbar(im, ax=ax, fraction=0.046, label="samples")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return covered, bins * bins


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="sparse", choices=["sparse", "very_sparse", "dense"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--size", type=int, default=10000, help="transitions per held-out set")
    ap.add_argument("--p-forward", type=float, default=0.55, help="probability of MOVE_FORWARD in the walk")
    ap.add_argument("--out", default="./vizdoom_holdout")
    ap.add_argument("--wad-dir", default="./vizdoom_scenarios")
    ap.add_argument("--doom-map", default="map01")
    cli = ap.parse_args()
    os.makedirs(cli.out, exist_ok=True)
    wad_path = os.path.join(cli.wad_dir, SCENARIO_WADS[cli.scenario])

    for seed in cli.seeds:
        args = Args(
            scenario=cli.scenario,
            holdout_dir=cli.out,
            wad_dir=cli.wad_dir,
            holdout_size=cli.size,
            doom_map=cli.doom_map,
        )
        obs, act, nxt, xs, ys = collect_holdout_transitions(args, seed, cli.size, cli.p_forward)
        npz_path = os.path.join(cli.out, f"holdout_{cli.scenario}_seed{seed}.npz")
        np.savez_compressed(npz_path, obs=obs, act=act, next_frame=nxt, x=xs, y=ys)
        png_path = os.path.join(cli.out, f"holdout_{cli.scenario}_seed{seed}_coverage.png")
        covered, total = save_coverage_heatmap(
            xs,
            ys,
            png_path,
            f"held-out coverage: {cli.scenario} seed {seed} (n={len(obs)})",
            wad_path=wad_path,
            doom_map=cli.doom_map,
        )
        print(f"seed {seed}: {npz_path}  ({len(obs)} transitions)  |  coverage {covered}/{total} cells  ->  {png_path}")

    print("\nDone. Eyeball the *_coverage.png heatmaps: samples should cover rooms and corridors")
    print("on the maze overlay, not just the spawn or wall-bump loops. If any look thin, raise --size or --p-forward.")


if __name__ == "__main__":
    main()
