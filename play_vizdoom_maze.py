# Walk the *exact* MyWayHome maze your agents trained on — with your own keyboard.
#
# Spectator mode: a window opens and YOU control the player. Great for giving an
# audience a feel for the sparse-exploration problem (find the vest, no other rewards).
#
# Mac usage (from repo root, in the env where `pip install vizdoom` succeeded):
#   python play_vizdoom_maze.py                 # sparse maze (the headline experiment)
#   python play_vizdoom_maze.py --scenario very_sparse
#   python play_vizdoom_maze.py --res 1280x720  # bigger window for a projector
#
# Controls (default Doom bindings in the FOCUSED window — click it first):
#   ↑ : move forward    ← / → : turn left / right    (mouse also turns)
#   (the agent itself only has forward + turn-left + turn-right — same as you)
#   Esc : quit          the episode auto-restarts when you find the vest.
import argparse
import os

import vizdoom as vzd

SCENARIO_WADS = {
    "sparse": "my_way_home_sparse.wad",
    "very_sparse": "my_way_home_verySparse.wad",
    "dense": "my_way_home_dense.wad",
}


def main():
    ap = argparse.ArgumentParser(description="Human-playable MyWayHome maze (ViZDoom spectator mode).")
    ap.add_argument("--scenario", default="sparse", choices=list(SCENARIO_WADS))
    ap.add_argument("--wad-dir", default="./vizdoom_scenarios")
    ap.add_argument("--doom-map", default="map01")
    ap.add_argument("--res", default="640x480", help="window resolution, e.g. 640x480 / 1280x720")
    ap.add_argument("--timeout-tics", type=int, default=20000, help="episode length cap (0 = unlimited)")
    ap.add_argument("--agent-view", action="store_true",
                    help="hide HUD/weapon to show exactly the spartan view the agent gets")
    cli = ap.parse_args()

    wad_path = os.path.join(cli.wad_dir, SCENARIO_WADS[cli.scenario])
    if not os.path.isfile(wad_path):
        raise SystemExit(f"scenario wad not found: {wad_path} (run from the repo root)")

    res_name = "RES_" + cli.res.upper().replace("X", "X")  # e.g. RES_640X480
    if not hasattr(vzd.ScreenResolution, res_name):
        raise SystemExit(f"unsupported --res {cli.res}; try 640x480, 800x600, 1024x768, 1280x720, 1920x1080")

    game = vzd.DoomGame()
    game.set_doom_scenario_path(wad_path)
    game.set_doom_map(cli.doom_map)
    game.set_screen_resolution(getattr(vzd.ScreenResolution, res_name))
    game.set_screen_format(vzd.ScreenFormat.RGB24)
    # match the trained agent's action set: forward + turn left/right only
    game.clear_available_buttons()
    for button in (vzd.Button.TURN_LEFT, vzd.Button.TURN_RIGHT, vzd.Button.MOVE_FORWARD):
        game.add_available_button(button)
    game.clear_available_game_variables()
    for gv in (vzd.GameVariable.POSITION_X, vzd.GameVariable.POSITION_Y, vzd.GameVariable.ANGLE):
        game.add_available_game_variable(gv)
    # render flags: clean by default; --agent-view strips everything to the agent's view
    game.set_render_hud(not cli.agent_view)
    game.set_render_crosshair(False)
    game.set_render_weapon(not cli.agent_view)
    game.set_render_decals(not cli.agent_view)
    game.set_render_particles(not cli.agent_view)
    game.set_living_reward(-0.0001)
    game.set_episode_timeout(cli.timeout_tics)
    game.set_window_visible(True)
    game.set_mode(vzd.Mode.SPECTATOR)  # <-- YOU drive
    game.init()

    print("\n" + "=" * 64)
    print(f"  MyWayHome [{cli.scenario}] — find the VEST. It's the only reward.")
    print("  Controls: ↑ forward, ←/→ turn (mouse also turns).  CLICK the window to focus it.")
    print("  Esc to quit.  Episode restarts automatically when you reach the vest.")
    print("=" * 64 + "\n")

    episode = 0
    try:
        while True:
            episode += 1
            game.new_episode()
            total = 0.0
            tics = 0
            while not game.is_episode_finished():
                game.advance_action()       # one tic of YOUR input
                r = game.get_last_reward()
                total += r
                tics += 1
                if r > 0.0:                  # the vest pickup is a positive reward
                    print(f"  >>> VEST FOUND! episode {episode} in {tics} tics  (return {total:+.3f})\n")
            st = game.get_state()
            if st is None or game.get_episode_time() <= 0:
                pass
            print(f"  episode {episode} ended  (tics={tics}, return={total:+.3f}) — restarting...")
    except KeyboardInterrupt:
        pass
    finally:
        game.close()
        print("\nclosed. thanks for playing.")


if __name__ == "__main__":
    main()
