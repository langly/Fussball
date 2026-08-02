"""Command line entry point.

    python -m football bots/tactician.py bots/chaser.py
    python -m football bots/a.py bots/b.py --headless --matches 20
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace

from .config import Rules
from .engine import AWAY, HOME, Match
from .loader import BotError, load_controller
from .sandbox import SandboxError, load_sandboxed


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="football",
        description="A 5-a-side football simulator driven by two Python scripts.",
    )
    p.add_argument("home", help="python script controlling the home team")
    p.add_argument("away", help="python script controlling the away team")
    p.add_argument("--headless", action="store_true", help="simulate with no window")
    p.add_argument("--matches", type=int, default=1, help="play N matches (implies --headless when N>1)")
    p.add_argument("--minutes", type=float, default=None, help="length of each half in simulated minutes")
    p.add_argument("--seconds", type=float, default=None, help="length of each half in simulated seconds")
    p.add_argument("--periods", type=int, default=2, help="number of halves (default 2)")
    p.add_argument("--seed", type=int, default=1, help="RNG seed; the same seed replays identically")
    p.add_argument("--speed", type=float, default=1.0, help="initial playback speed multiplier")
    p.add_argument("--window", default="1400x900", help="window size, e.g. 1920x1080")
    p.add_argument("--view", choices=("2d", "3d"), default="2d",
                   help="2d is the top-down debugging view (default); 3d uses Panda3D")
    p.add_argument("--camera", choices=("broadcast", "follow", "high"), default="broadcast",
                   help="starting camera for --view 3d")
    p.add_argument(
        "--trusted", action="store_true",
        help="run bots in-process with no sandbox. Faster and easier to debug, "
             "but a team script can then do anything you can. Only for scripts you wrote.",
    )
    p.add_argument(
        "--tick-timeout", type=float, default=None,
        help="seconds a sandboxed bot may take per tick before it is killed (default 0.25)",
    )
    p.add_argument(
        "--no-os-sandbox", action="store_true",
        help="skip the OS sandbox profile (macOS sandbox-exec); process limits still apply",
    )
    return p


def make_rules(args) -> Rules:
    r = Rules(periods=max(1, args.periods))
    if args.seconds is not None:
        r = replace(r, half_seconds=max(1.0, args.seconds))
    elif args.minutes is not None:
        r = replace(r, half_seconds=max(1.0, args.minutes * 60.0))
    return r


def print_result(m: Match) -> None:
    s = m.summary()
    home, away = s["home"], s["away"]
    print(f"\n  {home}  {s['score'][0]} – {s['score'][1]}  {away}")
    print(f"  possession  {s['possession'][0]}% / {s['possession'][1]}%"
          f"   shots {s['shots'][0]}/{s['shots'][1]}"
          f"   distance {s['distance_km'][0]}km / {s['distance_km'][1]}km")
    for t, team, scorer in s["goals"]:
        side = home if team == HOME else away
        print(f"    {int(t // 60)}:{int(t % 60):02d}  goal — {side}")
    for c in m.controllers:
        st = c.stats()
        flag = f"  ({st['errors']} errors)" if st["errors"] else ""
        if st.get("killed"):
            flag += f"  [SANDBOX KILLED: {st['killed']}]"
        print(f"    {st['name'][:24]:24s} think avg {st['avg_ms']:.3f} ms, worst {st['worst_ms']:.2f} ms{flag}")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    rules = make_rules(args)

    squad = rules.outfield_players + 1

    def make(path):
        """Bots are sandboxed unless explicitly trusted."""
        if args.trusted:
            return load_controller(path, squad)
        kw = {"os_sandbox": not args.no_os_sandbox}
        if args.tick_timeout is not None:
            kw["tick_timeout"] = args.tick_timeout
        return load_sandboxed(path, squad, **kw)

    try:
        home = make(args.home)
        away = make(args.away)
    except (BotError, SandboxError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    def factory(seed: int) -> Match:
        # reload keeps repeated matches independent of leftover bot state
        return Match(make(args.home), make(args.away), rules, seed=seed)

    if args.matches > 1 or args.headless:
        wins = [0, 0, 0]  # home, away, draw
        goals = [0, 0]
        for i in range(args.matches):
            m = factory(args.seed + i) if args.matches > 1 else Match(home, away, rules, args.seed)
            m.run()
            print_result(m)
            goals[HOME] += m.score[HOME]
            goals[AWAY] += m.score[AWAY]
            if m.score[HOME] > m.score[AWAY]:
                wins[0] += 1
            elif m.score[AWAY] > m.score[HOME]:
                wins[1] += 1
            else:
                wins[2] += 1
        if args.matches > 1:
            print(f"\n  === {args.matches} matches ===")
            print(f"  {home.name}: {wins[0]}W  draws: {wins[2]}  {away.name}: {wins[1]}W")
            print(f"  goals {goals[HOME]} – {goals[AWAY]}")
        return 0

    try:
        w, h = (int(v) for v in args.window.lower().split("x"))
    except ValueError:
        w, h = 1400, 900

    counter = {"n": 0}

    def viewer_factory() -> Match:
        seed = args.seed + counter["n"]
        counter["n"] += 1
        return factory(seed)

    # imported late so --headless needs neither a display nor these libraries
    if args.view == "3d":
        from .render3d import Viewer3D

        viewer = Viewer3D(viewer_factory, window=(w, h), speed=args.speed,
                          camera=args.camera)
    else:
        from .render import Viewer

        viewer = Viewer(viewer_factory, window=(w, h), speed=args.speed)
    match = viewer.run()
    print_result(match)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
