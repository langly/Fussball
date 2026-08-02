"""Load a user's team script and wrap it so a bad bot cannot crash the match."""

from __future__ import annotations

import importlib.util
import inspect
import sys
import time
import traceback
from pathlib import Path
from typing import Mapping, Sequence

from .api import Action, Team, parse_logo


class BotError(Exception):
    pass


class Controller:
    """Wraps a user `Team` object: times it, validates it, contains its crashes."""

    MAX_REPORTED_ERRORS = 3

    def __init__(self, team: Team, name: str, source: Path, squad_size: int = 5) -> None:
        self.team = team
        self.name = name
        self.source = source
        self.squad_size = squad_size
        self.errors = 0
        self.think_seconds = 0.0
        self.worst_tick_ms = 0.0
        self.calls = 0
        self._idle = {i: Action.idle() for i in range(squad_size)}

    # -- hooks ---------------------------------------------------------
    def on_match_start(self, info) -> None:
        self._guard(lambda: self.team.on_match_start(info), "on_match_start")

    def on_goal(self, scored_by_us: bool, state) -> None:
        self._guard(lambda: self.team.on_goal(scored_by_us, state), "on_goal")

    def on_match_end(self, state) -> None:
        self._guard(lambda: self.team.on_match_end(state), "on_match_end")

    # -- per-tick ------------------------------------------------------
    def act(self, state) -> dict[int, Action]:
        start = time.perf_counter()
        try:
            raw = self.team.act(state)
        except Exception:
            self._report("act")
            raw = None
        elapsed = time.perf_counter() - start
        self.think_seconds += elapsed
        self.worst_tick_ms = max(self.worst_tick_ms, elapsed * 1000.0)
        self.calls += 1
        return self._normalize(raw)

    def _normalize(self, raw) -> dict[int, Action]:
        if raw is None:
            return dict(self._idle)
        out = {i: Action.idle() for i in range(self.squad_size)}
        try:
            if isinstance(raw, Mapping):
                items = raw.items()
            elif isinstance(raw, Sequence):
                items = enumerate(raw)
            else:
                raise BotError(f"act() must return a mapping or sequence, got {type(raw).__name__}")
            for key, action in items:
                idx = int(key)
                if not 0 <= idx < self.squad_size:
                    continue
                if not isinstance(action, Action):
                    continue
                out[idx] = action.sanitized()
        except Exception:
            self._report("act result")
            return dict(self._idle)
        return out

    # -- error containment --------------------------------------------
    def _guard(self, fn, where: str) -> None:
        try:
            fn()
        except Exception:
            self._report(where)

    def _report(self, where: str) -> None:
        self.errors += 1
        if self.errors <= self.MAX_REPORTED_ERRORS:
            print(f"\n[{self.name}] error in {where}:", file=sys.stderr)
            traceback.print_exc()
            if self.errors == self.MAX_REPORTED_ERRORS:
                print(f"[{self.name}] further errors will be silenced.", file=sys.stderr)

    def squad_names(self, fallback: Sequence[str]) -> list[str]:
        """Resolve this team's shirt names, falling back per-slot on anything odd."""
        out = list(fallback)
        try:
            raw = getattr(self.team, "player_names", None)
            if raw is None:
                return out
            for i, value in enumerate(raw):
                if i >= len(out):
                    break
                if value is None:
                    continue  # str(None) would silently become "None"
                text = str(value).strip()
                if text:
                    out[i] = text[:14]
        except Exception:
            self._report("player_names")
        return out

    def logo(self):
        """The team's crest as (width, height, RGBA pixels), or None."""
        try:
            rows = getattr(self.team, "logo", None)
            if rows is None:
                return None
            return parse_logo(rows, getattr(self.team, "logo_colors", None))
        except Exception:
            self._report("logo")
            return None

    def logo_source(self):
        """The raw crest as the bot declared it, for sending over the wire."""
        try:
            rows = getattr(self.team, "logo", None)
            if not isinstance(rows, (list, tuple)):
                return None, None
            colours = getattr(self.team, "logo_colors", None)
            return list(rows), dict(colours) if isinstance(colours, dict) else None
        except Exception:
            return None, None

    def stats(self) -> dict:
        avg = (self.think_seconds / self.calls * 1000.0) if self.calls else 0.0
        return {
            "name": self.name,
            "errors": self.errors,
            "avg_ms": round(avg, 3),
            "worst_ms": round(self.worst_tick_ms, 2),
        }


# ---------------------------------------------------------------------------


def load_controller(path: str | Path, squad_size: int = 5) -> Controller:
    """Import `path` and find the team it defines.

    Accepted shapes, in priority order:
      1. a subclass of `football.api.Team`
      2. a `create_team()` factory returning such an object
      3. a module-level `act(state)` function
    """
    p = Path(path).expanduser().resolve()
    if not p.is_file():
        raise BotError(f"team script not found: {p}")

    module_name = f"botscript_{p.stem}_{abs(hash(str(p))) % 100000}"
    spec = importlib.util.spec_from_file_location(module_name, p)
    if spec is None or spec.loader is None:
        raise BotError(f"cannot import {p}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    # let the script import sibling modules from its own folder
    parent = str(p.parent)
    added = parent not in sys.path
    if added:
        sys.path.insert(0, parent)
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        raise BotError(f"{p.name} failed to import: {exc}") from exc
    finally:
        if added:
            try:
                sys.path.remove(parent)
            except ValueError:
                pass

    team = _instantiate(module, p)
    name = getattr(team, "name", None) or p.stem
    return Controller(team, str(name), p, squad_size)


def _instantiate(module, path: Path) -> Team:
    candidates = [
        obj
        for obj in vars(module).values()
        if inspect.isclass(obj)
        and issubclass(obj, Team)
        and obj is not Team
        and obj.__module__ == module.__name__
    ]
    if candidates:
        # prefer the most derived class if a script builds a small hierarchy
        leaf = [c for c in candidates if not any(other is not c and issubclass(other, c) for other in candidates)]
        return (leaf or candidates)[0]()

    factory = getattr(module, "create_team", None)
    if callable(factory):
        team = factory()
        if not isinstance(team, Team):
            raise BotError(f"{path.name}: create_team() must return a football.api.Team")
        return team

    fn = getattr(module, "act", None)
    if callable(fn):
        class FunctionTeam(Team):
            name = getattr(module, "TEAM_NAME", path.stem)

            def act(self, state):
                return fn(state)

        return FunctionTeam()

    raise BotError(
        f"{path.name}: no team found. Define a subclass of football.api.Team, "
        f"a create_team() factory, or a module-level act(state) function."
    )
