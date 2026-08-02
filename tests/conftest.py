"""Shared fixtures.

Everything here keeps tests headless and deterministic: no window is opened,
and matches are driven by scripted controllers rather than the example bots, so
a test never fails because a bot was retuned.
"""

from __future__ import annotations

import os
import sys
from dataclasses import replace
from pathlib import Path

import pytest

# pygame must not try to open a real window when render tests import it
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from football.config import Rules  # noqa: E402
from football.engine import Match  # noqa: E402
from football.vec import Vec2  # noqa: E402


class ScriptedController:
    """A controller that plays back whatever the test tells it to.

    Used instead of the example bots so that engine tests exercise the engine,
    not somebody's tactics.
    """

    def __init__(self, name="Test", actions=None, logo=None):
        self.name = name
        self.errors = 0
        self._actions = actions or {}
        self._logo = logo
        self.started = None
        self.goals = []
        self.ended = False

    # -- controller interface -----------------------------------------
    def squad_names(self, fallback):
        return list(fallback)

    def logo(self):
        return self._logo

    def on_match_start(self, info):
        self.started = info

    def on_goal(self, ours, state):
        self.goals.append(ours)

    def on_match_end(self, state):
        self.ended = True

    def act(self, state):
        if callable(self._actions):
            return self._actions(state)
        return dict(self._actions)

    def stats(self):
        return {"name": self.name, "errors": self.errors, "avg_ms": 0.0, "worst_ms": 0.0}


@pytest.fixture
def rules():
    """A short single-period match, so tests finish quickly."""
    return replace(Rules(), half_seconds=10.0, periods=1)


@pytest.fixture
def make_match(rules):
    def _make(home_actions=None, away_actions=None, seed=1, rules_override=None):
        return Match(
            ScriptedController("Home", home_actions),
            ScriptedController("Away", away_actions),
            rules_override or rules,
            seed=seed,
        )

    return _make


@pytest.fixture
def live_match(make_match):
    """A match in open play with every player parked well out of the way.

    Lets a test place the ball and a single player and observe one interaction
    without ten other bodies interfering.
    """

    def _make(**kw):
        m = make_match(**kw)
        m.phase = "play"
        m.setpiece = None
        for p in m.players:
            p.pos = Vec2(2.0, 2.0)
            p.vel = Vec2()
        m.owner = None
        m.loose_timer = 0.0
        m.events.clear()
        return m

    return _make


def step_ball(match, ticks=1, actions=None):
    """Advance only the ball, without the controllers or the clock."""
    empty = {0: {}, 1: {}}
    for _ in range(ticks):
        match._update_ball(actions or empty, match.rules.dt)
        match._check_boundaries()
