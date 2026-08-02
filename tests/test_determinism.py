"""Determinism, and the engine's tolerance of badly behaved bots."""

from __future__ import annotations

import pytest

from conftest import ScriptedController
from football.api import Action, Vec2
from football.config import Rules
from football.engine import Match


def snapshot(match):
    return (
        tuple(match.score),
        tuple((round(p.pos.x, 9), round(p.pos.y, 9)) for p in match.players),
        (round(match.ball.pos.x, 9), round(match.ball.pos.y, 9), round(match.ball.z, 9)),
    )


def chase(state):
    return {p.index: Action.go_to(p, state.ball.pos, sprint=True) for p in state.us}


class TestDeterminism:
    def _play(self, seed, rules):
        m = Match(ScriptedController("H", chase), ScriptedController("A", chase), rules, seed=seed)
        m.run()
        return m

    def test_same_seed_gives_an_identical_match(self, rules):
        a, b = self._play(7, rules), self._play(7, rules)
        assert snapshot(a) == snapshot(b)

    def test_different_seeds_diverge(self, rules):
        a, b = self._play(1, rules), self._play(2, rules)
        assert snapshot(a) != snapshot(b)

    def test_replaying_is_stable_across_many_seeds(self, rules):
        for seed in range(4):
            assert snapshot(self._play(seed, rules)) == snapshot(self._play(seed, rules))

    def test_the_clock_and_events_are_reproducible(self, rules):
        a, b = self._play(3, rules), self._play(3, rules)
        assert [(e.tick, e.kind, e.team) for e in a.events] == \
               [(e.tick, e.kind, e.team) for e in b.events]


class TestHostileBots:
    """A bad bot may lose its own match; it must not break the simulation.

    These go through the real `Controller` rather than a scripted stub, because
    that is where validation lives -- a stub would bypass the very code under
    test.
    """

    def _run(self, actions, rules):
        from football.api import Team
        from football.loader import Controller

        class Bad(Team):
            name = "Bad"

            def act(self, state):
                return actions(state)

        m = Match(Controller(Bad(), "Bad", __file__),
                  ScriptedController("Good", chase), rules, seed=1)
        m.run()
        return m

    def test_nan_actions_cannot_corrupt_the_pitch(self, rules):
        """One bot returning NaN once wrecked every player, both teams."""
        def nan_everything(state):
            bad = Vec2(float("nan"), float("nan"))
            return {p.index: Action(move=bad, kick=bad, kick_power=float("nan"))
                    for p in state.us}

        m = self._run(nan_everything, rules)
        for p in m.players:
            assert p.pos.x == p.pos.x and p.pos.y == p.pos.y, "NaN leaked into a player"
        assert m.ball.pos.x == m.ball.pos.x

    def test_enormous_actions_are_clamped(self, rules):
        def huge(state):
            return {p.index: Action(move=Vec2(1e12, -1e12), kick=Vec2(1e12, 0),
                                    kick_power=1e12, lift=1e12) for p in state.us}

        m = self._run(huge, rules)
        for p in m.players:
            assert abs(p.pos.x) <= m.rules.length + 1
            assert abs(p.pos.y) <= m.rules.width + 1

    @pytest.mark.parametrize("junk", [None, "nonsense", 5, {"x": "y"}, [1, 2, 3],
                                      {0: "not an action"}, {"0": 3.5}])
    def test_a_bot_returning_junk_just_idles(self, rules, junk):
        m = self._run(lambda s: junk, rules)
        assert m.finished

    def test_a_raising_bot_does_not_stop_the_match(self, rules):
        def explode(state):
            raise RuntimeError("boom")

        m = self._run(explode, rules)
        assert m.finished
        assert m.controllers[0].errors > 0

    def test_players_stay_on_the_pitch(self, rules):
        """Bodies are clamped to the pitch, then pushed apart, so allow a
        player-radius of slop where separation nudges someone over a line."""
        def run_away(state):
            return {p.index: Action(move=Vec2(-1, -1), sprint=True) for p in state.us}

        m = self._run(run_away, rules)
        slop = m.rules.player_radius
        for p in m.players:
            assert -slop <= p.pos.x <= m.rules.length + slop
            assert -slop <= p.pos.y <= m.rules.width + slop


class TestMatchFlow:
    def test_a_match_reaches_full_time(self, rules):
        m = Match(ScriptedController("H", chase), ScriptedController("A", chase), rules, seed=1)
        m.run()
        assert m.finished and m.phase == "full_time"

    def test_hooks_are_called(self, rules):
        home = ScriptedController("H", chase)
        m = Match(home, ScriptedController("A", chase), rules, seed=1)
        assert home.started is not None, "on_match_start before kickoff"
        m.run()
        assert home.ended, "on_match_end after the whistle"

    def test_two_halves_are_played(self):
        from dataclasses import replace

        rules = replace(Rules(), half_seconds=3.0, periods=2)
        m = Match(ScriptedController("H", chase), ScriptedController("A", chase), rules, seed=1)
        m.run()
        assert m.period == 2 and m.finished

    def test_possession_and_distance_are_tracked(self, rules):
        m = Match(ScriptedController("H", chase), ScriptedController("A", chase), rules, seed=1)
        m.run()
        summary = m.summary()
        assert sum(summary["possession"]) == pytest.approx(100.0, abs=0.2)
        assert all(d >= 0 for d in summary["distance_km"])
