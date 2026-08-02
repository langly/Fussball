"""Possession, duels and reach.

Most of these guard bugs that really happened: fast balls being vacuumed up on
contact, a keeper parrying its own kick forever, and a keeper sitting on the
ball for an entire match.
"""

from __future__ import annotations

import pytest

from conftest import step_ball
from football.vec import Vec2


def one_player(match, index=1, team=0, pos=(50, 34)):
    """Park everyone, then bring a single player back to `pos`."""
    for p in match.players:
        p.pos = Vec2(-50, -50)
        p.vel = Vec2()
    player = match.players[team * match.squad_size + index]
    player.pos = Vec2(*pos)
    return player


class TestTrapping:
    def test_slow_ball_is_collected(self, live_match):
        m = live_match()
        p = one_player(m)
        m.ball.pos = Vec2(50.5, 34)
        m.ball.vel = Vec2(1.0, 0)
        step_ball(m, 2)
        assert m.owner is p

    def test_fast_ball_rebounds_instead_of_being_vacuumed(self, live_match):
        """A 25 m/s shot must not be swallowed by the first body it passes."""
        m = live_match()
        one_player(m)
        m.ball.pos = Vec2(48.0, 34)
        m.ball.vel = Vec2(25.0, 0)
        for _ in range(20):
            step_ball(m, 1)
            if m.owner is not None:
                break
        assert m.owner is None, "a ball this fast cannot be brought under control"

    def test_ball_above_foot_height_cannot_be_trapped(self, live_match):
        m = live_match()
        p = one_player(m)
        m.ball.pos = Vec2(50.2, 34)
        m.ball.vel = Vec2(0.5, 0)
        m.ball.z = m.rules.reach_foot + 0.6
        step_ball(m, 2)
        assert m.owner is not p

    def test_ball_over_everyones_head_is_untouched(self, live_match):
        m = live_match()
        one_player(m)
        m.ball.pos = Vec2(48.0, 34)
        m.ball.vel = Vec2(6.0, 0)
        m.ball.z = m.rules.reach_head + 1.5
        m.ball.vz = 0.5
        before = m.ball.vel.x
        step_ball(m, 12)
        assert m.last_touch is None, "nobody can reach it"
        assert m.ball.vel.x == pytest.approx(before, rel=0.2)

    def test_keeper_reaches_higher_than_an_outfielder(self, live_match):
        m = live_match()
        assert m.rules.reach_keeper > m.rules.reach_head > m.rules.reach_foot


class TestDribbling:
    def test_owner_carries_the_ball_at_their_feet(self, live_match):
        m = live_match()
        p = one_player(m)
        m.ball.pos = Vec2(50.4, 34)
        step_ball(m, 2)
        assert m.owner is p
        p.pos = Vec2(60, 34)
        p.heading = Vec2(1, 0)
        step_ball(m, 1)
        assert m.ball.pos.dist(p.pos) == pytest.approx(m.rules.dribble_offset, abs=0.05)

    def test_a_shielded_ball_is_never_out_of_play(self, live_match):
        """Dribbling along the touchline must not concede a throw-in."""
        m = live_match()
        p = one_player(m, pos=(50, 0.2))
        m.ball.pos = Vec2(50, 0.3)
        step_ball(m, 2)
        assert m.owner is p
        p.heading = Vec2(0, -1)  # facing off the pitch
        step_ball(m, 5)
        assert not any(e.kind == "throw_in" for e in m.events)
        assert m.owner is p


class TestKeeper:
    def test_keeper_cannot_catch_its_own_clearance(self, live_match):
        """The keeper used to parry its own kick and re-catch it, forever."""
        from football.api import Action, PlayerView

        m = live_match()
        gk = one_player(m, index=0, pos=(3, 34))
        m.ball.pos = Vec2(3.6, 34)
        step_ball(m, 2)
        assert m.owner is gk

        view = PlayerView(index=0, name=gk.name, pos=gk.pos, vel=gk.vel,
                          heading=gk.heading, stamina=1.0, is_keeper=True, has_ball=True)
        kick = Action.kick_to(view, Vec2(80, 34), power=0.9)
        catching = Action.idle()
        catching.catch = True
        step_ball(m, 1, actions={0: {0: kick}, 1: {}})
        assert m.owner is None
        for _ in range(12):
            step_ball(m, 1, actions={0: {0: catching}, 1: {}})
        assert m.owner is not gk, "the keeper must not re-collect its own kick"

    def test_six_second_rule_forces_a_release(self, live_match):
        """A keeper that never kicks must still lose the ball."""
        m = live_match()
        gk = one_player(m, index=0, pos=(3, 34))
        m.ball.pos = Vec2(3.6, 34)
        step_ball(m, 2)
        assert m.owner is gk
        held = int((m.rules.keeper_max_hold + 0.5) / m.rules.dt)
        step_ball(m, held)
        assert m.owner is not gk
        assert any(e.kind == "six_seconds" for e in m.events)


class TestDuels:
    def test_only_the_nearest_opponent_challenges(self, live_match):
        """Swarming must not multiply the chance of winning the ball."""
        m = live_match()
        carrier = one_player(m, index=1, team=0, pos=(50, 34))
        m.ball.pos = Vec2(50.4, 34)
        step_ball(m, 2)
        assert m.owner is carrier

        # crowd four opponents around the carrier
        for i in range(1, 5):
            opp = m.players[m.squad_size + i]
            opp.pos = Vec2(50 + 0.4 * i, 34.5)
        m.rng.seed(1)
        steals = 0
        for _ in range(200):
            step_ball(m, 1)
            if m.owner is not carrier:
                steals += 1
                m.owner = carrier
        # with four challengers the rate should still be one challenger's worth
        assert steals < 200 * m.rules.tackle_rate * m.rules.dt * 2

    def test_a_tackle_can_win_the_ball(self, live_match):
        m = live_match()
        carrier = one_player(m, index=1, team=0, pos=(50, 34))
        m.ball.pos = Vec2(50.4, 34)
        step_ball(m, 2)
        opp = m.players[m.squad_size + 1]
        opp.pos = Vec2(50.6, 34)
        m.rng.seed(3)
        for _ in range(400):
            step_ball(m, 1)
            if m.owner is opp:
                break
        assert m.owner is opp
