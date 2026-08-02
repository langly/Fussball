"""Ball physics: rolling, flight, bounce and the frame."""

from __future__ import annotations

import math

import pytest

from conftest import step_ball
from football.vec import Vec2


def place(match, pos, vel=(0, 0), z=0.0, vz=0.0):
    match.ball.pos = Vec2(*pos)
    match.ball.vel = Vec2(*vel)
    match.ball.z = z
    match.ball.vz = vz


class TestRolling:
    def test_a_still_ball_stays_still(self, live_match):
        m = live_match()
        place(m, (50, 34))
        step_ball(m, 60)
        assert m.ball.pos.dist(Vec2(50, 34)) < 1e-9

    def test_rolling_ball_slows_and_stops(self, live_match):
        m = live_match()
        place(m, (50, 34), vel=(8, 0))
        step_ball(m, 600)
        assert m.ball.vel.length() == 0.0
        assert m.ball.pos.x > 50.0

    def test_roll_distance_matches_the_friction_model(self, live_match):
        """v0/k is the analytic distance for exponential decay."""
        m = live_match()
        speed = 10.0
        place(m, (10, 34), vel=(speed, 0))
        step_ball(m, 3000)
        expected = speed / m.rules.ball_friction
        assert m.ball.pos.x - 10.0 == pytest.approx(expected, rel=0.05)

    def test_grounded_ball_has_no_height(self, live_match):
        m = live_match()
        place(m, (50, 34), vel=(6, 0))
        step_ball(m, 30)
        assert m.ball.z == 0.0
        assert not m.ball.airborne


class TestFlight:
    def test_lofted_ball_rises_then_lands(self, live_match):
        m = live_match()
        place(m, (30, 34), vel=(10, 0), vz=8.0)
        peak = 0.0
        for _ in range(600):
            step_ball(m, 1)
            peak = max(peak, m.ball.z)
        assert peak > 2.0, "a ball launched upward should actually climb"
        assert m.ball.z == pytest.approx(0.0, abs=1e-6), "and come back down"

    def test_gravity_governs_the_peak(self, live_match):
        """Peak height should be close to v^2 / 2g, allowing for drag."""
        m = live_match()
        vz = 9.0
        place(m, (30, 34), vz=vz)
        peak = 0.0
        for _ in range(400):
            step_ball(m, 1)
            peak = max(peak, m.ball.z)
        ideal = vz * vz / (2 * m.rules.gravity)
        assert peak == pytest.approx(ideal, rel=0.15)

    def test_airborne_ball_keeps_its_pace(self, live_match):
        """Turf friction must not apply in the air, only light drag."""
        m = live_match()
        place(m, (20, 34), vel=(15, 0), vz=7.0)
        step_ball(m, 30)
        assert m.ball.vel.length() > 13.0

    def test_bounces_decay_and_settle(self, live_match):
        m = live_match()
        place(m, (50, 34), vz=6.0)
        peaks, rising = [], False
        for _ in range(1200):
            before = m.ball.vz
            step_ball(m, 1)
            if before > 0 >= m.ball.vz:
                peaks.append(m.ball.z)
        assert len(peaks) >= 2, "should bounce more than once"
        assert peaks == sorted(peaks, reverse=True), "each bounce must be lower"
        assert m.ball.vz == 0.0 and m.ball.z == 0.0, "and eventually settle"

    def test_lift_gets_the_ball_over_head_height(self, live_match):
        """What loft buys is height -- and with it, being untrappable.

        Note it does *not* shorten the kick: a flat ball is slowed by turf the
        whole way, while a lofted one flies, so lofted kicks actually travel
        further. The cost of loft is that nobody can control the ball until it
        comes back down.
        """
        from football.api import Action

        def peak_height(lift):
            m = live_match()
            striker = m.players[1]
            striker.pos = Vec2(20, 34)
            place(m, (20.5, 34))
            act = Action.kick_to(_view(striker), Vec2(90, 34), power=0.8, lift=lift)
            step_ball(m, 1, actions={0: {1: act}, 1: {}})
            peak = 0.0
            for _ in range(400):
                m._update_ball({0: {}, 1: {}}, m.rules.dt)  # no boundary checks
                peak = max(peak, m.ball.z)
            return peak

        assert peak_height(0.0) < 0.1, "a flat kick stays on the deck"
        assert peak_height(0.7) > m_reach(live_match), "a lofted one clears everyone"


def m_reach(live_match):
    return live_match().rules.reach_head


def _view(player):
    """Minimal stand-in for a PlayerView, enough for Action builders."""
    from football.api import PlayerView

    return PlayerView(
        index=player.index, name=player.name, pos=player.pos, vel=player.vel,
        heading=player.heading, stamina=1.0, is_keeper=player.is_keeper, has_ball=False,
    )


class TestWoodwork:
    def test_shot_under_the_bar_is_a_goal(self, live_match):
        m = live_match()
        place(m, (100, 34), vel=(20, 0), z=1.0)
        for _ in range(60):
            step_ball(m, 1)
            if m.score[0]:
                break
        assert m.score[0] == 1

    def test_shot_over_the_bar_is_not_a_goal(self, live_match):
        m = live_match()
        place(m, (100, 34), vel=(20, 0), z=4.0, vz=1.0)
        for _ in range(120):
            step_ball(m, 1)
        assert m.score[0] == 0
        assert any(e.kind in ("goal_kick", "corner") for e in m.events)

    def test_striking_a_post_deflects_the_ball(self, live_match):
        """The frame is solid. It may still go in off the post -- that is football."""
        m = live_match()
        place(m, (100, m.rules.goal_y0), vel=(18, 0), z=1.0)
        before = Vec2(m.ball.vel.x, m.ball.vel.y)
        for _ in range(60):
            step_ball(m, 1)
            if any(e.kind == "woodwork" for e in m.events):
                break
        assert any(e.kind == "woodwork" for e in m.events), "should hit the post"
        assert m.ball.vel.dist(before) > 1.0, "and be deflected by it"

    def test_ball_above_the_bar_never_scores(self, live_match):
        """Struck from close in, so it cannot drop under the bar in flight."""
        for z in (3.0, 4.5, 7.0):
            m = live_match()
            place(m, (103.5, 34), vel=(22, 0), z=z)
            for _ in range(90):
                step_ball(m, 1)
            assert m.score[0] == 0, f"ball at {z} m should not have scored"

    def test_a_dipping_ball_can_still_score(self, live_match):
        """The bar test is about height *at the line*, not height at the strike."""
        m = live_match()
        place(m, (95, 34), vel=(20, 0), z=3.2, vz=-3.0)
        for _ in range(90):
            step_ball(m, 1)
        assert m.score[0] == 1
