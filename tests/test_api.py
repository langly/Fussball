"""The bot-facing API: action validation, prediction, limits and crests."""

from __future__ import annotations

import math

import pytest

from football.api import Action, Vec2, parse_logo


class TestActionSanitising:
    def test_nan_move_is_scrubbed(self):
        """NaN once leaked into the physics and corrupted every player."""
        a = Action(move=Vec2(float("nan"), float("nan"))).sanitized()
        assert a.move.x == 0.0 and a.move.y == 0.0

    def test_infinite_move_is_scrubbed(self):
        a = Action(move=Vec2(float("inf"), -float("inf"))).sanitized()
        assert a.move.length() == 0.0

    def test_nan_kick_is_dropped(self):
        a = Action(kick=Vec2(float("nan"), 1.0), kick_power=1.0).sanitized()
        assert a.kick is None and a.kick_power == 0.0

    def test_nan_power_and_lift_become_zero(self):
        a = Action(kick=Vec2(1, 0), kick_power=float("nan"), lift=float("nan")).sanitized()
        assert a.kick_power == 0.0 and a.lift == 0.0

    def test_move_is_clamped_to_unit_length(self):
        a = Action(move=Vec2(90.0, 0.0)).sanitized()
        assert a.move.length() == pytest.approx(1.0)

    def test_power_and_lift_are_clamped(self):
        a = Action(kick=Vec2(1, 0), kick_power=8.0, lift=-3.0).sanitized()
        assert a.kick_power == 1.0 and a.lift == 0.0

    def test_kick_direction_is_normalised(self):
        a = Action(kick=Vec2(30.0, 40.0), kick_power=0.5).sanitized()
        assert a.kick.length() == pytest.approx(1.0)

    def test_flags_are_coerced_to_bool(self):
        a = Action(sprint="yes", catch=1).sanitized()
        assert a.sprint is True and a.catch is True


class TestLimits:
    def test_power_for_gives_a_receivable_pass(self, live_match):
        """A weighted pass must arrive slow enough for a team-mate to trap."""
        m = live_match()
        lim = m.state_for(0).limits
        for distance in (8.0, 18.0, 30.0):
            power = lim.power_for(distance, arrive_speed=5.0)
            launch = lim.kick_speed(power)
            arriving = launch - lim.ball_friction * distance
            assert arriving == pytest.approx(5.0, abs=1.5)
            assert arriving < lim.trap_speed

    def test_reach_of_matches_the_friction_model(self, live_match):
        lim = live_match().state_for(0).limits
        assert lim.reach_of(1.0) == pytest.approx(lim.max_kick_speed / lim.ball_friction)

    def test_kick_speed_spans_min_to_max(self, live_match):
        lim = live_match().state_for(0).limits
        assert lim.kick_speed(0.0) == pytest.approx(lim.min_kick_speed)
        assert lim.kick_speed(1.0) == pytest.approx(lim.max_kick_speed)

    def test_aerial_constants_are_exposed(self, live_match):
        lim = live_match().state_for(0).limits
        for field in ("gravity", "crossbar_height", "reach_foot", "reach_head", "reach_keeper"):
            assert getattr(lim, field) > 0.0


class TestPrediction:
    @pytest.mark.parametrize("vel,z,vz", [
        ((8.0, 0.0), 0.0, 0.0),
        ((-6.0, 4.0), 0.0, 0.0),
        ((12.0, -3.0), 2.5, 6.0),
        ((0.0, 0.0), 5.0, 0.0),
    ])
    def test_predict_ball_agrees_with_the_engine(self, live_match, vel, z, vz):
        """The predictor must step the same integration the engine does."""
        m = live_match()
        m.ball.pos = Vec2(50, 34)
        m.ball.vel = Vec2(*vel)
        m.ball.z, m.ball.vz = z, vz
        seconds = 0.8
        guess = m.state_for(0).predict_ball(seconds)
        for _ in range(int(seconds / m.rules.dt)):
            m._update_ball({0: {}, 1: {}}, m.rules.dt)
        assert guess.dist(m.ball.pos) < 1e-6

    def test_predict_height_tracks_the_ball(self, live_match):
        m = live_match()
        m.ball.pos = Vec2(50, 34)
        m.ball.vel = Vec2(4, 0)
        m.ball.z, m.ball.vz = 0.0, 7.0
        state = m.state_for(0)
        guess = state.ball.predict_height(0.5, m.rules.gravity, m.rules.ball_restitution,
                                          m.rules.air_drag)
        for _ in range(int(0.5 / m.rules.dt)):
            m._update_ball({0: {}, 1: {}}, m.rules.dt)
        assert guess == pytest.approx(m.ball.z, abs=1e-6)


class TestStateHelpers:
    def test_can_trap_rejects_a_high_ball(self, live_match):
        m = live_match()
        m.ball.pos = Vec2(50, 34)
        m.ball.vel = Vec2()
        m.ball.z = m.rules.reach_foot + 1.0
        state = m.state_for(0)
        assert not state.can_trap(state.us[1])

    def test_can_trap_rejects_a_fast_ball(self, live_match):
        m = live_match()
        m.ball.vel = Vec2(m.rules.trap_speed + 5.0, 0)
        state = m.state_for(0)
        assert not state.can_trap(state.us[1])

    def test_can_reach_follows_the_height_ceiling(self, live_match):
        m = live_match()
        m.ball.z = m.rules.reach_head + 0.5
        state = m.state_for(0)
        assert not state.can_reach(state.us[1])
        assert state.can_reach(state.keeper) or m.rules.reach_keeper < m.ball.z

    def test_lane_is_clear_sees_a_blocker(self, live_match):
        m = live_match()
        for p in m.players:
            p.pos = Vec2(-50, -50)
        m.players[m.squad_size + 1].pos = Vec2(50, 34)
        state = m.state_for(0)
        assert not state.lane_is_clear(Vec2(40, 34), Vec2(60, 34))
        assert state.lane_is_clear(Vec2(40, 10), Vec2(60, 10))

    def test_pressure_grows_as_opponents_close_in(self, live_match):
        m = live_match()
        for p in m.players:
            p.pos = Vec2(-50, -50)
        state = m.state_for(0)
        assert state.pressure_on(Vec2(50, 34)) == 0.0
        m.players[m.squad_size + 1].pos = Vec2(51, 34)
        assert m.state_for(0).pressure_on(Vec2(50, 34)) > 0.0


class TestCrest:
    def test_a_valid_crest_parses(self):
        w, h, px = parse_logo(["AB", ".A"], {"A": "#ff0000", "B": "#00ff00"})
        assert (w, h) == (2, 2)
        assert px[0] == (255, 0, 0, 255)
        assert px[2] == (0, 0, 0, 0), "'.' is transparent"

    def test_short_hex_and_tuples_are_accepted(self):
        _, _, px = parse_logo(["AB"], {"A": "#f00", "B": (0, 0, 255, 128)})
        assert px[0] == (255, 0, 0, 255)
        assert px[1] == (0, 0, 255, 128)

    def test_rows_are_padded_to_a_rectangle(self):
        w, h, px = parse_logo(["AAAA", "A"], {"A": "#fff"})
        assert (w, h) == (4, 2) and len(px) == 8
        assert px[5] == (0, 0, 0, 0)

    @pytest.mark.parametrize("rows,colours", [
        ([], {}),
        (None, {}),
        ("not a list", {}),
        ([1, 2], {}),
        (["X" * 99] * 99, {"X": "#fff"}),
        ([""], {}),
    ])
    def test_malformed_crests_are_rejected(self, rows, colours):
        assert parse_logo(rows, colours) is None

    @pytest.mark.parametrize("colour", ["nonsense", "#12345", "", None, object(), (1,)])
    def test_bad_colours_fall_back_to_transparent(self, colour):
        """A hostile palette must not raise -- the pixel just goes clear."""
        result = parse_logo(["A"], {"A": colour})
        assert result is not None
        assert result[2][0] == (0, 0, 0, 0)

    def test_oversized_crest_is_refused(self):
        from football.api import LOGO_MAX_SIZE

        assert parse_logo(["A" * (LOGO_MAX_SIZE + 1)], {"A": "#fff"}) is None
        assert parse_logo(["A"] * (LOGO_MAX_SIZE + 1), {"A": "#fff"}) is None
