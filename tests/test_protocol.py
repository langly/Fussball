"""The sandbox wire format.

The host must be able to hand a bot the whole world and get back only what it
is prepared to accept. `decode_actions` is the trust boundary, so most of this
file is about feeding it rubbish.
"""

from __future__ import annotations

import json
import math

import pytest

from football.api import Action, Vec2
from football.protocol import (
    decode_actions, decode_setup, decode_state, encode_actions, encode_setup,
    encode_state,
)


class TestStateRoundTrip:
    def test_state_survives_the_wire(self, live_match):
        m = live_match()
        m.ball.pos = Vec2(31.5, 22.25)
        m.ball.vel = Vec2(-4.5, 9.0)
        m.ball.z, m.ball.vz = 3.75, -2.5
        original = m.state_for(0)

        blob = json.loads(json.dumps(encode_state(original)))
        restored = decode_state(blob, original.pitch, original.limits,
                                [p.name for p in original.us],
                                [p.name for p in original.them])

        assert restored.ball.pos.dist(original.ball.pos) < 1e-9
        assert restored.ball.vel.dist(original.ball.vel) < 1e-9
        assert restored.tick == original.tick
        assert restored.our_score == original.our_score

    def test_ball_height_crosses_the_wire(self, live_match):
        """Omitting these once left every sandboxed bot seeing a flat world."""
        m = live_match()
        m.ball.z, m.ball.vz = 4.25, -3.5
        original = m.state_for(0)
        blob = encode_state(original)
        restored = decode_state(blob, original.pitch, original.limits,
                                [p.name for p in original.us],
                                [p.name for p in original.them])
        assert restored.ball.height == pytest.approx(4.25)
        assert restored.ball.vertical_speed == pytest.approx(-3.5)
        assert restored.ball.airborne

    def test_players_survive_the_wire(self, live_match):
        m = live_match()
        m.players[2].pos = Vec2(12.5, 44.0)
        m.players[2].stamina = 0.42
        original = m.state_for(0)
        blob = encode_state(original)
        restored = decode_state(blob, original.pitch, original.limits,
                                [p.name for p in original.us],
                                [p.name for p in original.them])
        assert restored.us[2].pos.dist(Vec2(12.5, 44.0)) < 1e-9
        assert restored.us[2].stamina == pytest.approx(0.42)
        assert restored.us[0].is_keeper and not restored.us[2].is_keeper

    def test_setup_carries_pitch_limits_and_names(self, live_match):
        m = live_match()
        m._notify_start()
        info = m.controllers[0].started
        blob = json.loads(json.dumps(encode_setup(info)))
        restored, names_us, names_them = decode_setup(blob)
        assert restored.pitch.length == pytest.approx(info.pitch.length)
        assert restored.limits.gravity == pytest.approx(info.limits.gravity)
        assert restored.limits.reach_keeper == pytest.approx(info.limits.reach_keeper)
        assert names_us == list(info.squad_names)
        assert names_them == list(info.opponent_names)


class TestActionRoundTrip:
    def test_actions_survive_the_wire(self):
        actions = {
            0: Action(move=Vec2(0.5, -0.5), sprint=True, catch=True),
            3: Action(kick=Vec2(1.0, 0.0), kick_power=0.75, lift=0.4),
        }
        restored = decode_actions(json.loads(json.dumps(encode_actions(actions, 5))), 5)
        assert restored[0].sprint is True and restored[0].catch is True
        assert restored[3].kick_power == pytest.approx(0.75)
        assert restored[3].lift == pytest.approx(0.4)

    def test_lift_crosses_the_wire(self):
        """Without this a sandboxed bot could never loft the ball."""
        encoded = encode_actions({1: Action(kick=Vec2(1, 0), kick_power=1.0, lift=0.9)}, 5)
        assert decode_actions(encoded, 5)[1].lift == pytest.approx(0.9)

    def test_missing_players_default_to_idle(self):
        restored = decode_actions(encode_actions({2: Action(sprint=True)}, 5), 5)
        assert len(restored) == 5
        assert restored[0].move.length() == 0.0 and restored[0].sprint is False


class TestHostileReplies:
    """Everything here is what an untrusted bot might send back."""

    @pytest.mark.parametrize("payload", [
        None, "a string", 42, [], {"not-an-index": [0] * 7},
        {"0": "not a list"}, {"0": [0] * 3}, {"0": None},
    ])
    def test_rubbish_becomes_idle_actions(self, payload):
        result = decode_actions(payload, 5)
        assert len(result) == 5
        assert all(a.move.length() == 0.0 for a in result.values())

    @pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
    def test_non_finite_numbers_are_scrubbed(self, bad):
        payload = {"1": [bad, bad, 0, [bad, bad], bad, 0, bad]}
        action = decode_actions(payload, 5)[1]
        assert math.isfinite(action.move.x) and math.isfinite(action.move.y)
        assert math.isfinite(action.kick_power) and math.isfinite(action.lift)
        assert action.move.length() == 0.0

    def test_out_of_range_indices_are_ignored(self):
        payload = {"99": [1, 0, 0, None, 0, 0, 0], "-4": [1, 0, 0, None, 0, 0, 0]}
        result = decode_actions(payload, 5)
        assert set(result) == {0, 1, 2, 3, 4}
        assert all(a.move.length() == 0.0 for a in result.values())

    def test_absurd_values_are_clamped(self):
        payload = {"0": [1e30, -1e30, 1, [1, 0], 1e9, 1, 1e9]}
        action = decode_actions(payload, 5)[0]
        assert action.move.length() <= 1.0 + 1e-9
        assert 0.0 <= action.kick_power <= 1.0
        assert 0.0 <= action.lift <= 1.0

    def test_a_flood_of_keys_is_bounded(self):
        payload = {str(i): [0, 0, 0, None, 0, 0, 0] for i in range(10_000)}
        assert len(decode_actions(payload, 5)) == 5

    def test_older_six_field_replies_still_work(self):
        """Pre-aerial replies should mean 'no lift', not be discarded."""
        payload = {"2": [1.0, 0.0, 1, [1.0, 0.0], 0.5, 0]}
        action = decode_actions(payload, 5)[2]
        assert action.kick_power == pytest.approx(0.5)
        assert action.lift == 0.0
        assert action.sprint is True
