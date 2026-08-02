"""Laws of the game: restarts, scoring, and the mirrored view each team sees."""

from __future__ import annotations

import pytest

from conftest import step_ball
from football.engine import AWAY, HOME
from football.vec import Vec2


def clear_pitch(match):
    for p in match.players:
        p.pos = Vec2(-50, -50)
        p.vel = Vec2()


def last_setpiece(match):
    for event in reversed(match.events):
        if event.kind in ("throw_in", "corner", "goal_kick"):
            return event
    return None


class TestRestarts:
    def test_ball_over_the_touchline_is_a_throw_in(self, live_match):
        m = live_match()
        clear_pitch(m)
        m.ball.pos = Vec2(50, 1.0)
        m.ball.vel = Vec2(0, -8.0)
        step_ball(m, 40)
        event = last_setpiece(m)
        assert event is not None and event.kind == "throw_in"

    def test_throw_in_goes_to_the_team_that_did_not_touch_it(self, live_match):
        m = live_match()
        clear_pitch(m)
        toucher = m.players[1]  # home
        m.last_touch = toucher
        m.ball.pos = Vec2(50, 1.0)
        m.ball.vel = Vec2(0, -8.0)
        step_ball(m, 40)
        event = last_setpiece(m)
        assert event.kind == "throw_in" and event.team == AWAY

    def test_defender_putting_it_behind_concedes_a_corner(self, live_match):
        m = live_match()
        clear_pitch(m)
        m.last_touch = m.players[m.squad_size + 1]  # an away defender
        m.ball.pos = Vec2(104, 20)
        m.ball.vel = Vec2(8, 0)
        step_ball(m, 40)
        event = last_setpiece(m)
        assert event.kind == "corner" and event.team == HOME

    def test_attacker_putting_it_behind_concedes_a_goal_kick(self, live_match):
        m = live_match()
        clear_pitch(m)
        m.last_touch = m.players[1]  # a home attacker, shooting at x = length
        m.ball.pos = Vec2(104, 20)
        m.ball.vel = Vec2(8, 0)
        step_ball(m, 40)
        event = last_setpiece(m)
        assert event.kind == "goal_kick" and event.team == AWAY

    def test_a_set_piece_puts_the_ball_where_it_went_out(self, live_match):
        m = live_match()
        clear_pitch(m)
        m.ball.pos = Vec2(30, 1.0)
        m.ball.vel = Vec2(0, -8.0)
        step_ball(m, 40)
        assert m.ball.pos.y in (0.0, m.rules.width)
        assert m.ball.vel.length() == 0.0
        assert m.phase == "setpiece"


class TestScoring:
    def test_a_goal_is_awarded_and_credited(self, live_match):
        m = live_match()
        clear_pitch(m)
        m.ball.pos = Vec2(104, 34)
        m.ball.vel = Vec2(10, 0)
        step_ball(m, 30)
        assert m.score[HOME] == 1 and m.score[AWAY] == 0
        assert any(e.kind == "goal" for e in m.events)

    def test_each_team_attacks_its_own_end(self, live_match):
        m = live_match()
        clear_pitch(m)
        m.ball.pos = Vec2(1.0, 34)
        m.ball.vel = Vec2(-10, 0)
        step_ball(m, 30)
        assert m.score[AWAY] == 1, "the away team scores at x = 0"

    def test_a_goal_restarts_with_a_kickoff(self, live_match):
        m = live_match()
        clear_pitch(m)
        m.ball.pos = Vec2(104, 34)
        m.ball.vel = Vec2(10, 0)
        step_ball(m, 30)
        assert m.phase == "goal"
        for _ in range(int((m.rules.goal_celebration + 0.1) / m.rules.dt)):
            m._advance_clock(m.rules.dt)
        assert m.phase == "kickoff"
        assert m.ball.pos.dist(Vec2(m.rules.length / 2, m.rules.width / 2)) < 2.0

    def test_ball_outside_the_posts_is_not_a_goal(self, live_match):
        m = live_match()
        clear_pitch(m)
        m.ball.pos = Vec2(104, m.rules.goal_y1 + 2.0)
        m.ball.vel = Vec2(10, 0)
        step_ball(m, 30)
        assert m.score[HOME] == 0


class TestMirroring:
    """Both teams must see themselves attacking +x, or bots can't be side-agnostic."""

    def test_both_teams_see_themselves_attacking_the_same_way(self, live_match):
        m = live_match()
        home = m.state_for(HOME)
        away = m.state_for(AWAY)
        assert home.pitch.their_goal.x == pytest.approx(m.rules.length)
        assert away.pitch.their_goal.x == pytest.approx(m.rules.length)

    def test_positions_are_mirrored_for_the_away_team(self, live_match):
        m = live_match()
        clear_pitch(m)
        m.players[1].pos = Vec2(20.0, 10.0)
        away = m.state_for(AWAY)
        opponent = away.them[1]
        assert opponent.pos.x == pytest.approx(m.rules.length - 20.0)
        assert opponent.pos.y == pytest.approx(m.rules.width - 10.0)

    def test_ball_height_is_not_mirrored(self, live_match):
        """Height is unaffected by which way you are facing."""
        m = live_match()
        m.ball.z, m.ball.vz = 3.0, 2.0
        for team in (HOME, AWAY):
            state = m.state_for(team)
            assert state.ball.height == pytest.approx(3.0)
            assert state.ball.vertical_speed == pytest.approx(2.0)

    def test_an_away_action_is_mirrored_back_into_the_world(self, live_match):
        from football.api import Action
        from football.engine import _mirror_action

        action = Action(move=Vec2(1.0, 0.0), kick=Vec2(0.0, 1.0), kick_power=0.5)
        mirrored = _mirror_action(action)
        assert mirrored.move.x == pytest.approx(-1.0)
        assert mirrored.kick.y == pytest.approx(-1.0)

    def test_an_away_bot_shooting_forward_scores_at_x_zero(self, make_match):
        """End to end: 'attack +x' in the away frame must mean x = 0 in the world."""
        from football.api import Action

        def always_shoot(state):
            me = state.us[1]
            return {1: Action.kick_to(me, state.pitch.their_goal, power=1.0)}

        m = make_match(away_actions=always_shoot)
        m.phase = "play"
        m.setpiece = None
        clear_pitch(m)
        shooter = m.players[m.squad_size + 1]
        shooter.pos = Vec2(8.0, 34.0)
        m.ball.pos = Vec2(7.4, 34.0)
        m.owner = None
        for _ in range(200):
            m.step()
            if m.score[AWAY]:
                break
        assert m.score[AWAY] == 1
