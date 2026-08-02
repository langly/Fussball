"""Loading team scripts, and containing the ones that misbehave."""

from __future__ import annotations

import textwrap

import pytest

from football.loader import BotError, load_controller

IDLE_BODY = """
    from football.api import Team, Action
    class T(Team):
        name = "Loaded"
        def act(self, state):
            return {p.index: Action.idle() for p in state.us}
"""


@pytest.fixture
def write_bot(tmp_path):
    def _write(source, name="bot.py"):
        path = tmp_path / name
        path.write_text(textwrap.dedent(source))
        return path

    return _write


class TestLoading:
    def test_loads_a_team_subclass(self, write_bot):
        c = load_controller(write_bot(IDLE_BODY))
        assert c.name == "Loaded"

    def test_loads_a_create_team_factory(self, write_bot):
        c = load_controller(write_bot("""
            from football.api import Team, Action
            class Inner(Team):
                name = "Factory"
                def act(self, state): return {}
            def create_team(): return Inner()
        """))
        assert c.name == "Factory"

    def test_loads_a_bare_act_function(self, write_bot):
        c = load_controller(write_bot("""
            TEAM_NAME = "Functional"
            def act(state): return {}
        """))
        assert c.name == "Functional"

    def test_a_missing_file_is_reported(self, tmp_path):
        with pytest.raises(BotError):
            load_controller(tmp_path / "nope.py")

    def test_a_file_with_no_team_is_reported(self, write_bot):
        with pytest.raises(BotError):
            load_controller(write_bot("x = 1\n"))

    def test_a_syntax_error_is_reported(self, write_bot):
        with pytest.raises(BotError):
            load_controller(write_bot("def broken(:\n"))

    def test_an_import_time_crash_is_reported(self, write_bot):
        with pytest.raises(BotError):
            load_controller(write_bot("raise RuntimeError('boom')\n"))


class TestNormalising:
    def test_a_sequence_of_actions_is_accepted(self, write_bot):
        c = load_controller(write_bot("""
            from football.api import Team, Action
            class T(Team):
                name = "Seq"
                def act(self, state):
                    return [Action(sprint=True) for _ in range(5)]
        """))
        result = c.act(None)
        assert len(result) == 5 and result[0].sprint is True

    def test_out_of_range_indices_are_dropped(self, write_bot):
        c = load_controller(write_bot("""
            from football.api import Team, Action
            class T(Team):
                name = "Wide"
                def act(self, state):
                    return {99: Action(sprint=True), 0: Action(sprint=True)}
        """))
        result = c.act(None)
        assert set(result) == {0, 1, 2, 3, 4}
        assert result[0].sprint is True

    def test_non_action_values_are_ignored(self, write_bot):
        c = load_controller(write_bot("""
            from football.api import Team
            class T(Team):
                name = "Bad"
                def act(self, state):
                    return {0: "not an action"}
        """))
        assert c.act(None)[0].move.length() == 0.0

    def test_an_exception_costs_only_the_tick(self, write_bot):
        c = load_controller(write_bot("""
            from football.api import Team
            class T(Team):
                name = "Boom"
                def act(self, state): raise ValueError("nope")
        """))
        result = c.act(None)
        assert len(result) == 5 and c.errors == 1


class TestSquadNames:
    def test_defaults_are_used_when_absent(self, write_bot):
        c = load_controller(write_bot(IDLE_BODY))
        assert c.squad_names(["a", "b", "c", "d", "e"]) == ["a", "b", "c", "d", "e"]

    def test_custom_names_are_taken(self, write_bot):
        c = load_controller(write_bot(IDLE_BODY.replace(
            'name = "Loaded"', 'name = "Loaded"\n        player_names = ("A","B","C","D","E")')))
        assert c.squad_names(["x"] * 5) == ["A", "B", "C", "D", "E"]

    def test_blanks_and_none_fall_back_per_slot(self, write_bot):
        c = load_controller(write_bot(IDLE_BODY.replace(
            'name = "Loaded"', 'name = "Loaded"\n        player_names = ("A","",None,"D")')))
        assert c.squad_names(["w", "x", "y", "z", "q"]) == ["A", "x", "y", "D", "q"]

    def test_long_names_are_truncated(self, write_bot):
        c = load_controller(write_bot(IDLE_BODY.replace(
            'name = "Loaded"', 'name = "Loaded"\n        player_names = ("X"*99,)')))
        assert len(c.squad_names(["a"] * 5)[0]) <= 14


class TestCrest:
    def test_a_crest_is_parsed(self, write_bot):
        c = load_controller(write_bot(IDLE_BODY.replace(
            'name = "Loaded"',
            'name = "Loaded"\n        logo = ("AB",".A")\n        logo_colors = {"A":"#f00","B":"#0f0"}')))
        assert c.logo()[:2] == (2, 2)

    def test_no_crest_is_fine(self, write_bot):
        assert load_controller(write_bot(IDLE_BODY)).logo() is None

    def test_a_broken_crest_does_not_raise(self, write_bot):
        c = load_controller(write_bot(IDLE_BODY.replace(
            'name = "Loaded"', 'name = "Loaded"\n        logo = 12345')))
        assert c.logo() is None
