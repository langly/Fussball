"""Sandbox containment.

Every test here writes a bot that genuinely attempts the attack, then asserts
it was stopped. These are slow (each spawns a process), so they are marked
`slow` -- run the fast suite with `-m "not slow"`.

Marked `skipif` on non-POSIX: the OS-level half of the sandbox (rlimits, and
the seatbelt profile on macOS) has no implementation on Windows yet.
"""

from __future__ import annotations

import os
import sys
import textwrap
from pathlib import Path

import pytest

from football.sandbox import SandboxedController, SandboxError

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(os.name != "posix", reason="OS sandbox is POSIX-only so far"),
]

IDLE = """
from football.api import Team, Action
class T(Team):
    name = "Probe"
    def act(self, state):
        return {p.index: Action.idle() for p in state.us}
"""


@pytest.fixture
def bot(tmp_path):
    """Write a bot that reports what it managed to do via its team name."""

    def _write(body, name="probe.py"):
        path = tmp_path / name
        path.write_text(textwrap.dedent(body))
        return path

    return _write


def load(path, **kw):
    return SandboxedController(path, 5, **kw)


def started_match(controller, rules):
    """Run the real handshake, then hand back a live state to act on.

    Calling act() straight after construction fails for an uninteresting
    reason -- the child has no pitch or squad names yet, so it cannot decode
    the state. Building a Match performs squad_names and on_match_start for us.
    """
    from conftest import ScriptedController
    from football.engine import Match

    match = Match(controller, ScriptedController("Away"), rules, seed=1)
    return match.state_for(0)


class TestImportTimeAttacks:
    """A team script runs arbitrary code at import, long before act() is called."""

    def test_opening_a_socket_is_blocked(self, bot):
        with pytest.raises(SandboxError) as err:
            load(bot("import socket\nsocket.socket()\n" + IDLE))
        assert "socket" in str(err.value)

    def test_writing_a_file_is_blocked(self, bot, tmp_path):
        target = tmp_path / "PWNED"
        with pytest.raises(SandboxError):
            load(bot(f"open({str(target)!r}, 'w').write('x')\n" + IDLE))
        assert not target.exists(), "nothing may be written"

    def test_spawning_a_process_is_blocked(self, bot):
        with pytest.raises(SandboxError) as err:
            load(bot("import subprocess\nsubprocess.run(['/bin/echo','hi'])\n" + IDLE))
        assert "subprocess" in str(err.value)

    def test_reading_outside_the_bots_folder_is_blocked(self, bot):
        """The bot reports its own result through the one channel it has: its name."""
        home = str(Path.home())
        c = load(bot(f"""
            import pathlib
            from football.api import Team, Action
            try:
                n = len(list(pathlib.Path({home!r}).iterdir()))
                RESULT = "LEAKED-%d" % n
            except Exception as e:
                RESULT = "denied"
            class T(Team):
                name = RESULT
                def act(self, state):
                    return {{p.index: Action.idle() for p in state.us}}
        """))
        try:
            assert not c.name.startswith("LEAKED"), "read the operator's home directory"
        finally:
            c.close()

    def test_the_environment_is_stripped(self, bot):
        """Secrets in the operator's shell must not be visible to a bot."""
        os.environ["FUSSBALL_SECRET_CANARY"] = "do-not-leak"
        try:
            c = load(bot("""
                import os
                from football.api import Team, Action
                RESULT = "SAW" if os.environ.get("FUSSBALL_SECRET_CANARY") else "clean"
                class T(Team):
                    name = RESULT
                    def act(self, state):
                        return {p.index: Action.idle() for p in state.us}
            """))
            try:
                assert c.name == "clean"
            finally:
                c.close()
        finally:
            del os.environ["FUSSBALL_SECRET_CANARY"]

    def test_a_bot_may_read_its_own_folder(self, bot, tmp_path):
        """Containment must not stop a bot loading its own data file."""
        (tmp_path / "data.txt").write_text("opening-book")
        c = load(bot("""
            import pathlib
            from football.api import Team, Action
            data = (pathlib.Path(__file__).parent / "data.txt").read_text().strip()
            class T(Team):
                name = data
                def act(self, state):
                    return {p.index: Action.idle() for p in state.us}
        """))
        try:
            assert c.name == "opening-book"
        finally:
            c.close()


class TestRuntimeContainment:
    def test_an_infinite_loop_is_killed(self, bot, rules):
        c = load(bot("""
            from football.api import Team
            class T(Team):
                name = "Hang"
                def act(self, state):
                    while True:
                        pass
        """), tick_timeout=0.25)
        try:
            state = started_match(c, rules)
            actions = c.act(state)
            assert c.killed_reason is not None, "a hung bot must be killed"
            assert len(actions) == 5, "and its team plays on, idle"
            assert all(a.move.length() == 0.0 for a in actions.values())
        finally:
            c.close()

    def test_a_killed_bot_keeps_returning_idle(self, bot, rules):
        c = load(bot("""
            from football.api import Team
            class T(Team):
                name = "Hang"
                def act(self, state):
                    while True: pass
        """), tick_timeout=0.2)
        try:
            state = started_match(c, rules)
            c.act(state)
            for _ in range(5):
                assert len(c.act(state)) == 5
        finally:
            c.close()

    def test_a_memory_bomb_does_not_take_the_host_down(self, bot, rules):
        c = load(bot("""
            from football.api import Team
            class T(Team):
                name = "Mem"
                def act(self, state):
                    x = bytearray(4_000_000_000)
                    return {}
        """), tick_timeout=1.5)
        try:
            assert len(c.act(started_match(c, rules))) == 5
        finally:
            c.close()

    def test_a_fork_bomb_is_contained(self, bot, rules):
        c = load(bot("""
            import os
            from football.api import Team, Action
            for _ in range(50):
                try: os.fork()
                except Exception: pass
            class T(Team):
                name = "Fork"
                def act(self, state):
                    return {p.index: Action.idle() for p in state.us}
        """), tick_timeout=1.0)
        try:
            assert len(c.act(started_match(c, rules))) == 5
        finally:
            c.close()


class TestSandboxParity:
    """A sandboxed bot must play exactly the same match as a trusted one."""

    def test_identical_results_either_side_of_the_boundary(self, tmp_path, rules):
        from football.engine import Match
        from football.loader import load_controller

        source = textwrap.dedent("""
            from football.api import Team, Action
            class T(Team):
                name = "Parity"
                player_names = ("A", "B", "C", "D", "E")
                logo = ("AB", ".A")
                logo_colors = {"A": "#ff0000", "B": "#00ff00"}
                def act(self, state):
                    me = state.us[1]
                    return {1: Action.intercept(me, state.ball.pos,
                                                state.pitch.their_goal, power=0.8, lift=0.3)}
        """)
        path = tmp_path / "parity.py"
        path.write_text(source)

        trusted = Match(load_controller(path), load_controller(path), rules, seed=5)
        trusted.run()

        home, away = load(path), load(path)
        try:
            sandboxed = Match(home, away, rules, seed=5)
            sandboxed.run()
        finally:
            home.close()
            away.close()

        assert sandboxed.score == trusted.score
        assert sandboxed.ball.pos.dist(trusted.ball.pos) < 1e-9
        assert sandboxed.ball.z == pytest.approx(trusted.ball.z)

    def test_a_crest_survives_the_boundary_intact(self, tmp_path):
        from football.loader import load_controller

        path = tmp_path / "crest.py"
        path.write_text(textwrap.dedent("""
            from football.api import Team, Action
            class T(Team):
                name = "Crest"
                logo = ("AB", ".A")
                logo_colors = {"A": "#ff0000", "B": "#00ff00"}
                def act(self, state): return {}
        """))
        c = load(path)
        try:
            c.squad_names(["a"] * 5)
            assert c.logo() == load_controller(path).logo()
        finally:
            c.close()

    def test_a_hostile_crest_is_rejected_host_side(self, tmp_path):
        """The host re-validates rather than trusting what the child sends."""
        path = tmp_path / "badcrest.py"
        path.write_text(textwrap.dedent("""
            from football.api import Team, Action
            class T(Team):
                name = "BadCrest"
                logo = ["X" * 500] * 500
                logo_colors = {"X": "#fff"}
                def act(self, state): return {}
        """))
        c = load(path)
        try:
            c.squad_names(["a"] * 5)
            assert c.logo() is None
        finally:
            c.close()
