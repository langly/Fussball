"""Host side of the sandbox: runs an untrusted team script in its own process.

`SandboxedController` is a drop-in replacement for `loader.Controller`, so the
engine neither knows nor cares whether a team is sandboxed.

The security model is: **the child process is expected to be hostile.**
Everything it says is parsed defensively, it is killed if it takes too long,
and it is handed a stripped environment so it cannot read the operator's
secrets. What actually contains it is the operating system -- process limits,
a scrubbed environment and, where available, an OS sandbox profile -- not any
Python-level filtering, which is a guardrail rather than a boundary.
"""

from __future__ import annotations

import json
import os
import select
import shutil
import subprocess
import sys
import time
from pathlib import Path

from .api import Action, parse_logo
from .protocol import MAX_LINE_BYTES, decode_actions, encode_setup, encode_state

#: Wall-clock a bot gets per tick before it is killed. Generous next to the
#: ~0.02 ms a normal bot takes, but finite -- an unkillable infinite loop in a
#: bot must not hang the match.
TICK_TIMEOUT = 0.25
#: Import-time can be slower (the bot may build tables), so it gets its own budget.
STARTUP_TIMEOUT = 10.0

PROJECT_ROOT = Path(__file__).resolve().parent.parent

def _sbpl_quote(path: str) -> str:
    return path.replace("\\", "\\\\").replace('"', '\\"')


def _macos_profile() -> str:
    """Deny-by-default seatbelt profile: no network, no writes, reads allowed.

    `process-exec` has to be granted for the interpreter itself, or the child
    cannot even start -- so it is scoped to the directories the interpreter
    actually lives in, which are discovered at runtime rather than hardcoded
    (a venv's python is typically a symlink into a completely different prefix).
    Anything exec'd this way still inherits the no-network, no-write rules.
    """
    roots = {
        "/usr", "/System", "/Library",
        str(PROJECT_ROOT),
        sys.prefix,
        sys.base_prefix,
        os.path.dirname(os.path.realpath(sys.executable)),
    }
    return _build_profile(roots, extra_reads=())


#: Read-denied even though reads are otherwise permitted. A strict read
#: allowlist was tried first and silently kills the interpreter -- CPython
#: touches more of the filesystem at startup than is practical to enumerate --
#: so this denies the places worth protecting instead. In SBPL the last
#: matching rule wins, so the project's own re-allow below still applies.
PRIVATE_READ_PATHS = ("/Users", "/private/var/root", "/private/etc/ssh")


def _build_profile(roots, extra_reads) -> str:
    denied = " ".join(f'(subpath "{_sbpl_quote(p)}")' for p in PRIVATE_READ_PATHS)
    allow_back = " ".join(
        f'(subpath "{_sbpl_quote(r)}")' for r in sorted({*roots, *extra_reads}) if r
    )
    execs = " ".join(f'(subpath "{_sbpl_quote(r)}")' for r in sorted(roots) if r)
    return (
        "(version 1)\n"
        "(deny default)\n"
        "(allow process-fork)\n"
        "(allow sysctl-read)\n"
        "(allow mach-lookup)\n"
        "(allow file-read-metadata)\n"
        "(allow file-read*)\n"
        f"(deny file-read* {denied})\n"
        # ...but the project and the bot's own folder stay readable even if
        # they happen to live inside a home directory.
        f"(allow file-read* {allow_back})\n"
        f"(allow process-exec {execs})\n"
        "(deny file-write*)\n"
        "(deny network*)\n"
    )


class SandboxError(Exception):
    pass


class SandboxedController:
    """Mirrors `loader.Controller`, but the bot lives in another process."""

    def __init__(self, script: str | Path, squad_size: int = 5, *,
                 tick_timeout: float = TICK_TIMEOUT, os_sandbox: bool = True) -> None:
        self.script = Path(script).expanduser().resolve()
        self.squad_size = squad_size
        self.tick_timeout = tick_timeout
        self.name = self.script.stem
        self.errors = 0
        self.think_seconds = 0.0
        self.worst_tick_ms = 0.0
        self.calls = 0
        self.killed_reason: str | None = None
        self.closed = False
        self._logo = None
        self._buf = b""
        self._idle = {i: Action.idle() for i in range(squad_size)}

        self.proc = self._spawn(os_sandbox)
        hello = self._exchange_raw(None, STARTUP_TIMEOUT)
        if hello is None or not hello.get("ok"):
            detail = (hello or {}).get("fatal", "bot process failed to start")
            self._kill(str(detail))
            raise SandboxError(f"{self.script.name}: {detail}")
        self.name = str(hello.get("name") or self.script.stem)[:40]

    # -- process ------------------------------------------------------
    def _spawn(self, os_sandbox: bool) -> subprocess.Popen:
        cmd = [sys.executable, "-B", "-m", "football.botrunner",
               str(self.script), str(self.squad_size)]
        self.os_sandboxed = False
        if os_sandbox and sys.platform == "darwin" and shutil.which("sandbox-exec"):
            roots = {
                "/usr", "/System", "/Library", str(PROJECT_ROOT),
                sys.prefix, sys.base_prefix,
                os.path.dirname(os.path.realpath(sys.executable)),
            }
            # the bot may legitimately read data files next to its own script
            profile = _build_profile(roots, extra_reads=(str(self.script.parent),))
            cmd = ["sandbox-exec", "-p", profile] + cmd
            self.os_sandboxed = True

        # A stripped environment: the bot inherits none of the operator's
        # variables, so tokens and keys in the parent shell stay invisible.
        env = {
            "PATH": "/usr/bin:/bin",
            "PYTHONPATH": str(PROJECT_ROOT),
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONHASHSEED": "0",  # keeps a sandboxed match reproducible
            "HOME": "/nonexistent",
            "TMPDIR": "/nonexistent",
        }
        try:
            proc = subprocess.Popen(
                cmd, cwd=str(PROJECT_ROOT), env=env,
                stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL, bufsize=0,
            )
        except OSError as exc:
            raise SandboxError(f"cannot start sandbox for {self.script.name}: {exc}") from exc
        os.set_blocking(proc.stdout.fileno(), False)
        return proc

    def _kill(self, reason: str | None) -> None:
        # `reason=None` is an ordinary shutdown, not a kill worth reporting
        if reason is not None and self.killed_reason is None and not self.closed:
            self.killed_reason = reason
        proc = getattr(self, "proc", None)
        if proc is None or proc.poll() is not None:
            return
        try:
            proc.kill()
            proc.wait(timeout=2.0)
        except Exception:
            pass

    def close(self) -> None:
        proc = getattr(self, "proc", None)
        if proc is None or proc.poll() is not None:
            self.closed = True
            return
        try:
            self._send({"t": "bye"})
            proc.wait(timeout=0.5)
        except Exception:
            pass
        self.closed = True
        self._kill(None)

    def __del__(self):
        try:
            self.closed = True
            self._kill(None)
        except Exception:
            pass

    # -- transport ----------------------------------------------------
    def _send(self, msg: dict) -> None:
        data = (json.dumps(msg, separators=(",", ":")) + "\n").encode()
        self.proc.stdin.write(data)
        self.proc.stdin.flush()

    def _read_line(self, timeout: float) -> bytes | None:
        """Read one line, or None on timeout, EOF or an oversized reply."""
        deadline = time.monotonic() + timeout
        fd = self.proc.stdout.fileno()
        while True:
            nl = self._buf.find(b"\n")
            if nl >= 0:
                line, self._buf = self._buf[:nl], self._buf[nl + 1:]
                return line
            if len(self._buf) > MAX_LINE_BYTES:
                self._kill("reply exceeded size limit")
                return None
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                return None
            try:
                ready, _, _ = select.select([fd], [], [], remaining)
            except (OSError, ValueError):
                return None
            if not ready:
                return None
            try:
                chunk = os.read(fd, 65536)
            except (BlockingIOError, InterruptedError):
                continue
            except OSError:
                return None
            if not chunk:  # child exited
                return None
            self._buf += chunk

    def _exchange_raw(self, msg: dict | None, timeout: float) -> dict | None:
        if self.killed_reason is not None:
            return None
        try:
            if msg is not None:
                self._send(msg)
        except (BrokenPipeError, OSError):
            self._kill("bot process closed the pipe")
            return None
        line = self._read_line(timeout)
        if line is None:
            self._kill(f"no reply within {timeout:.2f}s")
            return None
        try:
            reply = json.loads(line)
        except ValueError:
            self._kill("bot process sent malformed JSON")
            return None
        return reply if isinstance(reply, dict) else None

    def _exchange(self, msg: dict, timeout: float) -> dict | None:
        reply = self._exchange_raw(msg, timeout)
        if reply is None:
            return None
        if not reply.get("ok"):
            self._note_error(reply.get("err", "bot error"))
            return None
        return reply

    def _note_error(self, detail) -> None:
        self.errors += 1
        if self.errors <= 3:
            print(f"[{self.name}] {detail}", file=sys.stderr)
            if self.errors == 3:
                print(f"[{self.name}] further errors will be silenced.", file=sys.stderr)

    # -- controller interface -----------------------------------------
    def logo(self):
        """The crest the bot declared, re-validated here rather than trusted."""
        return self._logo

    def squad_names(self, fallback) -> list[str]:
        reply = self._exchange({"t": "names", "fallback": list(fallback)}, STARTUP_TIMEOUT)
        out = list(fallback)
        if reply is None:
            return out
        # Parsed with the same validator the trusted path uses, so a hostile
        # bot cannot smuggle anything through the crest.
        self._logo = parse_logo(reply.get("logo"), reply.get("logo_colors"))
        raw = reply.get("names")
        if not isinstance(raw, list):
            return out
        for i, value in enumerate(raw[: len(out)]):
            if value is None:
                continue
            text = str(value).strip()
            if text:
                out[i] = text[:14]
        return out

    def on_match_start(self, info) -> None:
        self._exchange({"t": "start", "d": encode_setup(info)}, STARTUP_TIMEOUT)

    def on_goal(self, scored_by_us: bool, state) -> None:
        self._exchange(
            {"t": "goal", "ours": bool(scored_by_us), "d": encode_state(state)},
            self.tick_timeout,
        )

    def on_match_end(self, state) -> None:
        self._exchange({"t": "end", "d": encode_state(state)}, self.tick_timeout)
        self.close()

    def act(self, state) -> dict[int, Action]:
        if self.killed_reason is not None:
            return dict(self._idle)
        start = time.perf_counter()
        reply = self._exchange({"t": "act", "d": encode_state(state)}, self.tick_timeout)
        elapsed = time.perf_counter() - start
        self.think_seconds += elapsed
        self.worst_tick_ms = max(self.worst_tick_ms, elapsed * 1000.0)
        self.calls += 1
        if reply is None:
            return dict(self._idle)
        return decode_actions(reply.get("a"), self.squad_size)

    def stats(self) -> dict:
        avg = (self.think_seconds / self.calls * 1000.0) if self.calls else 0.0
        return {
            "name": self.name,
            "errors": self.errors,
            "avg_ms": round(avg, 3),
            "worst_ms": round(self.worst_tick_ms, 2),
            "killed": self.killed_reason,
        }


def load_sandboxed(script, squad_size: int = 5, **kw) -> SandboxedController:
    return SandboxedController(script, squad_size, **kw)
