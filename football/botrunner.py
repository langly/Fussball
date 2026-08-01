"""Child process that runs exactly one untrusted team script.

Started by `football.sandbox`; not meant to be run by hand. It speaks JSON
lines on stdin/stdout (see `football.protocol`).

Order matters here: every restriction is installed *before* the bot module is
imported, because a team script runs arbitrary code at import time, long before
`act()` is ever called.

This process is the thing that is allowed to be compromised. The host assumes
it will be, kills it on timeout, and validates everything it says.
"""

from __future__ import annotations

import json
import os
import sys

# --- limits -----------------------------------------------------------------

MEMORY_BYTES = 512 * 1024 * 1024
CPU_SECONDS = 60  # hard ceiling for a whole match; per-tick timing is the host's job

#: Audit events refused outright. Prefix match, so "socket." covers the family.
BLOCKED_EVENTS = (
    "socket.",            # all networking bottoms out here
    "subprocess.",
    "os.system", "os.exec", "os.spawn", "os.fork", "os.posix_spawn", "os.startfile",
    "os.remove", "os.rename", "os.rmdir", "os.link", "os.symlink", "os.truncate",
    "os.chmod", "os.chown", "os.setuid", "os.setgid",
    "shutil.copy", "shutil.move", "shutil.rmtree",
    "ctypes.",            # direct memory / libc access would bypass everything
    "urllib.", "ftplib.", "smtplib.", "imaplib.", "poplib.", "nntplib.",
    "telnetlib.", "webbrowser.", "http.client",
    "pickle.find_class",  # arbitrary class resolution
    "pty.spawn",
)


class Denied(PermissionError):
    """Raised inside the bot process when it attempts a blocked operation."""


def _install_rlimits() -> None:
    try:
        import resource
    except ImportError:  # not POSIX
        return
    # Each is best-effort: a platform that refuses one should not stop the rest.
    for name, limit in (
        ("RLIMIT_FSIZE", 0),                 # cannot write any file (pipes still work)
        ("RLIMIT_CORE", 0),                  # no core dumps
        ("RLIMIT_AS", MEMORY_BYTES),         # address space
        ("RLIMIT_CPU", CPU_SECONDS),         # cpu seconds, SIGXCPU on breach
        ("RLIMIT_NOFILE", 64),               # few file descriptors
    ):
        res = getattr(resource, name, None)
        if res is None:
            continue
        try:
            soft, hard = resource.getrlimit(res)
            target = limit if hard in (resource.RLIM_INFINITY, -1) else min(limit, hard)
            resource.setrlimit(res, (target, target))
        except (ValueError, OSError):
            pass


def _writes(mode, flags) -> bool:
    if isinstance(mode, str) and any(c in mode for c in "wxa+"):
        return True
    if isinstance(flags, int):
        wr = os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_APPEND | os.O_TRUNC
        return bool(flags & wr)
    return False


def _read_roots(script: str) -> tuple[str, ...]:
    """Directories a bot is allowed to read from.

    Without this a bot can just read the operator's home directory by absolute
    path. The OS sandbox enforces the same thing on macOS; this is the portable
    half, and the only thing standing there on platforms with no profile.
    """
    roots = {
        sys.prefix, sys.base_prefix,
        os.path.dirname(os.path.realpath(sys.executable)),
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),  # project root
        os.path.dirname(os.path.abspath(script)),                     # the bot's own folder
        "/usr", "/System", "/Library", "/opt",
        "/dev/urandom", "/dev/null", "/dev/random",
    }
    return tuple(sorted(os.path.realpath(r) for r in roots if r))


def _install_audit_hook(script: str) -> None:
    allowed = _read_roots(script)

    def readable(path) -> bool:
        try:
            real = os.path.realpath(os.fspath(path))
        except (TypeError, ValueError, OSError):
            return False
        return real.startswith(allowed)

    def hook(event: str, args) -> None:
        if event.startswith(BLOCKED_EVENTS):
            raise Denied(f"blocked by sandbox: {event}")
        if event == "open":
            path = args[0] if args else None
            mode = args[1] if len(args) > 1 else None
            flags = args[2] if len(args) > 2 else None
            if _writes(mode, flags):
                raise Denied(f"blocked by sandbox: writing {path!r}")
            if path is not None and not readable(path):
                raise Denied(f"blocked by sandbox: reading {path!r}")
        elif event in ("os.listdir", "os.scandir") and args:
            target = args[0]
            if target is not None and not isinstance(target, int) and not readable(target):
                raise Denied(f"blocked by sandbox: listing {target!r}")

    # Audit hooks cannot be uninstalled once added -- there is deliberately no
    # removal API -- which is what makes this worth having even though it is a
    # guardrail rather than the boundary. The boundary is the OS limits above
    # and the fact that this whole process is disposable.
    sys.addaudithook(hook)


def harden(script: str) -> None:
    # Importing writes .pyc files, which RLIMIT_FSIZE would then kill.
    sys.dont_write_bytecode = True
    _install_rlimits()
    _install_audit_hook(script)


# --- main loop --------------------------------------------------------------


def main(argv: list[str]) -> int:
    if len(argv) < 3:
        print("usage: python -m football.botrunner <script> <squad_size>", file=sys.stderr)
        return 2
    script, squad_size = argv[1], int(argv[2])

    harden(script)

    # Imported only after hardening. The loader executes the bot's module-level
    # code, so this call is already running untrusted code.
    from .loader import BotError, load_controller
    from .protocol import decode_actions, decode_setup, decode_state, encode_actions

    out = sys.stdout
    try:
        team = load_controller(script, squad_size)
    except BotError as exc:
        out.write(json.dumps({"ok": False, "fatal": str(exc)}) + "\n")
        out.flush()
        return 1

    out.write(json.dumps({"ok": True, "name": team.name}) + "\n")
    out.flush()

    pitch = limits = None
    names_us: list[str] = []
    names_them: list[str] = []

    for line in sys.stdin:
        try:
            msg = json.loads(line)
        except ValueError:
            continue
        kind = msg.get("t")
        reply: dict = {"ok": True}
        try:
            if kind == "names":
                # asked before the match exists, so the host can label shirts
                reply["names"] = team.squad_names(list(msg.get("fallback") or []))
            elif kind == "start":
                info, names_us, names_them = decode_setup(msg["d"])
                pitch, limits = info.pitch, info.limits
                team.on_match_start(info)
            elif kind == "act":
                state = decode_state(msg["d"], pitch, limits, names_us, names_them)
                reply["a"] = encode_actions(team.act(state), squad_size)
            elif kind == "goal":
                state = decode_state(msg["d"], pitch, limits, names_us, names_them)
                team.on_goal(bool(msg.get("ours")), state)
            elif kind == "end":
                state = decode_state(msg["d"], pitch, limits, names_us, names_them)
                team.on_match_end(state)
            elif kind == "bye":
                return 0
        except Exception as exc:  # never let a bot kill the loop
            reply = {"ok": False, "err": f"{type(exc).__name__}: {exc}"[:400]}
        try:
            out.write(json.dumps(reply) + "\n")
            out.flush()
        except (BrokenPipeError, ValueError):
            return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
