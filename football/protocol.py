"""Wire format between the host and a sandboxed bot process.

JSON only. Never pickle: unpickling bytes produced by untrusted code executes
arbitrary code *in the host*, which would defeat the entire point of putting
the bot in another process.

The host must treat every value arriving from a bot as hostile. `decode_actions`
is the trust boundary, so it validates types, finiteness and ranges rather than
assuming the child is well behaved.
"""

from __future__ import annotations

import math

from .api import Action, BallView, GameState, Limits, MatchInfo, PitchInfo, PlayerView, Vec2

#: Refuse to even parse a line longer than this, so a bot cannot exhaust host
#: memory by replying with an enormous payload.
MAX_LINE_BYTES = 1 << 16

_LIMIT_FIELDS = (
    "control_radius", "keeper_control_radius", "kick_reach", "trap_speed",
    "keeper_trap_speed", "tackle_radius", "run_speed", "sprint_speed",
    "min_kick_speed", "max_kick_speed", "keeper_catch_radius",
    "keeper_catch_max_speed", "ball_friction",
)
_PITCH_FIELDS = (
    "length", "width", "goal_width", "penalty_depth", "penalty_width",
    "goal_area_depth", "goal_area_width",
)


# ---------------------------------------------------------------------------
# host -> child
# ---------------------------------------------------------------------------


def encode_setup(info: MatchInfo) -> dict:
    p, lim = info.pitch, info.limits
    return {
        "pitch": {f: getattr(p, f) for f in _PITCH_FIELDS},
        "limits": {f: getattr(lim, f) for f in _LIMIT_FIELDS},
        "half_seconds": info.half_seconds,
        "periods": info.periods,
        "dt": info.dt,
        "opponent_name": info.opponent_name,
        "playing_at_home": info.playing_at_home,
        "names_us": list(info.squad_names),
        "names_them": list(info.opponent_names),
    }


def decode_setup(d: dict) -> tuple[MatchInfo, list[str], list[str]]:
    p = d["pitch"]
    pitch = PitchInfo(
        center=Vec2(p["length"] / 2, p["width"] / 2),
        our_goal=Vec2(0.0, p["width"] / 2),
        their_goal=Vec2(p["length"], p["width"] / 2),
        **{f: float(p[f]) for f in _PITCH_FIELDS},
    )
    limits = Limits(**{f: float(d["limits"][f]) for f in _LIMIT_FIELDS})
    info = MatchInfo(
        pitch=pitch,
        limits=limits,
        half_seconds=float(d["half_seconds"]),
        periods=int(d["periods"]),
        dt=float(d["dt"]),
        opponent_name=str(d["opponent_name"]),
        playing_at_home=bool(d["playing_at_home"]),
        squad_names=tuple(d["names_us"]),
        opponent_names=tuple(d["names_them"]),
    )
    return info, list(d["names_us"]), list(d["names_them"])


def _encode_player(p: PlayerView) -> list:
    return [p.pos.x, p.pos.y, p.vel.x, p.vel.y, p.heading.x, p.heading.y,
            p.stamina, 1 if p.has_ball else 0]


def encode_state(s: GameState) -> dict:
    b = s.ball
    return {
        "k": [s.tick, s.time, s.period, s.time_left, s.our_score, s.their_score],
        "ph": s.phase,
        "sp": s.setpiece,
        "spo": s.setpiece_is_ours,
        "b": [b.pos.x, b.pos.y, b.vel.x, b.vel.y,
              -1 if b.owner_index is None else b.owner_index,
              1 if b.owned_by_us else 0, 1 if b.owned_by_them else 0,
              1 if b.held_by_keeper else 0],
        "u": [_encode_player(p) for p in s.us],
        "h": [_encode_player(p) for p in s.them],
    }


def _decode_player(a: list, index: int, name: str) -> PlayerView:
    return PlayerView(
        index=index,
        name=name,
        pos=Vec2(a[0], a[1]),
        vel=Vec2(a[2], a[3]),
        heading=Vec2(a[4], a[5]),
        stamina=a[6],
        is_keeper=index == 0,
        has_ball=bool(a[7]),
    )


def decode_state(d: dict, pitch: PitchInfo, limits: Limits,
                 names_us: list[str], names_them: list[str]) -> GameState:
    k = d["k"]
    b = d["b"]
    return GameState(
        tick=int(k[0]), time=float(k[1]), period=int(k[2]), time_left=float(k[3]),
        our_score=int(k[4]), their_score=int(k[5]),
        phase=d["ph"], setpiece=d["sp"], setpiece_is_ours=d["spo"],
        ball=BallView(
            pos=Vec2(b[0], b[1]), vel=Vec2(b[2], b[3]),
            owner_index=None if b[4] < 0 else int(b[4]),
            owned_by_us=bool(b[5]), owned_by_them=bool(b[6]),
            held_by_keeper=bool(b[7]),
        ),
        us=tuple(_decode_player(a, i, names_us[i]) for i, a in enumerate(d["u"])),
        them=tuple(_decode_player(a, i, names_them[i]) for i, a in enumerate(d["h"])),
        pitch=pitch,
        limits=limits,
    )


# ---------------------------------------------------------------------------
# child -> host   (the trust boundary: validate everything)
# ---------------------------------------------------------------------------


def encode_actions(actions, squad_size: int) -> dict:
    out = {}
    for i in range(squad_size):
        a = actions.get(i)
        if a is None:
            continue
        out[str(i)] = [
            a.move.x, a.move.y,
            1 if a.sprint else 0,
            None if a.kick is None else [a.kick.x, a.kick.y],
            a.kick_power,
            1 if a.catch else 0,
        ]
    return out


def _num(v, lo: float = -1e4, hi: float = 1e4) -> float:
    """Coerce to a real, bounded float. Anything odd becomes 0.0."""
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return 0.0
    f = float(v)
    if not math.isfinite(f):
        return 0.0
    return min(max(f, lo), hi)


def decode_actions(raw, squad_size: int) -> dict[int, Action]:
    """Turn a bot process's reply into actions, trusting none of it."""
    out = {i: Action.idle() for i in range(squad_size)}
    if not isinstance(raw, dict):
        return out
    for key, value in list(raw.items())[: squad_size * 2]:
        try:
            idx = int(key)
        except (TypeError, ValueError):
            continue
        if not 0 <= idx < squad_size:
            continue
        if not isinstance(value, (list, tuple)) or len(value) != 6:
            continue
        kick = None
        if isinstance(value[3], (list, tuple)) and len(value[3]) == 2:
            kick = Vec2(_num(value[3][0]), _num(value[3][1]))
        out[idx] = Action(
            move=Vec2(_num(value[0]), _num(value[1])),
            sprint=bool(value[2]),
            kick=kick,
            kick_power=_num(value[4], 0.0, 1.0),
            catch=bool(value[5]),
        ).sanitized()
    return out
