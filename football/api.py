"""The public API for team scripts.

A team script is a normal Python file that defines a subclass of `Team`::

    from football.api import Team, Action

    class MyTeam(Team):
        name = "My Team"

        def act(self, state):
            return {i: Action.go_to(state.us[i], state.ball.pos)
                    for i in range(5)}

Everything the bot sees is expressed in *its own* frame of reference:

  * your team always attacks +x (towards `state.pitch.their_goal`)
  * `state.us` are your five players, index 0 is always your goalkeeper
  * `state.them` are the five opponents

so the same script plays identically at home or away.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

from .vec import Vec2, from_angle

__all__ = [
    "Team",
    "Action",
    "GameState",
    "PlayerView",
    "BallView",
    "PitchInfo",
    "MatchInfo",
    "Limits",
    "Vec2",
    "from_angle",
    "KEEPER",
]

KEEPER = 0  # index of the goalkeeper in `state.us` / `state.them`


# ---------------------------------------------------------------------------
# What a bot sees
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PlayerView:
    """A single player as seen by a bot."""

    index: int
    name: str
    pos: Vec2
    vel: Vec2
    heading: Vec2
    stamina: float  # 1.0 = fresh, 0.0 = spent
    is_keeper: bool
    has_ball: bool

    @property
    def speed(self) -> float:
        return self.vel.length()


@dataclass(frozen=True, slots=True)
class BallView:
    pos: Vec2
    vel: Vec2
    owner_index: int | None  # index within the owning team, else None
    owned_by_us: bool
    owned_by_them: bool
    held_by_keeper: bool

    @property
    def loose(self) -> bool:
        return not (self.owned_by_us or self.owned_by_them)

    @property
    def speed(self) -> float:
        return self.vel.length()

    def predict(self, seconds: float, friction: float = 0.55) -> Vec2:
        """Where a rolling ball will be in `seconds` (ignores being tackled)."""
        import math

        if self.speed < 1e-6:
            return self.pos
        # integral of v0 * exp(-k t) dt from 0 to s
        travel = (1.0 - math.exp(-friction * seconds)) / friction
        return self.pos + self.vel * travel


@dataclass(frozen=True, slots=True)
class PitchInfo:
    length: float
    width: float
    goal_width: float
    penalty_depth: float
    penalty_width: float
    goal_area_depth: float
    goal_area_width: float
    center: Vec2
    our_goal: Vec2  # centre of the goal you defend (x = 0)
    their_goal: Vec2  # centre of the goal you attack (x = length)

    @property
    def our_post_left(self) -> Vec2:
        return Vec2(0.0, (self.width - self.goal_width) / 2)

    @property
    def our_post_right(self) -> Vec2:
        return Vec2(0.0, (self.width + self.goal_width) / 2)

    @property
    def their_post_left(self) -> Vec2:
        return Vec2(self.length, (self.width - self.goal_width) / 2)

    @property
    def their_post_right(self) -> Vec2:
        return Vec2(self.length, (self.width + self.goal_width) / 2)

    def in_our_penalty_area(self, p: Vec2) -> bool:
        half = self.penalty_width / 2
        return p.x <= self.penalty_depth and abs(p.y - self.width / 2) <= half

    def in_their_penalty_area(self, p: Vec2) -> bool:
        half = self.penalty_width / 2
        return p.x >= self.length - self.penalty_depth and abs(p.y - self.width / 2) <= half

    def clamp(self, p: Vec2, margin: float = 0.5) -> Vec2:
        """Clamp a point to the playing surface."""
        return Vec2(
            min(max(p.x, margin), self.length - margin),
            min(max(p.y, margin), self.width - margin),
        )


@dataclass(frozen=True, slots=True)
class Limits:
    """Engine constants you are allowed to plan against.

    The important one is `trap_speed`: a ball arriving faster than this
    (relative to you) cannot be brought under control, it just rebounds. Deciding
    whether to gather a ball or strike it is the single biggest call your bot
    makes, and it depends on this number, not on where the ball happens to be.
    """

    control_radius: float  # you can take a slow ball inside this range
    keeper_control_radius: float
    kick_reach: float  # you can strike a ball from this far, at any speed
    trap_speed: float  # max relative speed you can bring under control
    keeper_trap_speed: float
    tackle_radius: float
    run_speed: float
    sprint_speed: float
    min_kick_speed: float
    max_kick_speed: float
    keeper_catch_radius: float
    keeper_catch_max_speed: float
    ball_friction: float

    def kick_speed(self, power: float) -> float:
        """The speed a kick at `power` leaves your foot."""
        p = _clamp01(power)
        return self.min_kick_speed + p * (self.max_kick_speed - self.min_kick_speed)

    def power_for(self, distance: float, arrive_speed: float = 6.0) -> float:
        """Power needed so the ball is still moving at `arrive_speed` after `distance`.

        Use it for passes: the default leaves the ball slow enough for a
        team-mate to actually trap, instead of rebounding off them.
        """
        needed = arrive_speed + self.ball_friction * max(0.0, distance)
        span = self.max_kick_speed - self.min_kick_speed
        return _clamp01((needed - self.min_kick_speed) / span) if span > 0 else 1.0

    def reach_of(self, power: float) -> float:
        """How far a kick at `power` will roll before stopping."""
        return self.kick_speed(power) / self.ball_friction


@dataclass(frozen=True, slots=True)
class MatchInfo:
    """Handed to `on_match_start` once, before the first whistle."""

    pitch: PitchInfo
    limits: Limits
    half_seconds: float
    periods: int
    dt: float
    opponent_name: str
    playing_at_home: bool


@dataclass(frozen=True, slots=True)
class GameState:
    tick: int
    time: float  # seconds elapsed in the current period
    period: int  # 1-based
    time_left: float  # seconds left in the current period
    phase: str  # "kickoff" | "play" | "setpiece" | "goal" | "half_time" | "full_time"
    setpiece: str | None  # "kickoff" | "throw_in" | "corner" | "goal_kick"
    setpiece_is_ours: bool | None
    our_score: int
    their_score: int
    ball: BallView
    us: tuple[PlayerView, ...]
    them: tuple[PlayerView, ...]
    pitch: PitchInfo
    limits: Limits

    # -- convenience --------------------------------------------------
    def can_trap(self, me: PlayerView, margin: float = 1.0) -> bool:
        """True if this player could bring the ball under control right now.

        A ball closing faster than the trap limit rebounds off you instead, so
        the only way to play it is to strike it.
        """
        cap = self.limits.keeper_trap_speed if me.is_keeper else self.limits.trap_speed
        return (self.ball.vel - me.vel).length() < cap - margin

    @property
    def keeper(self) -> PlayerView:
        return self.us[KEEPER]

    @property
    def their_keeper(self) -> PlayerView:
        return self.them[KEEPER]

    @property
    def playing(self) -> bool:
        """True when the ball is live and can be played."""
        return self.phase in ("play", "setpiece")

    def nearest_teammate_to(self, point: Vec2, skip: int | None = None) -> PlayerView:
        pool = [p for p in self.us if p.index != skip]
        return min(pool, key=lambda p: p.pos.dist(point))

    def nearest_opponent_to(self, point: Vec2) -> PlayerView:
        return min(self.them, key=lambda p: p.pos.dist(point))

    def closest_to_ball(self, outfield_only: bool = True) -> PlayerView:
        pool = [p for p in self.us if not (outfield_only and p.is_keeper)] or list(self.us)
        return min(pool, key=lambda p: p.pos.dist(self.ball.pos))

    def opponents_within(self, point: Vec2, radius: float) -> list[PlayerView]:
        return [p for p in self.them if p.pos.dist(point) <= radius]

    def teammates_within(self, point: Vec2, radius: float, skip: int | None = None) -> list[PlayerView]:
        return [p for p in self.us if p.index != skip and p.pos.dist(point) <= radius]

    def pressure_on(self, point: Vec2, radius: float = 6.0) -> float:
        """0.0 = nobody near, grows as opponents crowd `point`."""
        total = 0.0
        for p in self.them:
            d = p.pos.dist(point)
            if d < radius:
                total += 1.0 - d / radius
        return total

    def lane_is_clear(self, start: Vec2, end: Vec2, corridor: float = 1.6) -> bool:
        """True if no opponent stands within `corridor` of the segment."""
        seg = end - start
        seg_len = seg.length()
        if seg_len < 1e-6:
            return True
        d = seg / seg_len
        for p in self.them:
            rel = p.pos - start
            along = rel.dot(d)
            if along <= 0.0 or along >= seg_len:
                continue
            if abs(rel.x * d.y - rel.y * d.x) < corridor:
                return False
        return True


# ---------------------------------------------------------------------------
# What a bot returns
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class Action:
    """One player's intent for one tick.

    `move` is a desired direction; its length (clamped to 1) scales the
    target speed. `kick` is a direction to strike the ball, applied only if
    the player can actually reach it this tick.
    """

    move: Vec2 = field(default_factory=Vec2)
    sprint: bool = False
    kick: Vec2 | None = None
    kick_power: float = 0.0  # 0..1
    catch: bool = False  # goalkeeper only, inside your own penalty area

    # -- builders -----------------------------------------------------
    @staticmethod
    def idle() -> "Action":
        return Action()

    @staticmethod
    def go_to(me: PlayerView, target: Vec2, sprint: bool = False, arrive: float = 1.2) -> "Action":
        """Run towards `target`, easing off over the last `arrive` metres."""
        delta = target - me.pos
        dist = delta.length()
        if dist < 1e-6:
            return Action(sprint=False)
        throttle = min(1.0, dist / arrive) if arrive > 0 else 1.0
        return Action(move=delta.normalized() * throttle, sprint=sprint and throttle > 0.5)

    @staticmethod
    def move_dir(direction: Vec2, sprint: bool = False) -> "Action":
        return Action(move=direction.clamped(1.0), sprint=sprint)

    @staticmethod
    def kick_to(me: PlayerView, target: Vec2, power: float = 1.0, sprint: bool = False) -> "Action":
        """Strike the ball towards `target` while holding position."""
        d = target - me.pos
        return Action(move=Vec2(), sprint=sprint, kick=d.normalized(), kick_power=_clamp01(power))

    @staticmethod
    def dribble(me: PlayerView, target: Vec2, power: float = 0.25, sprint: bool = True) -> "Action":
        """Knock the ball ahead towards `target` and chase it.

        You do NOT need this to carry the ball: the ball follows whoever owns
        it, so a plain `go_to` already dribbles. This deliberately releases
        possession to push the ball into space -- useful for knocking it past a
        defender, and a risk when someone else is closer to where it lands.

        To chase a *loose* ball use `Action.intercept`, which separates where
        you run from where you aim.
        """
        d = target - me.pos
        n = d.normalized()
        return Action(move=n, sprint=sprint, kick=n, kick_power=_clamp01(power))

    @staticmethod
    def intercept(
        me: PlayerView,
        ball_at: Vec2,
        kick_target: Vec2 | None = None,
        power: float = 1.0,
        sprint: bool = True,
    ) -> "Action":
        """Run at the ball, and strike it towards `kick_target` on arrival.

        Unlike `dribble` (which runs *and* kicks at the same point), this
        separates where you run from where you aim -- the usual want when
        chasing a loose ball you intend to clear or shoot.
        """
        a = Action.go_to(me, ball_at, sprint=sprint, arrive=0.0)
        if kick_target is not None:
            a.with_kick(me, kick_target, power)
        return a

    @staticmethod
    def pass_to(me: PlayerView, mate: PlayerView, power: float = 0.45, lead: float = 0.35) -> "Action":
        """Pass to a team-mate, leading them by `lead` seconds of their velocity."""
        target = mate.pos + mate.vel * lead
        return Action.kick_to(me, target, power)

    def with_kick(self, me: PlayerView, target: Vec2, power: float = 1.0) -> "Action":
        d = target - me.pos
        self.kick = d.normalized()
        self.kick_power = _clamp01(power)
        return self

    def sanitized(self) -> "Action":
        self.move = self.move.clamped(1.0)
        self.kick_power = _clamp01(self.kick_power)
        if self.kick is not None:
            k = self.kick.normalized()
            self.kick = k if k.length_sq() > 0 else None
        if self.kick is None:
            self.kick_power = 0.0
        return self


def _clamp01(v: float) -> float:
    return 0.0 if v < 0.0 else 1.0 if v > 1.0 else float(v)


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------


class Team:
    """Subclass this in your team script.

    Only `act` is required. The other hooks are optional.
    """

    #: Shown on the scoreboard.
    name: str = "Unnamed"

    #: Optional shirt names for your five players, index 0 first (the keeper).
    #: Short names read best -- they are drawn above whoever has the ball.
    #: Anything missing or blank falls back to the default roster.
    player_names: Sequence[str] | None = None

    def on_match_start(self, info: MatchInfo) -> None:
        """Called once before kickoff."""

    def act(self, state: GameState) -> Mapping[int, Action] | Sequence[Action]:
        """Return one `Action` per player, keyed by index 0..4 (0 = keeper).

        Missing indices are treated as `Action.idle()`. Raising an exception
        costs you the tick, not the match: your players simply stand still.
        """
        raise NotImplementedError

    def on_goal(self, scored_by_us: bool, state: GameState) -> None:
        """Called once each time a goal is scored."""

    def on_match_end(self, state: GameState) -> None:
        """Called once after the final whistle."""
