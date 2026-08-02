"""All tunable simulation constants live here.

Units are SI: metres, seconds, metres/second. The pitch is a standard
105 x 68 m field with the origin at the top-left corner:

    (0,0) ---------------- x -> (105, 0)
      |                            |
      | left goal            right goal
      |                            |
    (0,68) --------------------- (105,68)

The home team (team 0) always attacks +x. Bots never see this: the state
handed to a team is mirrored so that *every* bot attacks +x.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Rules:
    # --- time ---------------------------------------------------------
    dt: float = 1.0 / 60.0
    half_seconds: float = 90.0  # simulated seconds per half
    periods: int = 2

    # --- pitch --------------------------------------------------------
    length: float = 105.0
    width: float = 68.0
    # Wider than a real 7.32 m goal: with only five a side on a full-size
    # pitch, a regulation goal is very hard to beat and matches tend to 0-0.
    goal_width: float = 10.0
    penalty_depth: float = 16.5
    goal_area_depth: float = 5.5
    center_circle_r: float = 9.15

    # --- squad --------------------------------------------------------
    outfield_players: int = 4  # + 1 goalkeeper = 5 per team

    # --- players ------------------------------------------------------
    player_radius: float = 0.45
    run_speed: float = 7.0
    sprint_speed: float = 9.2
    accel: float = 22.0  # m/s^2 toward the desired velocity
    stamina_drain: float = 0.055  # per second while sprinting
    stamina_recover: float = 0.030  # per second while not sprinting
    stamina_speed_floor: float = 0.70  # speed factor at zero stamina
    stumble_seconds: float = 0.45  # slowdown after losing a tackle
    stumble_speed_factor: float = 0.40
    # How quickly a player who is standing still turns to face the ball. Without
    # this their heading freezes wherever they last ran, so idle players end up
    # all staring the same way.
    idle_turn_rate: float = 6.0  # radians per second, so a full turn takes ~0.5 s

    # --- ball ---------------------------------------------------------
    ball_radius: float = 0.11
    ball_friction: float = 0.55  # exponential decay constant
    ball_stop_speed: float = 0.15
    control_radius: float = 1.10  # distance at which a player takes control
    keeper_control_bonus: float = 0.90  # keepers reach further
    dribble_offset: float = 0.65  # ball sits this far in front of the owner
    kick_reach: float = 1.60  # a loose ball can be struck from this far
    min_kick_speed: float = 2.5  # low enough that a gentle touch is possible
    max_kick_speed: float = 28.0
    kick_cooldown: float = 0.35
    kick_momentum: float = 0.30  # share of the kicker's velocity added
    kick_spread: float = 0.055  # radians of noise at full power
    loose_after_kick: float = 0.22  # nobody may reclaim the ball this soon

    # A ball arriving faster than this (relative to the player) cannot be
    # brought under control -- it rebounds off them instead. Without this a
    # 28 m/s shot would be captured by anyone it happened to pass near.
    trap_speed: float = 10.0
    keeper_trap_bonus: float = 5.0
    deflection_restitution: float = 0.45
    deflection_spread: float = 0.18  # radians of noise on a rebound

    # --- aerial -------------------------------------------------------
    # Only the ball leaves the ground. Players stay two-dimensional and are
    # given a vertical *reach* instead, which keeps Vec2 correct everywhere
    # for positions and avoids a full 3D rewrite for no gameplay gain.
    gravity: float = 9.81
    # Bounces are damped hard on purpose: an airborne ball cannot be trapped,
    # so a lively ball that bounces five times keeps it out of play far longer
    # than real turf would and turns matches into pinball.
    ball_restitution: float = 0.42  # energy kept when it bounces off the turf
    air_drag: float = 0.06  # airborne balls are not slowed by turf friction
    settle_speed: float = 1.4  # below this a bounce stops bouncing
    max_launch_angle: float = 0.95  # radians at lift = 1.0 (~54 degrees)
    crossbar_height: float = 2.44
    post_radius: float = 0.06
    reach_foot: float = 0.55  # trap or dribble only below this
    # There is no jumping in the model, so reach is a hard ceiling. Set it at
    # heading height rather than standing height: too low and a lofted ball
    # simply sails over everyone untouched, which makes hoofing it downfield
    # a free way to bypass the whole pitch.
    reach_head: float = 2.60  # can still head or volley the ball below this
    reach_keeper: float = 2.95  # a keeper reaches higher again

    # --- duels --------------------------------------------------------
    tackle_radius: float = 1.35
    tackle_rate: float = 1.9  # base steal attempts per second
    shield_factor: float = 0.55  # rate multiplier when escaping at full speed
    # How much a stamina gap swings a duel: at 0.40, a fresh player robbing a
    # spent one is 40% more likely to win the ball, and 40% less the other way.
    tackle_stamina_swing: float = 0.40

    # --- goalkeeper ---------------------------------------------------
    keeper_catch_radius: float = 1.60  # how far a keeper can dive
    keeper_max_hold: float = 6.0
    keeper_catch_max_speed: float = 17.0  # faster than this is parried, not held
    parry_restitution: float = 0.55
    # After the six seconds are up the ball is released and that keeper may not
    # pick it up again for a while -- without this a keeper whose bot never
    # kicks would hold possession for the rest of the match.
    keeper_catch_lockout: float = 2.5

    # --- restarts -----------------------------------------------------
    setpiece_clearance: float = 4.5
    setpiece_timeout: float = 8.0
    kickoff_freeze: float = 1.2
    goal_celebration: float = 2.0

    # --- bot budget ---------------------------------------------------
    think_budget_ms: float = 8.0  # soft budget, reported at full time

    # --- derived ------------------------------------------------------
    # Both boxes are marked out from the goalposts, exactly as on a real
    # pitch, so they stay in proportion whatever `goal_width` is set to.
    @property
    def penalty_width(self) -> float:
        return self.goal_width + 2 * self.penalty_depth

    @property
    def goal_area_width(self) -> float:
        return self.goal_width + 2 * self.goal_area_depth

    @property
    def goal_y0(self) -> float:
        return (self.width - self.goal_width) / 2.0

    @property
    def goal_y1(self) -> float:
        return (self.width + self.goal_width) / 2.0

    @property
    def total_seconds(self) -> float:
        return self.half_seconds * self.periods


# Starting formation as fractions of (length, width) for a team attacking +x.
# Index 0 is always the goalkeeper.
FORMATION: tuple[tuple[float, float], ...] = (
    (0.045, 0.50),  # 0 keeper
    (0.230, 0.28),  # 1 left back
    (0.230, 0.72),  # 2 right back
    (0.420, 0.50),  # 3 midfielder
    (0.600, 0.50),  # 4 striker
)

ROLE_NAMES = ("GK", "LB", "RB", "MID", "FW")

# Shirt names used when a team script does not supply its own `player_names`.
DEFAULT_SQUAD_NAMES: tuple[tuple[str, ...], ...] = (
    ("Hansen", "Berg", "Dahl", "Ruud", "Moen"),
    ("Silva", "Costa", "Reis", "Neto", "Pinto"),
)


def default_squad_names(team: int, size: int) -> list[str]:
    """A full roster of `size` names for `team` (0 = home, 1 = away)."""
    base = DEFAULT_SQUAD_NAMES[team % len(DEFAULT_SQUAD_NAMES)]
    return [base[i] if i < len(base) else f"No.{i}" for i in range(size)]
