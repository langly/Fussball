"""The deterministic match simulation.

`Match.step()` advances exactly one fixed timestep. Given the same seed and
the same two controllers, a match replays identically every time.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

from .api import BallView, GameState, Limits, MatchInfo, PitchInfo, PlayerView
from .config import FORMATION, Rules, default_squad_names
from .vec import Vec2

HOME, AWAY = 0, 1

# Phases in which nobody may move or touch the ball.
FROZEN_PHASES = frozenset({"kickoff", "goal", "half_time", "full_time"})


@dataclass
class Player:
    team: int
    index: int
    name: str
    pos: Vec2
    vel: Vec2 = field(default_factory=Vec2)
    heading: Vec2 = field(default_factory=lambda: Vec2(1.0, 0.0))
    stamina: float = 1.0
    kick_cd: float = 0.0
    stumble: float = 0.0
    catch_cd: float = 0.0
    distance_run: float = 0.0

    @property
    def is_keeper(self) -> bool:
        return self.index == 0


@dataclass
class Ball:
    pos: Vec2
    vel: Vec2 = field(default_factory=Vec2)


@dataclass
class Event:
    tick: int
    time: float
    kind: str  # "goal" | "throw_in" | "corner" | "goal_kick" | "save" | "period"
    team: int | None
    detail: str = ""


class Match:
    """A single match between two controllers.

    A controller is anything with `.name`, `.act(state) -> {index: Action}`
    and the optional `on_match_start` / `on_goal` / `on_match_end` hooks.
    `football.loader.load_controller` wraps user scripts into one.
    """

    def __init__(self, home, away, rules: Rules | None = None, seed: int = 0) -> None:
        self.rules = rules or Rules()
        self.controllers = (home, away)
        self.rng = random.Random(seed)
        self.seed = seed

        r = self.rules
        self.squad_size = r.outfield_players + 1
        squad_names = [
            ctrl.squad_names(default_squad_names(t, self.squad_size))
            for t, ctrl in enumerate(self.controllers)
        ]
        self.players: list[Player] = [
            Player(team=t, index=i, name=squad_names[t][i], pos=self._formation_pos(t, i))
            for t in (HOME, AWAY)
            for i in range(self.squad_size)
        ]
        self.ball = Ball(pos=Vec2(r.length / 2, r.width / 2))

        self.score = [0, 0]
        self.tick = 0
        self.clock = 0.0  # seconds into the current period
        self.period = 1
        self.events: list[Event] = []

        # possession / restart bookkeeping
        self.owner: Player | None = None
        self.last_touch: Player | None = None
        self.loose_timer = 0.0
        self.keeper_hold = 0.0
        self.possession_ticks = [0, 0]
        self.shots = [0, 0]

        self.phase = "kickoff"
        self.phase_timer = r.kickoff_freeze
        self.setpiece: str | None = "kickoff"
        self.setpiece_team: int = HOME
        self.setpiece_clock = 0.0

        self._notify_start()
        self._setup_kickoff(HOME)

    # ------------------------------------------------------------------
    # setup helpers
    # ------------------------------------------------------------------
    def _formation_pos(self, team: int, index: int, own_half_only: bool = False) -> Vec2:
        r = self.rules
        fx, fy = FORMATION[index]
        if own_half_only:
            fx = min(fx, 0.46)
        p = Vec2(fx * r.length, fy * r.width)
        return p if team == HOME else Vec2(r.length - p.x, r.width - p.y)

    def _notify_start(self) -> None:
        r = self.rules
        for team, ctrl in enumerate(self.controllers):
            info = MatchInfo(
                pitch=self._pitch_info(),
                limits=self._limits(),
                half_seconds=r.half_seconds,
                periods=r.periods,
                dt=r.dt,
                opponent_name=self.controllers[1 - team].name,
                playing_at_home=(team == HOME),
            )
            ctrl.on_match_start(info)

    def _limits(self) -> Limits:
        r = self.rules
        return Limits(
            control_radius=r.control_radius,
            keeper_control_radius=r.control_radius + r.keeper_control_bonus,
            kick_reach=r.kick_reach,
            trap_speed=r.trap_speed,
            keeper_trap_speed=r.trap_speed + r.keeper_trap_bonus,
            tackle_radius=r.tackle_radius,
            run_speed=r.run_speed,
            sprint_speed=r.sprint_speed,
            min_kick_speed=r.min_kick_speed,
            max_kick_speed=r.max_kick_speed,
            keeper_catch_radius=r.keeper_catch_radius,
            keeper_catch_max_speed=r.keeper_catch_max_speed,
            ball_friction=r.ball_friction,
        )

    def _pitch_info(self) -> PitchInfo:
        r = self.rules
        return PitchInfo(
            length=r.length,
            width=r.width,
            goal_width=r.goal_width,
            penalty_depth=r.penalty_depth,
            penalty_width=r.penalty_width,
            goal_area_depth=r.goal_area_depth,
            goal_area_width=r.goal_area_width,
            center=Vec2(r.length / 2, r.width / 2),
            our_goal=Vec2(0.0, r.width / 2),
            their_goal=Vec2(r.length, r.width / 2),
        )

    def _setup_kickoff(self, taking: int) -> None:
        r = self.rules
        for p in self.players:
            p.pos = self._formation_pos(p.team, p.index, own_half_only=True)
            p.vel = Vec2()
            p.heading = Vec2(1.0, 0.0) if p.team == HOME else Vec2(-1.0, 0.0)
            p.kick_cd = 0.0
            p.stumble = 0.0
        # the team taking the kickoff puts its striker on the ball
        striker = self.players[taking * self.squad_size + self.squad_size - 1]
        offset = -1.2 if taking == HOME else 1.2
        striker.pos = Vec2(r.length / 2 + offset, r.width / 2)

        self.ball.pos = Vec2(r.length / 2, r.width / 2)
        self.ball.vel = Vec2()
        self.owner = None
        self.last_touch = None
        self.loose_timer = 0.0
        self.keeper_hold = 0.0
        self.phase = "kickoff"
        self.phase_timer = r.kickoff_freeze
        self.setpiece = "kickoff"
        self.setpiece_team = taking
        self.setpiece_clock = 0.0

    def _award_setpiece(self, kind: str, team: int, spot: Vec2) -> None:
        self.ball.pos = spot
        self.ball.vel = Vec2()
        self.owner = None
        self.last_touch = None
        self.loose_timer = 0.0
        self.keeper_hold = 0.0
        self.phase = "setpiece"
        self.phase_timer = 0.0
        self.setpiece = kind
        self.setpiece_team = team
        self.setpiece_clock = 0.0
        self.events.append(Event(self.tick, self.clock, kind, team))

    # ------------------------------------------------------------------
    # main loop
    # ------------------------------------------------------------------
    @property
    def finished(self) -> bool:
        return self.phase == "full_time"

    def step(self) -> None:
        if self.finished:
            return
        r = self.rules
        dt = r.dt

        actions = {
            HOME: self.controllers[HOME].act(self.state_for(HOME)),
            AWAY: self.controllers[AWAY].act(self.state_for(AWAY)),
        }
        # away acts in a mirrored frame; bring its intent back to world space
        actions[AWAY] = {i: _mirror_action(a) for i, a in actions[AWAY].items()}

        for p in self.players:
            self._move_player(p, actions[p.team].get(p.index), dt)
        self._separate_players()
        self._update_ball(actions, dt)
        self._check_boundaries()
        self._advance_clock(dt)
        self.tick += 1

    def run(self, on_tick=None) -> None:
        """Run to full time (headless)."""
        while not self.finished:
            self.step()
            if on_tick is not None:
                on_tick(self)

    # ------------------------------------------------------------------
    # players
    # ------------------------------------------------------------------
    def _move_player(self, p: Player, action, dt: float) -> None:
        r = self.rules
        p.kick_cd = max(0.0, p.kick_cd - dt)
        p.catch_cd = max(0.0, p.catch_cd - dt)
        if self.phase in FROZEN_PHASES or action is None:
            p.vel = p.vel * 0.80
            self._integrate(p, dt)
            return

        sprint = bool(action.sprint) and p.stamina > 0.05
        top = r.sprint_speed if sprint else r.run_speed
        top *= r.stamina_speed_floor + (1.0 - r.stamina_speed_floor) * p.stamina
        if p.stumble > 0.0:
            top *= r.stumble_speed_factor
            p.stumble -= dt

        desired = action.move.clamped(1.0) * top
        p.vel += (desired - p.vel).clamped(r.accel * dt)
        self._integrate(p, dt)

        if sprint and desired.length_sq() > 0.01:
            p.stamina = max(0.0, p.stamina - r.stamina_drain * dt)
        else:
            p.stamina = min(1.0, p.stamina + r.stamina_recover * dt)

    def _integrate(self, p: Player, dt: float) -> None:
        r = self.rules
        step = p.vel * dt
        p.distance_run += step.length()
        pos = p.pos + step
        # players stay on the pitch (they may stand on the line)
        p.pos = Vec2(
            min(max(pos.x, 0.0), r.length),
            min(max(pos.y, 0.0), r.width),
        )
        if p.vel.length_sq() > 0.04:
            p.heading = p.vel.normalized()

    def _separate_players(self) -> None:
        """Cheap pairwise push-apart so bodies do not overlap."""
        r = self.rules
        min_d = r.player_radius * 2.0
        n = len(self.players)
        for i in range(n):
            a = self.players[i]
            for j in range(i + 1, n):
                b = self.players[j]
                delta = b.pos - a.pos
                d = delta.length()
                if d >= min_d:
                    continue
                if d < 1e-6:
                    delta = Vec2(self.rng.uniform(-1, 1), self.rng.uniform(-1, 1)).normalized()
                    d = 1e-6
                push = delta.normalized() * ((min_d - d) * 0.5)
                a.pos = a.pos - push
                b.pos = b.pos + push

    # ------------------------------------------------------------------
    # ball
    # ------------------------------------------------------------------
    def _may_touch(self, p: Player) -> bool:
        if self.phase in FROZEN_PHASES:
            return False
        if self.phase == "setpiece":
            return p.team == self.setpiece_team
        return True

    def _control_radius(self, p: Player) -> float:
        r = self.rules
        return r.control_radius + (r.keeper_control_bonus if p.is_keeper else 0.0)

    def _update_ball(self, actions, dt: float) -> None:
        r = self.rules
        if self.loose_timer > 0.0:
            self.loose_timer -= dt

        self._resolve_keeper(actions, dt)
        kicked = self._resolve_kick(actions)

        if not kicked:
            self._resolve_control(dt)

        if self.owner is not None:
            # the ball travels at the dribbler's feet. A dribbler shields it
            # inside the touchlines, but may walk it over the goal line -- into
            # the net if they find the mouth.
            raw = self.owner.pos + self.owner.heading * r.dribble_offset
            in_mouth = r.goal_y0 <= raw.y <= r.goal_y1
            x = raw.x if in_mouth else min(max(raw.x, 0.0), r.length)
            self.ball.pos = Vec2(x, min(max(raw.y, 0.0), r.width))
            self.ball.vel = self.owner.vel
            self.possession_ticks[self.owner.team] += 1
        elif self.phase in FROZEN_PHASES or (self.phase == "setpiece" and self.ball.vel.length_sq() < 1e-6):
            self.ball.vel = Vec2()
        else:
            decay = math.exp(-r.ball_friction * dt)
            self.ball.vel = self.ball.vel * decay
            if self.ball.vel.length() < r.ball_stop_speed:
                self.ball.vel = Vec2()
            self.ball.pos = self.ball.pos + self.ball.vel * dt

        if self.phase == "setpiece":
            self._enforce_clearance(dt)

    def _resolve_keeper(self, actions, dt: float) -> None:
        """Catching, holding and the six-second rule."""
        r = self.rules
        pitch = self._pitch_info()

        # A keeper in possession inside their own area is on the clock however
        # they got the ball -- caught, trapped or dribbled. Otherwise a bot that
        # simply never kicks would keep the ball for the rest of the match.
        if (
            self.keeper_hold <= 0.0
            and self.owner is not None
            and self.owner.is_keeper
            and self._in_own_penalty_area(self.owner, pitch)
        ):
            self.keeper_hold = r.keeper_max_hold

        if self.keeper_hold > 0.0:
            self.keeper_hold = max(0.0, self.keeper_hold - dt)
            if self.keeper_hold <= 0.0 and self.owner is not None and self.owner.is_keeper:
                # six seconds are up: the ball is released in front of the keeper
                gk = self.owner
                forward = Vec2(1.0, 0.0) if gk.team == HOME else Vec2(-1.0, 0.0)
                self.ball.pos = gk.pos + forward * (r.dribble_offset + 0.6)
                self.ball.vel = forward * 2.0
                self.owner = None
                self.loose_timer = r.loose_after_kick
                gk.catch_cd = r.keeper_catch_lockout
                self.events.append(Event(self.tick, self.clock, "six_seconds", gk.team))

        if self.owner is not None:
            return
        for p in self.players:
            if not p.is_keeper or not self._may_touch(p):
                continue
            if p.kick_cd > 0.0 or p.catch_cd > 0.0:
                continue  # cannot catch a ball you just kicked or just released
            action = actions[p.team].get(p.index)
            if action is None or not action.catch:
                continue
            if p.pos.dist(self.ball.pos) > r.keeper_catch_radius:
                continue
            if not self._in_own_penalty_area(p, pitch):
                continue
            incoming = (self.ball.vel - p.vel).length()
            if incoming > r.keeper_catch_max_speed:
                # too hot to hold: parried away, and still a save
                self._deflect(p, r.parry_restitution)
                self.events.append(Event(self.tick, self.clock, "save", p.team, "parry"))
                return
            self.owner = p
            self.last_touch = p
            self.keeper_hold = r.keeper_max_hold
            self.ball.vel = Vec2()
            if incoming > 6.0:
                self.events.append(Event(self.tick, self.clock, "save", p.team, "caught"))
            if self.phase == "setpiece":
                self.phase = "play"
                self.setpiece = None
            return

    def _in_own_penalty_area(self, p: Player, pitch: PitchInfo) -> bool:
        r = self.rules
        half = r.penalty_width / 2
        if abs(p.pos.y - r.width / 2) > half:
            return False
        return p.pos.x <= r.penalty_depth if p.team == HOME else p.pos.x >= r.length - r.penalty_depth

    def _resolve_kick(self, actions) -> bool:
        r = self.rules
        best: tuple[float, Player, object] | None = None
        for p in self.players:
            action = actions[p.team].get(p.index)
            if action is None or action.kick is None or action.kick_power <= 0.0:
                continue
            if p.kick_cd > 0.0 or not self._may_touch(p):
                continue
            d = p.pos.dist(self.ball.pos)
            reach = self._control_radius(p) if self.owner is p else r.kick_reach
            if self.owner is not None and self.owner is not p:
                continue  # cannot kick a ball someone else is shielding
            if d > reach:
                continue
            if best is None or d < best[0]:
                best = (d, p, action)
        if best is None:
            return False

        _, p, action = best
        direction = action.kick.normalized()
        if direction.length_sq() < 1e-9:
            direction = p.heading
        power = max(0.0, min(1.0, action.kick_power))
        speed = r.min_kick_speed + power * (r.max_kick_speed - r.min_kick_speed)

        # accuracy degrades with power and with the striker's own speed
        spread = r.kick_spread * power * (1.0 + p.vel.length() / r.sprint_speed)
        direction = direction.rotated(self.rng.gauss(0.0, spread))

        self.ball.vel = direction * speed + p.vel * r.kick_momentum
        self.ball.pos = p.pos + direction * (self._control_radius(p) * 0.6)
        self.owner = None
        self.keeper_hold = 0.0
        self.loose_timer = r.loose_after_kick
        p.kick_cd = r.kick_cooldown
        self.last_touch = p

        goal_x = r.length if p.team == HOME else 0.0
        if abs(self.ball.pos.x - goal_x) > 1.0 and direction.x * (1 if p.team == HOME else -1) > 0.6 and speed > 18.0:
            self.shots[p.team] += 1

        if self.phase == "setpiece" and p.team == self.setpiece_team:
            self.phase = "play"
            self.setpiece = None
        return True

    def _resolve_control(self, dt: float) -> None:
        r = self.rules
        if self.owner is None:
            if self.loose_timer > 0.0 or self.phase in FROZEN_PHASES:
                return
            claimant: Player | None = None
            best = 1e9
            for p in self.players:
                if not self._may_touch(p) or p.kick_cd > 0.0:
                    continue
                d = p.pos.dist(self.ball.pos)
                if d <= self._control_radius(p) and d < best:
                    best, claimant = d, p
            if claimant is None:
                return
            # A ball travelling fast relative to the player cannot be trapped;
            # it rebounds off them. This is what keeps shots and clearances
            # alive instead of being swallowed by the first body they pass.
            relative = (self.ball.vel - claimant.vel).length()
            if relative > r.trap_speed + (r.keeper_trap_bonus if claimant.is_keeper else 0.0):
                self._deflect(claimant, r.deflection_restitution)
                return
            self.owner = claimant
            self.last_touch = claimant
            if self.phase == "setpiece" and claimant.team == self.setpiece_team:
                self.phase = "play"
                self.setpiece = None
            return

        # someone has it: opponents may challenge
        owner = self.owner
        if owner.is_keeper and self.keeper_hold > 0.0:
            return  # a held ball cannot be tackled
        # Only the nearest opponent actually challenges. Letting every nearby
        # opponent roll its own tackle each tick makes swarming the ball
        # strictly dominant -- four players would win it four times as fast, so
        # holding possession against a crowd would be impossible.
        challenger = None
        best = r.tackle_radius
        for p in self.players:
            if p.team == owner.team or p.kick_cd > 0.0:
                continue
            d = p.pos.dist(owner.pos)
            if d < best:
                best, challenger = d, p
        if challenger is None:
            return
        # Running away makes the ball harder to win, scaled by how fast you are
        # actually moving away -- an exhausted dribbler jogging clear should not
        # shield as well as a fresh one sprinting clear.
        away = (owner.pos - challenger.pos).normalized()
        escape_speed = max(0.0, owner.vel.dot(away))
        shield = 1.0 - (1.0 - r.shield_factor) * min(1.0, escape_speed / r.run_speed)
        # a fresh challenger robs a spent carrier more easily, and vice versa
        freshness = 1.0 + r.tackle_stamina_swing * (challenger.stamina - owner.stamina)
        rate = r.tackle_rate * shield * freshness
        if self.rng.random() < rate * dt:
            owner.stumble = r.stumble_seconds
            self.owner = challenger
            self.last_touch = challenger
            challenger.kick_cd = 0.10

    def _deflect(self, p: Player, restitution: float) -> None:
        """Bounce the ball off a player who could not bring it under control."""
        r = self.rules
        normal = (self.ball.pos - p.pos).normalized()
        if normal.length_sq() < 1e-9:
            normal = -self.ball.vel.normalized() if self.ball.vel.length_sq() > 1e-9 else Vec2(1.0, 0.0)
        rel = self.ball.vel - p.vel
        if rel.dot(normal) < 0.0:  # only reflect if it is arriving at them
            rel = rel - normal * (2.0 * rel.dot(normal))
        rel = rel.rotated(self.rng.gauss(0.0, r.deflection_spread))
        self.ball.vel = rel * restitution + p.vel
        # rest the ball against the player's body, not at the edge of their
        # control zone -- teleporting it metres away turns crowds into pinball
        self.ball.pos = p.pos + normal * (r.player_radius + r.ball_radius + 0.02)
        self.last_touch = p
        self.loose_timer = max(self.loose_timer, r.loose_after_kick * 0.5)

    def _enforce_clearance(self, dt: float) -> None:
        """Opponents must retreat from a set piece; nudge them out."""
        r = self.rules
        self.setpiece_clock += dt
        if self.setpiece_clock > r.setpiece_timeout:
            self.phase = "play"
            self.setpiece = None
            return
        for p in self.players:
            if p.team == self.setpiece_team:
                continue
            delta = p.pos - self.ball.pos
            d = delta.length()
            if d < r.setpiece_clearance:
                if d < 1e-6:
                    delta = Vec2(0.0, 1.0)
                    d = 1e-6
                p.pos = self.ball.pos + delta.normalized() * r.setpiece_clearance
                p.pos = Vec2(min(max(p.pos.x, 0.0), r.length), min(max(p.pos.y, 0.0), r.width))

    # ------------------------------------------------------------------
    # laws of the game
    # ------------------------------------------------------------------
    def _check_boundaries(self) -> None:
        if self.phase in FROZEN_PHASES:
            return
        r = self.rules
        b = self.ball.pos
        in_mouth = r.goal_y0 <= b.y <= r.goal_y1

        # a goal counts however the ball got there, dribbled or struck
        if b.x >= r.length and in_mouth:
            self._score(HOME)
            return
        if b.x <= 0.0 and in_mouth:
            self._score(AWAY)
            return

        # a shielded ball is never out of play
        if self.owner is not None:
            return

        if b.y < 0.0 or b.y > r.width:
            # throw-in to whoever did not touch it last
            side = 0.0 if b.y < 0.0 else r.width
            spot = Vec2(min(max(b.x, 1.0), r.length - 1.0), side)
            self._award_setpiece("throw_in", self._other_team(), spot)
            return

        if b.x < 0.0 or b.x > r.length:
            attacking_right = b.x > r.length  # ball went out over the right goal line
            defending = AWAY if attacking_right else HOME
            toucher = self.last_touch.team if self.last_touch else defending
            goal_line = r.length if attacking_right else 0.0
            if toucher == defending:
                # defender put it out -> corner for the attackers
                corner_y = 0.0 if b.y < r.width / 2 else r.width
                self._award_setpiece("corner", 1 - defending, Vec2(goal_line, corner_y))
            else:
                spot_x = goal_line + (-r.goal_area_depth if attacking_right else r.goal_area_depth)
                self._award_setpiece("goal_kick", defending, Vec2(spot_x, r.width / 2))

    def _other_team(self) -> int:
        if self.last_touch is None:
            return HOME
        return 1 - self.last_touch.team

    def _score(self, team: int) -> None:
        self.score[team] += 1
        self.events.append(
            Event(self.tick, self.clock, "goal", team, self.controllers[team].name)
        )
        for t, ctrl in enumerate(self.controllers):
            ctrl.on_goal(t == team, self.state_for(t))
        self.phase = "goal"
        self.phase_timer = self.rules.goal_celebration
        self.setpiece = None
        self.ball.vel = Vec2()
        self.owner = None
        self._pending_kickoff = 1 - team

    def _advance_clock(self, dt: float) -> None:
        r = self.rules
        if self.phase in ("goal", "half_time", "full_time", "kickoff"):
            self.phase_timer -= dt
            if self.phase_timer <= 0.0:
                if self.phase == "goal":
                    self._setup_kickoff(getattr(self, "_pending_kickoff", HOME))
                elif self.phase == "half_time":
                    self.period += 1
                    self.clock = 0.0
                    self._setup_kickoff(AWAY if self.period % 2 == 0 else HOME)
                elif self.phase == "kickoff":
                    self.phase = "setpiece"
                    self.setpiece_clock = 0.0
            return

        self.clock += dt
        if self.clock >= r.half_seconds:
            if self.period >= r.periods:
                self.phase = "full_time"
                self.events.append(Event(self.tick, self.clock, "period", None, "full time"))
                for ctrl in self.controllers:
                    ctrl.on_match_end(self.state_for(HOME))
            else:
                self.phase = "half_time"
                self.phase_timer = self.rules.goal_celebration
                self.events.append(Event(self.tick, self.clock, "period", None, "half time"))

    # ------------------------------------------------------------------
    # state export
    # ------------------------------------------------------------------
    def state_for(self, team: int) -> GameState:
        r = self.rules
        mirror = team == AWAY

        def pt(v: Vec2) -> Vec2:
            return Vec2(r.length - v.x, r.width - v.y) if mirror else v

        def vec(v: Vec2) -> Vec2:
            return Vec2(-v.x, -v.y) if mirror else v

        def view(p: Player) -> PlayerView:
            return PlayerView(
                index=p.index,
                name=p.name,
                pos=pt(p.pos),
                vel=vec(p.vel),
                heading=vec(p.heading),
                stamina=p.stamina,
                is_keeper=p.is_keeper,
                has_ball=self.owner is p,
            )

        us = tuple(view(p) for p in self.players if p.team == team)
        them = tuple(view(p) for p in self.players if p.team != team)
        ball = BallView(
            pos=pt(self.ball.pos),
            vel=vec(self.ball.vel),
            owner_index=self.owner.index if self.owner else None,
            owned_by_us=self.owner is not None and self.owner.team == team,
            owned_by_them=self.owner is not None and self.owner.team != team,
            held_by_keeper=self.keeper_hold > 0.0 and self.owner is not None and self.owner.is_keeper,
        )
        return GameState(
            tick=self.tick,
            time=self.clock,
            period=self.period,
            time_left=max(0.0, r.half_seconds - self.clock),
            phase=self.phase,
            setpiece=self.setpiece,
            setpiece_is_ours=None if self.setpiece is None else (self.setpiece_team == team),
            our_score=self.score[team],
            their_score=self.score[1 - team],
            ball=ball,
            us=us,
            them=them,
            pitch=self._pitch_info(),
            limits=self._limits(),
        )

    # ------------------------------------------------------------------
    def summary(self) -> dict:
        total = max(1, sum(self.possession_ticks))
        return {
            "seed": self.seed,
            "home": self.controllers[HOME].name,
            "away": self.controllers[AWAY].name,
            "score": tuple(self.score),
            "shots": tuple(self.shots),
            "possession": (
                round(100.0 * self.possession_ticks[HOME] / total, 1),
                round(100.0 * self.possession_ticks[AWAY] / total, 1),
            ),
            "distance_km": (
                round(sum(p.distance_run for p in self.players if p.team == HOME) / 1000.0, 2),
                round(sum(p.distance_run for p in self.players if p.team == AWAY) / 1000.0, 2),
            ),
            "goals": [
                (round(e.time, 1), e.team, e.detail) for e in self.events if e.kind == "goal"
            ],
        }


def _mirror_action(a):
    """Translate an away-team action from its mirrored frame into world space."""
    if a is None:
        return None
    a.move = Vec2(-a.move.x, -a.move.y)
    if a.kick is not None:
        a.kick = Vec2(-a.kick.x, -a.kick.y)
    return a
