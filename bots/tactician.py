"""A role-based reference team.

The idea it is built around: the single most important decision each tick is
whether the ball can be *gathered* or must be *struck*. A ball closing faster
than `limits.trap_speed` rebounds off you, so trying to collect it just gives
possession straight back. Get that call right and everything else follows,
because a player who owns the ball keeps it at their feet and can carry it.
"""

from football.api import Action, Team, Vec2

# Home positions as (x, y) fractions of the pitch, for a team attacking +x.
ZONES = {
    1: (0.24, 0.30),  # left back
    2: (0.24, 0.70),  # right back
    3: (0.46, 0.50),  # midfield
    4: (0.66, 0.50),  # striker
}


class Tactician(Team):
    name = "Tacticians"
    player_names = ("Vega", "Ferro", "Marek", "Oyelaran", "Bruhn")
    # A crest is a character grid plus a palette, so the bot stays one file
    # with no image assets beside it. '.' is transparent.
    logo = (
        "...HHHH...",
        "..HWWWWH..",
        ".HWWHHWWH.",
        "HWWH..HWWH",
        "HWH....HWH",
        "HWWH..HWWH",
        ".HWWHHWWH.",
        "..HWWWWH..",
        "...HHHH...",
    )
    logo_colors = {"H": "#1f4fa8", "W": "#eef3ff"}

    def on_match_start(self, info):
        self.pitch = info.pitch
        self.limits = info.limits
        self.marks = {}

    # -- shape ---------------------------------------------------------
    def home_spot(self, state, index):
        """Zonal anchor, slid up and down the pitch with the ball."""
        p = state.pitch
        fx, fy = ZONES[index]
        shift = (state.ball.pos.x - p.length / 2) / p.length  # -0.5 .. 0.5
        x = (fx + shift * 0.45) * p.length
        y = fy * p.width + (state.ball.pos.y - p.width / 2) * 0.30
        return p.clamp(Vec2(x, y), margin=2.0)

    def assign_marks(self, state, busy):
        """Give each free defender a distinct opponent, nearest threat first."""
        marks = {}
        free = [p for p in state.us if not p.is_keeper and p.index not in busy]
        threats = sorted(
            (o for o in state.them if not o.is_keeper),
            key=lambda o: o.pos.dist(state.pitch.our_goal),
        )
        for threat in threats:
            if not free:
                break
            picker = min(free, key=lambda d: d.pos.dist(threat.pos))
            marks[picker.index] = threat
            free.remove(picker)
        return marks

    # -- judgement -----------------------------------------------------
    def shot_quality(self, state, me):
        p = state.pitch
        goal = p.their_goal
        dist = me.pos.dist(goal)
        if dist > 32.0:
            return 0.0
        if abs(me.pos.y - goal.y) > p.goal_width / 2 + 10.0 + (32.0 - dist) * 0.5:
            return 0.0
        clear = state.lane_is_clear(me.pos, goal, corridor=1.5)
        return max(0.0, 1.0 - dist / 32.0) * (1.0 if clear else 0.30)

    def best_pass(self, state, me):
        """The most useful open team-mate, or None."""
        best, best_score = None, 0.6
        for mate in state.us:
            if mate.index == me.index or mate.is_keeper:
                continue
            d = me.pos.dist(mate.pos)
            if not 6.0 <= d <= 32.0:
                continue
            target = mate.pos + mate.vel * 0.35
            if not state.lane_is_clear(me.pos, target, corridor=1.8):
                continue
            progress = (mate.pos.x - me.pos.x) / 30.0
            space = 1.0 - min(1.0, state.pressure_on(mate.pos, 7.0))
            score = 0.5 + progress + space * 0.5
            if score > best_score:
                best, best_score = mate, score
        return best

    def send_pass(self, state, me, mate):
        """Weight the pass so it arrives slow enough to actually be trapped."""
        target = mate.pos + mate.vel * 0.35
        power = state.limits.power_for(me.pos.dist(target), arrive_speed=5.0)
        return Action.kick_to(me, target, power)

    # -- roles ---------------------------------------------------------
    def keeper(self, state, me):
        p, ball = state.pitch, state.ball

        if me.has_ball:
            mate = self.best_pass(state, me)
            if mate is not None:
                return self.send_pass(state, me, mate)
            # no one on: kick downfield, over anyone standing in the way
            downfield = Vec2(p.length * 0.66, p.width / 2)
            lift = 0.0 if state.lane_is_clear(me.pos, downfield, corridor=2.0) else 0.45
            return Action.kick_to(me, downfield, power=0.85, lift=lift)

        # come only for a ball we can actually get to first
        mine = me.pos.dist(ball.pos)
        if p.in_our_penalty_area(ball.pos) and not ball.owned_by_us:
            if mine < 6.0 or mine < state.nearest_opponent_to(ball.pos).pos.dist(ball.pos):
                a = Action.go_to(me, ball.pos, sprint=True, arrive=0.0)
                a.catch = True
                return a

        # otherwise guard the angle between the ball and the middle of the goal
        aim = ball.pos if ball.speed < 1.0 else ball.predict(0.35)
        to_ball = aim - p.our_goal
        depth = min(5.5, 1.4 + to_ball.length() * 0.05)
        spot = p.our_goal + to_ball.normalized() * depth
        spot = Vec2(
            max(0.8, spot.x),
            min(max(spot.y, p.our_post_left.y - 1.5), p.our_post_right.y + 1.5),
        )
        a = Action.go_to(me, spot, sprint=aim.x < p.length * 0.3)
        a.catch = True
        return a

    def on_ball(self, state, me):
        p = state.pitch

        if self.shot_quality(state, me) > 0.40:
            gk = state.their_keeper
            far_post = p.their_post_left if gk.pos.y > p.width / 2 else p.their_post_right
            aim = Vec2(far_post.x, far_post.y + (1.4 if gk.pos.y > p.width / 2 else -1.4))
            return Action.kick_to(me, aim, power=1.0)

        pressure = state.pressure_on(me.pos, 5.0)
        mate = self.best_pass(state, me)
        # Pass only when it is genuinely on. Winning the ball in a crowd and
        # immediately passing into that crowd is how possession gets thrown
        # straight back -- the receiver must actually have space.
        if (
            mate is not None
            and state.pressure_on(mate.pos, 6.5) < 0.45
            and (pressure > 0.9 or mate.pos.x > me.pos.x + 10.0)
        ):
            return self.send_pass(state, me, mate)

        if pressure > 1.6 and me.pos.x < p.length * 0.30:
            escape = Vec2(p.length * 0.82, p.width / 2)
            lift = 0.0 if state.lane_is_clear(me.pos, escape, corridor=2.0) else 0.45
            return Action.kick_to(me, escape, power=1.0, lift=lift)

        # Carry it into space. The ball follows its owner and a carrier moving
        # away from a challenge is much harder to rob, so sprint -- but go
        # *around* the nearest opponent, not directly away from them. Running
        # straight back cancels the goalward run, and when two teams line up
        # symmetrically that stalls into a permanent midfield scrum.
        return Action.move_dir(self.carry_direction(state, me), sprint=True)

    def carry_direction(self, state, me):
        p = state.pitch
        goalward = (p.their_goal - me.pos).normalized()
        opp = state.nearest_opponent_to(me.pos)
        if me.pos.dist(opp.pos) < 6.5:
            side = (opp.pos - me.pos).normalized().perp()
            if state.pressure_on(me.pos - side * 5.0, 7.0) < state.pressure_on(me.pos + side * 5.0, 7.0):
                side = -side
            # Keep the forward bias: any perpendicular component at all is
            # enough to break a symmetric stall, and weighting sideways too
            # heavily just surrenders territory.
            goalward = (goalward + side * 0.85).normalized()
        # stay off the touchlines, where there is nowhere left to run
        if me.pos.y < p.width * 0.16:
            goalward = (goalward + Vec2(0.0, 0.7)).normalized()
        elif me.pos.y > p.width * 0.84:
            goalward = (goalward + Vec2(0.0, -0.7)).normalized()
        return goalward

    def ball_winner(self, state, me):
        """Gather if the ball is collectable, otherwise strike it."""
        p, ball = state.pitch, state.ball

        # For a ball in the air, run to where it will come down rather than to
        # where it is now -- otherwise you stand under it and watch it sail on.
        if ball.airborne:
            aim = state.predict_ball(0.45)
        elif ball.speed > 2.0:
            aim = state.predict_ball(0.22)
        else:
            aim = ball.pos

        # can_trap now also fails when the ball is above foot height
        if state.can_trap(me):
            return Action.go_to(me, aim, sprint=True, arrive=0.0)

        # Too fast or too high to control: the only option is to hit it.
        if me.pos.x > p.length * 0.62 and self.shot_quality(state, me) > 0.25:
            target, lift = p.their_goal, 0.0  # keep shots down
        elif me.pos.x < p.length * 0.4:
            # Clearing our own third. Loft is for going *over* people, so only
            # pay its cost -- an airborne ball nobody can trap -- when someone
            # is actually in the way.
            target = Vec2(p.length * 0.8, p.width / 2)
            lift = 0.0 if state.lane_is_clear(me.pos, target, corridor=2.0) else 0.45
        else:
            mate = self.best_pass(state, me)
            target = (mate.pos + mate.vel * 0.3) if mate else p.their_goal
            lift = 0.0
        return Action.intercept(me, aim, target, power=0.85, lift=lift)

    def off_ball(self, state, me, presser, support):
        p, ball = state.pitch, state.ball

        if me.index == presser:
            return self.ball_winner(state, me)

        if me.index in support:
            # stack the supporting runners at different depths so they do not
            # pile onto the same square metre behind the presser
            depth = 3.5 + 4.5 * support.index(me.index)
            goal_side = (p.our_goal - ball.pos).normalized() * depth
            return Action.go_to(me, p.clamp(ball.pos + goal_side, 1.5), sprint=True, arrive=1.5)

        if ball.owned_by_us:
            spot = self.home_spot(state, me.index)
            return Action.go_to(me, p.clamp(Vec2(spot.x + 5.0, spot.y), 2.0),
                                sprint=ball.pos.x > p.length * 0.55)

        mark = self.marks.get(me.index)
        if mark is not None:
            goal_side = (p.our_goal - mark.pos).normalized() * 2.4
            return Action.go_to(me, p.clamp(mark.pos + goal_side, 1.0), sprint=True, arrive=2.0)
        return Action.go_to(me, self.home_spot(state, me.index))

    # -- entry point ---------------------------------------------------
    def act(self, state):
        ball = state.ball
        outfield = sorted((q for q in state.us if not q.is_keeper),
                          key=lambda q: q.pos.dist(ball.pos))
        presser = outfield[0].index
        # A two-man press, no more. Committing a third man to the ball was
        # measurably worse: it wins the ball no more often and leaves too few
        # players upfield to do anything with it.
        support = [q.index for q in outfield[1:2]] if not ball.owned_by_us else []
        carrier = next((q.index for q in state.us if q.has_ball), -1)
        self.marks = self.assign_marks(state, busy={presser, carrier, *support})

        actions = {}
        for me in state.us:
            if me.is_keeper:
                actions[me.index] = self.keeper(state, me)
            elif me.has_ball:
                actions[me.index] = self.on_ball(state, me)
            else:
                actions[me.index] = self.off_ball(state, me, presser, support)

        if state.setpiece and state.setpiece_is_ours:
            taker = min((q for q in state.us if not q.is_keeper),
                        key=lambda q: q.pos.dist(ball.pos))
            mate = self.best_pass(state, taker)
            if mate is not None:
                aim, power = mate.pos, state.limits.power_for(taker.pos.dist(mate.pos), 5.0)
            else:
                aim, power = state.pitch.their_goal, 0.9
            actions[taker.index] = Action.intercept(taker, ball.pos, aim, power=power)
        return actions
