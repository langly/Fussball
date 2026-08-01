"""Defend deep, win the ball, hit the striker on the counter.

A contrasting style to `tactician.py`, and a demonstration of `act` being a
plain function instead of a Team subclass.
"""

from football.api import Action, Vec2

TEAM_NAME = "Parked Bus"


def carry_direction(state, me):
    """Run at goal, but around the nearest opponent rather than away from them."""
    p = state.pitch
    goalward = (p.their_goal - me.pos).normalized()
    opp = state.nearest_opponent_to(me.pos)
    if me.pos.dist(opp.pos) < 6.5:
        side = (opp.pos - me.pos).normalized().perp()
        if state.pressure_on(me.pos - side * 5.0, 7.0) < state.pressure_on(me.pos + side * 5.0, 7.0):
            side = -side
        goalward = (goalward * 0.55 + side).normalized()
    if me.pos.y < p.width * 0.16:
        goalward = (goalward + Vec2(0.0, 0.7)).normalized()
    elif me.pos.y > p.width * 0.84:
        goalward = (goalward + Vec2(0.0, -0.7)).normalized()
    return goalward


def act(state):
    p = state.pitch
    ball = state.ball
    actions = {}

    keeper = state.us[0]
    if p.in_our_penalty_area(ball.pos) and not ball.owned_by_us:
        a = Action.go_to(keeper, ball.pos, sprint=True)
        a.catch = True
        actions[0] = a
    elif keeper.has_ball:
        # long ball to the striker
        actions[0] = Action.kick_to(keeper, state.us[4].pos + state.us[4].vel * 0.6, power=0.95)
    else:
        y = min(max(ball.pos.y, p.our_post_left.y - 1.0), p.our_post_right.y + 1.0)
        a = Action.go_to(keeper, Vec2(1.8, y), sprint=True)
        a.catch = True
        actions[0] = a

    # the two backs sit in a flat line 14 m out and squeeze towards the ball
    for idx, side in ((1, -1), (2, 1)):
        me = state.us[idx]
        y = p.width / 2 + side * 7.5 + (ball.pos.y - p.width / 2) * 0.45
        spot = p.clamp(Vec2(13.0, y), margin=2.0)
        if me.has_ball:
            actions[idx] = Action.kick_to(me, Vec2(p.length * 0.8, p.width / 2), power=1.0)
        elif ball.pos.x < 26.0 and me.pos.dist(ball.pos) < 11.0:
            actions[idx] = Action.intercept(me, ball.pos, Vec2(p.length * 0.85, p.width / 2), power=1.0)
        else:
            actions[idx] = Action.go_to(me, spot, sprint=ball.pos.x < 34.0)

    # the midfielder is the designated presser
    mid = state.us[3]
    striker = state.us[4]
    if mid.has_ball:
        if state.lane_is_clear(mid.pos, striker.pos, 2.0):
            target = striker.pos + striker.vel * 0.5
            power = state.limits.power_for(mid.pos.dist(target), arrive_speed=5.0)
            actions[3] = Action.kick_to(mid, target, power)
        else:
            actions[3] = Action.move_dir((p.their_goal - mid.pos).normalized(), sprint=True)
    else:
        chase = ball.predict(0.3) if ball.loose else ball.pos
        if state.can_trap(mid) and mid.pos.dist(ball.pos) < 14.0:
            actions[3] = Action.go_to(mid, chase, sprint=True, arrive=0.0)
        else:
            actions[3] = Action.intercept(mid, chase, Vec2(p.length * 0.9, p.width / 2), power=0.9)

    # the striker leads the counter
    fw = state.us[4]
    if fw.has_ball:
        if fw.pos.dist(p.their_goal) < 26.0:
            actions[4] = Action.kick_to(fw, p.their_goal, power=1.0)
        else:
            # carry it: the ball follows its owner, and shielding at a sprint is
            # what stops it being taken back. Go around the nearest opponent --
            # running straight away from them just cancels the forward run.
            actions[4] = Action.move_dir(carry_direction(state, fw), sprint=True)
    elif ball.owned_by_us or (ball.loose and ball.pos.x > p.length * 0.42):
        aim = ball.predict(0.25) if ball.loose else ball.pos
        if state.can_trap(fw) and fw.pos.dist(ball.pos) < 16.0:
            actions[4] = Action.go_to(fw, aim, sprint=True, arrive=0.0)
        else:
            actions[4] = Action.intercept(fw, aim, p.their_goal, power=1.0)
    else:
        # Sit higher than the halfway line. Holding exactly on it against
        # another deep block just produced a permanent midfield stalemate.
        hold = p.clamp(Vec2(p.length * 0.60, p.width / 2 + (ball.pos.y - p.width / 2) * 0.3), 2.0)
        actions[4] = Action.go_to(fw, hold)

    return actions
