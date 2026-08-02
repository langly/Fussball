"""The simplest thing that plays football: everyone runs at the ball.

Useful as a punching bag, and as a minimal example of the API.
"""

from football.api import Action, Team, Vec2


class Chaser(Team):
    name = "Chasers"

    def act(self, state):
        actions = {}
        for me in state.us:
            if me.is_keeper:
                # stay near the goal line, level with the ball
                goal = state.pitch.our_goal
                y = min(max(state.ball.pos.y, goal.y - 4.0), goal.y + 4.0)
                a = Action.go_to(me, Vec2(2.0, y), sprint=True)
                a.catch = True
                actions[me.index] = a
            else:
                # run at the ball and hammer it goalwards whenever it is reachable
                ball = state.ball
                aim = state.predict_ball(0.4 if ball.airborne else 0.2) if ball.loose else ball.pos
                actions[me.index] = Action.intercept(
                    me, aim, state.pitch.their_goal, power=1.0
                )
        return actions
