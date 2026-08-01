"""fussball — a 5-a-side football simulator controlled by Python scripts."""

from .api import Action, BallView, GameState, MatchInfo, PitchInfo, PlayerView, Team, Vec2
from .config import Rules
from .engine import Match

__version__ = "1.0.0"

__all__ = [
    "Action",
    "BallView",
    "GameState",
    "Match",
    "MatchInfo",
    "PitchInfo",
    "PlayerView",
    "Rules",
    "Team",
    "Vec2",
]
