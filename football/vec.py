"""Minimal 2D vector used everywhere in the simulation and the bot API."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Vec2:
    x: float = 0.0
    y: float = 0.0

    # -- arithmetic ---------------------------------------------------
    def __add__(self, o: "Vec2") -> "Vec2":
        return Vec2(self.x + o.x, self.y + o.y)

    def __sub__(self, o: "Vec2") -> "Vec2":
        return Vec2(self.x - o.x, self.y - o.y)

    def __mul__(self, k: float) -> "Vec2":
        return Vec2(self.x * k, self.y * k)

    __rmul__ = __mul__

    def __truediv__(self, k: float) -> "Vec2":
        return Vec2(self.x / k, self.y / k)

    def __neg__(self) -> "Vec2":
        return Vec2(-self.x, -self.y)

    def __iter__(self):
        yield self.x
        yield self.y

    # -- geometry -----------------------------------------------------
    def length(self) -> float:
        return math.hypot(self.x, self.y)

    def length_sq(self) -> float:
        return self.x * self.x + self.y * self.y

    def dot(self, o: "Vec2") -> float:
        return self.x * o.x + self.y * o.y

    def dist(self, o: "Vec2") -> float:
        return math.hypot(self.x - o.x, self.y - o.y)

    def normalized(self) -> "Vec2":
        n = self.length()
        return Vec2(self.x / n, self.y / n) if n > 1e-9 else Vec2()

    def clamped(self, max_len: float) -> "Vec2":
        n = self.length()
        if n <= max_len or n < 1e-9:
            return self
        k = max_len / n
        return Vec2(self.x * k, self.y * k)

    def rotated(self, radians: float) -> "Vec2":
        c, s = math.cos(radians), math.sin(radians)
        return Vec2(self.x * c - self.y * s, self.x * s + self.y * c)

    def angle(self) -> float:
        return math.atan2(self.y, self.x)

    def lerp(self, o: "Vec2", t: float) -> "Vec2":
        return Vec2(self.x + (o.x - self.x) * t, self.y + (o.y - self.y) * t)

    def perp(self) -> "Vec2":
        return Vec2(-self.y, self.x)

    def as_tuple(self) -> tuple[float, float]:
        return (self.x, self.y)

    def __repr__(self) -> str:  # compact, bots print these a lot
        return f"({self.x:.2f}, {self.y:.2f})"


def from_angle(radians: float, length: float = 1.0) -> Vec2:
    return Vec2(math.cos(radians) * length, math.sin(radians) * length)
