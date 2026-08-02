"""pygame-ce viewer for a match.

Controls
    space      pause / resume
    + / -      simulation speed
    tab        debug overlay (velocities, control radii, bot timings)
    r          restart the match with the same seed
    s          step one tick while paused
    esc / q    quit
"""

from __future__ import annotations

import math

import pygame

from .config import ROLE_NAMES, Rules
from .engine import AWAY, HOME, Match
from .vec import Vec2

HUD_H = 84
HINT_H = 24  # strip along the bottom for the key hints
MARGIN = 20
PITCH_PAD = 3.4  # metres of surround drawn outside the touchlines, for the goals

# How long the name plate stays up after its player last touched the ball,
# and how much of that tail is spent fading out (both in simulated seconds).
PLATE_LINGER = 1.6
PLATE_FADE = 0.5

# Players are drawn at this multiple and scaled back down, which anti-aliases
# the edges -- pygame's own primitives are hard-aliased and look crude at
# ~20 px per player.
SUPERSAMPLE = 4

GRASS_A = (34, 122, 62)
GRASS_B = (39, 133, 69)
LINE = (232, 240, 232)
HUD_BG = (18, 22, 28)
HUD_FG = (236, 240, 245)
HUD_DIM = (140, 152, 166)
BALL_C = (250, 250, 250)

TEAM_COLORS = (
    {
        "body": (48, 108, 224),
        "keeper": (250, 206, 74),
        "shorts": (236, 240, 248),
        "keeper_shorts": (54, 58, 68),
        "text": (255, 255, 255),
    },
    {
        "body": (216, 68, 56),
        "keeper": (74, 214, 192),
        "shorts": (38, 42, 50),
        "keeper_shorts": (236, 240, 248),
        "text": (255, 255, 255),
    },
)

# A little variety so the eleven dots read as people rather than counters.
SKIN_TONES = (
    (245, 200, 172), (226, 172, 138), (198, 134, 96), (146, 92, 58), (94, 58, 38),
)
HAIR_TONES = (
    (28, 24, 22), (58, 40, 28), (104, 68, 34), (168, 130, 72), (18, 18, 20),
)
SOCK_DARK = (28, 30, 36)


class NamePlateTracker:
    """Decides whose name to caption, and how opaque the plate should be.

    Possession alone is far too fleeting to read: an outfielder holds the ball
    for well under a second, while a keeper can sit on it for the full six. So
    following `match.owner` alone captions almost nothing but the goalkeeper.
    This follows the last player to *touch* the ball and lingers briefly after
    they lose it, which keeps a shot or a pass captioned with whoever struck it.

    Shared by both renderers so the fix cannot drift back out of one of them.
    """

    def __init__(self, linger: float = PLATE_LINGER, fade: float = PLATE_FADE) -> None:
        self.linger = linger
        self.fade = fade
        self.player = None
        self.since = 0

    def reset(self) -> None:
        self.player = None
        self.since = 0

    def update(self, match):
        """Return (player_or_None, alpha) for the current tick."""
        if match.owner is not None:
            self.player = match.owner
            self.since = match.tick
            return match.owner, 1.0

        toucher = match.last_touch
        if toucher is not self.player:
            self.player = toucher
            self.since = match.tick
        if self.player is None:
            return None, 0.0

        elapsed = (match.tick - self.since) * match.rules.dt
        if elapsed >= self.linger:
            return None, 0.0
        return self.player, min(1.0, (self.linger - elapsed) / self.fade)


def render_pitch_surface(rules, scale: float, pad_m: float = 0.0,
                         draw_goals: bool = True) -> pygame.Surface:
    """Draw the pitch and all its markings to a surface, `scale` px per metre.

    Shared by both renderers: the 3D view uploads this as the ground texture
    rather than re-implementing the line work in geometry. It takes `rules`
    instead of a Viewer so it can be called without a window, and skips the
    flat goal boxes when the caller draws real 3D frames instead.
    """
    r = rules
    pad = int(pad_m * scale)
    w, h = int(r.length * scale), int(r.width * scale)
    # padded so the goals, which sit behind the goal lines, are not clipped
    surf = pygame.Surface((w + pad * 2, h + pad * 2), pygame.SRCALPHA)

    def local(v: Vec2) -> tuple[int, int]:
        return (int((v.x + pad_m) * scale), int((v.y + pad_m) * scale))

    def metres(x: float) -> int:
        return max(1, int(x * scale))

    stripes = 12
    for i in range(stripes):
        x0 = pad + int(i * w / stripes)
        x1 = pad + int((i + 1) * w / stripes)
        surf.fill(GRASS_A if i % 2 == 0 else GRASS_B, pygame.Rect(x0, pad, x1 - x0, h))

    lw = max(1, int(0.12 * scale))

    def line(a: Vec2, b: Vec2) -> None:
        pygame.draw.line(surf, LINE, local(a), local(b), lw)

    def rect(x0: float, y0: float, x1: float, y1: float) -> None:
        pygame.draw.rect(
            surf, LINE,
            pygame.Rect(local(Vec2(x0, y0)),
                        (int((x1 - x0) * scale), int((y1 - y0) * scale))),
            lw,
        )

    rect(0, 0, r.length, r.width)
    line(Vec2(r.length / 2, 0), Vec2(r.length / 2, r.width))
    centre = local(Vec2(r.length / 2, r.width / 2))
    pygame.draw.circle(surf, LINE, centre, metres(r.center_circle_r), lw)
    pygame.draw.circle(surf, LINE, centre, max(2, lw + 1))

    cy = r.width / 2
    for side in (0, 1):
        sign = 1 if side == 0 else -1
        base = 0.0 if side == 0 else r.length
        rect(min(base, base + sign * r.penalty_depth), cy - r.penalty_width / 2,
             max(base, base + sign * r.penalty_depth), cy + r.penalty_width / 2)
        rect(min(base, base + sign * r.goal_area_depth), cy - r.goal_area_width / 2,
             max(base, base + sign * r.goal_area_depth), cy + r.goal_area_width / 2)
        spot = Vec2(base + sign * 11.0, cy)
        pygame.draw.circle(surf, LINE, local(spot), max(2, lw))
        arc_r = metres(9.15)
        box = pygame.Rect(0, 0, arc_r * 2, arc_r * 2)
        box.center = local(spot)
        a0 = -math.pi / 3 if side == 0 else math.pi * 2 / 3
        pygame.draw.arc(surf, LINE, box, a0, a0 + math.pi * 2 / 3, lw)

        if draw_goals:
            goal_depth = 2.2
            gx0 = -goal_depth if side == 0 else r.length
            goal_rect = pygame.Rect(local(Vec2(gx0, r.goal_y0)),
                                    (metres(goal_depth), metres(r.goal_width)))
            pygame.draw.rect(surf, (228, 232, 238, 70), goal_rect)  # netting
            step = max(3, metres(0.6))
            for gx in range(goal_rect.left, goal_rect.right, step):
                pygame.draw.line(surf, (250, 250, 252, 130), (gx, goal_rect.top), (gx, goal_rect.bottom))
            for gy in range(goal_rect.top, goal_rect.bottom, step):
                pygame.draw.line(surf, (250, 250, 252, 130), (goal_rect.left, gy), (goal_rect.right, gy))
            pygame.draw.rect(surf, (252, 252, 254), goal_rect, max(2, lw))
            post = max(2, int(lw * 1.6))
            for py in (r.goal_y0, r.goal_y1):
                pygame.draw.circle(surf, (252, 252, 254),
                                   local(Vec2(0.0 if side == 0 else r.length, py)), post)

    for cx, cyy, a0 in ((0, 0, 0.0), (r.length, 0, math.pi / 2),
                        (0, r.width, -math.pi / 2), (r.length, r.width, math.pi)):
        rr = metres(1.0)
        box = pygame.Rect(0, 0, rr * 2, rr * 2)
        box.center = local(Vec2(cx, cyy))
        pygame.draw.arc(surf, LINE, box, a0, a0 + math.pi / 2, lw)
    return surf


class Viewer:
    def __init__(self, match_factory, window=(1400, 900), speed: float = 1.0) -> None:
        pygame.init()
        pygame.display.set_caption("fussball — 5-a-side bot league")
        self.screen = pygame.display.set_mode(window, pygame.RESIZABLE)
        self.clock = pygame.time.Clock()
        self.match_factory = match_factory
        self.match: Match = match_factory()
        self.speed = speed
        self.paused = False
        self.debug = False
        self.single_step = False
        self.font = pygame.font.SysFont("menlo,dejavusansmono,monospace", 15)
        self.font_small = pygame.font.SysFont("menlo,dejavusansmono,monospace", 12)
        self.font_big = pygame.font.SysFont("helvetica,arial,sans-serif", 30, bold=True)
        self.font_mid = pygame.font.SysFont("helvetica,arial,sans-serif", 17, bold=True)
        self.font_name = pygame.font.SysFont("helvetica,arial,sans-serif", 14, bold=True)
        self._pitch_cache = None
        self._plate = NamePlateTracker()
        self._sprite_cache: dict = {}
        self._layout()

    # -- geometry ------------------------------------------------------
    def _layout(self) -> None:
        r = self.match.rules
        w, h = self.screen.get_size()
        # leave room for the goals and a strip of surround outside the lines
        avail_w = max(50, w - 2 * MARGIN)
        avail_h = max(50, h - HUD_H - HINT_H - 2 * MARGIN)
        self.scale = min(
            avail_w / (r.length + 2 * PITCH_PAD), avail_h / (r.width + 2 * PITCH_PAD)
        )
        pw, ph = r.length * self.scale, r.width * self.scale
        self.ox = (w - pw) / 2
        self.oy = HUD_H + (h - HUD_H - HINT_H - ph) / 2
        self._pitch_cache = None

    def to_px(self, v: Vec2) -> tuple[int, int]:
        return (int(self.ox + v.x * self.scale), int(self.oy + v.y * self.scale))

    def _local(self, v: Vec2) -> tuple[int, int]:
        """World metres -> pitch-surface pixels, including the surround padding."""
        return (int((v.x + PITCH_PAD) * self.scale), int((v.y + PITCH_PAD) * self.scale))

    def m(self, metres: float) -> int:
        return max(1, int(metres * self.scale))

    # -- main loop -----------------------------------------------------
    def run(self) -> Match:
        r = self.match.rules
        accumulator = 0.0
        running = True
        while running:
            frame_dt = self.clock.tick(60) / 1000.0
            running = self._handle_events()

            if self.single_step:
                self.match.step()
                self.single_step = False
            elif not self.paused and not self.match.finished:
                accumulator += frame_dt * self.speed
                # cap the catch-up so a slow bot cannot spiral the frame rate
                steps = 0
                while accumulator >= r.dt and steps < 400:
                    self.match.step()
                    accumulator -= r.dt
                    steps += 1
                    if self.match.finished:
                        accumulator = 0.0
                        break

            self._draw()
            pygame.display.flip()
        pygame.quit()
        return self.match

    def _handle_events(self) -> bool:
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                return False
            if e.type == pygame.VIDEORESIZE:
                self.screen = pygame.display.set_mode((e.w, e.h), pygame.RESIZABLE)
                self._layout()
            if e.type == pygame.KEYDOWN:
                if e.key in (pygame.K_ESCAPE, pygame.K_q):
                    return False
                if e.key == pygame.K_SPACE:
                    self.paused = not self.paused
                elif e.key == pygame.K_TAB:
                    self.debug = not self.debug
                elif e.key == pygame.K_r:
                    self.match = self.match_factory()
                    self._plate.reset()  # drop the stale reference into the old match
                    self._layout()
                elif e.key == pygame.K_s:
                    self.single_step = True
                    self.paused = True
                elif e.key in (pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS):
                    self.speed = min(16.0, self.speed * 2.0)
                elif e.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
                    self.speed = max(0.125, self.speed / 2.0)
        return True

    # -- drawing -------------------------------------------------------
    def _draw(self) -> None:
        self.screen.fill(HUD_BG)
        self._draw_pitch()
        m = self.match
        if self.debug:
            self._draw_debug_field()
        for p in m.players:
            self._draw_player(p)
        self._draw_ball()
        self._draw_owner_name()
        self._draw_hud()
        if self.paused:
            self._banner("PAUSED", (255, 255, 255))
        elif m.phase == "goal":
            team = m.events[-1].team if m.events else HOME
            self._banner("GOAL!", TEAM_COLORS[team]["body"])
        elif m.phase == "half_time":
            self._banner("HALF TIME", (255, 255, 255))
        elif m.phase == "full_time":
            self._banner("FULL TIME", (255, 255, 255))

    def _draw_pitch(self) -> None:
        if self._pitch_cache is None:
            self._pitch_cache = self._render_pitch()
        pad = int(PITCH_PAD * self.scale)
        self.screen.blit(self._pitch_cache, (int(self.ox) - pad, int(self.oy) - pad))

    def _render_pitch(self) -> pygame.Surface:
        return render_pitch_surface(self.match.rules, self.scale, PITCH_PAD)

    def _appearance(self, p):
        """Stable per-player skin and hair, so nobody changes mid-match."""
        h = p.team * 7 + p.index * 13
        return SKIN_TONES[h % len(SKIN_TONES)], HAIR_TONES[(h // 3) % len(HAIR_TONES)]

    def _player_sprite(self, p, rad: int) -> pygame.Surface:
        """A top-down player facing +x, drawn oversized for later downscaling.

        Everything here is rendered at SUPERSAMPLE times the final size; the
        caller rotates and then smoothscales down, which is what gives the
        sprite clean edges instead of pygame's hard-aliased primitives.
        """
        # Legs swing on a stride cycle driven by distance covered, so the
        # animation stays in step with the player's actual speed.
        speed = p.vel.length()
        amp = min(1.0, speed / self.match.rules.run_speed)
        phase = math.sin(p.distance_run * (2.0 * math.pi / 1.7)) * amp
        key = (p.team, p.is_keeper, p.index, rad, round(phase, 2))
        cached = self._sprite_cache.get(key)
        if cached is not None:
            return cached

        rad = rad * SUPERSAMPLE
        size = rad * 4
        surf = pygame.Surface((size, size), pygame.SRCALPHA)
        cx = cy = size / 2.0
        colors = TEAM_COLORS[p.team]
        jersey = colors["keeper"] if p.is_keeper else colors["body"]
        shorts = colors["keeper_shorts"] if p.is_keeper else colors["shorts"]
        skin, hair = self._appearance(p)

        # Sprite faces +x, so x is front-to-back depth and y is shoulder width.
        # Seen from directly above, shoulders are the widest thing and the head
        # is small -- getting that ratio wrong makes it read as a blob.
        def ellipse(color, x, y, depth, width):
            pygame.draw.ellipse(
                surf, color, pygame.Rect(x - depth / 2, y - width / 2, depth, width)
            )

        # Legs trail behind the hip and stride along the facing axis.
        for side in (-1, 1):
            swing = phase * side * rad * 0.38
            ellipse(shorts, cx - rad * 0.46 + swing, cy + side * rad * 0.34,
                    rad * 0.84, rad * 0.42)
            ellipse(SOCK_DARK, cx - rad * 0.80 + swing * 1.15, cy + side * rad * 0.34,
                    rad * 0.40, rad * 0.36)
        # arms tucked close, swinging opposite the legs
        for side in (-1, 1):
            swing = -phase * side * rad * 0.34
            ellipse(skin, cx + rad * 0.04 + swing, cy + side * rad * 0.70,
                    rad * 0.52, rad * 0.24)
        # torso: widest across the shoulders, with a dark edge to lift it
        ellipse((16, 18, 22), cx + rad * 0.08, cy, rad * 1.20, rad * 1.62)
        ellipse(jersey, cx + rad * 0.08, cy, rad * 1.06, rad * 1.46)
        # head: small from directly above, face just clearing the shoulders
        ellipse((16, 18, 22), cx + rad * 0.30, cy, rad * 0.68, rad * 0.66)
        ellipse(hair, cx + rad * 0.30, cy, rad * 0.58, rad * 0.56)
        ellipse(skin, cx + rad * 0.50, cy, rad * 0.26, rad * 0.42)

        self._sprite_cache[key] = surf
        if len(self._sprite_cache) > 4096:
            self._sprite_cache.clear()
        return surf

    def _draw_player(self, p) -> None:
        colors = TEAM_COLORS[p.team]
        center = self.to_px(p.pos)
        # Larger than the 0.45 m collision radius, because a true-to-scale
        # player is a ~5 px smudge at this zoom -- but not so large that the
        # keeper dwarfs a true-scale goal. This lands a keeper at ~14% of the
        # goal width against ~7% in real life; the floor is readability, since
        # below about 7 px the sprite turns to mush.
        rad = max(7, self.m(0.66))

        # contact shadow on the grass
        shadow = pygame.Surface((rad * 4, rad * 4), pygame.SRCALPHA)
        pygame.draw.ellipse(
            shadow, (0, 0, 0, 60),
            pygame.Rect(rad * 0.9, rad * 1.25, rad * 2.4, rad * 1.6),
        )
        self.screen.blit(shadow, (center[0] - rad * 2 + 2, center[1] - rad * 2 + 2))

        if self.match.owner is p:
            ring = pygame.Surface((rad * 5, rad * 5), pygame.SRCALPHA)
            pygame.draw.ellipse(
                ring, (255, 246, 140, 210),
                pygame.Rect(rad * 0.55, rad * 1.35, rad * 3.4, rad * 2.3), 2,
            )
            self.screen.blit(ring, (center[0] - rad * 2.5, center[1] - rad * 2.5))

        sprite = self._player_sprite(p, rad)
        angle = math.degrees(math.atan2(p.heading.y, p.heading.x))
        rot = pygame.transform.rotate(sprite, -angle)
        rot = pygame.transform.smoothscale(
            rot, (max(1, rot.get_width() // SUPERSAMPLE), max(1, rot.get_height() // SUPERSAMPLE))
        )
        self.screen.blit(rot, rot.get_rect(center=center))

        # Shirt numbers only in the debug overlay -- drawn over every player they
        # sit right on the head and undo the sprite entirely.
        if self.debug:
            label = self.font_small.render(str(p.index), True, colors["text"])
            box = label.get_rect(center=(center[0], center[1] - rad - 7))
            pad = box.inflate(5, 2)
            back = pygame.Surface(pad.size, pygame.SRCALPHA)
            back.fill((0, 0, 0, 150))
            self.screen.blit(back, pad.topleft)
            self.screen.blit(label, box)

        # Fatigue ring, debug only. On the plain view a bare arc under every
        # tiring player just reads as red litter round their feet.
        if self.debug and p.stamina < 0.995:
            box = pygame.Rect(0, 0, (rad + 4) * 2, (rad + 4) * 2)
            box.center = center
            frac = max(0.0, min(1.0, p.stamina))
            col = (120, 210, 130) if frac > 0.5 else (232, 196, 96) if frac > 0.25 else (226, 96, 84)
            pygame.draw.circle(self.screen, (0, 0, 0, 90), center, rad + 4, 2)
            pygame.draw.arc(self.screen, col, box, -math.pi / 2, -math.pi / 2 + frac * 2 * math.pi, 2)

    def _draw_ball(self) -> None:
        b = self.match.ball
        ground = self.to_px(b.pos)
        # Lift the ball up the screen by its height and leave the shadow on the
        # turf, so the top-down view still reads aerial play.
        lift = int(b.z * self.scale * 0.8)
        center = (ground[0], ground[1] - lift)
        rad = max(5, self.m(0.52))  # exaggerated: a true-scale ball is ~2 px

        # a struck ball smears along its direction of travel
        speed = b.vel.length()
        if speed > 6.0:
            trail = pygame.Surface(self.screen.get_size(), pygame.SRCALPHA)
            back = b.pos - b.vel.normalized() * min(2.2, speed * 0.06)
            pygame.draw.line(trail, (255, 255, 255, 70), self.to_px(back), center, max(2, rad))
            self.screen.blit(trail, (0, 0))

        shrink = max(0.45, 1.0 - b.z / 14.0)
        shadow = pygame.Surface((rad * 4, rad * 4), pygame.SRCALPHA)
        pygame.draw.ellipse(
            shadow, (0, 0, 0, int(90 * shrink)),
            pygame.Rect(rad * 0.9, rad * 1.5, rad * 2.2 * shrink, rad * 1.4 * shrink),
        )
        self.screen.blit(shadow, (ground[0] - rad * 2 + 1, ground[1] - rad * 2 + 1))
        if lift > 2:  # a line to the ground makes the height unambiguous
            pygame.draw.line(self.screen, (255, 255, 255, 70), ground, center, 1)

        pygame.draw.circle(self.screen, BALL_C, center, rad)
        # classic panels, and a highlight so it reads as a sphere
        if rad >= 4:
            pygame.draw.circle(self.screen, (32, 34, 38), center, max(1, int(rad * 0.34)))
            for i in range(3):
                a = i * (2 * math.pi / 3) + 0.4
                px = center[0] + math.cos(a) * rad * 0.66
                py = center[1] + math.sin(a) * rad * 0.66
                pygame.draw.circle(self.screen, (32, 34, 38), (int(px), int(py)), max(1, int(rad * 0.2)))
        pygame.draw.circle(self.screen, (210, 210, 214), center, rad, 1)

    def _plate_target(self):
        return self._plate.update(self.match)

    def _draw_owner_name(self) -> None:
        """Name plate above whoever is on the ball."""
        player, alpha = self._plate_target()
        if player is None or alpha <= 0.0:
            return
        center = self.to_px(player.pos)
        rad = max(6, self.m(0.62))
        label = self.font_name.render(player.name, True, (255, 255, 255))
        rect = label.get_rect(center=(center[0], center[1] - rad - 15))
        plate = rect.inflate(12, 7)
        surf = pygame.Surface(plate.size, pygame.SRCALPHA)
        pygame.draw.rect(surf, (0, 0, 0, int(175 * alpha)), surf.get_rect(), border_radius=7)
        pygame.draw.rect(
            surf, TEAM_COLORS[player.team]["body"], surf.get_rect(), width=2, border_radius=7
        )
        surf.blit(label, label.get_rect(center=surf.get_rect().center))
        surf.set_alpha(int(255 * alpha))
        self.screen.blit(surf, plate.topleft)

    def _draw_debug_field(self) -> None:
        m = self.match
        overlay = pygame.Surface(self.screen.get_size(), pygame.SRCALPHA)
        for p in m.players:
            c = self.to_px(p.pos)
            pygame.draw.circle(overlay, (255, 255, 255, 45), c, self.m(m._control_radius(p)), 1)
            if p.vel.length() > 0.2:
                pygame.draw.line(overlay, (255, 255, 0, 180), c, self.to_px(p.pos + p.vel * 0.4), 2)
        if m.ball.vel.length() > 0.2:
            pygame.draw.line(
                overlay, (120, 220, 255, 200), self.to_px(m.ball.pos),
                self.to_px(m.ball.pos + m.ball.vel * 0.4), 2,
            )
        self.screen.blit(overlay, (0, 0))

    def _draw_hud(self) -> None:
        m = self.match
        w = self.screen.get_width()
        pygame.draw.rect(self.screen, HUD_BG, pygame.Rect(0, 0, w, HUD_H))
        pygame.draw.line(self.screen, (44, 52, 62), (0, HUD_H - 1), (w, HUD_H - 1), 1)

        home = m.controllers[HOME].name
        away = m.controllers[AWAY].name
        score = f"{m.score[HOME]} – {m.score[AWAY]}"

        score_surf = self.font_big.render(score, True, HUD_FG)
        self.screen.blit(score_surf, score_surf.get_rect(center=(w // 2, 30)))

        hs = self.font_mid.render(home, True, TEAM_COLORS[HOME]["body"])
        self.screen.blit(hs, hs.get_rect(midright=(w // 2 - 52, 30)))
        as_ = self.font_mid.render(away, True, TEAM_COLORS[AWAY]["body"])
        self.screen.blit(as_, as_.get_rect(midleft=(w // 2 + 52, 30)))

        total = max(1, sum(m.possession_ticks))
        ph = 100.0 * m.possession_ticks[HOME] / total
        clock_txt = f"{int(m.clock // 60):01d}:{int(m.clock % 60):02d}"
        phase = m.setpiece.replace("_", " ") if m.setpiece else m.phase
        sub = f"H{m.period}  {clock_txt}   ·   {phase}   ·   possession {ph:.0f}% / {100 - ph:.0f}%   ·   x{self.speed:g}"
        ss = self.font.render(sub, True, HUD_DIM)
        self.screen.blit(ss, ss.get_rect(center=(w // 2, 60)))

        # possession bar
        bar = pygame.Rect(w // 2 - 180, 74, 360, 4)
        pygame.draw.rect(self.screen, TEAM_COLORS[AWAY]["body"], bar)
        pygame.draw.rect(
            self.screen, TEAM_COLORS[HOME]["body"],
            pygame.Rect(bar.x, bar.y, int(bar.width * ph / 100.0), bar.height),
        )

        if self.debug:
            lines = []
            for t in (HOME, AWAY):
                s = m.controllers[t].stats()
                lines.append(f"{s['name'][:16]:16s} avg {s['avg_ms']:6.3f}ms  max {s['worst_ms']:6.2f}ms  err {s['errors']}")
            lines.append(f"tick {m.tick}  seed {m.seed}  shots {m.shots[HOME]}/{m.shots[AWAY]}")
            for i, text in enumerate(lines):
                surf = self.font_small.render(text, True, HUD_DIM)
                self.screen.blit(surf, (12, 8 + i * 15))

        hint = self.font_small.render(
            "space pause · +/− speed · tab debug · s step · r restart · q quit",
            True, (86, 96, 108),
        )
        self.screen.blit(hint, hint.get_rect(center=(w // 2, self.screen.get_height() - HINT_H // 2)))

    def _banner(self, text: str, color) -> None:
        w, h = self.screen.get_size()
        surf = self.font_big.render(text, True, color)
        rect = surf.get_rect(center=(w // 2, HUD_H + (h - HUD_H) // 2))
        pad = pygame.Rect(rect.x - 26, rect.y - 14, rect.width + 52, rect.height + 28)
        box = pygame.Surface(pad.size, pygame.SRCALPHA)
        box.fill((0, 0, 0, 150))
        self.screen.blit(box, pad.topleft)
        self.screen.blit(surf, rect)
