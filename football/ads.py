"""Content for the pitch-side advertising hoardings.

Deliberately pygame-only: this module produces plain surfaces, and the 3D
renderer turns them into textures. That keeps the image handling testable
without a graphics context, and keeps Panda out of the loading path.

An "ad" is an `AdClip` -- one or more frames with per-frame durations. A still
image is simply a clip with a single frame, so animated and static content go
down exactly the same path.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import pygame

#: Hoardings are long and thin, so every ad is composited onto this canvas.
BOARD_PX = (512, 96)

STILL_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
ANIMATED_SUFFIXES = {".gif", ".webp"}


@dataclass
class AdClip:
    """A sequence of board-shaped frames with their durations, in seconds."""

    name: str
    frames: list[pygame.Surface]
    durations: list[float] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.durations:
            self.durations = [999.0] * len(self.frames)
        self.total = max(1e-3, sum(self.durations))

    @property
    def animated(self) -> bool:
        return len(self.frames) > 1

    def frame_index(self, t: float) -> int:
        """Which frame is showing `t` seconds in, looping."""
        if len(self.frames) == 1:
            return 0
        t %= self.total
        for i, d in enumerate(self.durations):
            if t < d:
                return i
            t -= d
        return len(self.frames) - 1


def _fit(surface: pygame.Surface, size=BOARD_PX, background=(14, 16, 20)) -> pygame.Surface:
    """Letterbox a frame onto a board-shaped canvas without distorting it."""
    out = pygame.Surface(size)
    out.fill(background)
    sw, sh = surface.get_size()
    if sw <= 0 or sh <= 0:
        return out
    scale = min(size[0] / sw, size[1] / sh)
    w, h = max(1, int(sw * scale)), max(1, int(sh * scale))
    # convert_alpha needs a display, which the 3D view has not got -- Panda
    # owns the window. It is only an optimisation, so skip it when absent.
    try:
        source = surface.convert_alpha()
    except pygame.error:
        source = surface
    scaled = pygame.transform.smoothscale(source, (w, h))
    out.blit(scaled, ((size[0] - w) // 2, (size[1] - h) // 2))
    return out


def _load_animated(path: Path) -> AdClip | None:
    """Decode an animated GIF/WebP. Needs Pillow; returns None without it."""
    try:
        from PIL import Image, ImageSequence
    except ImportError:
        return None
    try:
        image = Image.open(path)
    except Exception:
        return None
    frames, durations = [], []
    for frame in ImageSequence.Iterator(image):
        rgba = frame.convert("RGBA")
        surf = pygame.image.fromstring(rgba.tobytes(), rgba.size, "RGBA")
        frames.append(_fit(surf))
        durations.append(max(0.02, frame.info.get("duration", 100) / 1000.0))
        if len(frames) >= 240:  # a sanity cap on absurdly long clips
            break
    if not frames:
        return None
    return AdClip(path.stem, frames, durations)


def load_ad_clips(directory: str | Path) -> list[AdClip]:
    """Load every image in `directory` as an ad, animating GIFs where possible."""
    folder = Path(directory).expanduser()
    if not folder.is_dir():
        return []
    clips: list[AdClip] = []
    for path in sorted(folder.iterdir()):
        suffix = path.suffix.lower()
        clip = None
        if suffix in ANIMATED_SUFFIXES:
            clip = _load_animated(path)
        if clip is None and suffix in STILL_SUFFIXES | ANIMATED_SUFFIXES:
            try:
                clip = AdClip(path.stem, [_fit(pygame.image.load(str(path)))])
            except Exception:
                clip = None
        if clip is not None:
            clips.append(clip)
    return clips


# ---------------------------------------------------------------------------
# fallback content, so the hoardings are never blank out of the box
# ---------------------------------------------------------------------------

_PLACEHOLDERS = (
    ("FUSSBALL", (24, 92, 200), (255, 255, 255)),
    ("SANDBOX FC", (198, 44, 40), (255, 240, 200)),
    ("DETERMINISTIC", (18, 132, 96), (240, 255, 240)),
    ("PYTHON LEAGUE", (232, 168, 24), (30, 24, 12)),
)


def placeholder_clips() -> list[AdClip]:
    """Procedural boards, one of which sweeps, so animation is visible at once."""
    pygame.font.init()
    font = pygame.font.SysFont("helvetica,arial,sans-serif", 54, bold=True)
    clips = []
    for i, (text, bg, fg) in enumerate(_PLACEHOLDERS):
        if i == len(_PLACEHOLDERS) - 1:
            # a swept highlight, purely so the animation path is exercised
            frames, durations = [], []
            for step in range(24):
                surf = pygame.Surface(BOARD_PX)
                surf.fill(bg)
                x = int((step / 24.0) * (BOARD_PX[0] + 160)) - 80
                glow = pygame.Surface((80, BOARD_PX[1]), pygame.SRCALPHA)
                glow.fill((255, 255, 255, 60))
                surf.blit(glow, (x, 0))
                label = font.render(text, True, fg)
                surf.blit(label, label.get_rect(center=(BOARD_PX[0] // 2, BOARD_PX[1] // 2)))
                frames.append(surf)
                durations.append(0.06)
            clips.append(AdClip(text, frames, durations))
            continue
        surf = pygame.Surface(BOARD_PX)
        surf.fill(bg)
        pygame.draw.rect(surf, fg, surf.get_rect(), 3)
        label = font.render(text, True, fg)
        surf.blit(label, label.get_rect(center=(BOARD_PX[0] // 2, BOARD_PX[1] // 2)))
        clips.append(AdClip(text, [surf]))
    return clips
