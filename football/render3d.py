"""Panda3D view of a match.

A sibling to `render.py`, not a replacement: the 2D view stays the debugging
view. The simulation is untouched by either -- both just read `Match`.

Two things learned the hard way on macOS (see the notes in `_make_sphere`):
geometry must carry **normals** or lighting silently does nothing, and Panda's
auto-shader is unavailable here, so this uses the fixed-function pipeline of
the OpenGL compatibility profile.

Controls
    space      pause / resume
    + / -      simulation speed
    1 2 3      broadcast / follow-ball / high camera
    left drag  orbit the camera
    wheel      zoom
    tab        shirt numbers
    r          restart the match
    q / esc    quit
"""

from __future__ import annotations

import math

from panda3d.core import (
    AmbientLight, CardMaker, ClockObject, DirectionalLight, Geom, GeomNode,
    GeomTriangles, GeomVertexData, GeomVertexFormat, GeomVertexWriter, LVector3,
    Material, MouseButton, NodePath, Point3, TextNode, Texture,
    TransparencyAttrib, Vec4, loadPrcFileData,
)

from .engine import AWAY, HOME, Match
from .render import (
    GRASS_A, HAIR_TONES, SKIN_TONES, TEAM_COLORS, NamePlateTracker,
    render_pitch_surface,
)


def _appearance(p):
    """Same stable per-player skin and hair the 2D view uses."""
    h = p.team * 7 + p.index * 13
    skin = SKIN_TONES[h % len(SKIN_TONES)]
    hair = HAIR_TONES[(h // 3) % len(HAIR_TONES)]
    return _rgb01(skin), _rgb01(hair)

PITCH_TEXTURE_PX_PER_M = 12.0
SKY = (0.42, 0.60, 0.80, 1.0)


# ---------------------------------------------------------------------------
# procedural geometry -- everything here carries normals on purpose
# ---------------------------------------------------------------------------


def _make_sphere(radius: float = 1.0, segments: int = 16, rings: int = 12) -> NodePath:
    """A UV sphere with normals.

    Panda ships `models/misc/sphere`, but its vertex data has *no normal
    column*, so lighting cannot be computed and it renders as a flat silhouette.
    Generating it here guarantees normals exist.
    """
    fmt = GeomVertexFormat.getV3n3()
    vdata = GeomVertexData("sphere", fmt, Geom.UHStatic)
    vertex = GeomVertexWriter(vdata, "vertex")
    normal = GeomVertexWriter(vdata, "normal")

    for i in range(rings + 1):
        phi = math.pi * i / rings
        for j in range(segments + 1):
            theta = 2.0 * math.pi * j / segments
            n = LVector3(math.sin(phi) * math.cos(theta),
                         math.sin(phi) * math.sin(theta),
                         math.cos(phi))
            vertex.addData3(n * radius)
            normal.addData3(n)

    tris = GeomTriangles(Geom.UHStatic)
    row = segments + 1
    for i in range(rings):
        for j in range(segments):
            a = i * row + j
            b = a + row
            tris.addVertices(a, b, a + 1)
            tris.addVertices(a + 1, b, b + 1)
    geom = Geom(vdata)
    geom.addPrimitive(tris)
    node = GeomNode("sphere")
    node.addGeom(geom)
    return NodePath(node)


def _make_cylinder(radius: float = 1.0, height: float = 1.0, segments: int = 16) -> NodePath:
    """A capped cylinder along +Z, with normals, based at z = 0."""
    fmt = GeomVertexFormat.getV3n3()
    vdata = GeomVertexData("cyl", fmt, Geom.UHStatic)
    vertex = GeomVertexWriter(vdata, "vertex")
    normal = GeomVertexWriter(vdata, "normal")

    for j in range(segments + 1):
        t = 2.0 * math.pi * j / segments
        c, s = math.cos(t), math.sin(t)
        vertex.addData3(radius * c, radius * s, 0.0)
        normal.addData3(c, s, 0.0)
        vertex.addData3(radius * c, radius * s, height)
        normal.addData3(c, s, 0.0)

    tris = GeomTriangles(Geom.UHStatic)
    for j in range(segments):
        a = j * 2
        tris.addVertices(a, a + 1, a + 2)
        tris.addVertices(a + 2, a + 1, a + 3)

    top_start = (segments + 1) * 2
    vertex.addData3(0, 0, height)
    normal.addData3(0, 0, 1)
    for j in range(segments + 1):
        t = 2.0 * math.pi * j / segments
        vertex.addData3(radius * math.cos(t), radius * math.sin(t), height)
        normal.addData3(0, 0, 1)
    for j in range(segments):
        tris.addVertices(top_start, top_start + 1 + j, top_start + 2 + j)

    geom = Geom(vdata)
    geom.addPrimitive(tris)
    node = GeomNode("cylinder")
    node.addGeom(geom)
    return NodePath(node)


def _limb(pivot: NodePath, radius: float, length: float, material: Material,
          taper: float = 1.0) -> NodePath:
    """A limb segment hanging *down* from `pivot`, so rotating the pivot swings it.

    Geometry runs 0..+Z, so it is dropped by its own length to hang below the
    joint. `taper` narrows it towards the far end, which reads far better than
    a plain tube at this scale.
    """
    seg = _make_cylinder(radius, length, 10)
    seg.reparentTo(pivot)
    seg.setZ(-length)
    seg.setSx(taper)
    seg.setSy(taper)
    seg.setMaterial(material, 1)
    return seg


def _joint(parent: NodePath, radius: float, material: Material) -> NodePath:
    """A small sphere at a joint, so knees and elbows do not look snapped off."""
    ball = _make_sphere(radius, 10, 8)
    ball.reparentTo(parent)
    ball.setMaterial(material, 1)
    return ball


def _make_shadow_disc(radius: float, segments: int = 28, alpha: float = 0.55) -> NodePath:
    """A soft round shadow blob, as a vertex-coloured fan in the XY plane.

    A CardMaker quad is exactly that -- a square -- so an untextured shadow
    reads as a dark box on the grass. Fading the alpha out to zero at the rim
    gives a circular blob with soft edges, and doing it in vertex colours
    avoids a texture (and the ram-image plumbing that goes with one) entirely.
    """
    fmt = GeomVertexFormat.getV3c4()
    vdata = GeomVertexData("shadow", fmt, Geom.UHStatic)
    vertex = GeomVertexWriter(vdata, "vertex")
    color = GeomVertexWriter(vdata, "color")

    vertex.addData3(0.0, 0.0, 0.0)
    color.addData4(0.0, 0.0, 0.0, alpha)
    for i in range(segments + 1):
        t = 2.0 * math.pi * i / segments
        vertex.addData3(math.cos(t) * radius, math.sin(t) * radius, 0.0)
        color.addData4(0.0, 0.0, 0.0, 0.0)

    tris = GeomTriangles(Geom.UHStatic)
    for i in range(segments):
        tris.addVertices(0, i + 1, i + 2)
    geom = Geom(vdata)
    geom.addPrimitive(tris)
    node = GeomNode("shadow")
    node.addGeom(geom)
    return NodePath(node)


def _material(rgb, shininess: float = 18.0, ambient: float = 0.42) -> Material:
    m = Material()
    m.setDiffuse(Vec4(rgb[0], rgb[1], rgb[2], 1.0))
    m.setAmbient(Vec4(rgb[0] * ambient, rgb[1] * ambient, rgb[2] * ambient, 1.0))
    m.setSpecular(Vec4(0.18, 0.18, 0.18, 1.0))
    m.setShininess(shininess)
    return m


def _rgb01(c) -> tuple[float, float, float]:
    return (c[0] / 255.0, c[1] / 255.0, c[2] / 255.0)


#: Procedural hair, so a squad does not look like clones. Chosen per player.
HAIR_STYLES = ("crop", "afro", "mohawk", "ponytail", "swept", "bald", "topknot", "flattop")


def _build_hair(head: NodePath, style: str, hair: Material, radius: float) -> None:
    """Attach one of several hairstyles to a head of the given radius.

    Each piece takes the hair material as it is made -- sweeping the head's
    children afterwards would also catch the eyes and paint them hair-coloured.
    """

    def piece(r: float, scale, pos, segments: int = 12) -> None:
        np_ = _make_sphere(radius * r, segments, max(6, segments - 4))
        np_.reparentTo(head)
        np_.setScale(*scale)
        np_.setPos(*pos)
        np_.setMaterial(hair, 1)

    if style == "bald":
        return
    if style == "crop":
        piece(1.02, (1.0, 1.0, 0.55), (0, 0, radius * 0.36), 14)
    elif style == "flattop":
        piece(1.00, (1.0, 1.0, 0.34), (0, 0, radius * 0.58), 14)
    elif style == "afro":
        piece(1.26, (1.0, 1.0, 0.86), (0, 0, radius * 0.30), 16)
    elif style == "swept":
        piece(1.04, (1.0, 1.12, 0.50), (0, -radius * 0.12, radius * 0.38), 14)
    elif style == "mohawk":
        for i in range(5):
            t = (i / 4.0 - 0.5) * 2.0
            piece(0.22, (0.5, 1.0, 2.9 - abs(t) * 1.4), (0, -t * radius * 0.60, radius * 0.92), 8)
    elif style == "ponytail":
        piece(1.02, (1.0, 1.0, 0.50), (0, 0, radius * 0.40), 14)
        piece(0.30, (0.8, 1.7, 0.8), (0, -radius * 1.02, radius * 0.10), 10)
    elif style == "topknot":
        piece(1.02, (1.0, 1.0, 0.48), (0, 0, radius * 0.42), 14)
        piece(0.30, (1.0, 1.0, 1.0), (0, -radius * 0.12, radius * 0.98), 10)


class PlayerRig:
    """A stylised player, animated procedurally.

    Deliberately cartoon proportions -- an oversized head on a small body --
    rather than realistic ones. At the scale a match is actually watched from,
    a big head is what makes a player readable at all: it carries the skin
    tone, the hairstyle and the eyes that show which way they are facing.

    There is no rigged humanoid to load (Panda bundles only primitives and an
    animated panda), so the figure is built from jointed parts and driven by a
    run cycle keyed to `Player.distance_run` -- the same quantity the 2D sprite
    uses, so the stride stays in step with real ground speed.
    """

    HIP_H = 0.74
    SHOULDER_H = 1.16
    HEAD_R = 0.33
    STRIDE_M = 1.7  # metres per full cycle

    def __init__(self, render: NodePath, player, skin_rgb, hair_rgb) -> None:
        colors = TEAM_COLORS[player.team]
        shirt = _material(_rgb01(colors["keeper"] if player.is_keeper else colors["body"]), 26.0)
        shorts = _material(_rgb01(colors["keeper_shorts"] if player.is_keeper else colors["shorts"]), 26.0)
        skin = _material(skin_rgb, 22.0)
        hair = _material(hair_rgb, shininess=8.0)
        boot = _material((0.11, 0.11, 0.13), shininess=55.0)
        eye = _material((0.09, 0.09, 0.11), shininess=70.0)

        self.root = render.attachNewNode(f"p{player.team}{player.index}")
        self.body = self.root.attachNewNode("body")  # carries the forward lean
        self.hips = self.body.attachNewNode("hips")
        self.hips.setZ(self.HIP_H)

        torso_h = self.SHOULDER_H - self.HIP_H
        torso = _make_cylinder(0.19, torso_h, 14)
        torso.reparentTo(self.hips)
        torso.setSy(0.82)
        torso.setMaterial(shirt, 1)
        for z in (0.0, torso_h):  # round the shirt off at both ends
            cap = _joint(self.hips, 0.19, shirt)
            cap.setPos(0, 0, z)
            cap.setSy(0.82)

        self.hip_j, self.knee_j = {}, {}
        self.shoulder_j, self.elbow_j = {}, {}
        for side in (-1, 1):
            hip = self.hips.attachNewNode(f"hip{side}")
            hip.setPos(0.10 * side, 0, 0.02)
            _limb(hip, 0.082, 0.36, shorts, taper=0.95)
            knee = hip.attachNewNode("knee")
            knee.setZ(-0.36)
            _joint(knee, 0.078, skin)
            _limb(knee, 0.072, 0.36, skin, taper=0.92)
            # chunky rounded boot, not a flat slab
            foot = _make_sphere(0.10, 12, 8)
            foot.reparentTo(knee)
            foot.setScale(0.85, 2.0, 0.80)
            foot.setPos(0, 0.09, -0.40)
            foot.setMaterial(boot, 1)
            self.hip_j[side], self.knee_j[side] = hip, knee

            shoulder = self.hips.attachNewNode(f"sh{side}")
            shoulder.setPos(0.21 * side, 0, torso_h - 0.05)
            _joint(shoulder, 0.085, shirt)
            _limb(shoulder, 0.070, 0.26, shirt, taper=0.92)
            elbow = shoulder.attachNewNode("elbow")
            elbow.setZ(-0.26)
            _joint(elbow, 0.066, skin)
            _limb(elbow, 0.060, 0.24, skin, taper=0.9)
            mitt = _joint(elbow, 0.088, skin)  # rounded fist
            mitt.setZ(-0.26)
            self.shoulder_j[side], self.elbow_j[side] = shoulder, elbow

        self.head = self.hips.attachNewNode("head")
        self.head.setZ(torso_h + self.HEAD_R * 0.86)
        skull = _make_sphere(self.HEAD_R, 20, 16)
        skull.reparentTo(self.head)
        skull.setMaterial(skin, 1)
        # Eyes on the front face. They double as the clearest possible cue for
        # which way a player is turned.
        for side in (-1, 1):
            e = _make_sphere(self.HEAD_R * 0.17, 10, 8)
            e.reparentTo(self.head)
            e.setScale(0.85, 0.55, 1.15)
            e.setPos(side * self.HEAD_R * 0.34, self.HEAD_R * 0.90, self.HEAD_R * 0.10)
            e.setMaterial(eye, 1)
        self.hair_style = HAIR_STYLES[(player.team * 5 + player.index * 7) % len(HAIR_STYLES)]
        _build_hair(self.head, self.hair_style, hair, self.HEAD_R)

        self.label = None
        self.shadow = None

    def update(self, p, rules, dt: float) -> None:
        speed = p.vel.length()
        amp = min(1.0, speed / rules.run_speed)
        phase = p.distance_run * (2.0 * math.pi / self.STRIDE_M)
        swing = math.sin(phase)

        self.root.setPos(p.pos.x, p.pos.y, 0.0)
        self.root.setH(math.degrees(math.atan2(p.heading.y, p.heading.x)) - 90.0)
        # lean into the run, and bob on every half stride
        self.body.setP(amp * 9.0)
        self.hips.setZ(self.HIP_H + abs(math.sin(phase * 2.0)) * 0.035 * amp)

        # A fresh kick leaves kick_cd at its maximum, so it doubles as the
        # progress of a kick animation without the engine knowing about it.
        kick = max(0.0, p.kick_cd / rules.kick_cooldown) if rules.kick_cooldown else 0.0

        for side in (-1, 1):
            leg = swing * side
            self.hip_j[side].setP(leg * 44.0 * amp)
            # The trailing leg bends, the leading one straightens. The sign
            # matters: a knee folds the shin *backwards* (heel toward the
            # backside). Bending it the other way gives a bird's reverse joint
            # and the whole stride reads as running backwards.
            self.knee_j[side].setP(-max(0.0, -leg) * 62.0 * amp)
            self.shoulder_j[side].setP(-leg * 34.0 * amp - 6.0)
            # elbows fold the forearm forwards, again the opposite sign to knees
            self.elbow_j[side].setP(28.0 * amp + 8.0)

        if kick > 0.02:
            # right leg drives through, left plants
            self.hip_j[1].setP(75.0 * kick)   # driving leg swings through, forwards
            self.knee_j[1].setP(-10.0 * kick)
            self.shoulder_j[-1].setP(50.0 * kick)
            self.body.setP(amp * 9.0 - 8.0 * kick)  # rock back over the strike


# ---------------------------------------------------------------------------


class Viewer3D:
    """Drives a `Match` and draws it with Panda3D."""

    def __init__(self, match_factory, window=(1400, 900), speed: float = 1.0,
                 camera: str = "broadcast") -> None:
        loadPrcFileData("", f"win-size {window[0]} {window[1]}")
        loadPrcFileData("", "window-title fussball -- 3D")
        loadPrcFileData("", "audio-library-name null")
        loadPrcFileData("", "sync-video #t")
        # Ten rigs of a dozen joints each, all moving every frame, mint
        # thousands of unique transforms per second. Panda caches composed
        # transforms and render states, and here those tables grow faster than
        # they are collected: the frame rate decays steadily during a match
        # (measured 580 -> 101 fps over a few thousand frames). Turning the
        # caches off costs a few percent up front and holds a flat ~570 fps.
        loadPrcFileData("", "transform-cache false")
        loadPrcFileData("", "state-cache false")

        from direct.showbase.ShowBase import ShowBase

        self.base = ShowBase()
        self.match_factory = match_factory
        self.match: Match = match_factory()
        self.speed = speed
        self.paused = False
        self.debug = False
        self.camera_mode = camera
        self.orbit = [0.0, -22.0, 120.0]  # heading, pitch, distance
        self._accum = 0.0
        self._plate = NamePlateTracker()
        self._plate_name = None
        self._plate_alpha_step = -1

        self.base.setBackgroundColor(*SKY)
        self.base.disableMouse()
        self._build_scene()
        self._build_hud()
        self._bind_keys()
        self.base.taskMgr.add(self._tick, "sim")

    # -- scene ---------------------------------------------------------
    def _build_scene(self) -> None:
        r = self.match.rules
        render = self.base.render

        # Ground: the 2D renderer already draws every marking, so borrow it as
        # a texture instead of rebuilding the line work as geometry.
        surface = render_pitch_surface(r, PITCH_TEXTURE_PX_PER_M, pad_m=0.0, draw_goals=False)
        self.pitch_np = self._textured_ground(surface, r)

        # A wider apron of plain grass so the pitch does not float in the void.
        cm = CardMaker("apron")
        cm.setFrame(-40, r.length + 40, -40, r.width + 40)
        apron = render.attachNewNode(cm.generate())
        apron.setP(-90)
        apron.setZ(-0.02)
        apron.setMaterial(_material([c * 0.72 / 255.0 for c in GRASS_A]), 1)

        self._build_stadium(r)
        self._build_goals(r)
        self._build_players(r)

        ball_r = r.ball_radius * 2.4  # exaggerated, as in the 2D view
        self.ball = _make_sphere(ball_r, 22, 16)
        self.ball.reparentTo(render)
        self.ball.setMaterial(_material((0.97, 0.97, 0.97), 70.0), 1)
        # dark panels, so the spin below is actually visible
        dark = _material((0.13, 0.14, 0.17), 55.0)
        for hx, hy, hz in ((0, 0, 1), (0, 0, -1), (1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0)):
            patch = _make_sphere(ball_r * 0.42, 8, 6)
            patch.reparentTo(self.ball)
            patch.setPos(hx * ball_r * 0.86, hy * ball_r * 0.86, hz * ball_r * 0.86)
            patch.setMaterial(dark, 1)
        self.ball.flattenStrong()  # patches ride along with the ball's rotation
        self._ball_roll = 0.0
        self._last_ball_pos = self.match.ball.pos

        # A blob on the turf under the ball -- with real height in the sim,
        # this is what tells you how high a lofted ball actually is.
        self.ball_shadow = self._make_shadow(0.42)

        sun = DirectionalLight("sun")
        sun.setColor(Vec4(1.05, 1.02, 0.95, 1))
        sun_np = render.attachNewNode(sun)
        sun_np.setHpr(38, -52, 0)
        render.setLight(sun_np)
        fill = DirectionalLight("fill")
        fill.setColor(Vec4(0.28, 0.30, 0.36, 1))
        fill_np = render.attachNewNode(fill)
        fill_np.setHpr(-140, -30, 0)
        render.setLight(fill_np)
        amb = AmbientLight("amb")
        amb.setColor(Vec4(0.42, 0.44, 0.50, 1))
        render.setLight(render.attachNewNode(amb))

    def _textured_ground(self, surface, r) -> NodePath:
        import pygame

        # Panda textures start at the bottom-left, pygame surfaces at the
        # top-left, so the image has to be flipped or the pitch renders mirrored.
        flipped = pygame.transform.flip(surface, False, True)
        w, h = flipped.get_size()
        tex = Texture("pitch")
        tex.setup2dTexture(w, h, Texture.TUnsignedByte, Texture.FRgba)
        tex.setRamImage(pygame.image.tostring(flipped, "RGBA"))
        tex.setMagfilter(Texture.FTLinear)
        tex.setMinfilter(Texture.FTLinearMipmapLinear)

        cm = CardMaker("pitch")
        cm.setFrame(0, r.length, 0, r.width)
        np_ = self.base.render.attachNewNode(cm.generate())
        np_.setP(-90)
        np_.setTexture(tex)
        return np_

    def _make_shadow(self, radius: float) -> NodePath:
        np_ = _make_shadow_disc(radius)
        np_.reparentTo(self.base.render)
        np_.setTransparency(TransparencyAttrib.MAlpha)
        np_.setLightOff()  # keep the vertex alpha exactly as authored
        np_.setDepthWrite(False)  # shadows should not occlude one another
        np_.setZ(0.015)
        return np_

    def _box(self, size, pos, material, parent: NodePath | None = None) -> NodePath:
        """A unit box from Panda's bundled models, scaled into place.

        `models/box` is used rather than a generated cube because it is one of
        the few bundled models that actually carries normals.
        """
        np_ = self.base.loader.loadModel("models/box")
        np_.reparentTo(parent if parent is not None else self.base.render)
        np_.setTextureOff(1)  # it ships with a noisy placeholder texture
        np_.setScale(*size)
        np_.setPos(*pos)
        np_.setMaterial(material, 1)
        return np_

    def _build_stadium(self, r) -> None:
        """Raked terraces around the pitch, so it does not float in a void."""
        stadium = self.base.render.attachNewNode("stadium")
        gap = 7.0  # run-off between the touchline and the first step
        steps, rise, tread = 7, 1.5, 2.4
        concrete = _material((0.52, 0.53, 0.56), 6.0)
        seats = (_material((0.20, 0.32, 0.62), 8.0), _material((0.72, 0.25, 0.22), 8.0))

        for i in range(steps):
            h = (i + 1) * rise
            out = gap + i * tread
            mat = seats[i % 2] if i else concrete
            # long sides, running the length of the pitch
            span = r.length + 2 * (out + tread)
            self._box((span, tread, h), (-(out + tread), -out - tread, 0), mat, stadium)
            self._box((span, tread, h), (-(out + tread), r.width + out, 0), mat, stadium)
            # ends, behind each goal
            depth = r.width + 2 * (out + tread)
            self._box((tread, depth, h), (-out - tread, -(out + tread), 0), mat, stadium)
            self._box((tread, depth, h), (r.length + out, -(out + tread), 0), mat, stadium)

        # floodlight pylons at the corners
        steel = _material((0.34, 0.36, 0.40), 30.0)
        lamp = _material((1.0, 0.98, 0.88), 90.0)
        reach = gap + steps * tread + 3.0
        for cx in (-reach, r.length + reach):
            for cy in (-reach, r.width + reach):
                mast = _make_cylinder(0.5, 26.0, 10)
                mast.reparentTo(self.base.render)
                mast.setPos(cx, cy, 0)
                mast.setMaterial(steel, 1)
                mast.reparentTo(stadium)
                head = self._box((6.0, 1.2, 3.0), (cx - 3.0, cy - 0.6, 26.0), lamp, stadium)
                head.setLightOff()

        # None of this ever moves, so collapse it into as few draw calls as
        # possible -- unflattened it costs more than the entire rest of the
        # scene (measured: 42 fps before, ~10x that after).
        stadium.flattenStrong()

    def _build_goals(self, r) -> None:
        white = _material((0.95, 0.95, 0.96), 40.0)
        bar = r.crossbar_height
        # drawn thicker than the collision radius, which is invisible at this scale
        rad = max(r.post_radius, 0.11)
        for goal_x in (0.0, r.length):
            for py in (r.goal_y0, r.goal_y1):
                post = _make_cylinder(rad, bar, 12)
                post.reparentTo(self.base.render)
                post.setPos(goal_x, py, 0)
                post.setMaterial(white, 1)
            cross = _make_cylinder(rad, r.goal_width, 12)
            cross.reparentTo(self.base.render)
            # Cylinders run along +Z, so lay it across the mouth. Pitching by
            # -90 sends the local +Z to world +Y, spanning goal_y0 -> goal_y1;
            # +90 would send it the other way, outside the goal entirely.
            cross.setPos(goal_x, r.goal_y0, bar)
            cross.setHpr(0, -90, 0)
            cross.setMaterial(white, 1)
            self._build_net(goal_x, r)

    def _build_net(self, goal_x: float, r) -> None:
        """Three translucent panels forming the back and sides of the net."""
        depth = -2.2 if goal_x == 0.0 else 2.2
        panels = []

        # back panel: spans the full goal width, standing at the back of the net
        cm = CardMaker("net_back")
        cm.setFrame(0, r.goal_width, 0, r.crossbar_height)
        back = self.base.render.attachNewNode(cm.generate())
        back.setPos(goal_x + depth, r.goal_y0, 0)
        back.setH(90)  # local +X -> world +Y, so it spans the goal width
        panels.append(back)

        # side panels: run from each post back to the rear of the net
        for py in (r.goal_y0, r.goal_y1):
            cm = CardMaker("net_side")
            cm.setFrame(0, abs(depth), 0, r.crossbar_height)
            side = self.base.render.attachNewNode(cm.generate())
            side.setPos(goal_x if depth > 0 else goal_x + depth, py, 0)
            panels.append(side)

        for panel in panels:
            panel.setColor(1, 1, 1, 0.20)
            panel.setTransparency(TransparencyAttrib.MAlpha)
            panel.setTwoSided(True)
            panel.setLightOff()

    def _build_players(self, r) -> None:
        self.rigs = []
        for p in self.match.players:
            skin, hair = _appearance(p)
            rig = PlayerRig(self.base.render, p, skin, hair)

            label = TextNode(f"num{p.team}{p.index}")
            label.setText(str(p.index))
            label.setAlign(TextNode.ACenter)
            label.setTextColor(1, 1, 1, 1)
            label_np = rig.root.attachNewNode(label)
            label_np.setScale(0.5)
            label_np.setZ(2.3)
            label_np.setBillboardPointEye()
            label_np.setLightOff()
            label_np.hide()
            rig.label = label_np
            rig.shadow = self._make_shadow(0.40)
            self.rigs.append(rig)

    # -- hud -----------------------------------------------------------
    def _build_hud(self) -> None:
        from direct.gui.OnscreenText import OnscreenText

        # a2dTopCenter runs *downward* from the top edge, so these must be
        # negative or the text sits off the top of the screen
        self.hud_score = OnscreenText(text="", pos=(0, -0.12), scale=0.075,
                                      fg=(1, 1, 1, 1), shadow=(0, 0, 0, 0.7),
                                      mayChange=True, parent=self.base.a2dTopCenter)
        self.hud_sub = OnscreenText(text="", pos=(0, -0.20), scale=0.042,
                                    fg=(0.86, 0.90, 0.95, 1), shadow=(0, 0, 0, 0.7),
                                    mayChange=True, parent=self.base.a2dTopCenter)
        self.hud_hint = OnscreenText(
            text="space pause | +/- speed | 1 2 3 camera | drag orbit | tab numbers | r restart | q quit",
            pos=(0, 0.06), scale=0.035, fg=(0.80, 0.84, 0.90, 1),
            shadow=(0, 0, 0, 0.7), parent=self.base.a2dBottomCenter)
        self.name_plate = TextNode("carrier")
        self.name_plate.setAlign(TextNode.ACenter)
        self.name_plate.setTextColor(1, 1, 1, 1)
        self.name_plate.setCardColor(0, 0, 0, 0.55)
        self.name_plate.setCardAsMargin(0.25, 0.25, 0.12, 0.12)
        # Without this the card and the glyphs sit on the same plane and
        # z-fight, which hides the name inside its own box. Decal mode pushes
        # the card behind the text instead.
        self.name_plate.setCardDecal(True)
        self.name_np = self.base.render.attachNewNode(self.name_plate)
        self.name_np.setScale(0.62)
        self.name_np.setBillboardPointEye()
        self.name_np.setLightOff()
        self.name_np.setTransparency(TransparencyAttrib.MAlpha)
        self.name_np.setDepthWrite(False)
        self.name_np.setBin("fixed", 40)
        self.name_np.hide()

    # -- input ---------------------------------------------------------
    def _bind_keys(self) -> None:
        b = self.base
        b.accept("escape", self._quit)
        b.accept("q", self._quit)
        b.accept("space", self._toggle_pause)
        b.accept("tab", self._toggle_debug)
        b.accept("r", self._restart)
        for key in ("+", "=", "shift-="):
            b.accept(key, self._faster)
        b.accept("-", self._slower)
        b.accept("1", self._set_camera, ["broadcast"])
        b.accept("2", self._set_camera, ["follow"])
        b.accept("3", self._set_camera, ["high"])
        b.accept("wheel_up", self._zoom, [-8.0])
        b.accept("wheel_down", self._zoom, [8.0])
        self._drag = None

    def _quit(self):
        self.base.userExit()

    def _toggle_pause(self):
        self.paused = not self.paused

    def _toggle_debug(self):
        self.debug = not self.debug
        for rig in self.rigs:
            rig.label.show() if self.debug else rig.label.hide()

    def _restart(self):
        self.match = self.match_factory()
        self._last_ball_pos = self.match.ball.pos
        self._plate.reset()  # drop the stale reference into the old match

    def _faster(self):
        self.speed = min(16.0, self.speed * 2.0)

    def _slower(self):
        self.speed = max(0.125, self.speed / 2.0)

    def _set_camera(self, mode):
        self.camera_mode = mode

    def _zoom(self, amount):
        self.orbit[2] = max(25.0, min(240.0, self.orbit[2] + amount))

    # -- per-frame -----------------------------------------------------
    def _tick(self, task):
        from direct.task import Task

        dt = ClockObject.getGlobalClock().getDt()
        self._step_sim(dt)
        self._sync_scene(dt)
        self._move_camera(dt)
        self._update_hud()
        return Task.cont

    def _step_sim(self, frame_dt: float) -> None:
        """Fixed timestep, so the simulation never depends on frame rate."""
        if self.paused or self.match.finished:
            return
        r = self.match.rules
        self._accum += min(frame_dt, 0.25) * self.speed
        steps = 0
        while self._accum >= r.dt and steps < 400:
            self.match.step()
            self._accum -= r.dt
            steps += 1
            if self.match.finished:
                self._accum = 0.0
                break

    def _sync_scene(self, dt: float = 0.0) -> None:
        m = self.match
        for p, rig in zip(m.players, self.rigs):
            rig.update(p, m.rules, dt)
            rig.shadow.setPos(p.pos.x, p.pos.y, 0.015)

        b = m.ball
        self.ball.setPos(b.pos.x, b.pos.y, b.z + m.rules.ball_radius)
        self._spin_ball(b, m.rules)
        self.ball_shadow.setPos(b.pos.x, b.pos.y, 0.02)
        # the shadow shrinks and fades as the ball climbs. setColorScale, not
        # setColor -- the latter would replace the vertex alpha that gives the
        # blob its soft edge, turning it back into a hard disc.
        k = max(0.35, 1.0 - b.z / 12.0)
        self.ball_shadow.setScale(k)
        self.ball_shadow.setColorScale(1.0, 1.0, 1.0, k)

        # Follows the last toucher with a linger, not just the current owner:
        # keepers hold the ball for seconds and outfielders for a fraction of
        # one, so captioning `owner` alone names almost nobody but the keeper.
        carrier, alpha = self._plate.update(m)
        if carrier is None:
            self.name_np.hide()
        else:
            # Only touch the TextNode when something actually changed. Each of
            # setText/setTextColor/setCardColor forces the text geometry to be
            # rebuilt, and doing that every frame degrades the frame rate
            # steadily over a match (measured: 556 -> 221 fps).
            step = round(alpha * 12)
            if carrier.name != self._plate_name or step != self._plate_alpha_step:
                self._plate_name = carrier.name
                self._plate_alpha_step = step
                a = step / 12.0
                self.name_plate.setText(carrier.name)
                self.name_plate.setTextColor(1, 1, 1, a)
                self.name_plate.setCardColor(0, 0, 0, 0.55 * a)
            self.name_np.setPos(carrier.pos.x, carrier.pos.y, 2.15)
            self.name_np.show()

    def _spin_ball(self, b, rules) -> None:
        """Roll the ball by the distance it has travelled.

        Without this a moving ball reads as a sliding dot; the panels make the
        rotation visible, which is most of what sells the ball as a ball.
        """
        moved = b.pos.dist(self._last_ball_pos)
        self._last_ball_pos = b.pos
        speed = b.vel.length()
        if speed > 0.05:
            circumference = 2.0 * math.pi * max(rules.ball_radius, 0.01)
            self._ball_roll = (self._ball_roll + (moved / circumference) * 360.0) % 360.0
            heading = math.degrees(math.atan2(b.vel.y, b.vel.x)) - 90.0
            self.ball.setHpr(heading, self._ball_roll, 0.0)

    def _move_camera(self, dt: float) -> None:
        m = self.match
        r = m.rules
        cam = self.base.camera
        ball = m.ball.pos
        mid = Point3(r.length / 2, r.width / 2, 0)

        # absent when rendering offscreen, so this has to be optional
        mouse = self.base.mouseWatcherNode
        if mouse is not None and mouse.hasMouse() and mouse.isButtonDown(MouseButton.one()):
            mx, my = mouse.getMouseX(), mouse.getMouseY()
            if self._drag is not None:
                self.orbit[0] += (mx - self._drag[0]) * 120.0
                self.orbit[1] = max(-80.0, min(-8.0, self.orbit[1] + (my - self._drag[1]) * 80.0))
            self._drag = (mx, my)
        else:
            self._drag = None

        if self.camera_mode == "broadcast":
            target = Point3(mid.x + (ball.x - mid.x) * 0.35, mid.y, 0)
            dist, height = 78.0, 34.0
        elif self.camera_mode == "follow":
            target = Point3(ball.x, ball.y, 0)
            dist, height = 34.0, 16.0
        else:  # high -- far enough back that both goals stay in frame
            target = mid
            dist, height = 118.0, 88.0

        h = math.radians(self.orbit[0])
        want = Point3(target.x - math.sin(h) * dist, target.y - math.cos(h) * dist, height)
        # ease toward the wanted position so the camera does not snap about
        ease = min(1.0, dt * 3.5)
        cam.setPos(cam.getPos() * (1.0 - ease) + want * ease)
        cam.lookAt(target)

    def _update_hud(self) -> None:
        m = self.match
        home, away = m.controllers[HOME].name, m.controllers[AWAY].name
        self.hud_score.setText(f"{home}  {m.score[HOME]} - {m.score[AWAY]}  {away}")
        clock = f"{int(m.clock // 60):01d}:{int(m.clock % 60):02d}"
        phase = m.setpiece.replace("_", " ") if m.setpiece else m.phase
        extra = "  PAUSED" if self.paused else ""
        height = f"   ball {m.ball.z:.1f} m" if m.ball.z > 0.2 else ""
        self.hud_sub.setText(f"H{m.period}  {clock}   {phase}   x{self.speed:g}{height}{extra}")

    def run(self) -> Match:
        self.base.run()
        return self.match
