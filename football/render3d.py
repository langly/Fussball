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
from .render import GRASS_A, TEAM_COLORS, render_pitch_surface

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


def _material(rgb, shininess: float = 18.0, ambient: float = 0.42) -> Material:
    m = Material()
    m.setDiffuse(Vec4(rgb[0], rgb[1], rgb[2], 1.0))
    m.setAmbient(Vec4(rgb[0] * ambient, rgb[1] * ambient, rgb[2] * ambient, 1.0))
    m.setSpecular(Vec4(0.18, 0.18, 0.18, 1.0))
    m.setShininess(shininess)
    return m


def _rgb01(c) -> tuple[float, float, float]:
    return (c[0] / 255.0, c[1] / 255.0, c[2] / 255.0)


# ---------------------------------------------------------------------------


class Viewer3D:
    """Drives a `Match` and draws it with Panda3D."""

    def __init__(self, match_factory, window=(1400, 900), speed: float = 1.0,
                 camera: str = "broadcast") -> None:
        loadPrcFileData("", f"win-size {window[0]} {window[1]}")
        loadPrcFileData("", "window-title fussball -- 3D")
        loadPrcFileData("", "audio-library-name null")
        loadPrcFileData("", "sync-video #t")

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

        self._build_goals(r)
        self._build_players(r)

        self.ball = _make_sphere(r.ball_radius * 2.4, 20, 14)
        self.ball.reparentTo(render)
        self.ball.setMaterial(_material((0.96, 0.96, 0.96), 60.0), 1)

        # A blob on the turf under the ball -- with real height in the sim,
        # this is what tells you how high a lofted ball actually is.
        self.ball_shadow = self._make_shadow(0.5)

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
        cm = CardMaker("shadow")
        cm.setFrame(-radius, radius, -radius, radius)
        np_ = self.base.render.attachNewNode(cm.generate())
        np_.setP(-90)
        np_.setColor(0, 0, 0, 0.32)
        np_.setTransparency(TransparencyAttrib.MAlpha)
        np_.setLightOff()
        np_.setZ(0.015)
        return np_

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
        self.player_nodes = []
        for p in self.match.players:
            colors = TEAM_COLORS[p.team]
            shirt = _rgb01(colors["keeper"] if p.is_keeper else colors["body"])
            shorts = _rgb01(colors["keeper_shorts"] if p.is_keeper else colors["shorts"])

            root = self.base.render.attachNewNode(f"player{p.team}{p.index}")
            legs = _make_cylinder(0.20, 0.85, 10)
            legs.reparentTo(root)
            legs.setMaterial(_material(shorts), 1)
            torso = _make_cylinder(0.32, 0.75, 12)
            torso.reparentTo(root)
            torso.setZ(0.85)
            torso.setMaterial(_material(shirt), 1)
            head = _make_sphere(0.16, 14, 10)
            head.reparentTo(root)
            head.setZ(1.78)
            head.setMaterial(_material((0.85, 0.68, 0.54)), 1)
            # a stub facing forward, so you can read which way a player is turned
            nose = _make_cylinder(0.07, 0.22, 8)
            nose.reparentTo(root)
            nose.setPos(0, 0.16, 1.74)
            nose.setHpr(0, 90, 0)
            nose.setMaterial(_material((0.95, 0.95, 0.95)), 1)

            label = TextNode(f"num{p.team}{p.index}")
            label.setText(str(p.index))
            label.setAlign(TextNode.ACenter)
            label.setTextColor(1, 1, 1, 1)
            label_np = root.attachNewNode(label)
            label_np.setScale(0.55)
            label_np.setZ(2.35)
            label_np.setBillboardPointEye()
            label_np.setLightOff()
            label_np.hide()

            self.player_nodes.append({
                "root": root, "legs": legs, "label": label_np,
                "shadow": self._make_shadow(0.42),
            })

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
        self.name_plate.setCardAsMargin(0.3, 0.3, 0.15, 0.15)
        self.name_np = self.base.render.attachNewNode(self.name_plate)
        self.name_np.setScale(0.9)
        self.name_np.setBillboardPointEye()
        self.name_np.setLightOff()
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
        for node in self.player_nodes:
            node["label"].show() if self.debug else node["label"].hide()

    def _restart(self):
        self.match = self.match_factory()

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
        self._sync_scene()
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

    def _sync_scene(self) -> None:
        m = self.match
        for p, node in zip(m.players, self.player_nodes):
            node["root"].setPos(p.pos.x, p.pos.y, 0)
            node["root"].setH(math.degrees(math.atan2(p.heading.y, p.heading.x)) - 90.0)
            node["shadow"].setPos(p.pos.x, p.pos.y, 0.015)
            # legs bob on the stride cycle the 2D sprite already uses
            swing = math.sin(p.distance_run * (2.0 * math.pi / 1.7))
            amp = min(1.0, p.vel.length() / m.rules.run_speed)
            node["legs"].setZ(abs(swing) * amp * 0.09)

        b = m.ball
        self.ball.setPos(b.pos.x, b.pos.y, b.z + m.rules.ball_radius)
        self.ball_shadow.setPos(b.pos.x, b.pos.y, 0.02)
        # the shadow shrinks and fades as the ball climbs
        k = max(0.35, 1.0 - b.z / 12.0)
        self.ball_shadow.setScale(k)
        self.ball_shadow.setColor(0, 0, 0, 0.32 * k)

        owner = m.owner
        if owner is None:
            self.name_np.hide()
        else:
            self.name_plate.setText(owner.name)
            self.name_np.setPos(owner.pos.x, owner.pos.y, 2.9)
            self.name_np.show()

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
