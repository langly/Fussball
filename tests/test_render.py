"""Rendering, headless.

No window is opened. The 3D geometry helpers are pure Panda scene-graph calls,
so they need no graphics context at all -- which is what makes the normals
check below cheap enough to keep in the suite.
"""

from __future__ import annotations

import math

import pytest

pygame = pytest.importorskip("pygame")

from football.ads import AdClip, BOARD_PX, load_ad_clips, placeholder_clips  # noqa: E402
from football.render import NamePlateTracker, render_pitch_surface  # noqa: E402


class TestPitchSurface:
    def test_surface_is_the_expected_size(self, rules):
        surf = render_pitch_surface(rules, 4.0)
        assert surf.get_width() == int(rules.length * 4.0)
        assert surf.get_height() == int(rules.width * 4.0)

    def test_padding_widens_the_surface(self, rules):
        plain = render_pitch_surface(rules, 4.0, pad_m=0.0)
        padded = render_pitch_surface(rules, 4.0, pad_m=3.0)
        assert padded.get_width() > plain.get_width()

    def test_markings_are_actually_drawn(self, rules):
        """The halfway line should make the centre column noticeably pale."""
        surf = render_pitch_surface(rules, 6.0)
        mid_x = surf.get_width() // 2
        column = [surf.get_at((mid_x, y))[:3] for y in range(10, surf.get_height() - 10, 7)]
        pale = sum(1 for c in column if min(c) > 150)
        assert pale > len(column) * 0.5

    def test_a_centre_logo_is_blended_in(self, rules):
        logo = pygame.Surface((64, 64), pygame.SRCALPHA)
        logo.fill((255, 0, 255, 255))
        plain = render_pitch_surface(rules, 6.0)
        marked = render_pitch_surface(rules, 6.0, centre_logo=logo)
        centre = (marked.get_width() // 2, marked.get_height() // 2)
        assert marked.get_at(centre) != plain.get_at(centre)

    def test_goals_can_be_omitted_for_the_3d_view(self, rules):
        with_goals = render_pitch_surface(rules, 6.0, pad_m=3.0, draw_goals=True)
        without = render_pitch_surface(rules, 6.0, pad_m=3.0, draw_goals=False)
        assert with_goals.get_size() == without.get_size()


class TestNamePlateTracker:
    """Guards the bug where only the goalkeeper was ever captioned."""

    def test_the_owner_is_captioned(self, live_match):
        m = live_match()
        tracker = NamePlateTracker()
        m.owner = m.players[3]
        player, alpha = tracker.update(m)
        assert player is m.players[3] and alpha == 1.0

    def test_the_last_toucher_lingers_after_losing_the_ball(self, live_match):
        m = live_match()
        tracker = NamePlateTracker()
        striker = m.players[3]
        m.owner = striker
        tracker.update(m)
        m.owner = None
        m.last_touch = striker
        m.tick += 1
        player, alpha = tracker.update(m)
        assert player is striker, "a shot should stay captioned with its striker"
        assert 0.0 < alpha <= 1.0

    def test_the_caption_expires(self, live_match):
        m = live_match()
        tracker = NamePlateTracker(linger=0.5, fade=0.2)
        m.owner = m.players[3]
        tracker.update(m)
        m.owner = None
        m.last_touch = m.players[3]
        m.tick += int(1.0 / m.rules.dt)
        assert tracker.update(m)[0] is None

    def test_a_deflection_moves_the_caption(self, live_match):
        m = live_match()
        tracker = NamePlateTracker()
        m.owner = m.players[3]
        tracker.update(m)
        m.owner = None
        m.last_touch = m.players[m.squad_size + 2]
        m.tick += 1
        assert tracker.update(m)[0] is m.players[m.squad_size + 2]

    def test_reset_drops_a_stale_player(self, live_match):
        m = live_match()
        tracker = NamePlateTracker()
        m.owner = m.players[3]
        tracker.update(m)
        tracker.reset()
        assert tracker.player is None


class TestAds:
    def test_placeholders_exist_and_one_animates(self):
        clips = placeholder_clips()
        assert clips
        assert any(c.animated for c in clips), "at least one board should move"
        assert all(c.frames[0].get_size() == BOARD_PX for c in clips)

    def test_a_still_clip_never_changes_frame(self):
        surf = pygame.Surface(BOARD_PX)
        clip = AdClip("still", [surf])
        assert clip.frame_index(0.0) == clip.frame_index(99.0) == 0
        assert not clip.animated

    def test_an_animated_clip_cycles_and_loops(self):
        frames = [pygame.Surface(BOARD_PX) for _ in range(3)]
        clip = AdClip("anim", frames, [0.1, 0.1, 0.1])
        assert clip.frame_index(0.05) == 0
        assert clip.frame_index(0.15) == 1
        assert clip.frame_index(0.25) == 2
        assert clip.frame_index(0.35) == 0, "should wrap around"

    def test_a_missing_folder_yields_nothing(self, tmp_path):
        assert load_ad_clips(tmp_path / "absent") == []

    def test_images_in_a_folder_are_loaded(self, tmp_path):
        surf = pygame.Surface((80, 40))
        surf.fill((10, 200, 10))
        pygame.image.save(surf, str(tmp_path / "board.png"))
        clips = load_ad_clips(tmp_path)
        assert len(clips) == 1
        assert clips[0].frames[0].get_size() == BOARD_PX, "letterboxed to board shape"

    def test_junk_files_are_skipped(self, tmp_path):
        (tmp_path / "notes.txt").write_text("not an image")
        (tmp_path / "broken.png").write_bytes(b"still not an image")
        assert load_ad_clips(tmp_path) == []


class TestGeometry:
    """Panda geometry, built without a window."""

    def _columns(self, node_path):
        """Vertex columns present, including on the node itself.

        findAllMatches only walks descendants, and these helpers return the
        GeomNode directly, so it has to be considered too.
        """
        from panda3d.core import GeomNode

        names = set()
        targets = list(node_path.findAllMatches("**/+GeomNode"))
        if isinstance(node_path.node(), GeomNode):
            targets.append(node_path)
        for np_ in targets:
            geom_node = np_.node()
            for i in range(geom_node.getNumGeoms()):
                fmt = geom_node.getGeom(i).getVertexData().getFormat()
                names.update(str(c.getName()) for c in fmt.getColumns())
        assert targets, "no geometry found to inspect"
        return names

    def test_generated_spheres_carry_normals(self):
        """Without normals, lighting silently does nothing and everything is flat.

        Panda's own models/misc/sphere has no normal column, which is exactly
        how that bug arrived in the first place.
        """
        from football.render3d import _make_sphere

        assert "normal" in self._columns(_make_sphere(1.0))

    def test_generated_cylinders_carry_normals(self):
        from football.render3d import _make_cylinder

        assert "normal" in self._columns(_make_cylinder(1.0, 2.0))

    def test_the_shadow_disc_fades_out_at_its_rim(self):
        """A CardMaker quad would be square; this must be a soft circle."""
        from panda3d.core import GeomVertexReader

        from football.render3d import _make_shadow_disc

        disc = _make_shadow_disc(1.0, segments=12, alpha=0.5)
        geom = disc.node().getGeom(0)
        reader = GeomVertexReader(geom.getVertexData(), "color")
        alphas = []
        while not reader.isAtEnd():
            alphas.append(reader.getData4f()[3])
        # colours are stored as bytes, so allow for 8-bit quantisation
        assert alphas[0] == pytest.approx(0.5, abs=0.01), "opaque at the centre"
        assert all(a == pytest.approx(0.0, abs=0.01) for a in alphas[1:]), "clear at the rim"

    def test_every_hairstyle_builds(self):
        from panda3d.core import NodePath

        from football.render3d import HAIR_STYLES, _build_hair, _material

        for style in HAIR_STYLES:
            head = NodePath("head")
            _build_hair(head, style, _material((0.2, 0.2, 0.2)), 0.33)
            children = head.getChildren().getNumPaths()
            assert children == 0 if style == "bald" else children > 0

    def test_hair_never_recolours_the_eyes(self):
        """The crest of this bug: sweeping the head's children painted the eyes."""
        from panda3d.core import NodePath

        from football.render3d import _build_hair, _make_sphere, _material

        head = NodePath("head")
        eye = _make_sphere(0.05)
        eye.reparentTo(head)
        eye.setMaterial(_material((0.0, 0.0, 0.0)), 1)
        before = eye.getMaterial()
        _build_hair(head, "afro", _material((0.9, 0.7, 0.2)), 0.33)
        assert eye.getMaterial() == before
