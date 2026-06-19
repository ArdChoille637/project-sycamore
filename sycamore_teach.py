"""
Project Sycamore — Manim Educational Animation

Teaches the simulation science top-down:
  Top level : the physical problem
  Middle    : the governing equations for each track
  Bottom    : how each equation maps to the Python code

Scenes
------
  S1_Title          – title card
  S2_Architecture   – three-track problem decomposition diagram
  S3_BEM            – blade-element momentum theory, full derivation + code
  S4_Boids          – modern Boids local rules, equations, live simulation

Render (start with low quality for fast iteration):
  ~/ds/bin/manim -pql sycamore_teach.py S1_Title
  ~/ds/bin/manim -pql sycamore_teach.py S2_Architecture
  ~/ds/bin/manim -pql sycamore_teach.py S3_BEM
  ~/ds/bin/manim -pql sycamore_teach.py S4_Boids

Render all at medium quality (≈ 5 min):
  ~/ds/bin/manim -pqm sycamore_teach.py S1_Title S2_Architecture S3_BEM S4_Boids
"""

from manim import *
import numpy as np
from scipy.optimize import brentq

# ── Palette ────────────────────────────────────────────────────────────────
C_BLUE   = "#4fc3f7"
C_GREEN  = "#66bb6a"
C_ORANGE = "#ffa726"
C_RED    = "#ef5350"
C_PURPLE = "#ab47bc"
C_GOLD   = "#f0c040"
C_GRAY   = "#8b949e"
C_CODE   = "#e6b450"   # amber for code snippets
BG       = "#0d1117"

config.background_color = BG

def label(text, color=WHITE, size=22):
    return Text(text, color=color, font_size=size)

def math(tex, color=WHITE, size=32):
    return MathTex(tex, color=color, font_size=size)

def code_line(src, size=17):
    return Text(src, font="Courier New", font_size=size, color=C_CODE)

def box(content, stroke_color=WHITE, fill_color=None, pad=0.25):
    rect = SurroundingRectangle(
        content, corner_radius=0.12, buff=pad,
        color=stroke_color, stroke_width=1.5,
        fill_color=fill_color or stroke_color,
        fill_opacity=0.08,
    )
    return VGroup(rect, content)


# ══════════════════════════════════════════════════════════════════════════════
class S1_Title(Scene):
    """Title card — ~14 s."""
    def construct(self):
        title = Text("Project Sycamore", font_size=58, color=C_BLUE)
        sub   = Text("A Top-Down Guide to the Simulation Science",
                     font_size=24, color=WHITE).next_to(title, DOWN, buff=0.5)

        tracks = VGroup(
            Text("BEM  Aerodynamics",      font_size=19, color=C_BLUE),
            Text("Rothermel  Fire Spread", font_size=19, color=C_RED),
            Text("Boids  Swarm Control",   font_size=19, color=C_GREEN),
        ).arrange(RIGHT, buff=1.0).next_to(sub, DOWN, buff=0.7)

        sep = Line(LEFT * 5, RIGHT * 5, stroke_width=0.6, color=C_GRAY)
        sep.next_to(sub, DOWN, buff=0.35)

        self.play(Write(title), run_time=1.4)
        self.play(FadeIn(sub, shift=DOWN * 0.15), run_time=0.8)
        self.play(Create(sep), run_time=0.6)
        self.play(LaggedStart(
            *[FadeIn(t, shift=UP * 0.1) for t in tracks], lag_ratio=0.35
        ), run_time=1.2)
        self.wait(2.5)
        self.play(FadeOut(Group(*self.mobjects)))


# ══════════════════════════════════════════════════════════════════════════════
class S2_Architecture(Scene):
    """Three-track decomposition diagram — ~28 s."""
    def construct(self):
        hdr = Text("Top-Down Problem Decomposition",
                   font_size=30, color=WHITE).to_edge(UP, buff=0.4)
        self.play(Write(hdr))

        # Level 0 — problem
        prob_txt = VGroup(
            Text("Persistent Wildfire", font_size=20, color=WHITE),
            Text("Front Monitoring",    font_size=20, color=WHITE),
        ).arrange(DOWN, buff=0.1)
        prob = box(prob_txt, stroke_color=WHITE).shift(UP * 2.4)

        self.play(Create(prob), run_time=0.8)
        self.wait(0.3)

        # Level 1 — three tracks
        def track_box(title, q, color):
            t = VGroup(
                Text(title, font_size=18, color=color),
                Text(q,     font_size=14, color=C_GRAY),
            ).arrange(DOWN, buff=0.1)
            return box(t, stroke_color=color, pad=0.3)

        b_aero  = track_box("Aerodynamics", "how does it fly?",       C_BLUE)
        b_fire  = track_box("Fire Dynamics","where is fire going?",    C_RED)
        b_swarm = track_box("Swarm Control","how do UAVs coordinate?", C_GREEN)

        row1 = VGroup(b_aero, b_fire, b_swarm).arrange(RIGHT, buff=0.6).shift(UP * 0.5)

        arrows_down = VGroup(*[
            Arrow(prob.get_bottom(), b.get_top(),
                  stroke_width=1.5, color=C_GRAY, buff=0.1,
                  tip_length=0.18)
            for b in row1
        ])
        self.play(LaggedStart(*[Create(b) for b in row1], lag_ratio=0.25), run_time=1.2)
        self.play(LaggedStart(*[GrowArrow(a) for a in arrows_down], lag_ratio=0.2))

        # Level 2 — model names
        def model_box(name, color):
            t = Text(name, font_size=20, color=color, weight=BOLD)
            return box(t, stroke_color=color, pad=0.25)

        m_bem  = model_box("BEM",       C_BLUE)
        m_roth = model_box("Rothermel", C_RED)
        m_boid = model_box("Boids",     C_GREEN)
        row2 = VGroup(m_bem, m_roth, m_boid).arrange(RIGHT, buff=0.6).shift(DOWN * 1.3)

        arrows2 = VGroup(*[
            Arrow(row1[i].get_bottom(), row2[i].get_top(),
                  stroke_width=1.5, color=C_GRAY, buff=0.1, tip_length=0.18)
            for i in range(3)
        ])
        self.play(LaggedStart(*[Create(b) for b in row2], lag_ratio=0.25), run_time=1.0)
        self.play(LaggedStart(*[GrowArrow(a) for a in arrows2], lag_ratio=0.2))

        # Level 3 — what the model gives us
        outputs = [
            ("T = W,  Q = 0",   C_BLUE),
            ("R = f(wind, M, slope)", C_RED),
            ("a_i = Σ w_k f_k", C_GREEN),
        ]
        out_boxes = VGroup()
        for i, (txt, col) in enumerate(outputs):
            t = Text(txt, font_size=13, color=col)
            b = box(t, stroke_color=col, pad=0.18)
            out_boxes.add(b)
        out_boxes.arrange(RIGHT, buff=0.6).shift(DOWN * 2.6)

        arrows3 = VGroup(*[
            Arrow(row2[i].get_bottom(), out_boxes[i].get_top(),
                  stroke_width=1.2, color=C_GRAY, buff=0.1, tip_length=0.15)
            for i in range(3)
        ])
        self.play(LaggedStart(*[FadeIn(b) for b in out_boxes], lag_ratio=0.25))
        self.play(LaggedStart(*[GrowArrow(a) for a in arrows3], lag_ratio=0.2))
        self.wait(2.5)
        self.play(FadeOut(Group(*self.mobjects)))


# ══════════════════════════════════════════════════════════════════════════════
class S3_BEM(Scene):
    """
    Blade-Element Momentum theory — full top-down derivation.

    Sections
    --------
      1. Disk view   : rotating disk → single blade element extracted
      2. Velocity triangle : Ut = Ωr, Ua = Vd → Veff at inflow angle φ
      3. Forces      : Lift ⊥ Veff, Drag ∥ Veff
      4. Equations   : dT/dr and dQ/dr written from geometry
      5. Equilibrium : T = W  and  Q = 0  shown with boxed residuals
      6. Code map    : annotated snippet of bem_forces() lines
    """

    # ── convenience helpers ────────────────────────────────────────────────
    def _section(self, text):
        t = Text(text, font_size=18, color=C_GOLD)
        t.to_corner(UR, buff=0.3)
        self.play(FadeIn(t, shift=LEFT * 0.1), run_time=0.4)
        return t

    def construct(self):
        # ── 1. TITLE ──────────────────────────────────────────────────────
        title = Text("Blade-Element Momentum Theory",
                     font_size=34, color=C_BLUE).to_edge(UP, buff=0.35)
        self.play(Write(title), run_time=1.0)

        # ── 2. SINGLE BLADE ELEMENT EXTRACTED ─────────────────────────────
        sec1 = self._section("1 · blade element")

        # Rotor disk (top-down view, simplified)
        disk_lbl = Text("rotating disk (top view)", font_size=14, color=C_GRAY)
        disk  = Ellipse(width=2.8, height=0.5, stroke_width=1.5,
                        stroke_color=C_GRAY, fill_opacity=0)
        disk_g = VGroup(disk, disk_lbl.next_to(disk, DOWN, buff=0.1))
        disk_g.shift(LEFT * 4 + UP * 0.8)

        blade_disk = Line(
            disk.get_left(), disk.get_right(),
            stroke_width=4, color=C_ORANGE
        )
        self.play(Create(disk_g), Create(blade_disk), run_time=0.8)
        self.wait(0.4)

        # Zoom arrow → side-view element
        zoom_arr = Arrow(disk.get_right() + RIGHT * 0.1, RIGHT * 0.8 + UP * 0.8,
                         stroke_width=1.5, color=C_GRAY, tip_length=0.2, buff=0.05)
        zoom_lbl = Text("side view at radius r", font_size=13, color=C_GRAY)
        zoom_lbl.next_to(zoom_arr.get_center(), UP, buff=0.08).shift(RIGHT * 0.1)
        self.play(GrowArrow(zoom_arr), Write(zoom_lbl), run_time=0.7)

        # Rotor plane (horizontal dashed line)
        rotor_plane = DashedLine(
            LEFT * 1.5, RIGHT * 3.5, stroke_width=1.0, color=C_GRAY
        ).shift(DOWN * 0.6)
        rp_lbl = Text("rotor plane", font_size=12, color=C_GRAY)
        rp_lbl.next_to(rotor_plane.get_left(), LEFT, buff=0.08)
        self.play(Create(rotor_plane), Write(rp_lbl), run_time=0.7)

        # Blade element at angle θ = 35°
        θ_deg = 35
        θ     = np.deg2rad(θ_deg)
        elem_cen = rotor_plane.get_left() + RIGHT * 2.2
        half  = 0.75 * np.array([np.cos(θ), np.sin(θ), 0])
        blade = Line(
            elem_cen - half, elem_cen + half,
            stroke_width=6, color=C_ORANGE
        )
        te_lbl = Text("TE", font_size=11, color=C_ORANGE).next_to(elem_cen - half, DL, buff=0.05)
        le_lbl = Text("LE", font_size=11, color=C_ORANGE).next_to(elem_cen + half, UR, buff=0.05)
        self.play(Create(blade), Write(te_lbl), Write(le_lbl), run_time=0.8)

        # θ arc and label
        ref_pt = elem_cen.copy(); ref_pt[1] = rotor_plane.get_y()
        θ_arc  = Arc(radius=0.45, start_angle=0, angle=θ,
                     stroke_width=1.5, color=C_ORANGE).move_arc_center_to(ref_pt)
        θ_lbl  = MathTex(r"\theta = 35°", font_size=20, color=C_ORANGE)
        θ_lbl.next_to(θ_arc, RIGHT, buff=0.1).shift(DOWN * 0.05)
        self.play(Create(θ_arc), Write(θ_lbl), run_time=0.7)
        self.wait(0.5)

        # ── 3. VELOCITY TRIANGLE ──────────────────────────────────────────
        self.play(FadeOut(sec1)); sec2 = self._section("2 · velocity triangle")

        # Origin for velocity vectors
        O = rotor_plane.get_left() + RIGHT * 0.3 + DOWN * 1.2
        Ut_mag, Ua_mag = 2.4, 0.9

        Ut_arr = Arrow(O, O + RIGHT * Ut_mag,
                       stroke_width=2.0, color=C_BLUE, tip_length=0.18, buff=0)
        Ua_arr = Arrow(O, O + UP * Ua_mag,
                       stroke_width=2.0, color=C_GREEN, tip_length=0.18, buff=0)
        Veff_tip = O + RIGHT * Ut_mag + UP * Ua_mag
        Veff_arr = Arrow(O, Veff_tip,
                         stroke_width=2.0, color=WHITE, tip_length=0.18, buff=0)

        Ut_lbl  = MathTex(r"U_t = \Omega r", font_size=18, color=C_BLUE)
        Ua_lbl  = MathTex(r"U_a = V_d",      font_size=18, color=C_GREEN)
        Veff_lbl = MathTex(r"V_{eff}",        font_size=18, color=WHITE)
        Ut_lbl.next_to(Ut_arr.get_center(), DOWN, buff=0.12)
        Ua_lbl.next_to(Ua_arr.get_center(), LEFT, buff=0.12)
        Veff_lbl.next_to(Veff_arr.get_center(), UL, buff=0.08).shift(RIGHT * 0.1)

        self.play(GrowArrow(Ut_arr), Write(Ut_lbl), run_time=0.9)
        self.play(GrowArrow(Ua_arr), Write(Ua_lbl), run_time=0.9)
        self.play(GrowArrow(Veff_arr), Write(Veff_lbl), run_time=0.9)

        # φ arc
        φ_ang = np.arctan2(Ua_mag, Ut_mag)
        φ_arc = Arc(radius=0.55, start_angle=0, angle=φ_ang,
                    stroke_width=1.5, color=C_PURPLE).move_arc_center_to(O)
        φ_lbl = MathTex(r"\varphi", font_size=18, color=C_PURPLE)
        φ_lbl.next_to(φ_arc.get_center(), RIGHT, buff=0.08)
        self.play(Create(φ_arc), Write(φ_lbl), run_time=0.6)

        eq_phi = MathTex(
            r"\varphi = \arctan\!\left(\frac{V_d}{\Omega\, r}\right)",
            font_size=26, color=C_PURPLE
        ).to_edge(RIGHT, buff=0.6).shift(UP * 2.0)
        self.play(Write(eq_phi), run_time=1.0)
        self.wait(0.4)

        # α arc (between blade and Veff)
        blade_ang = θ
        veff_ang  = φ_ang
        # α = θ - φ, drawn at blade element centre
        α_arc = Arc(radius=0.35, start_angle=veff_ang, angle=(blade_ang - veff_ang),
                    stroke_width=1.5, color=C_GOLD).move_arc_center_to(elem_cen - half * 0.9)
        α_lbl = MathTex(r"\alpha = \theta - \varphi", font_size=18, color=C_GOLD)
        α_lbl.next_to(α_arc, RIGHT, buff=0.08)
        self.play(Create(α_arc), Write(α_lbl), run_time=0.7)
        self.wait(0.5)

        # ── 4. FORCES ─────────────────────────────────────────────────────
        self.play(FadeOut(sec2)); sec3 = self._section("3 · forces")

        # Lift: perpendicular to Veff (rotate Veff direction by 90°)
        veff_dir = np.array([np.cos(φ_ang), np.sin(φ_ang), 0])
        lift_dir = np.array([-np.sin(φ_ang), np.cos(φ_ang), 0])  # ⊥ to Veff
        drag_dir = -veff_dir                                        # opposing

        lift_start = elem_cen
        lift_arr  = Arrow(lift_start, lift_start + lift_dir * 1.2,
                          stroke_width=2.0, color=C_BLUE, tip_length=0.18, buff=0)
        drag_arr  = Arrow(lift_start, lift_start + drag_dir * 0.6,
                          stroke_width=2.0, color=C_RED,  tip_length=0.18, buff=0)
        L_lbl = MathTex(r"L'", font_size=20, color=C_BLUE)
        D_lbl = MathTex(r"D'", font_size=20, color=C_RED)
        L_lbl.next_to(lift_arr.get_tip(), UL, buff=0.07)
        D_lbl.next_to(drag_arr.get_tip(), DR, buff=0.07)

        perp_note = Text("⊥ Veff", font_size=12, color=C_BLUE)
        para_note = Text("∥ Veff  (opposing)", font_size=12, color=C_RED)
        perp_note.next_to(L_lbl, RIGHT, buff=0.1)
        para_note.next_to(D_lbl, RIGHT, buff=0.1)

        self.play(GrowArrow(lift_arr), Write(L_lbl), run_time=0.8)
        self.play(Write(perp_note), run_time=0.4)
        self.play(GrowArrow(drag_arr), Write(D_lbl), run_time=0.8)
        self.play(Write(para_note), run_time=0.4)

        # Airfoil equations on right panel
        eq_L = MathTex(
            r"L' = \tfrac{1}{2}\rho V_{eff}^2\,c\,C_L(\alpha)",
            font_size=23, color=C_BLUE
        ).next_to(eq_phi, DOWN, buff=0.4, aligned_edge=LEFT)
        eq_D = MathTex(
            r"D' = \tfrac{1}{2}\rho V_{eff}^2\,c\,C_D(\alpha)",
            font_size=23, color=C_RED
        ).next_to(eq_L, DOWN, buff=0.25, aligned_edge=LEFT)

        self.play(Write(eq_L), run_time=1.0)
        self.play(Write(eq_D), run_time=1.0)
        self.wait(0.5)

        # ── 5. RESOLVE: dT/dr, dQ/dr ──────────────────────────────────────
        self.play(FadeOut(sec3)); sec4 = self._section("4 · integrate")

        eq_dT = MathTex(
            r"\frac{dT}{dr} = N\!\left(L'\!\cos\varphi - D'\!\sin\varphi\right)",
            font_size=23, color=C_GREEN
        ).next_to(eq_D, DOWN, buff=0.4, aligned_edge=LEFT)
        eq_dQ = MathTex(
            r"\frac{dQ}{dr} = Nr\!\left(L'\!\sin\varphi - D'\!\cos\varphi\right)",
            font_size=23, color=C_ORANGE
        ).next_to(eq_dT, DOWN, buff=0.25, aligned_edge=LEFT)

        self.play(Write(eq_dT), run_time=1.2)
        self.play(Write(eq_dQ), run_time=1.2)
        self.wait(0.4)

        # ── 6. EQUILIBRIUM ────────────────────────────────────────────────
        self.play(FadeOut(sec4)); sec5 = self._section("5 · equilibrium")

        eq_TW = MathTex(r"\int_0^R \frac{dT}{dr}\,dr = W",
                        font_size=24, color=C_GREEN)
        eq_Q0 = MathTex(r"\int_0^R \frac{dQ}{dr}\,dr = 0",
                        font_size=24, color=C_ORANGE)
        solve_note = Text("solve simultaneously → (V_d*, Ω*)",
                          font_size=15, color=C_GOLD)

        equil = VGroup(eq_TW, eq_Q0, solve_note).arrange(DOWN, buff=0.28)
        equil.next_to(eq_dQ, DOWN, buff=0.45, aligned_edge=LEFT)

        self.play(Write(eq_TW), run_time=0.9)
        self.play(Write(eq_Q0), run_time=0.9)
        self.play(Write(solve_note), run_time=0.7)

        # Box the equilibrium conditions
        self.play(
            Create(SurroundingRectangle(VGroup(eq_TW, eq_Q0),
                                        color=C_GOLD, buff=0.2, corner_radius=0.1)),
            run_time=0.6
        )
        self.wait(0.8)

        # ── 7. CODE MAP ───────────────────────────────────────────────────
        self.play(FadeOut(sec5)); sec6 = self._section("6 · code")

        # Fade out diagram, keep equations
        self.play(
            FadeOut(VGroup(disk_g, blade_disk, zoom_arr, zoom_lbl,
                           rotor_plane, rp_lbl, blade, te_lbl, le_lbl,
                           θ_arc, θ_lbl, Ut_arr, Ut_lbl, Ua_arr, Ua_lbl,
                           Veff_arr, Veff_lbl, φ_arc, φ_lbl, α_arc, α_lbl,
                           lift_arr, L_lbl, drag_arr, D_lbl,
                           perp_note, para_note)),
            run_time=0.5
        )

        # Code snippet
        code_lines = VGroup(
            code_line("# bem_forces(Vd, Omega):"),
            code_line("phi   = np.arctan2(Vd, Omega * r_e)      # inflow angle"),
            code_line("alpha = theta_e - phi                      # AoA"),
            code_line("cl, cd = cl_cd(alpha)                      # thin plate + Viterna"),
            code_line("q   = 0.5 * RHO * Veff**2 * c_e"),
            code_line("dT  = N_BL*np.sum((q*cl*cos(phi)"),
            code_line("                  -q*cd*sin(phi))*dr)      # → T=W"),
            code_line("dQ  = N_BL*np.sum(r_e*(q*cl*sin(phi)"),
            code_line("                      -q*cd*cos(phi))*dr)  # → Q=0"),
        ).arrange(DOWN, buff=0.08, aligned_edge=LEFT)
        code_lines.scale(0.85).to_edge(LEFT, buff=0.5).shift(UP * 0.5)

        code_bg = BackgroundRectangle(code_lines, fill_opacity=0.12,
                                      fill_color=WHITE, buff=0.2)
        self.play(FadeIn(code_bg), FadeIn(code_lines, shift=RIGHT * 0.1), run_time=1.0)

        # Arrows connecting equations to code
        ann1 = Arrow(eq_phi.get_left(), code_lines[1].get_right(),
                     color=C_PURPLE, stroke_width=1.3, tip_length=0.14,
                     buff=0.05, max_tip_length_to_length_ratio=0.3)
        ann2 = Arrow(eq_dT.get_left(), code_lines[5].get_right(),
                     color=C_GREEN, stroke_width=1.3, tip_length=0.14,
                     buff=0.05, max_tip_length_to_length_ratio=0.3)
        ann3 = Arrow(eq_dQ.get_left(), code_lines[7].get_right(),
                     color=C_ORANGE, stroke_width=1.3, tip_length=0.14,
                     buff=0.05, max_tip_length_to_length_ratio=0.3)

        self.play(GrowArrow(ann1), run_time=0.6)
        self.play(GrowArrow(ann2), run_time=0.6)
        self.play(GrowArrow(ann3), run_time=0.6)
        self.wait(2.5)
        self.play(FadeOut(Group(*self.mobjects)))


# ══════════════════════════════════════════════════════════════════════════════
class S4_Boids(Scene):
    """
    Modern Boids for wildfire UAV swarm.

    Sections
    --------
      1. Local-only world   : drone sees only within perception radius
      2. Rule 1 Separation  : inverse-square repulsion
      3. Rule 2 Alignment   : steer toward mean neighbour heading
      4. Rule 3 Cohesion    : drift toward neighbour centroid
      5. Rule 4 Fire track  : attract to nearest perimeter point
      6. Combined update    : weighted sum → velocity update
      7. Live simulation    : boids evolving on fire perimeter in Manim
    """

    # ── Boids physics (inline, Manim-scale units) ──────────────────────────
    FIRE_A, FIRE_B = 3.8, 2.0
    R_PERC, R_SEP  = 1.8, 0.6
    V_MAX, V_CRZ   = 1.0, 0.55
    W_SEP, W_ALI   = 2.5, 0.7
    W_COH, W_FIRE  = 0.5, 1.8
    SIGMA           = 0.18
    DT_SIM          = 0.07
    N_BOIDS         = 16

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        N_P = 300
        φ_p = np.linspace(0, 2*np.pi, N_P, endpoint=False)
        self._perim = np.column_stack([
            self.FIRE_A * np.cos(φ_p), self.FIRE_B * np.sin(φ_p)
        ])
        self._rng = np.random.default_rng(7)

    def _init_agents(self):
        N = self.N_BOIDS
        angles = np.linspace(0, 2*np.pi, N, endpoint=False)
        angles += self._rng.uniform(0, 2*np.pi/N, N)
        pos = np.column_stack([
            1.3 * self.FIRE_A * np.cos(angles),
            1.3 * self.FIRE_B * np.sin(angles),
        ])
        tan = np.column_stack([-np.sin(angles), np.cos(angles)])
        sgn = np.where(self._rng.random(N) > 0.5, 1., -1.)[:, None]
        vel = tan * sgn * self.V_CRZ * self._rng.uniform(0.7, 1.1, (N, 1))
        return pos, vel

    def _step(self, pos, vel):
        N = len(pos)
        disp = pos[None,:,:] - pos[:,None,:]
        dist = np.linalg.norm(disp, axis=2)
        np.fill_diagonal(dist, np.inf)

        perc = dist < self.R_PERC
        sep  = (dist < self.R_SEP) & (dist > 1e-5)
        nN   = perc.sum(axis=1).clip(min=1)

        sw   = np.where(sep, 1.0/dist**2, 0.0)
        f_sep = -np.einsum('ij,ijk->ik', sw, disp)

        v_hat = vel / np.linalg.norm(vel, axis=1, keepdims=True).clip(min=1e-5)
        f_ali = np.einsum('ij,jk->ik', perc.astype(float), v_hat) / nN[:,None]

        cen   = np.einsum('ij,jk->ik', perc.astype(float), pos) / nN[:,None]
        dcen  = np.where(nN[:,None]>0, cen-pos, 0.)
        f_coh = dcen / np.linalg.norm(dcen, axis=1, keepdims=True).clip(min=1e-5)

        dp   = np.linalg.norm(self._perim[None,:,:]-pos[:,None,:], axis=2)
        pstar = self._perim[np.argmin(dp, axis=1)]
        tp    = pstar - pos
        f_fire = tp / np.linalg.norm(tp, axis=1, keepdims=True).clip(min=1e-5)

        spd  = np.linalg.norm(vel, axis=1, keepdims=True).clip(min=1e-5)
        f_spd = v_hat * (self.V_CRZ - spd)

        accel = (self.W_SEP*f_sep + self.W_ALI*f_ali
                 + self.W_COH*f_coh + self.W_FIRE*f_fire + 0.3*f_spd)

        na  = self._rng.normal(0, self.SIGMA, N)
        cn, sn = np.cos(na), np.sin(na)
        rot = np.column_stack([vel[:,0]*cn-vel[:,1]*sn, vel[:,0]*sn+vel[:,1]*cn])

        vn  = vel + self.DT_SIM*accel + (rot-vel)
        spn = np.linalg.norm(vn, axis=1, keepdims=True).clip(min=1e-5)
        vn  = np.where(spn > self.V_MAX, vn/spn*self.V_MAX, vn)
        vn  = np.where(spn < 0.05,       vn/spn*0.05,       vn)
        return pos + self.DT_SIM*vn, vn

    # ── Construct ──────────────────────────────────────────────────────────
    def construct(self):
        # ── TITLE ─────────────────────────────────────────────────────────
        title = Text("Boids: Local Rules → Emergent Patrol",
                     font_size=30, color=C_GREEN).to_edge(UP, buff=0.35)
        self.play(Write(title))

        N_DEMO = 7  # small set for rule illustrations
        # Fixed positions so every demo (sep/ali/coh/fire) always has subjects:
        #  [0] = ego at origin, [1] very close (separation), [2-4] mid-range
        #  (perception), [5-6] far outside perception ring
        demo_pos = np.array([
            [ 0.0,  0.0],   # ego (C_GOLD)
            [ 0.28, 0.22],  # inside R_SEP — separation target
            [ 0.8,  0.5],   # inside R_PERC
            [-0.7,  0.85],  # inside R_PERC
            [ 1.1, -0.55],  # inside R_PERC
            [-2.0,  1.7],   # outside R_PERC
            [ 2.1, -1.4],   # outside R_PERC
        ])

        # ── 1. PERCEPTION CIRCLE ──────────────────────────────────────────
        sec = Text("1 · local world (perception radius)",
                   font_size=16, color=C_GOLD).to_corner(UR, buff=0.25)
        self.play(Write(sec))

        dots = VGroup(*[
            Dot(point=[*p, 0], radius=0.09,
                color=C_BLUE if i > 0 else C_GOLD)
            for i, p in enumerate(demo_pos)
        ])
        self.play(Create(dots), run_time=0.8)

        perc_circ = Circle(radius=self.R_PERC, stroke_width=1.2,
                           color=C_GOLD, stroke_opacity=0.7)
        perc_lbl  = MathTex(r"R_{perc}", font_size=18, color=C_GOLD)
        perc_lbl.next_to(perc_circ.get_right(), RIGHT, buff=0.1)

        note = Text("A drone only observes neighbours within R_perc",
                    font_size=14, color=C_GRAY).to_edge(DOWN, buff=0.4)
        self.play(Create(perc_circ), Write(perc_lbl))
        self.play(FadeIn(note, shift=UP * 0.1))
        self.wait(1.2)
        self.play(FadeOut(sec), FadeOut(note))

        # ── 2. SEPARATION ─────────────────────────────────────────────────
        sec = Text("Rule 1 · Separation — avoid crowding",
                   font_size=16, color=C_RED).to_corner(UR, buff=0.25)
        self.play(Write(sec))

        close_pos = demo_pos[
            np.linalg.norm(demo_pos - demo_pos[0], axis=1) < self.R_SEP
        ]
        sep_arrows = VGroup(*[
            Arrow(ORIGIN, [*(-(p - demo_pos[0])*0.8), 0],
                  stroke_width=2.0, color=C_RED, tip_length=0.17, buff=0)
            for p in close_pos if np.linalg.norm(p - demo_pos[0]) > 0.05
        ])
        eq_sep = MathTex(
            r"\mathbf{f}_{sep} = -\sum_{j\!:\,d<R_{sep}}"
            r"\frac{\mathbf{x}_j - \mathbf{x}_i}{\|\mathbf{x}_j - \mathbf{x}_i\|^2}",
            font_size=24, color=C_RED
        ).to_edge(RIGHT, buff=0.5).shift(UP * 0.5)

        if len(sep_arrows) > 0:
            self.play(LaggedStart(*[GrowArrow(a) for a in sep_arrows], lag_ratio=0.2))
        self.play(Write(eq_sep), run_time=1.2)
        self.wait(1.4)
        self.play(FadeOut(sep_arrows), FadeOut(sec))

        # ── 3. ALIGNMENT ──────────────────────────────────────────────────
        sec = Text("Rule 2 · Alignment — match heading",
                   font_size=16, color=C_BLUE).to_corner(UR, buff=0.25)
        self.play(Write(sec))

        rng3  = np.random.default_rng(55)
        vel_demo = rng3.uniform(-0.5, 0.5, (N_DEMO, 2))
        vel_demo /= np.linalg.norm(vel_demo, axis=1, keepdims=True)

        neigh_mask = np.linalg.norm(demo_pos - demo_pos[0], axis=1) < self.R_PERC
        neigh_mask[0] = False
        mean_dir = (vel_demo[neigh_mask].mean(axis=0)
                    if neigh_mask.any() else np.array([1.0, 0.0]))
        mean_dir /= max(np.linalg.norm(mean_dir), 1e-5)

        vel_arrows = VGroup(*[
            Arrow([*demo_pos[i], 0],
                  [*(demo_pos[i] + vel_demo[i]*0.5), 0],
                  stroke_width=1.8, color=C_BLUE if neigh_mask[i] else C_GRAY,
                  tip_length=0.14, buff=0)
            for i in range(N_DEMO)
        ])
        mean_arrow = Arrow(ORIGIN, [*mean_dir*0.7, 0],
                           stroke_width=2.5, color=C_GOLD, tip_length=0.2, buff=0)
        mean_lbl = MathTex(r"\bar{\hat{v}}", font_size=18, color=C_GOLD)
        mean_lbl.next_to(mean_arrow.get_tip(), UR, buff=0.08)

        eq_ali = MathTex(
            r"\mathbf{f}_{ali} = \frac{1}{|N|}\sum_{j\in N}\hat{\mathbf{v}}_j",
            font_size=24, color=C_BLUE
        ).next_to(eq_sep, DOWN, buff=0.4, aligned_edge=LEFT)

        self.play(Create(vel_arrows), run_time=0.7)
        self.play(GrowArrow(mean_arrow), Write(mean_lbl), run_time=0.8)
        self.play(Write(eq_ali), run_time=1.0)
        self.wait(1.4)
        self.play(FadeOut(vel_arrows), FadeOut(mean_arrow),
                  FadeOut(mean_lbl), FadeOut(sec))

        # ── 4. COHESION ───────────────────────────────────────────────────
        sec = Text("Rule 3 · Cohesion — drift to centroid",
                   font_size=16, color=C_GREEN).to_corner(UR, buff=0.25)
        self.play(Write(sec))

        neigh_pos = demo_pos[neigh_mask]
        centroid  = neigh_pos.mean(axis=0)
        cen_dot   = Dot(point=[*centroid, 0], radius=0.11,
                        color=C_GREEN, fill_opacity=0.8)
        cen_lbl   = MathTex(r"\bar{\mathbf{x}}_N", font_size=18, color=C_GREEN)
        cen_lbl.next_to(cen_dot, UR, buff=0.08)
        coh_arr   = Arrow(ORIGIN, [*((centroid-demo_pos[0])*0.85), 0],
                          stroke_width=2.5, color=C_GREEN, tip_length=0.2, buff=0)
        eq_coh = MathTex(
            r"\mathbf{f}_{coh} = \bar{\mathbf{x}}_N - \mathbf{x}_i",
            font_size=24, color=C_GREEN
        ).next_to(eq_ali, DOWN, buff=0.4, aligned_edge=LEFT)

        self.play(Create(cen_dot), Write(cen_lbl), run_time=0.7)
        self.play(GrowArrow(coh_arr), run_time=0.7)
        self.play(Write(eq_coh), run_time=0.9)
        self.wait(1.4)
        self.play(FadeOut(cen_dot), FadeOut(cen_lbl),
                  FadeOut(coh_arr), FadeOut(sec))

        # ── 5. FIRE TRACKING ──────────────────────────────────────────────
        sec = Text("Rule 4 · Fire tracking — nearest perimeter point",
                   font_size=16, color=C_RED).to_corner(UR, buff=0.25)
        self.play(Write(sec))

        perim_curve = ParametricFunction(
            lambda t: [self.FIRE_A*np.cos(t)*0.6, self.FIRE_B*np.sin(t)*0.6, 0],
            t_range=[0, 2*np.pi], color=C_RED, stroke_width=2.0
        )
        # Nearest point on this scaled perim to ORIGIN
        phi_arr = np.linspace(0, 2*np.pi, 200, endpoint=False)
        pscale = np.column_stack([
            self.FIRE_A*0.6*np.cos(phi_arr),
            self.FIRE_B*0.6*np.sin(phi_arr)
        ])
        nearest_idx = np.argmin(np.linalg.norm(pscale, axis=1))
        p_star = pscale[nearest_idx]

        p_dot  = Dot(point=[*p_star, 0], radius=0.1, color=C_RED)
        p_lbl  = MathTex(r"p^*", font_size=16, color=C_RED)
        p_lbl.next_to(p_dot, UR, buff=0.06)
        fire_arr = Arrow(ORIGIN, [*p_star*0.85, 0],
                         stroke_width=2.5, color=C_RED, tip_length=0.2, buff=0)
        eq_fire = MathTex(
            r"\mathbf{f}_{fire} = \frac{p^* - \mathbf{x}_i}{\|p^* - \mathbf{x}_i\|}",
            font_size=24, color=C_RED
        ).next_to(eq_coh, DOWN, buff=0.4, aligned_edge=LEFT)

        self.play(Create(perim_curve), run_time=0.8)
        self.play(Create(p_dot), Write(p_lbl), GrowArrow(fire_arr), run_time=0.8)
        self.play(Write(eq_fire), run_time=1.0)
        self.wait(1.4)
        self.play(FadeOut(p_dot), FadeOut(p_lbl),
                  FadeOut(fire_arr), FadeOut(sec))

        # ── 6. COMBINED UPDATE ────────────────────────────────────────────
        sec = Text("Combined update", font_size=16, color=C_GOLD).to_corner(UR, buff=0.25)
        self.play(Write(sec))

        eq_combo = MathTex(
            r"\mathbf{a}_i = w_1\mathbf{f}_{sep}"
            r"+ w_2\mathbf{f}_{ali}"
            r"+ w_3\mathbf{f}_{coh}"
            r"+ w_4\mathbf{f}_{fire}",
            font_size=23, color=C_GOLD
        )
        eq_update = MathTex(
            r"\mathbf{v}_i \leftarrow "
            r"\mathrm{clip}\!\left(\mathbf{v}_i + \Delta t\,\mathbf{a}_i + \eta,\;v_{max}\right)",
            font_size=23, color=WHITE
        )
        VGroup(eq_combo, eq_update).arrange(DOWN, buff=0.35).to_edge(DOWN, buff=0.65)

        self.play(Write(eq_combo), run_time=1.2)
        self.play(Write(eq_update), run_time=1.0)

        code_update = VGroup(
            code_line("accel = W_SEP*f_sep + W_ALI*f_ali"),
            code_line("      + W_COH*f_coh + W_FIRE*f_fire"),
            code_line("vel_new = clip(vel + DT*accel + noise, V_MAX)"),
            code_line("pos_new = pos + DT*vel_new"),
        ).arrange(DOWN, buff=0.08, aligned_edge=LEFT)
        code_update.scale(0.85).next_to(eq_combo, UP, buff=0.3).to_edge(LEFT, buff=0.5)

        self.play(FadeIn(code_update), run_time=0.8)
        self.wait(1.5)
        self.play(FadeOut(VGroup(sec, dots, perc_circ, perc_lbl,
                                  eq_sep, eq_ali, eq_coh, eq_fire,
                                  eq_combo, eq_update, code_update,
                                  perim_curve)))

        # ── 7. LIVE BOIDS SIMULATION ──────────────────────────────────────
        self.play(FadeOut(title))
        live_title = Text("Live swarm — fire perimeter patrol",
                          font_size=26, color=C_GREEN).to_edge(UP, buff=0.3)
        self.play(Write(live_title))

        # Fire perimeter
        fire_ell = ParametricFunction(
            lambda t: [self.FIRE_A*np.cos(t), self.FIRE_B*np.sin(t), 0],
            t_range=[0, 2*np.pi], color=C_RED, stroke_width=2.0
        )
        self.play(Create(fire_ell), run_time=0.8)

        # Initialise boids state
        pos_live, vel_live = self._init_agents()
        state = {'p': pos_live, 'v': vel_live}

        # Create dot mobjects
        live_dots = VGroup(*[
            Dot(point=[*pos_live[i], 0], radius=0.08, color=C_BLUE)
            for i in range(self.N_BOIDS)
        ])

        # Velocity arrow collection (quiver-style)
        live_arrs = VGroup(*[
            Arrow(ORIGIN, RIGHT*0.3, stroke_width=1.5,
                  color=C_GREEN, tip_length=0.13, buff=0)
            for _ in range(self.N_BOIDS)
        ])
        for i in range(self.N_BOIDS):
            live_arrs[i].move_to([*pos_live[i], 0])

        self.play(Create(live_dots), run_time=0.6)
        self.add(live_arrs)

        STEPS_PER_FRAME = 4

        def sim_advance(mob, dt_render):
            for _ in range(STEPS_PER_FRAME):
                state['p'], state['v'] = self._step(state['p'], state['v'])
            for i in range(self.N_BOIDS):
                live_dots[i].move_to([*state['p'][i], 0])
                # Update velocity arrow
                p2d = state['p'][i]
                v2d = state['v'][i]
                vn  = np.linalg.norm(v2d)
                if vn > 1e-4:
                    tip = p2d + v2d / vn * 0.35
                    live_arrs[i].put_start_and_end_on([*p2d, 0], [*tip, 0])

        live_dots.add_updater(sim_advance)
        self.add(live_dots)
        self.wait(10)    # run simulation for 10 seconds
        live_dots.clear_updaters()

        self.wait(0.5)
        self.play(FadeOut(Group(*self.mobjects)))
