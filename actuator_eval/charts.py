"""
Charts, as dependency-free SVG.

Every chart describes exactly ONE actuator configuration -- the same one the
text report describes. Overlaying several candidates on one axis made it
impossible to tell which duty-cycle points belonged to which configuration, so
comparisons are done by generating one chart set per configuration.

Three plots:

  1. Torque-speed envelope -- WHERE do this configuration's operating points
     sit relative to its limits? A table says "peak margin 2.8x"; the plot
     shows whether that margin is all at low speed and nearly gone at the top
     of the stroke. These are the points of ONE motion cycle, split by phase of
     the trapezoid; "duty" in the duty-factor sense belongs to chart 2 only.
  2. Thermal warm-up, for a FAMILY of duty cycles -- how hard can this
     configuration be worked before heat becomes the constraint? The duty
     cycles differ in rest time, which is the knob a designer actually has.
  3. Margin bars -- which constraint binds, at a glance.

Implementation note: draw calls are QUEUED and only converted to pixels in
render(), after the axis range is final. Computing pixels eagerly means
anything drawn before the last fit() lands against a stale scale.
"""

from __future__ import annotations
import math
import re
from typing import Callable, List, Optional, Sequence, Tuple

from . import physics as phys
from .evaluate import Evaluation, PASS, MARGINAL, FAIL

RPM = 30.0 / math.pi

# Okabe-Ito, colourblind safe.
SERIES = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9"]
INK = "#1a1a1a"
MUTED = "#6b6b6b"
GRID = "#dcdcdc"
STATUS_COLOR = {PASS: "#009E73", MARGINAL: "#E69F00", FAIL: "#D55E00"}

# Phases of the trapezoidal move, in legend order. Deliberately avoids the
# colours already spoken for on chart 1: SERIES[0] is the voltage envelope,
# SERIES[2] the continuous line, and #D55E00 the peak-demand marker.
#
# Accel and decel land in different torque bands, so each direction gets a
# light/dark pair of one hue: the hue says which stroke, the shade says which
# way the joint is being worked.
PHASE_STYLE = [
    ("Move_Pos_Accel", "#7C3AED"),   # violet, dark  -- upper band
    ("Move_Pos_Decel", "#C4A5F5"),   # violet, light -- lower band
    ("Move_Neg_Accel", "#0E7490"),   # teal,   dark  -- lower band
    ("Move_Neg_Decel", "#7DD3E8"),   # teal,   light -- upper band
    ("Cruise",         "#B45309"),   # at max_velocity, zero acceleration
    ("Dwell",          "#3a3a3a"),   # dwell_time rest, holding against gravity
]


def _esc(s) -> str:
    return (str(s).replace("&", "&amp;").replace("<", "&lt;")
            .replace(">", "&gt;").replace('"', "&quot;"))


def _fmt(v: float) -> str:
    if v == 0:
        return "0"
    a = abs(v)
    if a >= 1000:
        return f"{v:,.0f}"
    if a >= 10:
        return f"{v:.0f}"
    if a >= 1:
        return f"{v:.1f}"
    return f"{v:.2f}"


def _nice_ticks(lo: float, hi: float, target: int = 6) -> List[float]:
    if hi <= lo:
        hi = lo + 1.0
    raw = (hi - lo) / max(target, 2)
    mag = 10 ** math.floor(math.log10(raw)) if raw > 0 else 1.0
    for m in (1, 2, 2.5, 5, 10):
        if raw / mag <= m:
            step = m * mag
            break
    else:
        step = 10 * mag
    start = math.floor(lo / step) * step
    ticks, v = [], start
    while v <= hi + step * 0.5:
        if v >= lo - step * 1e-9:
            ticks.append(round(v, 10))
        v += step
    return ticks


class Plot:
    """Linear-axis SVG plotter with deferred rendering."""

    def __init__(self, width=680, height=400, title="", subtitle="",
                 xlabel="", ylabel="", margin=(66, 20, 54, 70)):
        self.w, self.h = width, height
        self.title, self.subtitle = title, subtitle
        self.xlabel, self.ylabel = xlabel, ylabel
        self.mt, self.mr, self.mb, self.ml = margin
        self._queue: List[Callable[[], str]] = []
        self.legend: List[Tuple[str, str, str]] = []
        self.xlo = self.xhi = self.ylo = self.yhi = None
        self.ymin_floor = None          # force the y axis to start here
        # Optional tick relabelling. A chart that plots a transformed value
        # (log10 of seconds, say) sets this to render the axis in the units the
        # reader thinks in. None means format the raw value as usual.
        self.xtick_fmt = None
        self.xticks = None              # force these tick positions
        self.footnote = ""              # one line under the plot

    # -- legend sizing ----------------------------------------------------
    def _legend_rows(self) -> int:
        """How many rows the legend wraps onto at the current plot width."""
        if not self.legend:
            return 0
        rows, x = 1, 0.0
        for label, _, _ in self.legend:
            w_item = 30 + 6.1 * len(label)
            if x > 0 and x + w_item > self.pw:
                rows, x = rows + 1, w_item
            else:
                x += w_item
        return rows

    def _grow_for_legend(self):
        """Reserve top margin for legend rows beyond the first."""
        extra = 15 * (self._legend_rows() - 1)
        if extra > 0:
            self.mt += extra

    # -- footnote sizing --------------------------------------------------
    def _footnote_lines(self) -> List[str]:
        """
        The footnote, wrapped to the plot width. These lines are long -- they
        carry a whole sentence of interpretation -- so without wrapping they run
        off the right edge, and without reserved height they land on top of the
        x-axis label.
        """
        if not self.footnote:
            return []
        budget = max(int(self.pw / 5.6), 20)     # ~5.6 px per char at 11px
        words, lines, cur = self.footnote.split(), [], ""
        for w in words:
            trial = f"{cur} {w}".strip()
            if cur and len(trial) > budget:
                lines.append(cur)
                cur = w
            else:
                cur = trial
        if cur:
            lines.append(cur)
        return lines

    def _footnote_h(self) -> int:
        n = len(self._footnote_lines())
        return 0 if n == 0 else 14 * n + 6

    def _grow_for_footnote(self):
        """Reserve bottom margin so the footnote never sits on the x label."""
        self.mb += self._footnote_h()

    # -- range ------------------------------------------------------------
    def fit(self, xs: Sequence[float] = (), ys: Sequence[float] = ()):
        xs = [x for x in xs if x is not None and math.isfinite(x)]
        ys = [y for y in ys if y is not None and math.isfinite(y)]
        if xs:
            self.xlo = min(xs) if self.xlo is None else min(self.xlo, min(xs))
            self.xhi = max(xs) if self.xhi is None else max(self.xhi, max(xs))
        if ys:
            self.ylo = min(ys) if self.ylo is None else min(self.ylo, min(ys))
            self.yhi = max(ys) if self.yhi is None else max(self.yhi, max(ys))

    def _finalise(self):
        if self.xlo is None:
            self.xlo, self.xhi = 0.0, 1.0
        if self.ylo is None:
            self.ylo, self.yhi = 0.0, 1.0
        if self.xhi - self.xlo < 1e-12:
            self.xhi = self.xlo + 1.0
        if self.yhi - self.ylo < 1e-12:
            self.yhi = self.ylo + 1.0
        self.yhi += 0.10 * (self.yhi - self.ylo)     # headroom for labels
        if self.ymin_floor is not None:
            self.ylo = self.ymin_floor
        elif 0 <= self.ylo < 0.35 * self.yhi:
            self.ylo = 0.0
        self.xhi += 0.02 * (self.xhi - self.xlo)

    @property
    def pw(self):
        return self.w - self.ml - self.mr

    @property
    def ph(self):
        return self.h - self.mt - self.mb

    def px(self, x):
        return self.ml + (x - self.xlo) / (self.xhi - self.xlo) * self.pw

    def py(self, y):
        return self.mt + self.ph - (y - self.ylo) / (self.yhi - self.ylo) * self.ph

    def _clampy(self, y):
        return min(max(y, self.mt), self.mt + self.ph)

    # -- deferred draw primitives -----------------------------------------
    def line(self, xs, ys, colour, label=None, width=2.2, dash=None, opacity=1.0):
        self.fit(xs, ys)
        data = list(zip(xs, ys))

        def draw():
            pts = [(self.px(x), self.py(y)) for x, y in data
                   if math.isfinite(x) and math.isfinite(y)]
            if len(pts) < 2:
                return ""
            d = " ".join(f"{x:.2f},{y:.2f}" for x, y in pts)
            da = f' stroke-dasharray="{dash}"' if dash else ""
            return (f'<polyline points="{d}" fill="none" stroke="{colour}" '
                    f'stroke-width="{width}" stroke-linejoin="round" '
                    f'stroke-linecap="round" opacity="{opacity}"{da}/>')
        self._queue.append(draw)
        if label:
            self.legend.append((label, colour, "dash" if dash else "line"))

    def area(self, xs, ys, colour, opacity=0.10):
        self.fit(xs, ys)
        data = list(zip(xs, ys))

        def draw():
            pts = [(self.px(x), self.py(y)) for x, y in data
                   if math.isfinite(x) and math.isfinite(y)]
            if len(pts) < 2:
                return ""
            base = self.py(self.ylo)
            d = (f"M {pts[0][0]:.2f},{base:.2f} "
                 + " ".join(f"L {x:.2f},{y:.2f}" for x, y in pts)
                 + f" L {pts[-1][0]:.2f},{base:.2f} Z")
            return f'<path d="{d}" fill="{colour}" opacity="{opacity}" stroke="none"/>'
        self._queue.append(draw)

    def scatter(self, xs, ys, colour, label=None, r=2.8, opacity=0.8,
                stroke="#ffffff", stroke_w=0.6, marker="dot"):
        """
        Point series. marker="square" distinguishes a second series by SHAPE as
        well as colour, which matters when two scatters share a plot and the
        reader may be colourblind or printing in mono.
        """
        self.fit(xs, ys)
        data = list(zip(xs, ys))

        def draw():
            out = []
            for x, y in data:
                if not (math.isfinite(x) and math.isfinite(y)):
                    continue
                cx, cy = self.px(x), self.py(y)
                if marker == "square":
                    s = r * 1.8
                    out.append(f'<rect x="{cx-s/2:.2f}" y="{cy-s/2:.2f}" '
                               f'width="{s:.2f}" height="{s:.2f}" '
                               f'fill="{colour}" opacity="{opacity}" '
                               f'stroke="{stroke}" stroke-width="{stroke_w}"/>')
                else:
                    out.append(f'<circle cx="{cx:.2f}" cy="{cy:.2f}" '
                               f'r="{r}" fill="{colour}" opacity="{opacity}" '
                               f'stroke="{stroke}" stroke-width="{stroke_w}"/>')
            return "".join(out)
        self._queue.append(draw)
        if label:
            self.legend.append((label, colour, marker))

    def marker(self, x, y, colour, label=None, r=5.5):
        self.fit([x], [y])

        def draw():
            cx, cy = self.px(x), self.py(y)
            pts = []
            for i in range(10):
                ang = -math.pi / 2 + i * math.pi / 5
                rad = r if i % 2 == 0 else r * 0.45
                pts.append(f"{cx + rad*math.cos(ang):.2f},{cy + rad*math.sin(ang):.2f}")
            return (f'<polygon points="{" ".join(pts)}" fill="{colour}" '
                    f'stroke="#ffffff" stroke-width="0.9"/>')
        self._queue.append(draw)
        if label:
            self.legend.append((label, colour, "star"))

    def hline(self, y, colour, label=None, dash="7 4", width=1.8):
        self.fit(ys=[y])

        def draw():
            return (f'<line x1="{self.ml}" y1="{self.py(y):.2f}" '
                    f'x2="{self.ml + self.pw}" y2="{self.py(y):.2f}" '
                    f'stroke="{colour}" stroke-width="{width}" '
                    f'stroke-dasharray="{dash}"/>')
        self._queue.append(draw)
        if label:
            self.legend.append((label, colour, "dash"))

    def text_at(self, x, y, s, colour=MUTED, anchor="start", dx=0, dy=-7,
                size=10.5, weight="400"):
        def draw():
            return (f'<text x="{self.px(x)+dx:.2f}" y="{self._clampy(self.py(y))+dy:.2f}" '
                    f'font-size="{size}" font-weight="{weight}" fill="{colour}" '
                    f'text-anchor="{anchor}">{_esc(s)}</text>')
        self._queue.append(draw)

    # -- render ------------------------------------------------------------
    def render(self) -> str:
        self._grow_for_legend()
        self._grow_for_footnote()
        self._finalise()
        o = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {self.w} {self.h}" '
             f'width="100%" style="max-width:{self.w}px;height:auto;font-family:'
             f'system-ui,-apple-system,sans-serif">',
             f'<rect width="{self.w}" height="{self.h}" fill="#ffffff"/>']

        for x in (self.xticks if self.xticks is not None
                  else _nice_ticks(self.xlo, self.xhi)):
            if not (self.xlo <= x <= self.xhi):
                continue
            o.append(f'<line x1="{self.px(x):.2f}" y1="{self.mt}" '
                     f'x2="{self.px(x):.2f}" y2="{self.mt+self.ph}" '
                     f'stroke="{GRID}" stroke-width="1"/>')
            _lab = self.xtick_fmt(x) if self.xtick_fmt else _fmt(x)
            o.append(f'<text x="{self.px(x):.2f}" y="{self.mt+self.ph+18}" '
                     f'font-size="11" fill="{MUTED}" text-anchor="middle">'
                     f'{_esc(_lab)}</text>')
        for y in _nice_ticks(self.ylo, self.yhi):
            if not (self.ylo <= y <= self.yhi):
                continue
            o.append(f'<line x1="{self.ml}" y1="{self.py(y):.2f}" '
                     f'x2="{self.ml+self.pw}" y2="{self.py(y):.2f}" '
                     f'stroke="{GRID}" stroke-width="1"/>')
            o.append(f'<text x="{self.ml-8}" y="{self.py(y)+4:.2f}" '
                     f'font-size="11" fill="{MUTED}" text-anchor="end">'
                     f'{_esc(_fmt(y))}</text>')

        o.append(f'<clipPath id="pa"><rect x="{self.ml}" y="{self.mt}" '
                 f'width="{self.pw}" height="{self.ph}"/></clipPath>')
        o.append('<g clip-path="url(#pa)">')
        for d in self._queue:
            o.append(d())
        o.append('</g>')

        o.append(f'<rect x="{self.ml}" y="{self.mt}" width="{self.pw}" '
                 f'height="{self.ph}" fill="none" stroke="{MUTED}" stroke-width="1"/>')
        if self.title:
            o.append(f'<text x="{self.ml}" y="20" font-size="14" '
                     f'font-weight="600" fill="{INK}">{_esc(self.title)}</text>')
        if self.subtitle:
            o.append(f'<text x="{self.ml}" y="36" font-size="11.5" '
                     f'fill="{MUTED}">{_esc(self.subtitle)}</text>')
        if self.xlabel:
            # The footnote, when there is one, owns the bottom of the frame --
            # _grow_for_footnote() has reserved room for it below this label.
            _xl_y = self.h - 12 - self._footnote_h()
            o.append(f'<text x="{self.ml+self.pw/2}" y="{_xl_y}" font-size="12" '
                     f'fill="{INK}" text-anchor="middle">{_esc(self.xlabel)}</text>')
        if self.ylabel:
            cy = self.mt + self.ph / 2
            o.append(f'<text x="17" y="{cy}" font-size="12" fill="{INK}" '
                     f'text-anchor="middle" transform="rotate(-90 17 {cy})">'
                     f'{_esc(self.ylabel)}</text>')

        if self.legend:
            # Wrap onto extra rows rather than running off the right edge: the
            # number of series is not known when the margins are chosen, so a
            # single row silently overflows as soon as labels get long. Rows
            # start high enough that the last one lands just above the plot;
            # _grow_for_legend() has already widened the top margin to suit.
            rows = self._legend_rows()
            x = self.ml
            y = self.mt - 10 - 15 * (rows - 1)
            for label, colour, kind in self.legend:
                w_item = 30 + 6.1 * len(label)
                if x > self.ml and x + w_item > self.ml + self.pw:
                    x, y = self.ml, y + 15
                if kind == "dot":
                    o.append(f'<circle cx="{x+6}" cy="{y-4}" r="3.6" fill="{colour}"/>')
                elif kind == "star":
                    o.append(f'<circle cx="{x+6}" cy="{y-4}" r="4.2" fill="{colour}"/>')
                elif kind == "square":
                    o.append(f'<rect x="{x+2.4}" y="{y-7.6}" width="7.2" '
                             f'height="7.2" fill="{colour}"/>')
                else:
                    da = ' stroke-dasharray="5 3"' if kind == "dash" else ""
                    o.append(f'<line x1="{x}" y1="{y-4}" x2="{x+16}" y2="{y-4}" '
                             f'stroke="{colour}" stroke-width="2.4"{da}/>')
                o.append(f'<text x="{x+22}" y="{y}" font-size="11" fill="{INK}">'
                         f'{_esc(label)}</text>')
                x += w_item
        _fl = self._footnote_lines()
        for _k, _line in enumerate(_fl):
            _y = self.h - 10 - 14 * (len(_fl) - 1 - _k)
            o.append(f'<text x="{self.ml}" y="{_y}" font-size="11" '
                     f'fill="{MUTED}">{_esc(_line)}</text>')
        o.append("</svg>")
        return "\n".join(o)


# ---------------------------------------------------------------------------
# Duty-cycle family
# ---------------------------------------------------------------------------

def _is_idle(seg) -> bool:
    return abs(seg[2]) < 1e-6


def duty_variants(ev: Evaluation, rest_scales=(0.0, 0.5, 1.0, 2.0)):
    """
    Build a family of duty cycles for ONE actuator configuration by varying the
    rest time between moves, which is the knob a designer actually has: the
    motion itself is set by the task, but how often it repeats is negotiable.

    Returns [(label, segments, duty_factor, is_nominal)] where segments are
    per-actuator output, as simulate_duty expects.

    Note that "rest" is not free: on a gravity-loaded joint the actuator still
    draws holding current at zero speed, so extra rest reduces heating but
    never to zero.
    """
    segs = ev.duty_segments
    if not segs:
        return []
    moving = [s for s in segs if not _is_idle(s)]
    idle = [s for s in segs if _is_idle(s)]
    t_move = sum(s[0] for s in moving)
    t_idle = sum(s[0] for s in idle)

    if not moving:
        return [("as specified", list(segs), 1.0, True)]

    # No rest in the nominal cycle: synthesise it by holding at the last torque.
    synth = not idle
    if synth:
        hold_tau = segs[-1][1]
        idle = [(t_move, hold_tau, 0.0)]
        t_idle = t_move
        rest_scales = (0.0, 0.25, 0.5, 1.0)

    out = []
    seen = set()
    for k in rest_scales:
        if k <= 0:
            new_idle = []
            t_new_idle = 0.0
        else:
            new_idle = [(s[0] * k, s[1], s[2]) for s in idle]
            t_new_idle = t_idle * k
        total = t_move + t_new_idle
        df = t_move / total if total > 0 else 1.0
        key = round(df, 4)
        if key in seen:
            continue
        seen.add(key)

        # keep the original ordering: moves and rests interleaved as authored
        merged = []
        for s in segs:
            if _is_idle(s):
                if k > 0:
                    merged.append((s[0] * k, s[1], s[2]))
            else:
                merged.append(s)
        if synth and k > 0:
            merged = list(segs) + list(new_idle)

        nominal = (abs(k - 1.0) < 1e-9) and not synth
        if k == 0:
            label = f"no rest ({df*100:.0f}% duty)"
        elif nominal:
            label = f"as specified ({df*100:.0f}% duty)"
        else:
            label = f"{k:g}x rest ({df*100:.0f}% duty)"
        out.append((label, merged, df, nominal))

    out.sort(key=lambda r: -r[2])           # hardest duty first
    return out


# ---------------------------------------------------------------------------
# Chart 1: torque-speed envelope, single configuration
# ---------------------------------------------------------------------------

def torque_speed(ev: Evaluation, width=1020, height=630) -> str:
    a, j = ev.actuator, ev.joint
    n = j.n_actuators
    cfg = f"{n}x {a.name}"
    gain = n * float(j.ratio) * float(j.ratio_eff)
    v_bus = j.bus_v(a)

    p = Plot(width, height,
             title="Torque-speed envelope at the joint",
             subtitle=f"{cfg}  |  {v_bus:.0f} V bus  |  "
                      f"{j.T_ambient:.0f} degC ambient  |  {j.name}",
             xlabel="joint speed (rpm)", ylabel="joint torque (N.m)")
    p.ymin_floor = 0.0

    # The envelope that actually drives the verdicts. When the actuator has a
    # measured T-N curve this IS that curve; otherwise it is computed from Ke,
    # R_phase and L_phase.
    curve, scaled = phys.tn_curve_for(a, v_bus)
    w_max_out = phys.no_load_speed_out(a, v_bus)
    xs, ys = [], []
    for k in range(161):
        w_out = w_max_out * k / 160.0
        xs.append(w_out / max(float(j.ratio), 1e-9) * RPM)
        ys.append(phys.max_torque_at_speed(a, w_out, v_bus) * gain)
    p.area(xs, ys, SERIES[0], opacity=0.09)
    p.line(xs, ys, SERIES[0],
           "peak (measured curve)" if curve else "peak (voltage limited)",
           width=2.4)

    # With a measured curve driving the envelope, plot the electrical model
    # alongside it. The two are derived independently, so the gap between them is
    # a readable diagnostic rather than decoration: a model that tracks the
    # measurement validates R_phase and L_phase, and one that does not shows at
    # a glance WHERE it fails -- high-speed divergence implicates L_phase,
    # a uniform offset implicates the current limit or Kt.
    if curve:
        mxs, mys = [], []
        for k in range(161):
            w_out = w_max_out * k / 160.0
            mxs.append(w_out / max(float(j.ratio), 1e-9) * RPM)
            mys.append(phys.modelled_torque_at_speed(a, w_out, v_bus) * gain)
        p.line(mxs, mys, SERIES[3], "our model (cross-check)",
               dash="6 4", width=1.8)
        # The measured points themselves, so digitisation is visible as data
        # rather than implied by a smooth line.
        #
        # NOTE the unit change: the lines above are built from w_out in rad/s and
        # so need the RPM factor, but TNCurve.points and tn_crosscheck() rows are
        # ALREADY in output rpm. Applying RPM here too would scale them by 9.55.
        p.scatter([s / max(float(j.ratio), 1e-9) for s, _ in curve.points],
                  [t * gain for _, t in curve.points],
                  SERIES[0], "vendor data points", r=3.4, opacity=0.95)
        rows, rms = phys.tn_crosscheck(a, v_bus)
        if rows and rms and rms > 0.25:
            # Anchor the callout at the worst disagreement. rows are in rpm.
            wr, wv, wm, _ = max(rows, key=lambda r: abs(r[3]))
            p.text_at(wr / max(float(j.ratio), 1e-9),
                      max(wv, wm) * gain,
                      f"model off by {rms*100:.0f}% RMS -- see report",
                      SERIES[3], dx=9, dy=-9, anchor="start", weight="600")

    tc = ev.extras.get("tau_cont_cap_joint")
    if tc:
        p.hline(tc, SERIES[2], "continuous (thermal)", dash="7 4", width=1.9)

    segs = getattr(ev, "joint_segments", None) or []
    if segs:
        # Operating points over ONE motion cycle -- not a duty percentage.
        # Both torque and speed are magnitudes, so accel and decel of the SAME
        # stroke would otherwise land in different bands under one label: with
        # inertia dominating gravity, what sets the band is whether the joint is
        # speeding up or slowing down. Hence accel/decel x direction, split out.
        phases = getattr(ev, "motion_phases", None) or []
        if len(phases) == len(segs):
            for name, colour in PHASE_STYLE:
                idx = [k for k, ph in enumerate(phases) if ph == name]
                if not idx:
                    continue
                p.scatter([abs(segs[k][2]) * RPM for k in idx],
                          [abs(segs[k][1]) for k in idx],
                          colour, name, r=2.7, opacity=0.72)
        else:
            # No trajectory to classify (explicit duty_segments in the JSON).
            p.scatter([abs(w) * RPM for _, _, w in segs],
                      [abs(t) for _, t, _ in segs],
                      "#3a3a3a", "operating points", r=2.7, opacity=0.62)
        i = max(range(len(segs)), key=lambda k: abs(segs[k][1]))
        p.marker(abs(segs[i][2]) * RPM, abs(segs[i][1]), "#D55E00",
                 "peak demand", r=6.0)
        p.text_at(abs(segs[i][2]) * RPM, abs(segs[i][1]),
                  f"{abs(segs[i][1]):.1f} N.m @ {abs(segs[i][2])*RPM:.0f} rpm",
                  "#D55E00", dx=9, dy=-9, anchor="start", weight="600")

    if tc:
        p.text_at(0, tc, f"  {tc:.1f} N.m continuous", SERIES[2], dy=-6,
                  weight="600")
    return p.render()


# ---------------------------------------------------------------------------
# Chart 2: thermal warm-up across the duty-cycle family
# ---------------------------------------------------------------------------

def thermal(ev: Evaluation, width=1020, height=630) -> str:
    a, j = ev.actuator, ev.joint
    n = j.n_actuators
    cfg = f"{n}x {a.name}"
    limit = float(a.T_winding_max)
    t_amb = j.T_ambient

    p = Plot(width, height,
             title="Winding temperature from cold, across duty cycles",
             subtitle=f"{cfg}  |  per actuator  |  same motion, varying rest time",
             xlabel="time (minutes)", ylabel="winding temperature (degC)")
    p.ymin_floor = min(t_amb - 4, t_amb * 0.9)

    variants = duty_variants(ev)
    ceiling = limit * 1.6
    duration = None

    for i, (label, segs, df, nominal) in enumerate(variants):
        c = SERIES[i % len(SERIES)]
        ts, tw, tc = phys.warmup_curve(a, segs, t_amb, duration=duration)
        if duration is None:
            duration = ts[-1]
        xm = [t / 60.0 for t in ts]
        tw = [(v if (math.isfinite(v) and v <= ceiling) else float("nan"))
              for v in tw]
        p.line(xm, tw, c, label, width=2.6 if nominal else 1.9,
               dash=None if nominal else "5 3", opacity=1.0 if nominal else 0.9)

        # where does this duty cross the insulation limit, if it does?
        for k in range(1, len(tw)):
            if math.isfinite(tw[k]) and tw[k] >= limit > tw[k - 1]:
                p.marker(xm[k], limit, c, None, r=4.6)
                p.text_at(xm[k], limit, f" {xm[k]:.0f} min", c, dy=14,
                          size=10, weight="600")
                break

    p.hline(limit, "#D55E00", f"insulation limit ({limit:.0f} degC)",
            dash="8 4", width=2.0)
    p.hline(t_amb, MUTED, f"ambient ({t_amb:.0f} degC)", dash="2 4", width=1.2)
    return p.render()


# ---------------------------------------------------------------------------
# Chart 3: overload endurance -- measurement, model, and this duty's demands
# ---------------------------------------------------------------------------

def overload_endurance(ev: Evaluation, condition: str = "rotating",
                       width=1020, height=630) -> str:
    """
    How long the actuator holds a given torque: what the model predicts, what
    the vendor measured if they published it, and where this duty cycle sits.

    Each layer answers a different question. The model line and the duty
    exposure curve together say whether this application is anywhere near the
    thermal limit -- the question most users actually have, and one that needs
    no vendor data. Vendor points, when they exist, add the second question:
    whether the model deserves to be believed, since Rth_wc, Rth_ca, C_w and C_c
    are otherwise four estimates with nothing checking them.

    The vendor layer is therefore OPTIONAL, not a precondition. Drawing this
    only for actuators with a published table would hide it exactly where the
    model is least verified, which is backwards. Where there is no measurement
    the chart says so plainly rather than implying the model has been checked.

    Returns "" only for the stall condition with no stall table -- there is no
    "modelled stall endurance" worth drawing on its own, because the two-node
    model cannot represent single-phase stall heating in the first place.
    """
    a, j = ev.actuator, ev.joint
    curve = a.overload_curve(condition)
    if curve:
        rows, log_rms = phys.overload_crosscheck(a, condition)
    else:
        rows, log_rms = None, None
        # Without a table there is nothing to anchor a stall plot to, and the
        # model is known not to represent stall heating; skip rather than draw
        # a line that would be believed.
        if condition != "rotating":
            return ""

    # Time spans decades, so it goes on a log axis. Plot log10(seconds) and
    # relabel the ticks back into seconds; that keeps the shared Plot linear.
    def lg(s):
        return math.log10(max(float(s), 1e-3))

    def fmt_t(v):
        # The last tick carries the continuous-rating point, whose claim is
        # "indefinitely", not "28 hours". Labelling it with a finite duration
        # would misread the one point on this chart that is explicitly an
        # asymptote.
        if v >= 5.0 - 1e-9:
            return "indefinite"
        s = 10.0 ** v
        if s >= 3600:
            return f"{s/3600:.0f} h"
        if s >= 60:
            return f"{s/60:.0f} min"
        return f"{s:.0f} s" if s >= 1 else f"{s:.1f} s"

    cfg = f"{j.n_actuators}x {a.name}"

    # Joint-referred, like charts 1 and 2, so a reader comparing them is not
    # silently switching frames. Chart 1's continuous line is a joint torque;
    # plotting this one per-actuator made the same actuator look half as
    # capable here as there (9.2 N.m vs 5.0 N.m for the 2x RS00 elbow) with
    # nothing on either chart to say why.
    #
    # But everything BELOW this line is per-actuator physics: the vendor
    # measured one unit, and time_to_thermal_limit integrates one winding. So
    # gain is applied only where a torque crosses into or out of plot space --
    # never inside the thermal integration. Scaling the model's OUTPUT instead
    # of its INPUT would silently model a single actuator carrying the whole
    # joint load, which is the one thing this chart must not imply.
    gain = j.n_actuators * float(j.ratio) * float(j.ratio_eff)

    # Test conditions: the vendor's where there is a table, otherwise this
    # application's, since that is then what the model is being run at.
    if curve:
        test_rpm = curve.speed_rpm_output or 0.0
        test_amb = curve.ambient_C if curve.ambient_C is not None else 25.0
        cond_bits = []
        if curve.speed_rpm_output is not None:
            cond_bits.append("stalled" if curve.speed_rpm_output == 0
                             else f"{curve.speed_rpm_output:.0f} rpm")
        if curve.ambient_C is not None:
            cond_bits.append(f"{curve.ambient_C:.0f} degC")
        if curve.mounting:
            cond_bits.append(curve.mounting)
        where = "vendor test: " + ", ".join(cond_bits)
        title = f"Overload endurance ({condition}): measured vs model"
    else:
        test_rpm = 0.0
        test_amb = j.T_ambient
        where = (f"modelled at {test_amb:.0f} degC ambient, "
                 f"{j.mounting} -- no vendor measurement to check it")
        title = "Overload endurance (modelled -- UNVERIFIED)"

    # The frame note earns its place only when there is more than one actuator:
    # at n=1 joint torque and actuator torque are the same number and the
    # caveat would be noise.
    frame = ("joint torque; actuators assumed to be loaded identically"
             if j.n_actuators > 1 else "per actuator")

    p = Plot(width, height, title=title,
             subtitle=f"{cfg}  |  {frame}  |  {where}",
             xlabel="time  (curve/points: how long the torque is held before the "
                    "winding hits its limit  |  squares: one duty cycle period)",
             ylabel="joint torque (N.m)" if j.n_actuators > 1
                    else "output torque (N.m)")
    p.xtick_fmt = fmt_t
    p.xticks = [float(k) for k in range(-1, 6)]

    # Torque range to sweep: the measured span where there is one, otherwise
    # from the continuous limit up to the actuator's peak, which is the range
    # over which "how long can I hold this?" is a meaningful question.
    if curve:
        lo, hi = curve.torque_range
    else:
        lo = phys.continuous_torque_limit(
            a, test_amb, omega_rotor=test_rpm * math.pi / 30.0 * float(a.gear_ratio))
        hi = float(a.tau_peak_out_spec) if a.tau_peak_out_spec else lo * 2.5
        lo = min(lo * 1.02, hi * 0.98)      # just above continuous: finite time

    # lo/hi are per-actuator torques and stay that way through the integration;
    # only the plotted value is scaled to the joint.
    taus, ts = [], []
    for k in range(41):
        tau = lo + (hi - lo) * k / 40.0
        secs = phys.time_to_thermal_limit(
            a, tau, test_rpm * math.pi / 30.0, test_amb)
        if math.isfinite(secs) and secs > 0:
            taus.append(tau * gain)
            ts.append(lg(secs))
    if ts:
        p.line(ts, taus, SERIES[3],
               "our model (cross-check)" if curve else "our model (unverified)",
               width=2.2, dash="7 4")

    if curve:
        # The vendor measured ONE actuator. Plotting n x that is a statement
        # about the pair that no one tested -- it holds only if the load really
        # does split evenly and both units heat alike, which is what the
        # subtitle declares. The legend repeats it here so the claim travels
        # with the points rather than living only in the header.
        p.scatter([lg(s) for _, s in curve.points],
                  [t * gain for t, _ in curve.points],
                  SERIES[0],
                  f"vendor measured ({curve.source})" + (
                      f", x{j.n_actuators}" if j.n_actuators > 1 else ""),
                  r=4.0, opacity=0.95)

    # The rated torque is the ASYMPTOTE of this same curve, not an independent
    # reference level: the vendor's tables literally end "...6 N.m for 285 s,
    # 5 rated". Drawing it as a horizontal line spanning the whole time axis
    # claimed something false at the left -- that 5 N.m is the limit at one
    # second, when 5 N.m for a second is trivially fine -- and competed visually
    # with the curve it terminates. So it goes at the far right of the time
    # axis, where "indefinitely" belongs, in the same colour as the measured
    # points it continues.
    rated_per_act = (curve.rated_torque if curve and curve.rated_torque
                     else (float(a.tau_cont_out_spec)
                           if a.tau_cont_out_spec else None))
    if rated_per_act:
        rated = rated_per_act * gain
        x_inf = p.xticks[-1] if p.xticks else lg(1e5)
        is_measured = bool(curve and curve.rated_torque)
        col = SERIES[0] if is_measured else SERIES[2]
        # A dotted lead-in from the last measured point: the vendor tested
        # neither this stretch nor the asymptote's approach, so it is drawn as
        # explicitly unmeasured rather than interpolated.
        if curve and curve.points:
            t_last, s_last = curve.points[0]
            p.line([lg(s_last), x_inf], [t_last * gain, rated], col, None,
                   width=1.4, dash="2 4", opacity=0.75)
        p.scatter([x_inf], [rated], col,
                  ("vendor rated, continuous (held indefinitely)" if is_measured
                   else "vendor rated, continuous (spec table)"),
                  r=4.6, opacity=0.95)
        p.text_at(x_inf, rated, f"{rated:.1f} N.m  ", col, anchor="end",
                  dy=-9, size=10, weight="600")

    # The application, as one point per duty variant -- the same family chart 2
    # sweeps -- at (cycle period, RMS-equivalent torque).
    #
    # RMS is the only torque that compares honestly against the two reference
    # lines here. Both the vendor rated line and the endurance curve describe
    # SUSTAINED operation, so plotting the profile's instantaneous peak against
    # them invites a comparison that does not hold: the elbow touches 5.1 N.m
    # for 23 ms of every 1.6 s, which sits right on the 5.0 N.m rated line and
    # reads as marginal, while its RMS-equivalent torque is 2.3 N.m -- half the
    # continuous limit. An earlier version of this chart drew a cumulative
    # "time at or above" curve and made exactly that error.
    #
    # One point per variant also answers "what if I ran this harder?", which a
    # single point cannot, and lands the whole family on the axis the reader is
    # already reading the reference lines against.
    variants = duty_variants(ev)
    vx, vy, vlab = [], [], []
    for label, segs, df, nominal in variants:
        period = sum(float(s[0]) for s in segs)
        if period <= 0:
            continue
        i_rms = phys.rms_current_of_duty(a, segs, t_est=float(a.T_winding_max))
        # segs are per-actuator output (see duty_variants), so torque_out gives
        # a per-actuator torque; scale to the joint for plotting.
        tau_rms = abs(a.torque_out(i_rms, t_magnet=float(a.T_winding_max)))
        if tau_rms <= 0:
            continue
        vx.append(lg(period))
        vy.append(tau_rms * gain)
        vlab.append((label, nominal, df))
    if vx:
        p.scatter(vx, vy, SERIES[1],
                  "application profile (RMS-equivalent, per cycle)",
                  r=5.0, opacity=0.95, marker="square")
        for k, (label, nominal, df) in enumerate(vlab):
            if nominal:
                p.text_at(vx[k], vy[k],
                          f"  as specified: {df*100:.0f}% duty, "
                          f"{vy[k]:.2f} N.m RMS over {10**vx[k]:.2f} s",
                          SERIES[1], dy=-9, size=10, weight="600")
            else:
                p.text_at(vx[k], vy[k], f"  {df*100:.0f}%", SERIES[1],
                          dy=-8, size=9)

    # The instantaneous peak is a different question -- can it be reached at all,
    # not can it be held -- so it is annotated, not drawn on the endurance axis.
    peaks = [abs(float(s[1])) for s in (ev.duty_segments or [])
             if float(s[0]) > 0]
    if peaks and vx:
        p.footnote_extra = (
            f"profile peaks at {max(peaks) * gain:.1f} N.m instantaneously; "
            f"that is a torque-capability question (chart 1), not an "
            f"endurance one")

    # One line of plain language about what the comparison means. Two separate
    # things matter and they are easy to conflate: whether the model agrees with
    # the measurement, and whether this duty is anywhere near either of them.
    bits = []
    fin = [r[3] for r in (rows or []) if math.isfinite(r[3]) and r[3] > 0]
    if fin:
        gm = math.exp(sum(math.log(r) for r in fin) / len(fin))
        if gm > 1.25:
            bits.append(f"model outlasts the measurement by {gm:.1f}x "
                        f"(OPTIMISTIC -- treat the model line as an upper bound)")
        elif gm < 0.8:
            bits.append(f"model is {1/gm:.1f}x conservative against the "
                        f"measurement (verdicts safe, capability may be unused)")
        else:
            bits.append(f"model tracks the measurement within "
                        f"{abs(gm-1)*100:.0f}%")
    elif not curve:
        bits.append("no vendor endurance data for this actuator -- the curve "
                    "above is the model alone, resting on four estimated "
                    "thermal parameters. Treat it as indicative, not a rating")
    if vy:
        worst_rms = max(vy)                      # already joint-referred
        lo_t = (curve.torque_range[0] if curve else lo) * gain
        if worst_rms < lo_t:
            bits.append(f"even at 100% duty this profile is {worst_rms:.1f} N.m "
                        f"RMS, below the {lo_t:.1f} N.m where "
                        + ("the overload data starts" if curve
                           else "the actuator stops being continuous-rated")
                        + " -- sustained load is not the binding concern here")
    if getattr(p, "footnote_extra", ""):
        bits.append(p.footnote_extra)
    p.footnote = ";  ".join(bits)
    return p.render()


# ---------------------------------------------------------------------------
# Chart 4: margins
# ---------------------------------------------------------------------------

def margins(ev: Evaluation, width=1020) -> str:
    crits = [c for c in ev.criteria if c.margin_meaningful]
    row_h, ml, mr = 26, 196, 74
    height = 82 + row_h * len(crits)
    pw = width - ml - mr
    x_max = 3.0
    cfg = f"{ev.joint.n_actuators}x {ev.actuator.name}"

    def bx(v):
        return ml + min(max(v, 0.0), x_max) / x_max * pw

    o = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
         f'width="100%" style="max-width:{width}px;height:auto;font-family:'
         f'system-ui,-apple-system,sans-serif">',
         f'<rect width="{width}" height="{height}" fill="#ffffff"/>',
         f'<text x="14" y="21" font-size="14" font-weight="600" fill="{INK}">'
         f'Margin by criterion (capability / demand)</text>',
         f'<text x="14" y="37" font-size="11.5" fill="{MUTED}">{_esc(cfg)}</text>']

    top, bot = 50, height - 26
    for g in (0.5, 1.0, 1.5, 2.0, 2.5, 3.0):
        o.append(f'<line x1="{bx(g):.1f}" y1="{top}" x2="{bx(g):.1f}" y2="{bot}" '
                 f'stroke="{GRID}" stroke-width="1"/>')
        o.append(f'<text x="{bx(g):.1f}" y="{height-9}" font-size="10" '
                 f'fill="{MUTED}" text-anchor="middle">{g:g}x</text>')

    y = top + 6
    for c in crits:
        m = c.margin if math.isfinite(c.margin) else x_max
        col = STATUS_COLOR.get(c.status, MUTED)
        o.append(f'<text x="{ml-10}" y="{y+12}" font-size="11.5" fill="{INK}" '
                 f'text-anchor="end">{_esc(c.name)}</text>')
        o.append(f'<rect x="{ml}" y="{y+2}" width="{max(bx(m)-ml,1.5):.1f}" '
                 f'height="15" fill="{col}" opacity="0.85" rx="2"/>')
        txt = "no limit" if not math.isfinite(c.margin) else f"{c.margin:.2f}x"
        o.append(f'<text x="{min(bx(m)+7, width-6):.1f}" y="{y+14}" font-size="11" '
                 f'fill="{MUTED}">{txt}</text>')
        y += row_h

    o.append(f'<line x1="{bx(1.0):.1f}" y1="{top}" x2="{bx(1.0):.1f}" y2="{bot}" '
             f'stroke="{INK}" stroke-width="1.8"/>')
    o.append(f'<text x="{bx(1.0)+5:.1f}" y="{top-4}" font-size="10" fill="{INK}">'
             f'1.0x = just adequate</text>')
    o.append("</svg>")
    return "\n".join(o)


# ---------------------------------------------------------------------------
# HTML wrapper -- one configuration per file
# ---------------------------------------------------------------------------

_CSS = """
body{font-family:system-ui,-apple-system,sans-serif;margin:0;padding:32px;
background:#fff;color:#1a1a1a;line-height:1.5}
.wrap{max-width:1140px;margin:0 auto}
h1{font-size:21px;margin:0 0 4px}
h2{font-size:15px;margin:34px 0 10px;font-weight:600}
.sub{color:#6b6b6b;font-size:13px;margin-bottom:26px}
.verdict{display:inline-block;padding:3px 11px;border-radius:3px;font-size:12px;
font-weight:600;color:#fff;letter-spacing:.02em}
figure{margin:0 0 8px}
figcaption{font-size:12px;color:#6b6b6b;margin-top:6px}
pre{background:#f6f6f4;padding:14px;border-radius:4px;font-size:11.5px;
overflow-x:auto;border:1px solid #e6e6e2;line-height:1.45}
"""


def _endurance_section(ev: Evaluation) -> List[str]:
    """
    The endurance figures, one per test condition the vendor published, or
    nothing at all when there is no measured table. Most actuators have none,
    and an empty frame would imply the check was run and passed.
    """
    out = []
    has_measured = any(ev.actuator.overload_curve(c)
                       for c in ("rotating", "stalled"))
    # Torques here are joint-referred, to match charts 1 and 2. That means the
    # vendor's single-unit measurement is multiplied by n, which is a real
    # assumption about load sharing and is stated rather than left implicit.
    n_act = ev.joint.n_actuators
    frame = ("" if n_act <= 1 else
             f" Torques are at the joint, as in charts 1 and 2, so the "
             f"vendor's per-actuator figures are scaled by {n_act}x; this "
             f"assumes the {n_act} actuators are loaded identically and heat "
             f"alike, which the vendor did not test.")
    for cond in ("rotating", "stalled"):
        svg = overload_endurance(ev, cond)
        if not svg:
            continue
        if not out:
            out.append("<h2>@. Overload endurance"
                       + (" vs the vendor's measurement" if has_measured
                          else " (modelled)")
                       + "</h2>")
        if cond != "rotating":
            extra = (" Stall is the harsher case: with the rotor stationary the "
                     "three phases heat unevenly (the vendor puts single-phase "
                     "heating at 1.414x rotating), and the two-node model "
                     "assumes even heating, so it reads optimistic here by "
                     "construction.")
        elif has_measured:
            extra = (" The measurement and the model together say whether the "
                     "thermal model can be trusted &mdash; it is otherwise four "
                     "estimated parameters with nothing to check them against.")
        else:
            extra = (" There is no published endurance table for this actuator, "
                     "so the model line is <em>unverified</em>: it rests on "
                     "estimated values for Rth_wc, Rth_ca, C_w and C_c. It still "
                     "shows how far this duty sits from the thermal limit, which "
                     "is the practical question, but do not read it as a rating. "
                     "Adding a vendor endurance table to the db entry turns it "
                     "into a checked figure.")
        out.append(
            f"<figure>{svg}<figcaption>How long the winding lasts at each "
            f"torque: what this tool's thermal model predicts (dashed)"
            + (", what the vendor measured (dots)" if has_measured else "")
            + f", and where the application sits (squares). The squares are the "
            f"same duty-cycle family as chart 2, each plotted at its cycle "
            f"period and its <em>RMS-equivalent</em> torque &mdash; the torque "
            f"that heats the winding as much as the real varying profile does. "
            f"RMS is what belongs on this axis: the rated point and the "
            f"endurance curve both describe sustained load, so comparing an "
            f"instantaneous peak against them would read as marginal when it is "
            f"not. The continuous rating sits at the far right because it is the "
            f"<em>asymptote</em> of the measured series, not a separate limit "
            f"&mdash; a vendor endurance table ends by naming the torque that "
            f"can be held indefinitely &mdash; and the dotted lead-in to it "
            f"marks the stretch the vendor never tested.{frame}{extra}"
            f"</figcaption></figure>")
    return out


def write_html(ev: Evaluation, path: str, text_report: str = "") -> str:
    """
    Write a self-contained HTML file for ONE actuator configuration.

    The charts describe exactly the configuration in `ev`, so they always match
    the text report embedded alongside them. To compare configurations, call
    this once per configuration.
    """
    if isinstance(ev, (list, tuple)):
        if len(ev) != 1:
            raise ValueError(
                "write_html takes a single Evaluation: charts describe one "
                "actuator configuration so they cannot disagree with the text "
                "report. Call it once per configuration.")
        ev = ev[0]

    cfg = f"{ev.joint.n_actuators}x {ev.actuator.name}"
    col = STATUS_COLOR.get(ev.verdict, MUTED)
    b = ev.binding
    parts = [
        "<!doctype html><html><head><meta charset='utf-8'>",
        f"<title>{_esc(ev.joint.name)} - {_esc(cfg)}</title>",
        f"<style>{_CSS}</style></head><body><div class='wrap'>",
        f"<h1>{_esc(ev.joint.name)} &mdash; {_esc(cfg)}</h1>",
        f"<div class='sub'>verdict <span class='verdict' style='background:{col}'>"
        f"{ev.verdict}</span> &nbsp; binding constraint: {_esc(b.name)} "
        f"({b.margin:.2f}x)</div>",

        "<h2>1. Where the motion sits against the limits</h2>",
        f"<figure>{torque_speed(ev)}<figcaption>The peak curve falls away with "
        f"speed as back-EMF eats the available bus voltage. Every operating "
        f"point must sit under it; the cycle's RMS must sit under the "
        f"continuous line, which individual points may exceed. Points shown are "
        f"the operating points over one motion cycle as specified, split by "
        f"phase of the trapezoidal move: the two directions of travel carry "
        f"different torque because gravity aids one and opposes the other. "
        f"Duty <em>factor</em> is varied in chart 2, not here."
        f"</figcaption></figure>",

        "<h2>2. Thermal margin across duty cycles</h2>",
        f"<figure>{thermal(ev)}<figcaption>Same motion and the same "
        f"{_esc(cfg)} configuration throughout, varying only the rest time "
        f"between moves. A curve that flattens below the limit runs "
        f"indefinitely; one that crosses it is marked with the time it takes. "
        f"Rest is not free &mdash; a gravity-loaded joint still draws holding "
        f"current at zero speed.</figcaption></figure>",

        *_endurance_section(ev),

        "<h2>@. Margin by criterion</h2>",
        f"<figure>{margins(ev)}<figcaption>Anything left of the 1.0x line "
        f"fails. Advisory criteria are omitted.</figcaption></figure>",
    ]
    # Number the sections here rather than in the literals: the endurance
    # section is only present for actuators with a measured table, and hard
    # numbers would read 1, 2, 4 for every actuator without one.
    _n = 0
    for _i, _part in enumerate(parts):
        if _part.startswith("<h2>") and "Full text report" not in _part:
            _n += 1
            parts[_i] = re.sub(r"<h2>[@\d]+\.", f"<h2>{_n}.", _part)
    if text_report:
        parts += ["<h2>Full text report</h2>", f"<pre>{_esc(text_report)}</pre>"]
    parts.append("</div></body></html>")

    html = "\n".join(parts)
    with open(path, "w") as f:
        f.write(html)
    return path
