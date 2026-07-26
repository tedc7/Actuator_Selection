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
                stroke="#ffffff", stroke_w=0.6):
        self.fit(xs, ys)
        data = list(zip(xs, ys))

        def draw():
            out = []
            for x, y in data:
                if not (math.isfinite(x) and math.isfinite(y)):
                    continue
                out.append(f'<circle cx="{self.px(x):.2f}" cy="{self.py(y):.2f}" '
                           f'r="{r}" fill="{colour}" opacity="{opacity}" '
                           f'stroke="{stroke}" stroke-width="{stroke_w}"/>')
            return "".join(out)
        self._queue.append(draw)
        if label:
            self.legend.append((label, colour, "dot"))

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
        self._finalise()
        o = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {self.w} {self.h}" '
             f'width="100%" style="max-width:{self.w}px;height:auto;font-family:'
             f'system-ui,-apple-system,sans-serif">',
             f'<rect width="{self.w}" height="{self.h}" fill="#ffffff"/>']

        for x in _nice_ticks(self.xlo, self.xhi):
            if not (self.xlo <= x <= self.xhi):
                continue
            o.append(f'<line x1="{self.px(x):.2f}" y1="{self.mt}" '
                     f'x2="{self.px(x):.2f}" y2="{self.mt+self.ph}" '
                     f'stroke="{GRID}" stroke-width="1"/>')
            o.append(f'<text x="{self.px(x):.2f}" y="{self.mt+self.ph+18}" '
                     f'font-size="11" fill="{MUTED}" text-anchor="middle">'
                     f'{_esc(_fmt(x))}</text>')
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
            o.append(f'<text x="{self.ml+self.pw/2}" y="{self.h-12}" font-size="12" '
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
                else:
                    da = ' stroke-dasharray="5 3"' if kind == "dash" else ""
                    o.append(f'<line x1="{x}" y1="{y-4}" x2="{x+16}" y2="{y-4}" '
                             f'stroke="{colour}" stroke-width="2.4"{da}/>')
                o.append(f'<text x="{x+22}" y="{y}" font-size="11" fill="{INK}">'
                         f'{_esc(label)}</text>')
                x += w_item
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

    w_max_out = phys.no_load_speed_out(a, v_bus)
    xs, ys = [], []
    for k in range(161):
        w_out = w_max_out * k / 160.0
        xs.append(w_out / max(float(j.ratio), 1e-9) * RPM)
        ys.append(phys.max_torque_at_speed(a, w_out, v_bus) * gain)
    p.area(xs, ys, SERIES[0], opacity=0.09)
    p.line(xs, ys, SERIES[0], "peak (voltage limited)", width=2.4)

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
# Chart 3: margins
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

        "<h2>3. Margin by criterion</h2>",
        f"<figure>{margins(ev)}<figcaption>Anything left of the 1.0x line "
        f"fails. Advisory criteria are omitted.</figcaption></figure>",
    ]
    if text_report:
        parts += ["<h2>Full text report</h2>", f"<pre>{_esc(text_report)}</pre>"]
    parts.append("</div></body></html>")

    html = "\n".join(parts)
    with open(path, "w") as f:
        f.write(html)
    return path
