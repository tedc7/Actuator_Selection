"""
Application envelopes: what a joint ACTUALLY did, summarised from a log.

A `profile` is a designer's guess at one simple move; an envelope is a record of
a real session. Sizing against the guess is over-confident in exactly the place
it matters most -- thermal -- because a real joint runs a mix of tasks, payloads
and idle periods whose torque distribution is nothing like one trapezoid
repeated forever.

The design rests on one property:

    a cell (tau, omega, seconds) IS a duty segment (dt, tau, omega)

so an envelope lowers onto the `List[(dt, tau, omega)]` interface that every
function in physics.py already consumes, and none of them need to change.

WHAT EACH ARTIFACT IS AUTHORITATIVE FOR
---------------------------------------
This split is the honesty of the whole feature and is reported verbatim.

  occupancy   : cycle-mean loss -> case temperature, and RMS current. Both are
                pure functions of the MULTISET of (tau, omega, dt) triples with
                no order dependence (see physics._mean_loss and
                rms_current_of_duty), so binning loses NO information those two
                depend on beyond the width of a cell. Nothing else reads it.

                How close that is in practice, measured on a realistic 20 k
                sample log at the default resolution: RMS current within 6e-5
                relative, cycle-mean loss within 3e-4, total time exact. Both
                fall roughly in proportion to cell width, so the knobs are
                there if a case ever needs them.

                It is NOT bitwise exact and cannot be. Two reasons, both real:
                Actuator.current_for_torque inverts the cubic
                tau = k*i - k*s*i^3, so no single representative torque
                reproduces a cell's mean CURRENT; and iron and mechanical loss
                depend on omega, which is binned too. Storing each cell's RMS
                torque rather than its plain mean is what makes the residual
                negligible -- ~100x better than the mean in isolation -- since
                copper loss dominates and goes as tau^2.

                3e-4 is several orders below the tolerance on any actuator
                parameter feeding the loss model (5% for measured, 100% for a
                guess), so this is comfortably not the limiting error.
  boundary    : the reached torque-speed outline -- the highest torque seen
                in each narrow speed slice, from raw samples. Free to rise and
                fall as the data does, so it traces the region actually
                reached rather than a monotone or convex envelope over it.
  outliers    : every excursion above p99, verbatim, with its duration. The
                guarantee that no critical point is rounded away.
  windows     : ordered worst sequences at a ladder of timescales -> winding
                peak and burst endurance. The only artifact with time ordering.
  extremes    : percentile and max demands.

The rule that follows, and which this module enforces by construction: a peak
winding temperature is NEVER taken from the occupancy-derived segment list. It
has no time ordering, so any ripple read off it would be an artefact of how the
cells happened to be sorted. It comes from the windows.

WHY THE HISTOGRAM IS ONLY AN OCCUPANCY RECORD
---------------------------------------------
A fixed-rate PI controller's output jitters continuously; two consecutive
commands are similar but never identical. So "time spent at 7.44 N.m" does not
exist -- only time spent in a NEIGHBOURHOOD, which is a property of the
summarisation rather than of the robot. Every question about sustained
behaviour is therefore answered by walking the time series (see the window
ladder in logread.py), and the histogram is kept only for the one question that
is genuinely order-free.
"""

from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from .params import (P, VENDOR_MEASURED, VENDOR_DERIVED, ESTIMATED)


# Speed below which a sample counts as "holding" rather than moving. Measured
# stationary is never exactly zero, unlike an authored profile's dwell.
HOLD_SPEED = 0.05          # rad/s at the joint

# torque_source -> provenance. Converting a current to a torque goes through Kt,
# so it is derived however good the ammeter was; only a torque transducer earns
# `measured`. This mapping is what stops a user casually claiming `measured` off
# a current log.
TORQUE_SOURCE_RANK = {
    "torque_sensor": VENDOR_MEASURED,
    "commanded_current": VENDOR_DERIVED,
    "measured_current": VENDOR_DERIVED,
    "commanded_torque": VENDOR_DERIVED,
    "simulated": ESTIMATED,
}


# ----------------------------------------------------------------------------
# Pieces
# ----------------------------------------------------------------------------

@dataclass
class Cell:
    """
    One occupancy bin: how long the joint spent near this (torque, speed).

    `tau` is the cell's time-weighted RMS torque carrying the sign of its mean,
    not the plain mean. Copper loss goes as tau^2 and dominates, so matching the
    cell's mean SQUARE is what makes the duty's RMS current come out right --
    measured ~100x closer than the plain mean on a realistic log. The signed
    mean is kept separately because the sign, not the magnitude, decides whether
    the gearbox efficiency multiplies or divides in actuator_output_demand.
    """
    ti: int                 # torque bin index
    si: int                 # speed bin index
    seconds: float
    tau: float              # time-weighted RMS within the cell, signed, joint N.m
    omega: float            # time-weighted mean within the cell, joint rad/s
    tau_mean: float = 0.0   # time-weighted plain mean, joint N.m


@dataclass
class Window:
    """
    An ordered excerpt of the log: the worst real sequence at one timescale.

    Selected by integrating a thermal kernel over the samples rather than by a
    torque threshold, so it is the stretch that actually heated the winding
    most, not merely the one with the biggest number in it.
    """
    duration: float                      # s, the window length
    found_at: float                      # s into the session
    rms_torque: float                    # N.m at the joint
    segments: List[Tuple[float, float, float]]   # ordered (dt, tau, omega)
    selected_by_tau_w: List[float] = field(default_factory=list)
    selection_metric: str = "tau_squared_single_pole"


@dataclass
class Outlier:
    """
    One excursion, stored verbatim because a summary would round it away.

    Duration is the point: a 4 ms spike at 3x torque and a 1.9 s excursion at
    1.5x are different events failing different criteria, and only the duration
    tells them apart.
    """
    t_start: float
    duration: float
    tau_peak: float
    omega_at_peak: float
    tau_mean: float
    tau_rms: float = 0.0     # what the excursion contributes thermally


@dataclass
class Extremes:
    """Percentiles and maxima, from a fine pre-histogram rather than the grid."""
    torque: Dict[str, float] = field(default_factory=dict)   # p50/p95/p99/p99_9/max/max_raw
    speed: Dict[str, float] = field(default_factory=dict)

    def t(self, key: str, default: float = 0.0) -> float:
        return float(self.torque.get(key, default))

    def w(self, key: str, default: float = 0.0) -> float:
        return float(self.speed.get(key, default))


@dataclass
class Capture:
    """Provenance of the session this envelope came from."""
    captured_at: str = ""
    robot_id: str = ""
    joint_id: str = ""
    firmware: str = ""
    duration: float = 0.0
    sample_rate: Optional[float] = None
    coverage: float = 1.0
    gaps: List[Tuple[float, float]] = field(default_factory=list)  # (t, duration)
    source_logs: List[dict] = field(default_factory=list)
    tool_version: str = ""


@dataclass
class Despike:
    """What the median filter removed. An editorial act, so it is reported."""
    filter: str = "median"
    width_samples: int = 0
    width_s: Optional[float] = None
    samples_changed: int = 0
    worst_change: float = 0.0

    @property
    def applied(self) -> bool:
        return self.width_samples > 1


# ----------------------------------------------------------------------------
# The envelope
# ----------------------------------------------------------------------------

@dataclass
class Envelope:
    """
    A joint's measured operating envelope. Joint-referred, always.

    Joint-referred because the whole point is evaluating DIFFERENT actuators
    against ONE captured reality: an actuator-referred envelope would bake in
    n_actuators, ratio and ratio_eff and be useless the moment you compare 2x a
    small unit against 1x a big one. evaluate() maps each segment through
    Joint.actuator_output_demand per candidate, exactly as it does for an
    authored duty_segments list.
    """
    name: str = "envelope"

    cells: List[Cell] = field(default_factory=list)
    total_time: float = 0.0
    torque_edges: List[float] = field(default_factory=list)
    speed_edges: List[float] = field(default_factory=list)
    binned_below: Optional[float] = None    # |tau| above this went to outliers

    boundary: List[Tuple[float, float]] = field(default_factory=list)      # (omega, tau)
    boundary_raw: List[Tuple[float, float]] = field(default_factory=list)
    outliers: List[Outlier] = field(default_factory=list)
    windows: List[Window] = field(default_factory=list)
    duration_curve: List[Tuple[float, float]] = field(default_factory=list)  # (window_s, rms_tau)
    event_counts: Dict[str, dict] = field(default_factory=dict)
    extremes: Extremes = field(default_factory=Extremes)

    capture: Capture = field(default_factory=Capture)
    despike: Despike = field(default_factory=Despike)

    # referred_to
    ratio: float = 1.0
    n_actuators_at_capture: Optional[int] = None
    torque_source: str = "commanded_current"
    sign_convention: str = ""
    fixed_rate: bool = True
    dt_nominal: Optional[float] = None

    source: Optional[str] = None
    tol: Optional[float] = None
    note: str = ""
    source_path: Optional[str] = None

    # ------------------------------------------------------------------
    def provenance(self) -> P:
        """
        The envelope as a parameter, so its quality propagates into every
        criterion that reads it.

        A `commanded_current` envelope is `derived` however careful the capture
        was, because the torque went through Kt to get there. That correctly
        caps the confidence of every thermal verdict built on it -- the
        profile-based reports are over-confident, not these ones under-confident.
        """
        src = self.source or TORQUE_SOURCE_RANK.get(self.torque_source, ESTIMATED)
        note = f"envelope '{self.name}'"
        if self.torque_source:
            note += f", torque from {self.torque_source}"
        if self.capture.duration:
            note += f", {self.capture.duration/3600.0:.2f} hr session"
        return P(1.0, "-", src, self.tol, note, name="envelope")

    # ------------------------------------------------------------------
    def negate_torque(self) -> "Envelope":
        """
        Flip the torque sense, in place, across every artifact.

        For an application whose frame is the opposite of the logger's. Only
        torque is negated, never speed: the pair's RELATIVE sign is what decides
        motoring versus regenerating, so negating both would be a no-op and
        negating torque alone is precisely the correction wanted.
        """
        for c in self.cells:
            c.tau = -c.tau
            c.tau_mean = -c.tau_mean
        for o in self.outliers:
            o.tau_peak = -o.tau_peak
            o.tau_mean = -o.tau_mean
            o.tau_rms = -o.tau_rms
        for w in self.windows:
            w.segments = [(dt, -tau, om) for dt, tau, om in w.segments]
        # boundary, extremes and the duration curve are magnitudes, so they are
        # unaffected by construction.
        return self

    def as_duty_segments(self) -> List[Tuple[float, float, float]]:
        """
        The occupancy record as a duty cycle: [(dt, tau_joint, omega_joint)].

        Ordered by ascending tau^2. The ordering is ARBITRARY -- the artifact it
        comes from has no time ordering at all -- and is fixed only so that the
        same envelope always yields the same list and reports diff cleanly. Any
        quantity that depends on the order (physics.simulate_duty's
        t_winding_peak) is therefore discarded by evaluate() in favour of the
        window computation, which does have real ordering. See binding_window().
        """
        return [(c.seconds, c.tau, c.omega) for c in self._ordered_cells()]

    def _ordered_cells(self) -> List[Cell]:
        """Occupied cells in the fixed ascending-tau^2 order used everywhere."""
        return sorted((c for c in self.cells if c.seconds > 0),
                      key=lambda c: c.tau * c.tau)

    def motion_phases(self) -> List[str]:
        """
        Regime tag per segment of as_duty_segments(), aligned 1:1 with it.

        Not the trapezoid tags: an envelope has no commanded trajectory to
        classify, so the split that carries information is holding vs driving vs
        regenerating. Charts key off `len(phases) == len(segs)` and colour the
        series accordingly.
        """
        out = []
        for c in self._ordered_cells():
            if abs(c.omega) < HOLD_SPEED:
                out.append("Env_Hold")
            else:
                # Signed MEAN torque decides motoring vs regenerating; the RMS
                # in c.tau is a magnitude and carries no usable sign here.
                kind = "Drive" if c.tau_mean * c.omega > 0 else "Regen"
                out.append(f"Env_{kind}_{'Pos' if c.omega > 0 else 'Neg'}")
        return out

    # ------------------------------------------------------------------
    def window_segments(self, duration: float) -> List[Tuple[float, float, float]]:
        """Ordered segments of the stored window closest to `duration`."""
        w = self.window_for_duration(duration)
        return list(w.segments) if w else []

    def window_for_duration(self, duration: float) -> Optional[Window]:
        if not self.windows:
            return None
        return min(self.windows, key=lambda w: abs(math.log(max(w.duration, 1e-9))
                                                   - math.log(max(duration, 1e-9))))

    def binding_window(self, tau_w: float) -> Optional[Window]:
        """
        The window that binds for an actuator whose winding time constant is
        `tau_w` (its C_w * Rth_wc).

        Prefer a window that names this constant as one of its selectors, since
        that window was literally chosen by a kernel of this shape. Otherwise
        fall back to the nearest duration in log space.
        """
        for w in self.windows:
            for t in w.selected_by_tau_w:
                if abs(math.log(max(t, 1e-9)) - math.log(max(tau_w, 1e-9))) < 0.35:
                    return w
        return self.window_for_duration(tau_w)

    # ------------------------------------------------------------------
    def peak_torque_demand(self, percentile: str = "p99_9") -> float:
        """
        Sizing torque. A percentile, not the max: one sensor glitch or a single
        collision should not size the whole joint. The true max is still
        reported alongside, and drives the DRIVE CURRENT criterion instead --
        a fault threshold is a real maximum, unlike a mechanical margin.
        """
        return self.extremes.t(percentile) or self.extremes.t("max")

    def peak_speed_demand(self, percentile: str = "p99_9") -> float:
        return self.extremes.w(percentile) or self.extremes.w("max")

    def tail_is_outlier_dominated(self, factor: float = 2.0) -> bool:
        """max far above p99.9 means one event is setting the peak. Worth saying."""
        p = self.extremes.t("p99_9")
        return bool(p > 0 and self.extremes.t("max") > factor * p)

    def time_in_regime(self) -> Dict[str, Tuple[float, float]]:
        """{regime: (fraction of time, mean |tau|)} for the report."""
        acc: Dict[str, List[float]] = {}
        for c in self.cells:
            if c.seconds <= 0:
                continue
            if abs(c.omega) < HOLD_SPEED:
                key = "holding"
            else:
                key = "driving" if c.tau_mean * c.omega > 0 else "regenerating"
            a = acc.setdefault(key, [0.0, 0.0])
            a[0] += c.seconds
            a[1] += abs(c.tau_mean) * c.seconds
        total = sum(a[0] for a in acc.values()) or 1e-9
        return {k: (a[0] / total, a[1] / max(a[0], 1e-9)) for k, a in acc.items()}

    def sustained_torque_at(self, window_s: float) -> Optional[float]:
        """RMS joint torque sustainable over `window_s`, from the duration curve."""
        if not self.duration_curve:
            return None
        return min(self.duration_curve,
                   key=lambda p: abs(math.log(max(p[0], 1e-9))
                                     - math.log(max(window_s, 1e-9))))[1]

    def summary(self) -> str:
        return (f"{self.name}: {len(self.cells)} cells, "
                f"{self.capture.duration/3600.0:.2f} hr, "
                f"{len(self.windows)} windows, {len(self.outliers)} outliers")


# ----------------------------------------------------------------------------
# Building one from samples
# ----------------------------------------------------------------------------

def _quantile(sorted_vals: Sequence[float], weights: Sequence[float], q: float) -> float:
    """Time-weighted quantile of an already-sorted value list."""
    total = sum(weights)
    if total <= 0:
        return 0.0
    target = q * total
    acc = 0.0
    for v, w in zip(sorted_vals, weights):
        acc += w
        if acc >= target:
            return v
    return sorted_vals[-1]


def quantile_edges(values: Sequence[float], weights: Sequence[float],
                   n: int) -> List[float]:
    """
    Bin edges at fixed quantiles of the distribution, not uniform spacing.

    Uniform bins are indefensible here: the width that suits 4000 s of low-torque
    holding is far too coarse for a brief high-torque move, and since copper loss
    goes as tau^2 the tail is where the resolution actually matters. Quantile
    spacing puts narrow bins where the samples are dense and lets the tail have
    its own.

    Zero is forced to be an edge so that no cell straddles it: a cell spanning
    tau=0 could have a mean whose sign disagrees with most of its samples, which
    would put it in the wrong quadrant and land the gearbox efficiency the wrong
    way in Joint.actuator_output_demand.
    """
    pairs = sorted(zip(values, weights))
    if not pairs:
        return [-1.0, 0.0, 1.0]
    vs = [p[0] for p in pairs]
    ws = [p[1] for p in pairs]
    edges = {0.0, vs[0], vs[-1]}
    for k in range(1, n):
        edges.add(_quantile(vs, ws, k / n))

    # Quantile spacing collapses when the distribution is a few spikes rather
    # than a spread: a log that is 90% holding at 1 N.m and 10% moving at 30
    # puts almost every quantile on 1.0, leaving one enormous cell that averages
    # the two populations together and makes the move disappear. Backfill with
    # linear edges so a cell can never span the whole range, which is the case
    # where binning does real damage.
    if len(edges) < max(n // 2, 4):
        lo, hi = vs[0], vs[-1]
        if hi > lo:
            for k in range(n + 1):
                edges.add(lo + (hi - lo) * k / n)

    out = sorted(edges)
    # Guarantee a strictly increasing sequence spanning the data.
    if out[0] > vs[0]:
        out.insert(0, vs[0])
    if out[-1] < vs[-1]:
        out.append(vs[-1])
    return out


def _bin_index(edges: Sequence[float], v: float) -> int:
    """Index of the cell containing v; clamped to the outermost cell."""
    lo, hi = 0, len(edges) - 1
    if v <= edges[0]:
        return 0
    if v >= edges[-1]:
        return max(hi - 1, 0)
    while lo < hi - 1:
        mid = (lo + hi) // 2
        if v < edges[mid]:
            hi = mid
        else:
            lo = mid
    return lo


def boundary_scan(points: Sequence[Tuple[float, float]],
                  slices: int = 96) -> List[Tuple[float, float]]:
    """
    Outer outline of the reached (|omega|, |tau|) region: max torque per speed.

    Divide the speed range into narrow slices and keep the highest torque seen
    in each. The outline is then free to rise AND fall as the data does, which
    is what "the region the joint actually reached" means.

    Deliberately not monotone, and deliberately not a convex hull. Both of those
    are global-max constructions that fabricate an outline the robot never
    traced, and being conservative in a way the reader cannot see is the failure
    mode this whole artifact exists to avoid.

    An earlier version walked samples in descending torque and kept one only if
    its speed exceeded every sample kept so far. That forced a monotonically
    decreasing staircase, so on a real capture -- whose peak torque occurs at
    LOW speed, because that is where a gravity-loaded joint works hardest -- the
    entire low-speed region below the peak was discarded by construction. The
    chart then showed nothing below the peak point while excursions and window
    samples were plainly visible there, which is how the bug was spotted.

    Caller must pass DESPIKED samples: a per-slice max is still a max operation,
    so on raw data it would trace the sensor's noise ceiling rather than the
    envelope. Envelope.boundary_raw holds the undespiked version for comparison.
    """
    if not points:
        return []
    mags = [(abs(w), abs(t)) for w, t in points]
    w_hi = max(w for w, _ in mags)
    if w_hi <= 0:
        return [(0.0, max(t for _, t in mags))]

    width = w_hi / max(slices, 1)
    best: Dict[int, Tuple[float, float]] = {}
    for om, tau in mags:
        k = min(int(om / width), slices - 1)
        cur = best.get(k)
        if cur is None or tau > cur[1]:
            best[k] = (om, tau)
    return [best[k] for k in sorted(best)]


def from_samples(samples: Sequence[Tuple[float, float, float]],
                 name: str = "envelope",
                 n_torque_bins: int = 24,
                 n_speed_bins: int = 20,
                 outlier_percentile: float = 0.99,
                 **kw) -> Envelope:
    """
    Build an envelope from [(dt, tau_joint, omega_joint)] samples.

    Pure and in-memory: this is the testable core. The streaming, two-pass
    version that never holds a 400 MB log is logread.py, which calls the same
    helpers. Everything above the outlier percentile is kept VERBATIM and is
    never binned, so the region where bin width is hardest to defend simply has
    no bins in it.
    """
    samples = [(float(dt), float(tau), float(om))
               for dt, tau, om in samples if float(dt) > 0]
    env = Envelope(name=name, **kw)
    if not samples:
        return env

    dts = [s[0] for s in samples]
    taus = [s[1] for s in samples]
    oms = [s[2] for s in samples]
    env.total_time = sum(dts)
    env.capture.duration = env.total_time

    # ---- extremes, from the raw sample population -------------------------
    abs_t = sorted(zip((abs(t) for t in taus), dts))
    abs_w = sorted(zip((abs(w) for w in oms), dts))
    tv, tw = [p[0] for p in abs_t], [p[1] for p in abs_t]
    wv, ww = [p[0] for p in abs_w], [p[1] for p in abs_w]
    env.extremes = Extremes(
        torque={"p50": _quantile(tv, tw, 0.50), "p95": _quantile(tv, tw, 0.95),
                "p99": _quantile(tv, tw, 0.99), "p99_9": _quantile(tv, tw, 0.999),
                "max": tv[-1]},
        speed={"p50": _quantile(wv, ww, 0.50), "p95": _quantile(wv, ww, 0.95),
               "p99": _quantile(wv, ww, 0.99), "p99_9": _quantile(wv, ww, 0.999),
               "max": wv[-1]},
    )

    # ---- boundary, from every sample, no binning --------------------------
    env.boundary = boundary_scan([(w, t) for _, t, w in samples])

    # ---- split the tail out before binning anything -----------------------
    # The cut must sit strictly BELOW the peak, otherwise nothing is above it
    # and the excursion vanishes from the outlier list. That happens whenever a
    # single high-torque state occupies more than (1 - percentile) of the
    # session -- a busy joint doing real work, i.e. exactly the case that must
    # not be lost. Back the cut off to the largest value under the peak.
    cut = _quantile(tv, tw, outlier_percentile)
    peak = tv[-1]
    if cut >= peak:
        below = [v for v in tv if v < peak]
        cut = below[-1] if below else peak * 0.999
    env.binned_below = cut
    env.outliers, kept_idx = _extract_outliers(
        samples, cut, env.extremes.w("p99"), with_indices=True)

    # Bin everything the outlier list did NOT keep, including the excursions the
    # cap dropped: those are still time the joint spent under load, and losing
    # them would break the invariant everything else rests on -- that the
    # segment list totals the session -- while understating the thermal average
    # by exactly the busiest moments in the log.
    #
    # Selected by sample INDEX, not by an accumulated clock. Summing dt over
    # tens of thousands of samples drifts by enough to misclassify a sample at a
    # span boundary, and every one lost that way is time that silently leaves
    # the session.
    binnable = [s for i, s in enumerate(samples) if i not in kept_idx]
    if not binnable:
        binnable = samples

    # ---- occupancy, quantile-spaced, signed both ways ---------------------
    env.torque_edges = quantile_edges([s[1] for s in binnable],
                                      [s[0] for s in binnable], n_torque_bins)
    env.speed_edges = quantile_edges([s[2] for s in binnable],
                                     [s[0] for s in binnable], n_speed_bins)

    acc: Dict[Tuple[int, int], List[float]] = {}
    for dt, tau, om in binnable:
        key = (_bin_index(env.torque_edges, tau), _bin_index(env.speed_edges, om))
        a = acc.setdefault(key, [0.0, 0.0, 0.0, 0.0])
        a[0] += dt
        a[1] += tau * dt          # time-weighted, so the cell mean is unbiased
        a[2] += om * dt
        a[3] += tau * tau * dt    # the term copper loss actually depends on
    env.cells = [
        Cell(ti, si, s, math.copysign(math.sqrt(sq / s), tsum or 1.0),
             wsum / s, tsum / s)
        for (ti, si), (s, tsum, wsum, sq) in sorted(acc.items()) if s > 0
    ]

    # The outliers were pulled out of the binning above, but they are still time
    # the joint spent somewhere and must not vanish from the thermal average.
    # Re-add them as their own unbinned cells, keyed off the grid so they are
    # distinguishable, so as_duty_segments() still totals the full session.
    for k, o in enumerate(env.outliers):
        env.cells.append(Cell(-1, -(k + 1), o.duration, o.tau_rms,
                              o.omega_at_peak, o.tau_mean))

    return env


# ----------------------------------------------------------------------------
# Serialisation
# ----------------------------------------------------------------------------

def _rows(rows: Sequence[Sequence[float]], indent: str = "      ") -> str:
    """
    One record per line.

    json.dumps(indent=2) turns a 400-cell array into 2400 lines and makes the
    git diff of a recapture unreadable, which defeats the point of committing
    the envelope as a reviewable artifact. One row per line keeps a recapture's
    diff proportional to what actually changed.
    """
    if not rows:
        return "[]"
    body = ",\n".join(indent + "[" + ", ".join(_num(v) for v in r) + "]"
                      for r in rows)
    return "[\n" + body + "\n" + indent[:-2] + "]"


def _num(v) -> str:
    if isinstance(v, int):
        return str(v)
    f = float(v)
    if f == int(f) and abs(f) < 1e15:
        return str(int(f))
    return f"{f:.6g}"


_EXT_ORDER = ("p50", "p95", "p99", "p99_9", "max", "max_raw")


def _ext_block(d: Dict[str, float], unit: str) -> dict:
    """Percentiles in reading order, worst last, with the unit alongside."""
    out = {k: round(d[k], 4) for k in _EXT_ORDER if k in d}
    for k in sorted(d):
        out.setdefault(k, round(d[k], 4))
    out["units"] = unit
    return out


def to_json(env: Envelope) -> str:
    """Serialise an envelope in the committed on-disk format."""
    import json as _json

    def blk(o):
        return _json.dumps(o, indent=2, sort_keys=False)

    cap = env.capture
    capture = {"captured_at": cap.captured_at, "robot_id": cap.robot_id,
               "joint_id": cap.joint_id, "firmware": cap.firmware,
               "source_logs": cap.source_logs,
               "duration": {"value": round(cap.duration, 3), "units": "s"},
               "coverage": round(cap.coverage, 6),
               "gaps": [{"t": round(t, 3), "duration": round(d, 4)}
                        for t, d in cap.gaps],
               "tool_version": cap.tool_version}
    if cap.sample_rate:
        capture["sample_rate"] = {"value": cap.sample_rate, "units": "Hz"}

    referred = {"frame": "joint",
                "ratio": {"value": env.ratio, "units": "-",
                          "note": "must match the application's ratio; checked at load"},
                "n_actuators_at_capture": env.n_actuators_at_capture,
                "torque_source": env.torque_source,
                "sign_convention": env.sign_convention,
                "controller": {"fixed_rate": env.fixed_rate,
                               "dt": {"value": env.dt_nominal, "units": "s"}}
                if env.dt_nominal else {"fixed_rate": env.fixed_rate},
                "despike": {"filter": env.despike.filter,
                            "width_samples": env.despike.width_samples,
                            "width": {"value": env.despike.width_s, "units": "s"},
                            "samples_changed": env.despike.samples_changed,
                            "worst_change": {"value": env.despike.worst_change,
                                             "units": "N.m"},
                            "note": "median rejects isolated single-step spikes; "
                                    "a moving average would also erase genuine "
                                    "short events"}}

    parts = [
        '{',
        f'  "schema": "actuator_eval.envelope/1",',
        f'  "name": {_json.dumps(env.name)},',
        '',
        '  "capture": ' + blk(capture).replace("\n", "\n  ") + ',',
        '',
        '  "referred_to": ' + blk(referred).replace("\n", "\n  ") + ',',
        '',
        '  "occupancy": {',
        '    "_note": "time-occupancy ONLY; no criterion reads a peak from this",',
        f'    "total_time": {{"value": {env.total_time:.4f}, "units": "s"}},',
    ]
    if env.binned_below is not None:
        parts.append(f'    "binned_below": {{"value": {env.binned_below:.4f}, '
                     f'"units": "N.m", "note": "everything above this went to '
                     f'outliers, unbinned"}},')
    parts += [
        '    "torque_edges": {"units": "N.m", "quantile_spaced": true, "edges": ['
        + ", ".join(_num(e) for e in env.torque_edges) + ']},',
        '    "speed_edges": {"units": "rad/s", "quantile_spaced": true, "edges": ['
        + ", ".join(_num(e) for e in env.speed_edges) + ']},',
        '    "cells_units": ["-", "-", "s", "N.m", "rad/s", "N.m"],',
        '    "_cells_note": "[ti, si, seconds, tau_rms, omega_mean, tau_mean]; '
        'ti=-1 marks an unbinned outlier cell",',
        '    "cells": ' + _rows([[c.ti, c.si, round(c.seconds, 6), round(c.tau, 4),
                                  round(c.omega, 4), round(c.tau_mean, 4)]
                                 for c in env.cells]) + '',
        '  },',
        '',
        '  "boundary": {"_note": "max torque per speed slice, from despiked samples; '
        'free to rise and fall, so neither monotone nor convex",',
        '    "units": ["rad/s", "N.m"],',
        '    "points": ' + _rows([[round(w, 4), round(t, 4)] for w, t in env.boundary])
        + '},',
    ]
    if env.boundary_raw:
        parts += [
            '',
            '  "boundary_raw": {"_note": "same scan without despiking, so the '
            'filtering is visible rather than silent",',
            '    "units": ["rad/s", "N.m"],',
            '    "points": ' + _rows([[round(w, 4), round(t, 4)]
                                      for w, t in env.boundary_raw]) + '},',
        ]
    parts += [
        '',
        '  "outliers": {',
        '    "criterion": "|tau| above the binning cut, or |omega| above p99",',
        '    "rows_units": ["s", "s", "N.m", "rad/s", "N.m", "N.m"],',
        '    "_rows_note": "[t_start, duration, tau_peak, omega_at_peak, '
        'tau_mean, tau_rms] per excursion; duration is what separates a drive-'
        'current event from a thermal one",',
        '    "rows": ' + _rows([[round(o.t_start, 4), round(o.duration, 5),
                                 round(o.tau_peak, 4), round(o.omega_at_peak, 4),
                                 round(o.tau_mean, 4), round(o.tau_rms, 4)]
                                for o in env.outliers]),
        '  },',
        '',
        '  "event_counts": ' + blk(env.event_counts).replace("\n", "\n  ") + ',',
        '',
        '  "extremes": {',
        '    "torque_abs": ' + _json.dumps(_ext_block(env.extremes.torque, "N.m")) + ',',
        '    "speed_abs": ' + _json.dumps(_ext_block(env.extremes.speed, "rad/s")),
        '  },',
        '',
        '  "duration_curve": {"units": "N.m",',
        '    "_note": "highest RMS joint torque sustained over each averaging window",',
        '    "points": ' + _rows([[w, round(t, 4)] for w, t in env.duration_curve])
        + '},',
        '',
        '  "windows": [',
    ]
    wparts = []
    for w in env.windows:
        wparts.append(
            '    {"selection_metric": ' + _json.dumps(w.selection_metric) + ',\n'
            '     "selected_by_tau_w": {"value": ['
            + ", ".join(_num(x) for x in w.selected_by_tau_w) + '], "units": "s"},\n'
            f'     "duration": {{"value": {w.duration:g}, "units": "s"}},\n'
            f'     "found_at": {{"value": {w.found_at:g}, "units": "s"}},\n'
            f'     "rms_torque": {{"value": {w.rms_torque:g}, "units": "N.m"}},\n'
            '     "segments_units": ["s", "N.m", "rad/s"],\n'
            '     "segments": ' + _rows([[round(s[0], 6), round(s[1], 4),
                                          round(s[2], 4)] for s in w.segments],
                                        indent="       ") + '}')
    parts.append(",\n".join(wparts))
    parts += [
        '  ],',
        '',
        f'  "source": {_json.dumps(env.source or TORQUE_SOURCE_RANK.get(env.torque_source, ESTIMATED))},',
        f'  "tol": {env.tol if env.tol is not None else 0.05},',
        f'  "note": {_json.dumps(env.note)}',
        '}',
        '',
    ]
    return "\n".join(parts)


def write_json(env: Envelope, path: str) -> str:
    with open(path, "w") as f:
        f.write(to_json(env))
    return path


def _extract_outliers(samples: Sequence[Tuple[float, float, float]],
                      tau_cut: float, omega_cut: float,
                      limit: int = 200, with_indices: bool = False):
    """
    Contiguous runs above the cut, kept verbatim with their durations.

    A run rather than a sample, because duration is what separates a 4 ms spike
    (a drive-current question) from a 1.9 s excursion (a thermal one), and both
    must survive summarisation as themselves.

    With `with_indices`, also returns the set of sample indices the kept runs
    cover, so the caller can bin everything else without re-deriving the spans
    from timestamps.
    """
    out: List[Outlier] = []
    spans: List[Tuple[int, int]] = []      # [start, end) sample indices
    run: List[Tuple[float, float, float]] = []
    t_clock = 0.0
    run_start = 0.0
    run_i0 = 0

    def flush(i_end):
        if not run:
            return
        dur = sum(s[0] for s in run)
        pk = max(run, key=lambda s: abs(s[1]))
        mean = sum(s[1] * s[0] for s in run) / max(dur, 1e-12)
        rms = math.sqrt(sum(s[1] * s[1] * s[0] for s in run) / max(dur, 1e-12))
        out.append(Outlier(run_start, dur, pk[1], pk[2], mean,
                           math.copysign(rms, mean or 1.0)))
        spans.append((run_i0, i_end))

    for i, (dt, tau, om) in enumerate(samples):
        hot = abs(tau) > tau_cut or (omega_cut > 0 and abs(om) > omega_cut)
        if hot:
            if not run:
                run_start = t_clock
                run_i0 = i
            run.append((dt, tau, om))
        else:
            flush(i)
            run = []
        t_clock += dt
    flush(len(samples))

    # Keep the most severe by peak AND the most severe by duration: they fail
    # different criteria, so trimming by either alone would hide the other.
    # Index-based dedup, not id(): two orderings of the same list share objects,
    # and collapsing them by identity silently halves the kept set.
    if len(out) > limit:
        order = sorted(range(len(out)), key=lambda i: -abs(out[i].tau_peak))
        keep = set(order[:limit // 2])
        order = sorted(range(len(out)), key=lambda i: -out[i].duration)
        for i in order:
            if len(keep) >= limit:
                break
            keep.add(i)
        out = [o for i, o in enumerate(out) if i in keep]
        spans = [s for i, s in enumerate(spans) if i in keep]

    if not with_indices:
        return out
    covered = set()
    for lo, hi in spans:
        covered.update(range(lo, hi))
    return out, covered
