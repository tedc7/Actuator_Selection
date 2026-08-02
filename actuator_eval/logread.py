"""
Turn a controller log into an envelope, without ever holding the log in memory.

THE LOGGER CONTRACT
-------------------
Three columns, at whatever fixed rate the controller already runs, appended to a
file: time, position, torque. That is the whole ask. No real-time computation,
no envelope built on the robot, no synchronisation -- the summarising happens
here, afterwards, where it is cheap and repeatable.

Speed is used when logged (the controller's own filtered estimate beats anything
post-hoc differentiation can produce) and derived from position otherwise. A
current column substitutes for torque given a declared joint-referred Kt.

WHY TWO PASSES AND NOT ONE
--------------------------
The percentile thresholds that define an excursion are not known until the whole
distribution has been seen, and binning cannot start until the quantile edges
are known. So: pass 1 accumulates a fine pre-histogram, the boundary outline
and a decimated tau^2 series; pass 2 re-reads with the thresholds in hand to
extract excursions, event counts, the occupancy grid and the winning windows.
Neither pass keeps more than a bounded amount of state, so a 400 MB log costs
the same memory as a 4 MB one.

WHY A MEDIAN AND NOT A MOVING AVERAGE
-------------------------------------
A PI controller's output contains isolated one-sample excursions -- quantisation,
sensor noise, a single aggressive correction -- that are not physically
meaningful. Short REAL events (a collision, a hard stop) also exist and the
drive genuinely sees them. A moving average attenuates by duration and so cannot
separate the two: on a 1 kHz log, a 50 ms mean rejects isolated 40 N.m glitches
but also flattens a genuine 24 N.m event lasting 8 ms to 9 N.m. A median removes
anything narrower than half its window and passes anything wider through
undistorted, which is exactly the distinction wanted.
"""

from __future__ import annotations

import csv
import hashlib
import math
import os
from dataclasses import dataclass, field
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

from . import units as U
from .envelope import (Cell, Envelope, Extremes, Outlier, Window,
                       boundary_scan, quantile_edges, _bin_index)


# Winding time constants to probe. The floor is always present so envelopes stay
# comparable across captures; it brackets every actuator in the shipped db
# (10.7-17.1 s) by two decades on each side and reaches the case-node timescale.
TAU_W_LADDER = (1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0)

DURATION_CURVE_WINDOWS = (0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 100.0, 300.0,
                          1000.0, 1800.0)

MAX_WINDOW_SEGMENTS = 160        # matches the house `samples` convention
DECIMATED_DT = 0.1               # resolution of the in-memory tau^2 series


class LogError(Exception):
    """A log that cannot be read honestly, with a pointed reason."""


# ---------------------------------------------------------------------------
# Column mapping
# ---------------------------------------------------------------------------

FIELD_DIMENSIONS = {
    "time": "time",
    "position": "angle",
    "speed": "angular_velocity",
    "torque": "torque",
    "current": "current",
}


@dataclass
class ColumnMap:
    """Which CSV column is which quantity, and in what units."""
    columns: Dict[str, str] = field(default_factory=dict)
    units: Dict[str, str] = field(default_factory=dict)

    def validate(self) -> None:
        for k in self.columns:
            if k not in FIELD_DIMENSIONS:
                raise LogError(
                    f"'{k}' is not a loggable quantity. Map one or more of: "
                    f"{', '.join(sorted(FIELD_DIMENSIONS))}.")
        if "torque" not in self.columns and "current" not in self.columns:
            raise LogError(
                "the log must carry either a torque or a current column; "
                "without one there is nothing to size an actuator against.")
        if "speed" not in self.columns and "position" not in self.columns:
            raise LogError(
                "the log must carry either a speed or a position column; speed "
                "is preferred (the controller's filtered estimate beats "
                "post-hoc differentiation) and is derived from position "
                "otherwise.")
        for k in self.columns:
            if k not in self.units:
                raise LogError(
                    f"no units declared for '{k}'. Units are never assumed for "
                    f"logged data -- a torque read as N.m when it was really "
                    f"kgf.cm is wrong by 100x and still evaluates. Valid: "
                    f"{', '.join(U.valid_units(FIELD_DIMENSIONS[k]))}.")
            # Fail here rather than 4 GB into the file.
            U.convert(1.0, self.units[k], FIELD_DIMENSIONS[k])

    def scale(self, field_name: str) -> float:
        return U.convert(1.0, self.units[field_name], FIELD_DIMENSIONS[field_name])


def parse_map_arg(spec: str, into: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    """`time=t_s,torque=tau_cmd` -> {'time': 't_s', 'torque': 'tau_cmd'}."""
    out = dict(into or {})
    for piece in spec.split(","):
        piece = piece.strip()
        if not piece:
            continue
        if "=" not in piece:
            raise LogError(f"expected key=value in a mapping, got {piece!r}")
        k, v = piece.split("=", 1)
        out[k.strip()] = v.strip()
    return out


# ---------------------------------------------------------------------------
# Streaming primitives
# ---------------------------------------------------------------------------

class MedianFilter:
    """
    Running median over an odd window. O(k) per sample with a sorted deque,
    which is bounded memory and fast enough at k=5.

    Reports how many samples it moved MATERIALLY rather than at all: a median
    nudges nearly every sample of a noisy signal by a fraction of the jitter,
    and a "2.8 M samples changed" line would say nothing about the log.
    """

    def __init__(self, width: int = 5):
        self.width = max(int(width), 1)
        if self.width % 2 == 0:
            self.width += 1
        self.buf: List[float] = []
        self.changes: List[float] = []

    @property
    def active(self) -> bool:
        return self.width > 1

    def push(self, v: float) -> float:
        if not self.active:
            return v
        self.buf.append(v)
        if len(self.buf) > self.width:
            self.buf.pop(0)
        s = sorted(self.buf)
        out = s[len(s) // 2]
        self.changes.append(abs(v - out))
        return out

    def summary(self) -> Tuple[int, float]:
        """(materially changed, worst change)."""
        if not self.changes:
            return 0, 0.0
        ordered = sorted(self.changes)
        sigma = ordered[len(ordered) // 2] or 1e-9
        material = sum(1 for c in self.changes if c > 10 * sigma)
        return material, max(self.changes)


class Quantiles:
    """
    Streaming quantiles via a fine fixed pre-histogram.

    4096 bins over a range that grows as needed gives 0.025% resolution, which
    is far finer than the stored occupancy grid and is what lets every
    percentile in the envelope be a raw-sample fact rather than a binned one.
    """

    def __init__(self, bins: int = 4096):
        self.bins = bins
        self.hi = 1e-6
        self.counts = [0.0] * bins
        self.total = 0.0
        self.max_seen = 0.0

    def _grow(self, new_hi: float) -> None:
        factor = int(math.ceil(new_hi / self.hi))
        if factor < 2:
            factor = 2
        merged = [0.0] * self.bins
        for i, c in enumerate(self.counts):
            if c:
                merged[min(i // factor, self.bins - 1)] += c
        self.counts = merged
        self.hi *= factor

    def add(self, v: float, w: float) -> None:
        v = abs(v)
        self.max_seen = max(self.max_seen, v)
        if v >= self.hi:
            self._grow(v * 1.05)
        i = min(int(v / self.hi * self.bins), self.bins - 1)
        self.counts[i] += w
        self.total += w

    def quantile(self, q: float) -> float:
        if self.total <= 0:
            return 0.0
        target = q * self.total
        acc = 0.0
        for i, c in enumerate(self.counts):
            acc += c
            if acc >= target:
                return (i + 0.5) * self.hi / self.bins
        return self.max_seen

    def summary(self) -> Dict[str, float]:
        return {"p50": self.quantile(0.50), "p95": self.quantile(0.95),
                "p99": self.quantile(0.99), "p99_9": self.quantile(0.999),
                "max": self.max_seen}


def sg_velocity(window: Sequence[float], dt: float) -> float:
    """
    Centred least-squares first derivative -- the Savitzky-Golay derivative for
    a linear or quadratic fit, which reduces to sum(k*y_k) / (dt * sum(k^2)).

    Differentiated encoder position at 1 kHz is noise-dominated, and that noise
    lands straight in omega, which drives iron loss and back-EMF headroom. A
    plain two-point difference would make the speed axis junk.
    """
    n = len(window)
    if n < 3:
        return 0.0
    h = n // 2
    num = sum((i - h) * window[i] for i in range(n))
    den = dt * sum((i - h) ** 2 for i in range(n))
    return num / den if den else 0.0


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------

@dataclass
class ReadStats:
    rows: int = 0
    dropped_nan: int = 0
    gaps: List[Tuple[float, float]] = field(default_factory=list)
    covered_time: float = 0.0
    span: float = 0.0
    dt_median: float = 0.0
    out_of_order: int = 0

    @property
    def coverage(self) -> float:
        return self.covered_time / self.span if self.span > 0 else 1.0


def _rows(paths: Sequence[str], cmap: ColumnMap) -> Iterator[Dict[str, float]]:
    """Yield {field: value in canonical SI} per row, streaming, never buffering."""
    scales = {k: cmap.scale(k) for k in cmap.columns}
    t_offset = 0.0
    last_t = None
    for path in paths:
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None:
                raise LogError(f"{path}: no header row")
            missing = [c for c in cmap.columns.values()
                       if c not in reader.fieldnames]
            if missing:
                raise LogError(
                    f"{path}: column(s) {', '.join(missing)} not in the header "
                    f"({', '.join(reader.fieldnames)})")
            file_first = None
            file_last = 0.0
            for raw in reader:
                out = {}
                bad = False
                for field_name, col in cmap.columns.items():
                    txt = (raw.get(col) or "").strip()
                    if not txt:
                        bad = True
                        break
                    try:
                        v = float(txt)
                    except ValueError:
                        bad = True
                        break
                    if v != v or v in (float("inf"), float("-inf")):
                        bad = True
                        break
                    out[field_name] = v * scales[field_name]
                if bad:
                    yield {"_nan": 1.0}
                    continue
                if "time" in out:
                    if file_first is None:
                        file_first = out["time"]
                    file_last = out["time"]
                    out["time"] += t_offset
                yield out
            # Multiple files are concatenated in the order given; the clock is
            # rebased so a second file that restarts at zero does not look like
            # a gigantic backwards jump.
            if file_first is not None and last_t is not None and \
                    file_first + t_offset < last_t:
                t_offset = last_t - file_first
            if file_first is not None:
                last_t = file_last + t_offset


def accepted_rows(paths: Sequence[str], cmap: ColumnMap,
                  dt_override: Optional[float] = None) -> Iterator[Dict[str, float]]:
    """
    Rows both passes agree to keep, in order.

    Every sweep must make the same accept/reject decisions, because the window
    spans are SAMPLE INDICES into this sequence. If pass 2 kept a row that pass
    1 skipped -- an out-of-order timestamp, a blank field -- the two indexings
    would slide apart and the stored windows would gradually point at the wrong
    part of the session. Routing both through one generator makes that
    impossible rather than merely unlikely.
    """
    prev_t = None
    for row in _rows(paths, cmap):
        if "_nan" in row:
            continue
        if "time" in row:
            if prev_t is not None and row["time"] - prev_t <= 0:
                continue
            prev_t = row["time"]
        elif dt_override is None:
            raise LogError(
                "the log has no time column, so --dt must state the "
                "controller's fixed update interval.")
        yield row


@dataclass
class Pass1:
    """Everything learnable in one streaming sweep."""
    tau_q: Quantiles
    om_q: Quantiles
    boundary_pts: List[Tuple[float, float]]
    boundary_raw_pts: List[Tuple[float, float]]
    energy: List[float]           # decimated mean tau^2, one per energy_dt
    energy_dt: float              # true seconds per element; see read_pass1
    energy_stride: int            # ACCEPTED samples per element (exact)
    stats: ReadStats
    tau_max_raw: float = 0.0      # peak before despiking, so filtering is visible
    despike_changed: int = 0
    despike_worst: float = 0.0
    pos_hist: Dict[int, float] = field(default_factory=dict)
    grav_pos: List[float] = field(default_factory=list)
    grav_tau: List[float] = field(default_factory=list)


def _staircase(points: Iterable[Tuple[float, float]],
               keep: int = 4000) -> List[Tuple[float, float]]:
    """Thin a point cloud before the outline scan, keeping the outer edge."""
    best: Dict[int, Tuple[float, float]] = {}
    for om, tau in points:
        k = int(abs(om) * 200)
        cur = best.get(k)
        if cur is None or abs(tau) > abs(cur[1]):
            best[k] = (om, tau)
    return boundary_scan(list(best.values()))


def read_pass1(paths: Sequence[str], cmap: ColumnMap, *,
               despike: int = 5, dt_override: Optional[float] = None,
               gap_factor: float = 5.0, kt_joint: Optional[float] = None,
               flip_torque: bool = False, flip_speed: bool = False) -> Pass1:
    """
    First sweep: distribution, outline, and a decimated energy series.

    Nothing here needs a threshold, which is exactly why it can be done before
    the thresholds exist.
    """
    tau_q, om_q = Quantiles(), Quantiles()
    med_t, med_w = MedianFilter(despike), MedianFilter(despike)
    stats = ReadStats()
    raw_pts: List[Tuple[float, float]] = []
    dsp_pts: List[Tuple[float, float]] = []
    energy: List[float] = []
    e_acc = 0.0
    e_n = 0
    e_stride = 0
    e_warmup: List[float] = []          # samples per decimated element; fixed once dt is known
    pos_hist: Dict[int, float] = {}
    grav_pos: List[float] = []
    grav_tau: List[float] = []

    pos_win: List[float] = []
    prev_t = None
    dts: List[float] = []
    first_t = None
    tau_max_raw = 0.0

    for row in _rows(paths, cmap):
        if "_nan" in row:
            stats.dropped_nan += 1
            continue
        # ---- timebase ----------------------------------------------------
        # A row rejected here must not be counted, because pass 2 walks the same
        # accept/reject decisions and the two indexings have to agree exactly --
        # the window spans are sample indices into that shared sequence.
        if "time" in row:
            t = row["time"]
            if first_t is None:
                first_t = t
            if prev_t is not None:
                dt = t - prev_t
                if dt <= 0:
                    stats.out_of_order += 1
                    continue
                if len(dts) < 5000:
                    dts.append(dt)
            prev_t = t
            stats.rows += 1
        else:
            if dt_override is None:
                raise LogError(
                    "the log has no time column, so --dt must state the "
                    "controller's fixed update interval.")
            stats.rows += 1
            t = (first_t or 0.0) + stats.rows * dt_override
            if first_t is None:
                first_t = 0.0
            prev_t = t

        # ---- torque ------------------------------------------------------
        if "torque" in row:
            tau = row["torque"]
        else:
            if kt_joint is None:
                raise LogError(
                    "the log carries current rather than torque, so "
                    "--kt-joint must state the joint-referred N.m per amp. "
                    "That conversion is actuator-specific and cannot be "
                    "guessed without silently baking in an assumption.")
            tau = row["current"] * kt_joint
        if flip_torque:
            tau = -tau

        # ---- speed -------------------------------------------------------
        if "speed" in row:
            om = row["speed"]
        else:
            pos_win.append(row["position"])
            if len(pos_win) > max(5, despike | 1):
                pos_win.pop(0)
            om = sg_velocity(pos_win, dt_override or
                             (dts[len(dts) // 2] if dts else 0.001))
        if flip_speed:
            om = -om

        raw_pts.append((om, tau))
        tau_max_raw = max(tau_max_raw, abs(tau))
        tau_d, om_d = med_t.push(tau), med_w.push(om)
        dsp_pts.append((om_d, tau_d))
        if len(raw_pts) > 200000:            # thin as we go, bounded memory
            raw_pts = _staircase(raw_pts)
            dsp_pts = _staircase(dsp_pts)

        step = dt_override or (dts[len(dts) // 2] if dts else 0.001)
        tau_q.add(tau_d, step)
        om_q.add(om_d, step)

        # Decimated tau^2 series for the window search. Bucketed by a fixed
        # SAMPLE COUNT rather than by accumulating dt until it passes a
        # threshold: the accumulate-and-compare form overshoots by up to one
        # sample every bucket, and at 1 kHz that 1% drift compounds to seconds
        # by the end of a long log, so every window lands early.
        #
        # Element k must cover exactly samples [k*stride, (k+1)*stride) so that
        # pass 2 can turn an element index straight back into a sample index.
        # The stride is not known until enough intervals have been seen, so the
        # opening samples are buffered rather than emitted short -- emitting
        # them would insert extra elements and slide every later index.
        if e_stride == 0:
            e_warmup.append(tau_d * tau_d)
            if dt_override or len(dts) >= 64:
                e_stride = max(1, int(round(DECIMATED_DT / max(step, 1e-9))))
                for v in e_warmup:                # replay, now correctly sized
                    e_acc += v
                    e_n += 1
                    if e_n >= e_stride:
                        energy.append(e_acc / e_n)
                        e_acc, e_n = 0.0, 0
                e_warmup = []
        else:
            e_acc += tau_d * tau_d
            e_n += 1
            if e_n >= e_stride:
                energy.append(e_acc / e_n)
                e_acc, e_n = 0.0, 0

        if "position" in row:
            k = int(math.degrees(row["position"]) // 10)
            pos_hist[k] = pos_hist.get(k, 0.0) + step
            # Near-stationary samples carry the gravity signature, which is the
            # free independent check on the torque sign convention.
            if abs(om_d) < 0.05 and len(grav_pos) < 20000:
                grav_pos.append(row["position"])
                grav_tau.append(tau_d)

    if e_stride == 0 and e_warmup:      # log shorter than the warm-up
        e_stride = max(1, int(round(DECIMATED_DT / max(stats.dt_median or 0.001, 1e-9))))
        for v in e_warmup:
            e_acc += v
            e_n += 1
            if e_n >= e_stride:
                energy.append(e_acc / e_n)
                e_acc, e_n = 0.0, 0
    # A trailing partial bucket is dropped: an element covering fewer
    # samples than the stride would break the index->sample mapping the
    # window search depends on, and it is at most 0.1 s of the session.

    dts.sort()
    stats.dt_median = dt_override or (dts[len(dts) // 2] if dts else 0.001)
    if prev_t is not None and first_t is not None:
        stats.span = prev_t - first_t + stats.dt_median
    stats.covered_time = stats.rows * stats.dt_median
    changed, worst = med_t.summary()

    if e_stride == 0:
        e_stride = max(1, int(round(DECIMATED_DT / max(stats.dt_median, 1e-9))))

    return Pass1(tau_q=tau_q, om_q=om_q,
                 boundary_pts=_staircase(dsp_pts),
                 boundary_raw_pts=_staircase(raw_pts),
                 energy=energy, energy_dt=e_stride * stats.dt_median,
                 energy_stride=e_stride, stats=stats,
                 tau_max_raw=tau_max_raw,
                 despike_changed=changed, despike_worst=worst,
                 pos_hist=pos_hist, grav_pos=grav_pos, grav_tau=grav_tau)


# ---------------------------------------------------------------------------
# Window search
# ---------------------------------------------------------------------------

def kernel_worst_moment(energy: Sequence[float], tau_w: float,
                        dt: float = DECIMATED_DT) -> int:
    """
    Index of the moment a winding of time constant `tau_w` was hottest.

    A single-pole low-pass on tau^2 IS the shape of the winding's thermal
    response, and Kt, R and Rth all cancel out of the ARGMAX -- they scale the
    output but not which moment wins. That is what makes this selection
    actuator-independent, which is what lets one capture be evaluated against
    any candidate. Measured across the shipped db (tau_w 10.7-17.1 s) the same
    moment is selected throughout, and stays stable from 5 s to 25 s.
    """
    if not energy:
        return 0
    alpha = math.exp(-dt / max(tau_w, 1e-9))
    y = 0.0
    best_y, best_i = -1.0, 0
    for i, e in enumerate(energy):
        y = alpha * y + (1 - alpha) * e
        if y > best_y:
            best_y, best_i = y, i
    return best_i


def worst_window(energy: Sequence[float], window_s: float,
                 dt: float = DECIMATED_DT) -> Optional[Tuple[int, float]]:
    """
    (start index, RMS torque) of the worst contiguous window of this length.

    A boxcar over the window's own length, not the exponential kernel. The
    kernel answers "when was the winding hottest", which is the right question
    for CHOOSING which timescales matter; but once a length is fixed, the window
    to store is the one that actually maximises the mean tau^2 over that length.
    Using the kernel's argmax for both put the window trailing the peak instead
    of centred on the event, so a 100 s window around a 45 s burst carried 55 s
    of idle and reported an RMS well below the duration curve's own value for
    100 s -- two numbers in the same report disagreeing about the same fact.
    """
    n = int(round(window_s / dt))
    if n < 1 or n > len(energy):
        return None
    run = sum(energy[:n])
    best, best_i = run, 0
    for i in range(n, len(energy)):
        run += energy[i] - energy[i - n]
        if run > best:
            best, best_i = run, i - n + 1
    return best_i, math.sqrt(max(best, 0.0) / n)


def sustained_rms(energy: Sequence[float], window_s: float,
                  dt: float = DECIMATED_DT) -> Optional[float]:
    """Highest RMS torque sustained over any window of this length."""
    got = worst_window(energy, window_s, dt)
    return got[1] if got else None


def bucket_by_energy(chunk: Sequence[Tuple[float, float, float]],
                     max_segments: int = MAX_WINDOW_SEGMENTS
                     ) -> List[Tuple[float, float, float]]:
    """
    Compress an ordered excerpt to <= max_segments, preserving order.

    Bucketed by equal tau^2 ENERGY rather than equal time, so a 300 s window
    does not smear its 2 s torque spike into a 2 s average. Total time and total
    energy both come out exact; resolution goes where the loss is.
    """
    if not chunk:
        return []
    total_e = sum(t * t * dt for dt, t, _ in chunk)
    if total_e <= 0:
        dur = sum(c[0] for c in chunk)
        return [(dur, 0.0, sum(c[2] * c[0] for c in chunk) / max(dur, 1e-12))]
    per = total_e / max_segments
    out: List[Tuple[float, float, float]] = []
    acc_e = acc_t = acc_sq = acc_w = 0.0
    for dt, tau, om in chunk:
        acc_e += tau * tau * dt
        acc_t += dt
        acc_sq += tau * tau * dt
        acc_w += om * dt
        if acc_e >= per and acc_t > 0:
            out.append((acc_t, math.sqrt(acc_sq / acc_t), acc_w / acc_t))
            acc_e = acc_t = acc_sq = acc_w = 0.0
    if acc_t > 0:
        out.append((acc_t, math.sqrt(acc_sq / acc_t), acc_w / acc_t))
    return out


def build_ladder(session_s: float, dt: float,
                 extra: Sequence[float] = ()) -> List[float]:
    """
    Winding constants to probe: a fixed floor, extended to suit this log.

    The floor keeps envelopes comparable across captures. The extensions cover
    the cases the floor cannot: a session too short for its longest windows, and
    a log fast enough to resolve sub-second ones.
    """
    lo = max(10 * dt, 0.1)
    hi = session_s / 3.0
    out = {t for t in TAU_W_LADDER if lo <= t <= hi}
    out.update(t for t in extra if lo <= t <= hi)
    if not out:
        out = {max(lo, min(hi, session_s / 4.0))}
    return sorted(out)


def refine_ladder(energy: Sequence[float], ladder: Sequence[float],
                  dt: float = DECIMATED_DT) -> List[float]:
    """
    Add constants where the binding event CHANGES CHARACTER.

    When two neighbouring constants pick moments far apart in the session, the
    crossover between them is where a different excursion takes over as the
    worst one -- exactly the transition the sweep exists to find. Bisecting
    there beats assuming the fixed ladder happened to straddle it.
    """
    if len(ladder) < 2 or not energy:
        return list(ladder)
    picks = {t: kernel_worst_moment(energy, t, dt) for t in ladder}
    extra = []
    for a, b in zip(ladder, ladder[1:]):
        # "Far apart" means the two windows do not even overlap.
        if abs(picks[a] - picks[b]) * dt > max(a, b):
            extra.append(math.sqrt(a * b))
    return sorted(set(list(ladder) + extra))


# ---------------------------------------------------------------------------
# Pass 2: thresholds known, so excursions and occupancy can be built
# ---------------------------------------------------------------------------

def build_envelope(paths: Sequence[str], cmap: ColumnMap, p1: Pass1, *,
                   name: str = "envelope", despike: int = 5,
                   dt_override: Optional[float] = None,
                   kt_joint: Optional[float] = None,
                   flip_torque: bool = False, flip_speed: bool = False,
                   n_torque_bins: int = 24, n_speed_bins: int = 20,
                   outlier_percentile: float = 0.99,
                   tau_w_ladder: Optional[Sequence[float]] = None) -> Envelope:
    """
    Second sweep. Everything that needed a threshold happens here.

    Deliberately re-reads the file rather than caching pass 1's samples: caching
    would make peak memory proportional to log length, which is the one thing
    this module promises never to do.
    """
    env = Envelope(name=name)
    stats = p1.stats
    dt = stats.dt_median
    tau_cut = p1.tau_q.quantile(outlier_percentile)
    om_cut = p1.om_q.quantile(outlier_percentile)

    # The cut must sit strictly below the peak or nothing is above it, which
    # happens whenever one busy state occupies more than (1 - percentile) of the
    # session -- exactly the case that must not be lost.
    if tau_cut >= p1.tau_q.max_seen:
        tau_cut = p1.tau_q.max_seen * 0.999
    env.binned_below = tau_cut

    # ---- stream once, splitting excursions from the binnable population ----
    med_t, med_w = MedianFilter(despike), MedianFilter(despike)
    pos_win: List[float] = []
    samples_for_edges: List[Tuple[float, float]] = []
    outliers: List[Outlier] = []
    run: List[Tuple[float, float, float]] = []
    run_start = 0.0
    clock = 0.0
    thresholds = {k: p1.tau_q.quantile(q)
                  for k, q in (("p95", 0.95), ("p99", 0.99), ("p99_9", 0.999))}
    ev_counts = {k: {"threshold": v, "events": 0, "total_time": 0.0,
                     "max_duration": 0.0, "_cur": 0.0}
                 for k, v in thresholds.items()}
    binnable: Dict[Tuple[int, int], List[float]] = {}
    pending: List[Tuple[float, float, float]] = []

    def flush_run():
        if not run:
            return
        dur = sum(s[0] for s in run)
        pk = max(run, key=lambda s: abs(s[1]))
        mean = sum(s[1] * s[0] for s in run) / max(dur, 1e-12)
        rms = math.sqrt(sum(s[1] * s[1] * s[0] for s in run) / max(dur, 1e-12))
        outliers.append(Outlier(run_start, dur, pk[1], pk[2], mean,
                                math.copysign(rms, mean or 1.0)))

    for row in accepted_rows(paths, cmap, dt_override):
        tau = (row["torque"] if "torque" in row
               else row["current"] * (kt_joint or 0.0))
        if flip_torque:
            tau = -tau
        if "speed" in row:
            om = row["speed"]
        else:
            pos_win.append(row["position"])
            if len(pos_win) > max(5, despike | 1):
                pos_win.pop(0)
            om = sg_velocity(pos_win, dt)
        if flip_speed:
            om = -om
        tau, om = med_t.push(tau), med_w.push(om)

        for key, c in ev_counts.items():
            if abs(tau) > c["threshold"]:
                c["_cur"] += dt
            elif c["_cur"] > 0:
                c["events"] += 1
                c["total_time"] += c["_cur"]
                c["max_duration"] = max(c["max_duration"], c["_cur"])
                c["_cur"] = 0.0

        hot = abs(tau) > tau_cut or (om_cut > 0 and abs(om) > om_cut)
        if hot:
            if not run:
                run_start = clock
            run.append((dt, tau, om))
        else:
            flush_run()
            run = []
            pending.append((dt, tau, om))
            if len(samples_for_edges) < 400000:
                samples_for_edges.append((tau, om))
        clock += dt
    flush_run()
    for c in ev_counts.values():
        if c["_cur"] > 0:
            c["events"] += 1
            c["total_time"] += c["_cur"]
            c["max_duration"] = max(c["max_duration"], c["_cur"])
        c.pop("_cur", None)
        c["mean_duration"] = c["total_time"] / max(c["events"], 1)

    # Keep the worst by peak AND by duration: they fail different criteria, so
    # trimming by either alone would hide the other.
    if len(outliers) > 200:
        order = sorted(range(len(outliers)), key=lambda i: -abs(outliers[i].tau_peak))
        keep = set(order[:100])
        for i in sorted(range(len(outliers)), key=lambda i: -outliers[i].duration):
            if len(keep) >= 200:
                break
            keep.add(i)
        dropped = [o for i, o in enumerate(outliers) if i not in keep]
        outliers = [o for i, o in enumerate(outliers) if i in keep]
        # Dropped excursions are still time the joint spent under load. They go
        # back into the binned population rather than vanishing -- losing them
        # would understate the thermal average by the busiest moments in the log.
        for o in dropped:
            pending.append((o.duration, o.tau_rms, o.omega_at_peak))
            samples_for_edges.append((o.tau_rms, o.omega_at_peak))
    env.outliers = outliers

    # ---- occupancy on quantile edges, now that the population is known -----
    if samples_for_edges:
        w = [dt] * len(samples_for_edges)
        env.torque_edges = quantile_edges([s[0] for s in samples_for_edges], w,
                                          n_torque_bins)
        env.speed_edges = quantile_edges([s[1] for s in samples_for_edges], w,
                                         n_speed_bins)
    else:
        env.torque_edges = env.speed_edges = [-1.0, 0.0, 1.0]

    for seg_dt, tau, om in pending:
        key = (_bin_index(env.torque_edges, tau), _bin_index(env.speed_edges, om))
        a = binnable.setdefault(key, [0.0, 0.0, 0.0, 0.0])
        a[0] += seg_dt
        a[1] += tau * seg_dt
        a[2] += om * seg_dt
        a[3] += tau * tau * seg_dt
    env.cells = [
        Cell(ti, si, s, math.copysign(math.sqrt(sq / s), tsum or 1.0),
             wsum / s, tsum / s)
        for (ti, si), (s, tsum, wsum, sq) in sorted(binnable.items()) if s > 0
    ]
    for k, o in enumerate(env.outliers):
        env.cells.append(Cell(-1, -(k + 1), o.duration, o.tau_rms,
                              o.omega_at_peak, o.tau_mean))
    env.total_time = sum(c.seconds for c in env.cells)

    # ---- artifacts that need no second look --------------------------------
    env.extremes = Extremes(torque=p1.tau_q.summary(), speed=p1.om_q.summary())
    # The unfiltered peak, so the report can show what despiking removed
    # rather than quietly presenting the filtered number as the truth.
    if p1.tau_max_raw > 0:
        env.extremes.torque["max_raw"] = p1.tau_max_raw
    env.boundary = list(p1.boundary_pts)
    env.boundary_raw = list(p1.boundary_raw_pts)
    env.event_counts = ev_counts
    env.duration_curve = [
        (w, v) for w, v in ((w, sustained_rms(p1.energy, w, p1.energy_dt))
                            for w in DURATION_CURVE_WINDOWS) if v is not None]

    # ---- the window ladder --------------------------------------------------
    session = stats.covered_time
    ladder = list(tau_w_ladder) if tau_w_ladder else build_ladder(session, dt)
    ladder = refine_ladder(p1.energy, ladder, p1.energy_dt)
    env.windows = _extract_windows(paths, cmap, p1, ladder, dt=dt,
                                   despike=despike, kt_joint=kt_joint,
                                   flip_torque=flip_torque,
                                   flip_speed=flip_speed,
                                   dt_override=dt_override)

    env.dt_nominal = dt
    env.despike.width_samples = despike if despike > 1 else 0
    env.despike.width_s = despike * dt if despike > 1 else None
    env.despike.samples_changed = p1.despike_changed
    env.despike.worst_change = p1.despike_worst
    env.capture.duration = session
    env.capture.coverage = stats.coverage
    env.capture.gaps = list(stats.gaps)
    env.capture.sample_rate = 1.0 / dt if dt > 0 else None
    return env


def _extract_windows(paths, cmap, p1: Pass1, ladder: Sequence[float], *,
                     dt: float, despike: int, kt_joint, flip_torque,
                     flip_speed, dt_override=None) -> List[Window]:
    """
    One window per probed constant, at that constant's own duration.

    A 3 s window and a 300 s window are different lengths holding different
    numbers of points, so they are always distinct objects answering different
    questions -- can the winding survive three seconds of this, versus can the
    case survive five minutes of it -- even when they overlap the same busy
    stretch. Only identical (constant, moment) results are merged.
    """
    if not ladder or not p1.energy:
        return []

    # Spans are held as SAMPLE INDEX ranges, converted from the decimated
    # series' own stride. Comparing an accumulated clock against a start time
    # would drift, and pass 2's clock need not even share an origin with pass
    # 1's -- the log's first timestamp is rarely zero. Indices are exact.
    stride = max(1, p1.energy_stride)
    wanted = []
    for tau_w in ladder:
        got = worst_window(p1.energy, tau_w, p1.energy_dt)
        if got is None:
            continue
        start_i, _ = got
        i0 = start_i * stride
        n = max(1, int(round(tau_w / max(dt, 1e-9))))
        wanted.append([tau_w, i0, n, []])

    # One more sweep, collecting only the samples inside a wanted span.
    med_t, med_w = MedianFilter(despike), MedianFilter(despike)
    pos_win: List[float] = []
    idx = 0
    for row in accepted_rows(paths, cmap, dt_override):
        tau = (row["torque"] if "torque" in row
               else row["current"] * (kt_joint or 0.0))
        if flip_torque:
            tau = -tau
        if "speed" in row:
            om = row["speed"]
        else:
            pos_win.append(row["position"])
            if len(pos_win) > max(5, despike | 1):
                pos_win.pop(0)
            om = sg_velocity(pos_win, dt)
        if flip_speed:
            om = -om
        tau, om = med_t.push(tau), med_w.push(om)
        for w in wanted:
            if w[1] <= idx < w[1] + w[2]:
                w[3].append((dt, tau, om))
        idx += 1

    out: List[Window] = []
    merged: Dict[Tuple[int, int], Window] = {}
    for tau_w, i0, n, chunk in wanted:
        if not chunk:
            continue
        segs = bucket_by_energy(chunk)
        total = sum(s[0] for s in segs) or 1e-12
        rms = math.sqrt(sum(s[1] * s[1] * s[0] for s in segs) / total)
        key = (n, i0)
        if key in merged:
            merged[key].selected_by_tau_w.append(tau_w)
            continue
        w = Window(duration=total, found_at=i0 * dt, rms_torque=rms,
                   segments=segs, selected_by_tau_w=[tau_w])
        merged[key] = w
        out.append(w)
    out.sort(key=lambda w: w.duration)
    return out
