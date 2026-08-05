#!/usr/bin/env python3
"""
Summarise a joint's controller log into an application envelope.

    ./envelope_from_log.py --log elbow_2026-07-28.csv \
        --map time=t_s,position=q_rad,torque=tau_cmd_Nm \
        --units time=s,position=rad,torque=N.m \
        --ratio 1.0 --torque-source commanded_current \
        --name elbow_pick_place_2026_07 --robot arm-004 --joint-id elbow \
        --out envelopes/elbow_pick_place_2026_07.json

The logger only has to append three columns -- time, position, torque -- at
whatever fixed rate the controller already runs. All the summarising happens
here, afterwards, so nothing has to be computed on the robot.

The result is tens of KB of reviewable JSON that an application file can point
at, and that `eval_actuator.py` will size any candidate actuator against. Commit
it: a recapture then produces a diff you can read.

Deliberately a separate script rather than a subcommand of eval_actuator.py.
The two have disjoint arguments and disjoint outputs, and eval_actuator.py's
flat argparse would have to be restructured for every existing user to gain a
subcommand it does not otherwise need.
"""

import argparse
import hashlib
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from actuator_eval import logread as LR                      # noqa: E402
from actuator_eval import units as U                         # noqa: E402
from actuator_eval.envelope import TORQUE_SOURCE_RANK, write_json  # noqa: E402

RPM = 30.0 / math.pi


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def gravity_sign_check(p1: LR.Pass1, gravity_angle_deg=None):
    """
    Advisory: does positive torque hold the link UP against gravity?

    This is a convenience for whoever OWNS THE APPLICATION, not a requirement on
    the log. What the evaluation actually depends on is the sign of tau*omega --
    whether the joint is motoring or regenerating, which decides whether gearbox
    efficiency divides or multiplies the demand. Flipping torque and speed
    together leaves every criterion identical (measured: bit-for-bit); flipping
    torque alone moves the thermal margin by ~5% on a geared joint.

    So the absolute sense of "positive torque" is a property of the APPLICATION,
    not of the capture, and a logger should never have to know about gravity
    angles to hand over a usable file. Run this when you happen to have position
    and a gravity_angle to hand; it costs nothing and catches a genuinely
    inverted convention. It does not gate the capture.

    At rest the holding torque opposes gravity, so tau correlates NEGATIVELY
    with sin(theta - theta_gravity).

    Returns (verdict, correlation, n) with verdict one of ok / flipped /
    inconclusive.
    """
    xs, ys = p1.grav_pos, p1.grav_tau
    n = len(xs)
    if n < 200:
        return "inconclusive", 0.0, n
    theta_g = math.radians(gravity_angle_deg or 0.0)
    a = [math.sin(x - theta_g) for x in xs]
    if max(a) - min(a) < 0.15:
        # The joint barely moved while stationary, so there is no lever arm
        # variation to correlate against.
        return "inconclusive", 0.0, n
    ma = sum(a) / n
    mb = sum(ys) / n
    num = sum((x - ma) * (y - mb) for x, y in zip(a, ys))
    da = math.sqrt(sum((x - ma) ** 2 for x in a))
    db = math.sqrt(sum((y - mb) ** 2 for y in ys))
    if da <= 0 or db <= 0:
        return "inconclusive", 0.0, n
    r = num / (da * db)
    if r < -0.3:
        return "ok", r, n
    if r > 0.3:
        return "flipped", r, n
    return "inconclusive", r, n


def regen_fraction(p1: LR.Pass1) -> float:
    """Fraction of moving time with tau*omega < 0, i.e. the load driving back."""
    return p1.regen_time / max(p1.moving_time, 1e-9)


def torque_speed_consistency(p1: LR.Pass1):
    """
    Are torque and speed signed in the same frame? Decided from physics alone.

    Net mechanical energy over a session must be POSITIVE for a joint doing real
    work: friction dissipates, and gravity returns at most what it took, so a
    joint can never give back more than it consumed. If the log says otherwise,
    torque and speed disagree about which way is positive.

    This is why the log format asks for no sign convention at all. The absolute
    sense of "positive" is unobservable -- negating torque and speed together is
    a change of frame that leaves every criterion identical -- and the relative
    sense, the only part that matters, is recoverable here without anyone having
    to declare anything.

    Note the regen FRACTION cannot do this job: a joint that lifts and lowers
    the same load spends comparable time in each, so the fraction sits near 0.5
    whichever way the sign goes (measured: 0.505 vs 0.495 on the bundled
    example). Energy is asymmetric where time is not.

    Returns (verdict, net_joules) with verdict one of ok / inverted /
    inconclusive.
    """
    net = p1.energy_motoring - p1.energy_regen
    scale = p1.energy_motoring + p1.energy_regen
    if scale <= 0:
        return "inconclusive", 0.0
    # A margin, because a nearly-lossless joint legitimately sits near zero.
    if net > 0.02 * scale:
        return "ok", net
    if net < -0.02 * scale:
        return "inverted", net
    return "inconclusive", net


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--log", action="append", required=True, metavar="FILE.csv",
                    help="controller log; repeat for a multi-file session")
    ap.add_argument("--map", action="append", default=[], metavar="K=COL",
                    help="column mapping, e.g. time=t_s,torque=tau_cmd_Nm. "
                         "Needs time + (torque|current) + (speed|position)")
    ap.add_argument("--map-file", default=None,
                    help="JSON file holding {'map': {...}, 'units': {...}} so a "
                         "fleet's standard mapping is committed once")
    ap.add_argument("--units", action="append", default=[], metavar="K=UNIT",
                    help="units per mapped column, e.g. torque=N.m. Never "
                         "assumed: a torque read as N.m when it was kgf.cm is "
                         "wrong by 100x and still evaluates")
    ap.add_argument("--dt", type=float, default=None,
                    help="controller update interval, if the log has no time column")
    ap.add_argument("--kt-joint", type=float, default=None,
                    help="joint-referred N.m per logged amp, if logging current "
                         "rather than torque")
    ap.add_argument("--despike", type=int, default=5, metavar="N",
                    help="median filter width in samples (default 5, 0 to "
                         "disable). A median rejects isolated single-sample "
                         "spikes while passing genuine short events through "
                         "undistorted; a moving average cannot do both")
    ap.add_argument("--flip-torque", action="store_true",
                    help="negate the logged torque")
    ap.add_argument("--flip-speed", action="store_true",
                    help="negate the logged speed")
    ap.add_argument("--gravity-angle", type=float, default=None, metavar="DEG",
                    help="joint angle where the CG points straight down, from "
                         "the application file; enables the sign check")
    ap.add_argument("--min-coverage", type=float, default=0.95,
                    help="refuse below this fraction of the session accounted "
                         "for (default 0.95)")
    ap.add_argument("--allow-gaps", action="store_true",
                    help="proceed despite coverage below --min-coverage")
    ap.add_argument("--ratio", type=float, default=1.0,
                    help="post-actuator ratio the capture was taken at; must "
                         "match the application's")
    ap.add_argument("--n-actuators", type=int, default=None,
                    help="how many actuators drove the joint during capture "
                         "(informational: joint-referred torque does not depend "
                         "on it)")
    ap.add_argument("--torque-source", default="commanded_current",
                    choices=sorted(TORQUE_SOURCE_RANK),
                    help="where the torque number came from; sets the "
                         "envelope's provenance and caps every verdict built "
                         "on it")
    ap.add_argument("--tau-w", default=None, metavar="LIST",
                    help="override the winding-constant ladder, e.g. 2,20,200")
    ap.add_argument("--name", default=None, help="envelope name")
    ap.add_argument("--robot", default="", help="robot id, for provenance")
    ap.add_argument("--joint-id", default="", help="joint id, for provenance")
    ap.add_argument("--firmware", default="", help="firmware version")
    ap.add_argument("--captured-at", default="", help="ISO timestamp of capture")
    ap.add_argument("--note", default="", help="free text recorded in the file")
    ap.add_argument("-o", "--out", default=None,
                    help="where to write the envelope json")
    args = ap.parse_args(argv)

    # ---- column mapping -------------------------------------------------
    cmap = LR.ColumnMap()
    if args.map_file:
        with open(args.map_file) as f:
            spec = json.load(f)
        cmap.columns.update(spec.get("map", {}))
        cmap.units.update(spec.get("units", {}))
    try:
        for m in args.map:
            cmap.columns = LR.parse_map_arg(m, cmap.columns)
        for m in args.units:
            cmap.units = LR.parse_map_arg(m, cmap.units)
        cmap.validate()
    except (LR.LogError, U.UnitError) as e:
        ap.error(str(e))

    for p in args.log:
        if not os.path.exists(p):
            ap.error(f"no such log file: {p}")

    despike = max(args.despike, 0)
    name = args.name or os.path.splitext(os.path.basename(args.log[0]))[0]

    print(f"reading {len(args.log)} log file(s)...")
    try:
        p1 = LR.read_pass1(args.log, cmap, despike=despike,
                           dt_override=args.dt, kt_joint=args.kt_joint,
                           flip_torque=args.flip_torque,
                           flip_speed=args.flip_speed)
    except (LR.LogError, U.UnitError) as e:
        ap.error(str(e))

    st = p1.stats
    if st.rows == 0:
        ap.error("the log had no usable rows")

    # ---- integrity, before anything is written --------------------------
    print()
    print(f"  rows                : {st.rows:,}"
          + (f"  ({st.dropped_nan:,} dropped as blank/NaN)"
             if st.dropped_nan else ""))
    print(f"  sample interval     : {st.dt_median*1e3:.3f} ms "
          f"({1.0/max(st.dt_median,1e-9):.0f} Hz)")
    print(f"  session length      : {st.covered_time:.1f} s "
          f"({st.covered_time/3600.0:.2f} hr)")
    print(f"  coverage            : {st.coverage*100:.2f}%")
    if st.out_of_order:
        print(f"  ** {st.out_of_order:,} out-of-order timestamps were skipped")
    if st.coverage < args.min_coverage and not args.allow_gaps:
        print()
        print(f"REFUSING: coverage {st.coverage*100:.1f}% is below "
              f"{args.min_coverage*100:.0f}%. A histogram missing a large "
              f"fraction of the session has a meaningless mean. Pass "
              f"--allow-gaps to override.")
        return 2

    changed, worst = p1.despike_changed, p1.despike_worst
    if despike > 1:
        print(f"  despike (median {despike}) : {changed:,} samples materially "
              f"changed, worst by {worst:.2f} N.m")
        raw_pk = max((abs(t) for _, t in p1.boundary_raw_pts), default=0.0)
        dsp_pk = max((abs(t) for _, t in p1.boundary_pts), default=0.0)
        if raw_pk > dsp_pk * 1.5 and dsp_pk > 0:
            print(f"  ** raw peak {raw_pk:.1f} N.m vs despiked {dsp_pk:.1f} N.m: "
                  f"the log has substantial single-sample noise. That ratio is "
                  f"worth understanding before trusting the capture.")

    # The sign relationship the evaluation actually consumes. Needs no gravity
    # angle and no absolute convention -- only that torque and speed agree with
    # each other about which way the joint is going.
    if p1.moving_time > 0:
        rf = regen_fraction(p1)
        print(f"  motoring/regen split: {(1-rf)*100:.0f}% / {rf*100:.0f}% "
              f"of moving time")
        verdict, net = torque_speed_consistency(p1)
        if verdict == "ok":
            print(f"  torque/speed frames : consistent (net mechanical energy "
                  f"{net:+.0f} J)")
        elif verdict == "inverted":
            print()
            print(f"REFUSING: torque and speed are signed in OPPOSITE frames.")
            print(f"Net mechanical energy over the session is {net:+.0f} J, "
                  f"i.e. the log says the")
            print(f"joint returned more energy than it consumed. Friction "
                  f"dissipates and gravity")
            print(f"returns at most what it took, so that is not physical -- "
                  f"one of the two signals")
            print(f"is negated relative to the other.")
            print(f"Re-run with --flip-torque (or --flip-speed) to correct it.")
            return 2
        else:
            print(f"  torque/speed frames : inconclusive (net mechanical "
                  f"energy {net:+.0f} J, too near zero to judge)")

    # Advisory only. Which direction is "positive torque" is a property of the
    # APPLICATION, not of the capture, so a bad answer here is a note rather
    # than a refusal -- see --flip-torque, or set the application's frame.
    verdict, r, n = gravity_sign_check(p1, args.gravity_angle)
    if verdict == "ok":
        print(f"  gravity sign (fyi)  : positive torque lifts the link "
              f"(r={r:+.2f} over {n:,} stationary samples)")
    elif verdict == "flipped":
        print(f"  gravity sign (fyi)  : positive torque LOWERS the link "
              f"(r={r:+.2f}). That is a valid convention; the application must "
              f"declare it.")

    # ---- pass 2 ----------------------------------------------------------
    ladder = None
    if args.tau_w:
        ladder = [float(x) for x in args.tau_w.replace(",", " ").split()]
    print()
    print("summarising...")
    env = LR.build_envelope(args.log, cmap, p1, name=name, despike=despike,
                            dt_override=args.dt, kt_joint=args.kt_joint,
                            flip_torque=args.flip_torque,
                            flip_speed=args.flip_speed,
                            tau_w_ladder=ladder)

    env.ratio = args.ratio
    env.n_actuators_at_capture = args.n_actuators
    env.torque_source = args.torque_source
    env.sign_convention = ("positive torque raises the link, matching "
                           "load.gravity_angle in the application"
                           + ("  [--flip-torque applied]" if args.flip_torque else ""))
    env.note = args.note
    env.capture.robot_id = args.robot
    env.capture.joint_id = args.joint_id
    env.capture.firmware = args.firmware
    env.capture.captured_at = args.captured_at
    env.capture.tool_version = "envelope_from_log 1"
    env.capture.source_logs = [
        {"file": os.path.basename(p), "sha256": _sha256(p),
         "bytes": os.path.getsize(p)} for p in args.log]

    # ---- summary, so a bad capture is obvious before it is committed -----
    e = env.extremes
    print()
    print(f"  occupancy           : {len(env.cells)} cells "
          f"({len(env.outliers)} excursions kept verbatim)")
    print(f"  peak torque         : p99 {e.t('p99'):.2f}  p99.9 "
          f"{e.t('p99_9'):.2f}  max {e.t('max'):.2f} N.m")
    print(f"  peak speed          : p99.9 {e.w('p99_9')*RPM:.0f} rpm  "
          f"max {e.w('max')*RPM:.0f} rpm")
    for key in ("p95", "p99", "p99_9"):
        c = env.event_counts.get(key) or {}
        if c.get("events"):
            print(f"    above {key.replace('_', '.'):<5} "
                  f"{c['threshold']:6.2f} N.m : {c['events']:6d} events, "
                  f"{c['total_time']:8.1f} s total, longest "
                  f"{c['max_duration']:.2f} s")
    print()
    print("  sustained joint torque by averaging window (RMS N.m):")
    for w, v in env.duration_curve:
        print(f"    {w:8.4g} s  {v:6.2f}")
    print()
    print("  worst sequences found:")
    for w in env.windows:
        sel = ", ".join(f"{x:g}" for x in w.selected_by_tau_w)
        print(f"    {w.duration:8.1f} s at t={w.found_at:8.1f} s  "
              f"RMS {w.rms_torque:6.2f} N.m  (winding constants: {sel} s)")

    if env.total_time > 0:
        hold = sum(c.seconds for c in env.cells if abs(c.omega) < 0.05)
        frac = hold / env.total_time
        if frac > 0.95:
            print()
            print(f"  ** {frac*100:.0f}% of this session was stationary. A "
                  f"capture that never really works the joint")
            print(f"     will size an actuator optimistically. Consider "
                  f"recapturing during real duty.")

    out = args.out or os.path.join("envelopes", f"{name}.json")
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    write_json(env, out)
    print()
    print(f"wrote {out}  ({os.path.getsize(out):,} bytes)")
    print(f"reference it from an application file with:")
    print(f'    "envelope": "{os.path.splitext(os.path.basename(out))[0]}"')
    return 0


if __name__ == "__main__":
    sys.exit(main())
