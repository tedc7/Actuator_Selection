"""Human-readable report generation."""

from __future__ import annotations
import math
from typing import List

from .evaluate import Evaluation, PASS, MARGINAL, FAIL, UNKNOWN
from .params import P
from . import physics as phys

MARK = {PASS: "PASS", MARGINAL: "MARG", FAIL: "FAIL", UNKNOWN: "????"}
RPM = 30.0 / math.pi


def _bar(margin: float, width: int = 12) -> str:
    if not math.isfinite(margin):
        return "[" + "=" * width + "]"
    f = max(0.0, min(margin / 2.0, 1.0))
    n = int(round(f * width))
    return "[" + "=" * n + "." * (width - n) + "]"


def render(ev: Evaluation, verbose: bool = True) -> str:
    a, j = ev.actuator, ev.joint
    L: List[str] = []
    add = L.append

    n = j.n_actuators
    add("=" * 78)
    add(f"  {j.name.upper()}  <-  {n} x {a.name}"
        + (f"  ({a.vendor})" if a.vendor else ""))
    add("=" * 78)

    # --- configuration summary ---
    add("")
    add("CONFIGURATION")
    add(f"  actuators in parallel : {n}")
    add(f"  actuator gear ratio   : {float(a.gear_ratio):.3g}:1 ({a.gear_type}, "
        f"eff {float(a.gear_eff):.2f})")
    add(f"  post-actuator ratio   : {float(j.ratio):.3g}:1 ({j.ratio_type}, "
        f"eff {float(j.ratio_eff):.2f})")
    add(f"  total reduction       : {float(a.gear_ratio)*float(j.ratio):.3g}:1")
    _vb = j.bus_v(a)
    _vsrc = "application" if j.bus_v_is_from_app() else "actuator nominal"
    add(f"  bus voltage           : {_vb:.1f} V  ({_vsrc}; actuator accepts "
        f"{float(a.V_bus_min) if a.V_bus_min is not None else _vb:.0f}-"
        f"{float(a.V_bus_max) if a.V_bus_max is not None else _vb:.0f} V)")
    if j.max_surface_temp is not None:
        add(f"  surface temp limit    : {j.max_surface_temp:.0f} degC (user safety)")
    add(f"  ambient (internal)    : {j.T_ambient:.0f} degC     mounting: {j.mounting}")

    # Which vendor document these numbers came from. Vendors reissue datasheets
    # without renaming them, so a report that does not say what it was based on
    # cannot be audited once the source has moved on.
    _ds = a.datasheet_id()
    if _ds:
        add(f"  actuator datasheet    : {_ds}")
    else:
        add("  actuator datasheet    : NOT DECLARED - values cannot be traced "
            "to a source")
    if a.entry_revision:
        add(f"  db entry revised      : {a.entry_revision}")
    add(f"  actuation mass        : {n*float(a.mass)*1000:.0f} g")
    if a.price_usd:
        add(f"  actuation cost        : ${n*a.price_usd:,.0f}")

    # --- capability envelope ---
    add("")
    add("JOINT CAPABILITY (all n actuators, at the joint)")
    tp = abs(a.torque_out(float(a.I_peak_rms))) * n * float(j.ratio) * float(j.ratio_eff)
    tc = ev.extras.get("tau_cont_cap_joint", 0.0)
    w0 = phys.no_load_speed_out(a) / max(float(j.ratio), 1e-9)
    add(f"  peak torque           : {tp:8.2f} N.m")
    add(f"  continuous torque     : {tc:8.2f} N.m   (thermal, at "
        f"{j.T_ambient:.0f} degC ambient)")
    add(f"  no-load speed         : {w0*RPM:8.0f} rpm  ({w0:.1f} rad/s)")

    # --- duty cycle ---
    if ev.duty_segments:
        per = sum(s[0] for s in ev.duty_segments)
        tmax = max(abs(t) for _, t, _ in ev.duty_segments)
        wmax = max(abs(w) for _, _, w in ev.duty_segments)
        add("")
        add("DUTY CYCLE (per actuator, at its output shaft)")
        add(f"  cycle period          : {per:.2f} s")
        add(f"  peak torque demand    : {tmax:8.2f} N.m")
        add(f"  RMS current           : {ev.extras.get('i_rms_duty', 0):8.2f} A_rms")
        add(f"  peak speed            : {wmax*RPM:8.0f} rpm")

    # --- commanded motion: solved, not prescribed -------------------------
    # move_time is an OUTPUT of the limit solver, so it is reported rather than
    # echoed back. The binding limit is the actionable number: it names which
    # of the three controller limits to change to make the move faster or
    # easier on the actuator.
    if "move_time" in ev.extras and j.profile is not None:
        p = j.profile
        reg = ev.extras.get("regime", "?")
        add("")
        add("COMMANDED MOTION (solved from the controller's limits)")
        add(f"  stroke                : {math.degrees(p.stroke):8.1f} deg  "
            f"({math.degrees(p.stroke_start):.1f} -> "
            f"{math.degrees(p.stroke_end):.1f})")
        add(f"  move time (solved)    : {p.move_time*1e3:8.0f} ms per traverse"
            f"   <- {reg}-limited")
        add(f"  dwell at each end     : {p.dwell_time*1e3:8.0f} ms")
        def _lim(reached: float, limit: float, unit: str) -> str:
            at = "AT LIMIT" if reached >= limit * 0.999 else "below"
            return f"{reached:8.1f} {unit} (limit {limit:.1f}, {at})"
        add(f"  peak velocity         : "
            f"{_lim(p.peak_velocity*RPM, p.max_velocity*RPM, 'rpm')}")
        add(f"  peak acceleration     : "
            f"{_lim(p.peak_accel, p.max_accel, 'rad/s^2')}")
        add(f"  jerk                  : {p.max_jerk:8.1f} rad/s^3 "
            f"(always at limit during transitions)")

        # The acceleration limit restated as the torque it demands, for a
        # controller whose ceiling is a torque rather than an acceleration.
        # Gravity, friction and external torque are NOT in this figure -- it is
        # the inertial term alone, so it matches an inertia-only torque cap.
        if "tau_inertial_cmd" in ev.extras:
            j_load = ev.extras["J_load_joint"]
            j_refl = ev.extras["J_reflected_joint"]
            tau_in = ev.extras["tau_inertial_cmd"]
            add(f"  inertia about joint   : {j_load + j_refl:8.4f} kg.m^2 "
                f"(load {j_load:.4f} + reflected rotor {j_refl:.4f})")
            add(f"  implied inertial torq : {tau_in:8.2f} N.m at the joint "
                f"(= J_total * peak accel)")
            tau_per = ev.extras["tau_inertial_cmd_per_actuator"]
            add(f"  {'':22}  {tau_per:8.2f} N.m per actuator output "
                f"({n}x, ratio {float(j.ratio):.2f})")
            add(f"  {'':22}  inertia only -- gravity, friction and external "
                f"torque are extra")

    # --- criteria table ---
    add("")
    add("CRITERIA")
    add(f"  {'':4}  {'criterion':<24} {'demand':>10} {'capability':>11} "
        f"{'margin':>7}  confidence")
    add("  " + "-" * 74)
    for c in ev.criteria:
        m = "  inf" if not math.isfinite(c.margin) else f"{c.margin:5.2f}x"
        tag = c.name + (" (fyi)" if c.advisory else "")
        add(f"  {MARK[c.status]:<4}  {tag:<24} {c.demand:>10.3g} "
            f"{c.capability:>11.3g} {m:>7}  {c.confidence}")
        if verbose and c.detail:
            add(f"        {c.detail}")
    add("  " + "-" * 74)

    v = ev.verdict
    b = ev.binding
    add("")
    add(f"  VERDICT: {v}    (binding constraint: {b.name}, margin {b.margin:.2f}x)")

    # --- thermal detail ---
    if ev.thermal:
        t = ev.thermal
        add("")
        add("THERMAL DETAIL (per actuator)")
        if t.runaway:
            add("  *** THERMAL RUNAWAY: no equilibrium exists for this duty cycle ***")
            add("      (copper resistance rises with temperature, which raises")
            add("       the loss, which raises the temperature -- no fixed point)")
            tl = ev.extras.get("time_to_limit_s")
            if tl is not None:
                add(f"  burst capability      : {tl:6.0f} s from cold before the "
                    f"winding hits {float(a.T_winding_max):.0f} degC")
        else:
            add(f"  peak winding temp     : {t.t_winding_peak:6.1f} degC "
                f"(limit {float(a.T_winding_max):.0f})")
            add(f"  settled winding temp  : {t.t_winding_final:6.1f} degC")
            add(f"  settled case temp     : {t.t_case_final:6.1f} degC")
            add(f"  mean heat dissipation : {t.mean_loss_W:6.2f} W")
            tw = float(a.C_w) * float(a.Rth_wc)
            tc_ = float(a.C_c) * float(a.Rth_ca)
            add(f"  time constants        : winding {tw:.0f} s, case {tc_:.0f} s")
            add(f"  duty cycles simulated : {t.duty_cycles_simulated}"
                + ("  (converged)" if t.settled else "  (NOT converged - still heating)"))
            if "time_to_limit_s" in ev.extras:
                tl = ev.extras["time_to_limit_s"]
                if tl is None:
                    add("  burst capability      : indefinite (never reaches the limit)")
                else:
                    add(f"  burst capability      : {tl:6.0f} s from cold before the "
                        f"winding hits {float(a.T_winding_max):.0f} degC")

    # --- unit audit ---
    audits = [a for a in getattr(ev, "unit_audits", []) if a and a.entries]
    if audits:
        add("")
        add("UNIT AUDIT  (as written in the input files -> canonical SI)")
        for a_ in audits:
            add(a_.render())

    # --- assumptions ---
    if ev.assumptions:
        add("")
        add("ASSUMED PARAMETERS  (defaults, not vendor data)")
        for p in sorted(ev.assumptions, key=lambda x: x.name):
            add(f"  - {p.describe()}")

    # --- sensitivity ---
    add("")
    if ev.sensitivity:
        add("SENSITIVITY  --  assumptions that change a verdict within their error band")
        for k, flips in ev.sensitivity.items():
            add(f"  {k}:")
            for f in flips:
                add(f"      {f}")
        add("")
        add("  MEASURE THESE FIRST: " + ", ".join(ev.sensitivity.keys()))
    else:
        add("SENSITIVITY: no assumed parameter flips a verdict within its error")
        add("  band. The result is robust to the guesses that were made.")

    # --- torque-speed envelope provenance ---
    # Which envelope drove the verdicts, and if it was a measured curve, how far
    # the independent electrical model disagrees with it. A large spread does not
    # invalidate the verdict (the measurement wins) but it does say the model's
    # R_phase / L_phase are wrong, which still matters for thermal predictions.
    _vb = j.bus_v(a)
    _curve, _scaled = phys.tn_curve_for(a, _vb)
    add("")
    add("TORQUE-SPEED ENVELOPE")
    if _curve:
        _lo, _hi = _curve.speed_range_rpm
        add(f"  source   : measured curve ({_curve.source}), {len(_curve.points)} points, "
            f"{_lo:.0f}-{_hi:.0f} rpm")
        if _scaled:
            add(f"  WARNING  : curve was measured at "
                f"{a.tn_curve.bus_voltage:.0f} V, rescaled to {_vb:.0f} V. "
                "Speed axis scaled by voltage ratio; torque axis assumed unchanged.")
        add("  verdicts use the measured curve; the model below is an independent check")
        rows, rms = phys.tn_crosscheck(a, _vb)
        if rows:
            add("")
            add(f"  {'rpm':>6}  {'measured':>9}  {'model':>8}  {'delta':>8}")
            for rpm, tv, tm, rel in rows:
                flag = "  <-- " if abs(rel) > 0.25 else ""
                add(f"  {rpm:>6.0f}  {tv:>9.2f}  {tm:>8.2f}  {rel*100:>+7.1f}%{flag}")
            add(f"  RMS disagreement: {rms*100:.0f}%"
                "   (over points above 15% of peak torque; the near-no-load"
                " tail is excluded as both curves approach zero there)")
            if rms > 0.25:
                add("")
                add("  The electrical model disagrees materially with the measured curve.")
                add("  The verdicts above are unaffected -- they use the measurement -- but")
                add("  the model's parameters are evidently wrong, and R_phase still")
                add("  drives the thermal model.")
                # Which parameter to suspect is readable from the SHAPE of the
                # disagreement, so say so rather than making the reader guess.
                # Judged on the same trimmed set as the RMS: the near-zero tail
                # always diverges and would make every case look high-speed.
                _tmax = max(r[1] for r in rows)
                _sig = [r for r in rows if r[1] >= 0.15 * _tmax] or rows
                lo_err = abs(_sig[0][3])
                hi_err = max(abs(r[3]) for r in _sig[len(_sig) // 2:])
                spread = max(abs(r[3]) for r in _sig) - min(abs(r[3]) for r in _sig)
                if lo_err < 0.10 and hi_err > 0.25:
                    add("  The model tracks the curve at low speed and falls away at high")
                    add("  speed: that is the voltage/reactance term. Suspect L_phase")
                    add("  (a placeholder guess unless you have set it).")
                elif spread < 0.20:
                    add("  The error is roughly constant across the whole speed range, which")
                    add("  is not a rolloff problem. Suspect a scaling term: L_phase forcing")
                    add("  an early current limit, or Kt/gear_eff/kt_sat_derate.")
                else:
                    add("  Suspect L_phase first (usually a placeholder), then R_phase.")
    else:
        add("  source   : computed from Ke, R_phase, L_phase (no measured curve)")
        if a.tn_curve is None and isinstance(a.datasheet, dict):
            add("  no vendor T-N curve recorded for this actuator; if one exists,")
            add("  adding it to the db entry will tighten the envelope considerably")

    # --- overload endurance: the thermal model's only measured check ---
    for _cond in ("rotating", "stalled"):
        _oc = a.overload_curve(_cond)
        if not _oc:
            continue
        _rows, _lrms = phys.overload_crosscheck(a, _cond)
        if not _rows:
            continue
        add("")
        add(f"OVERLOAD ENDURANCE CROSS-CHECK ({_cond})")
        _cond_bits = []
        if _oc.speed_rpm_output is not None:
            _cond_bits.append("stalled" if _oc.speed_rpm_output == 0
                              else f"{_oc.speed_rpm_output:.0f} rpm output")
        if _oc.ambient_C is not None:
            _cond_bits.append(f"{_oc.ambient_C:.0f} degC ambient")
        if _oc.mounting:
            _cond_bits.append(f"mounting '{_oc.mounting}'")
        add(f"  source   : measured ({_oc.source}), {len(_oc.points)} points"
            + (f", +/-{_oc.tol*100:.0f}%" if _oc.tol else ""))
        if _cond_bits:
            add(f"  vendor test conditions: {', '.join(_cond_bits)}")
        add("  the model is run at those conditions, NOT this application's")
        add("")
        add(f"  {'N.m':>6}  {'vendor':>9}  {'model':>9}  {'ratio':>7}")
        for _tau, _vs, _ms, _ratio in _rows:
            _ms_s = "never" if math.isinf(_ms) else f"{_ms:.1f}"
            if math.isinf(_ratio):
                _r_s, _flag = "  inf", "  <-- "
            elif _ratio != _ratio:
                _r_s, _flag = "    ?", "  <-- "
            else:
                _r_s = f"{_ratio:.2f}x"
                _flag = "  <-- " if (_ratio > 1.5 or _ratio < 0.67) else ""
            add(f"  {_tau:>6.1f}  {_vs:>8.0f}s  {_ms_s:>9}  {_r_s:>7}{_flag}")
        add(f"  log-RMS disagreement: {_lrms:.2f}"
            "   (log space: endurance spans three decades, so a 2x miss counts"
            " the same at 3 s and at 1400 s)")

        # Direction is the safety-relevant part, so say it in words.
        _fin = [r[3] for r in _rows if math.isfinite(r[3]) and r[3] > 0]
        if _fin:
            _gm = math.exp(sum(math.log(r) for r in _fin) / len(_fin))
            add("")
            if _gm > 1.25:
                add(f"  The model OUTLASTS the vendor's measurement by {_gm:.1f}x on"
                    " average, i.e. it")
                add("  is OPTIMISTIC: a duty cycle sized against it will overheat"
                    " sooner than")
                add("  this report implies. Treat the continuous-torque figure above as"
                    " an")
                add("  upper bound, and prefer the vendor's rated torque"
                    + (f" ({_oc.rated_torque:.1f} N.m)" if _oc.rated_torque else "")
                    + " until R_phase")
                add("  is measured.")
                if _cond == "stalled":
                    add("  Expected here: the two-node model assumes even three-phase")
                    add("  heating, but a stalled winding concentrates it (1.414x).")
            elif _gm < 0.8:
                add(f"  The model is CONSERVATIVE by {1/_gm:.1f}x against this"
                    " measurement -- it")
                add("  predicts overheating sooner than the vendor measured. Verdicts"
                    " are safe")
                add("  but may be leaving capability unused.")
            else:
                add(f"  The model tracks the measurement within {abs(_gm-1)*100:.0f}%"
                    " on average, so the")
                add("  thermal estimates are doing their job on this actuator.")

    # --- vendor cross-check ---
    cc = a.consistency_report()
    if cc:
        add("")
        add("VENDOR DATA CROSS-CHECK")
        for line in cc:
            add(f"  {line}")

    add("")
    return "\n".join(L)


def compare(evals: List[Evaluation]) -> str:
    """Side-by-side summary of several candidate actuators against one joint."""
    L = []
    add = L.append
    add("=" * 78)
    add(f"  COMPARISON  --  joint: {evals[0].joint.name}")
    add("=" * 78)
    add(f"  {'config':<28} {'verdict':<9} {'binding':<22} {'margin':>7}")
    add("  " + "-" * 74)
    for ev in sorted(evals, key=lambda e: -e.binding.margin):
        cfg = f"{ev.joint.n_actuators}x {ev.actuator.name}"
        b = ev.binding
        add(f"  {cfg:<28} {ev.verdict:<9} {b.name:<22} {b.margin:>6.2f}x")
    add("  " + "-" * 74)
    add("")
    add(f"  {'config':<28} {'Tpeak':>8} {'Tcont':>8} {'mass':>8} {'Twind':>8}")
    add("  " + "-" * 74)
    for ev in sorted(evals, key=lambda e: -e.binding.margin):
        a, j = ev.actuator, ev.joint
        cfg = f"{j.n_actuators}x {a.name}"
        tp = abs(a.torque_out(float(a.I_peak_rms))) * j.n_actuators * float(j.ratio) * float(j.ratio_eff)
        tc = ev.extras.get("tau_cont_cap_joint", 0.0)
        mass = j.n_actuators * float(a.mass) * 1000
        tw = ev.thermal.t_winding_peak if ev.thermal else float("nan")
        add(f"  {cfg:<28} {tp:>7.1f}N {tc:>7.1f}N {mass:>7.0f}g {tw:>7.0f}C")
    add("")
    return "\n".join(L)
