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
