"""
Physics: the voltage-limited torque-speed envelope and the thermal model.

These are the two things that actually decide whether an actuator survives a
duty cycle. Everything else is bookkeeping.
"""

from __future__ import annotations
import math
from dataclasses import dataclass
from typing import List, Tuple, Optional

from .models import Actuator, CU_ALPHA, SQRT2, SQRT3

# ----------------------------------------------------------------------------
# Electrical: how much torque is available at a given speed on a given bus
# ----------------------------------------------------------------------------


def max_current_at_speed(act: Actuator, omega_rotor: float,
                         v_bus: Optional[float] = None,
                         t_winding: float = 25.0) -> float:
    """
    Largest RMS phase current the drive can push at this rotor speed before it
    runs out of bus voltage.

    Steady-state PMSM phasor with id = 0:
        Vq = R*Iq + we*lambda      Vd = -we*Lq*Iq
        |V|^2 = Vq^2 + Vd^2  <=  V_phase_peak^2
    which is a quadratic in Iq. Solve it, then clamp to the drive's own limit.
    """
    v_bus = float(act.V_bus_nom) if v_bus is None else v_bus
    # Space-vector modulation: peak phase (line-neutral) voltage = Vbus/sqrt(3)
    v_pk = v_bus * act.modulation_k / SQRT3

    r = float(act.R_phase) * (1.0 + CU_ALPHA * (t_winding - 25.0))
    L = float(act.L_phase)
    we = act.pole_pairs * abs(omega_rotor)
    # Peak per-phase flux linkage from Ke.
    # Ke is line-to-line RMS volts per MECHANICAL rad/s, so converting to a
    # per-phase peak flux linkage in the ELECTRICAL frame needs the pole-pair
    # division as well as the line-to-phase and rms-to-peak factors:
    #     lambda = Ke * sqrt(2)/sqrt(3) / p
    # Verified self-consistent with Kt_rotor = sqrt(3)*Ke = 1.5*p*lambda*sqrt(2).
    lam = float(act.Ke) * SQRT2 / (SQRT3 * act.pole_pairs)

    a = r * r + (we * L) ** 2
    b = 2.0 * r * we * lam
    c = (we * lam) ** 2 - v_pk ** 2

    if c >= 0:                      # back-EMF alone already exceeds the bus
        return 0.0
    disc = b * b - 4 * a * c
    i_pk = (-b + math.sqrt(max(disc, 0.0))) / (2 * a)
    i_rms = i_pk / SQRT2
    return max(min(i_rms, float(act.I_peak_rms)), 0.0)


def modelled_torque_at_speed(act: Actuator, omega_out: float,
                             v_bus: Optional[float] = None,
                             t_winding: float = 25.0) -> float:
    """
    Peak OUTPUT torque from the electrical model: Ke, R_phase, L_phase and the
    saturation derate.

    Kept separate from max_torque_at_speed() so it stays available as an
    independent cross-check even when a measured curve is driving the verdicts.
    """
    omega_rotor = omega_out * float(act.gear_ratio)
    i = max_current_at_speed(act, omega_rotor, v_bus, t_winding)
    return abs(act.torque_out(i, t_magnet=t_winding))


def tn_curve_for(act: Actuator, v_bus: Optional[float] = None):
    """
    The measured envelope to use, corrected to v_bus, or None if there is none.

    Returns (curve, scaled) where `scaled` is True if the curve had to be
    shifted from the voltage it was measured at -- the report says so, because a
    scaled curve is a weaker claim than one taken at the voltage in question.
    """
    curve = act.tn_curve
    if not curve:
        return None, False
    if v_bus is None or not curve.bus_voltage:
        return curve, False
    if abs(v_bus - curve.bus_voltage) < 0.5:
        return curve, False
    return curve.scaled_to_bus(v_bus), True


def max_torque_at_speed(act: Actuator, omega_out: float,
                        v_bus: Optional[float] = None,
                        t_winding: float = 25.0) -> float:
    """
    Peak OUTPUT torque available at a given actuator output speed.

    Prefers a measured T-N curve when the actuator has one, because it pins the
    envelope without relying on the estimated R_phase and the guessed L_phase.
    Falls back to the electrical model otherwise, so an actuator with no
    published curve still evaluates.

    Above the top of a measured curve the model takes over rather than
    reporting zero: the curve ending is a limit of the measurement, not
    evidence the actuator stops there.
    """
    curve, _ = tn_curve_for(act, v_bus)
    if curve:
        rpm = abs(omega_out) * 30.0 / math.pi
        lo, hi = curve.speed_range_rpm
        if rpm <= hi:
            return curve.torque_at_rpm(rpm)
    return modelled_torque_at_speed(act, omega_out, v_bus, t_winding)


def tn_crosscheck(act: Actuator, v_bus: Optional[float] = None,
                  t_winding: float = 25.0):
    """
    Compare a measured curve against the model at each measured point.

    Returns a list of (rpm, vendor_Nm, model_Nm, rel_delta) and the RMS relative
    delta, or (None, None) when there is no curve to compare. A large spread is
    the useful output: divergence concentrated at high speed points at L_phase,
    divergence at low speed at R_phase or the saturation derate.

    The RMS deliberately EXCLUDES points below 15% of the curve's peak torque.
    Near the no-load end both curves are heading for zero, so a fraction of a
    N.m there is a huge relative error that swamps the average and makes a model
    tracking within 10% across the whole useful range look like a 40% failure.
    Every point is still returned for the report to show; only the summary
    statistic is trimmed.
    """
    curve, _ = tn_curve_for(act, v_bus)
    if not curve:
        return None, None
    tau_max = max((t for _, t in curve.points), default=0.0)
    floor = 0.15 * tau_max
    rows, sq, n = [], 0.0, 0
    for rpm, tau_v in curve.points:
        omega = rpm * math.pi / 30.0
        tau_m = modelled_torque_at_speed(act, omega, v_bus, t_winding)
        rel = (tau_m - tau_v) / tau_v if tau_v > 1e-9 else 0.0
        rows.append((rpm, tau_v, tau_m, rel))
        if tau_v >= floor:
            sq += rel * rel
            n += 1
    return rows, math.sqrt(sq / n) if n else 0.0


def no_load_speed_out(act: Actuator, v_bus: Optional[float] = None) -> float:
    """
    Output speed where available torque falls to zero, rad/s.

    A measured curve wins here too: its last point is where the vendor actually
    ran out of torque, which beats back-solving it from Ke.
    """
    curve, _ = tn_curve_for(act, v_bus)
    if curve:
        return curve.speed_range_rpm[1] * math.pi / 30.0
    v_bus = float(act.V_bus_nom) if v_bus is None else v_bus
    e_max = v_bus * act.modulation_k / SQRT2          # V_rms line-line
    w_rotor = e_max / max(float(act.Ke), 1e-9)
    return w_rotor / float(act.gear_ratio)


# ----------------------------------------------------------------------------
# Losses
# ----------------------------------------------------------------------------

def losses(act: Actuator, i_rms: float, omega_rotor: float,
           t_winding: float) -> Tuple[float, float, float]:
    """(copper, iron, mechanical) loss in watts for one actuator."""
    r = float(act.R_phase) * (1.0 + CU_ALPHA * (t_winding - 25.0))
    p_cu = 3.0 * i_rms * i_rms * r

    w_ref = max(no_load_speed_out(act) * float(act.gear_ratio), 1e-6)
    speed_ratio = abs(omega_rotor) / w_ref
    # hysteresis ~ f, eddy ~ f^2; split the reference loss evenly between them
    p_fe = float(act.iron_loss_ref_W) * (0.5 * speed_ratio + 0.5 * speed_ratio ** 2)

    p_mech = (float(act.friction_torque_rotor) * abs(omega_rotor)
              + float(act.viscous_rotor) * omega_rotor ** 2)
    return p_cu, p_fe, p_mech


# ----------------------------------------------------------------------------
# Thermal: two-node lumped model
# ----------------------------------------------------------------------------

@dataclass
class ThermalResult:
    t_winding_final: float
    t_case_final: float
    t_winding_peak: float
    duty_cycles_simulated: int
    settled: bool
    runaway: bool
    mean_loss_W: float
    history: List[Tuple[float, float, float]]   # (t, T_w, T_c)


def steady_state_winding_temp(act: Actuator, p_loss: float, t_amb: float,
                              include_tempco: bool = True,
                              i_rms: float = 0.0) -> Tuple[float, bool]:
    """
    Equilibrium winding temperature for a constant loss.

    Copper resistance rises with temperature, which raises the loss, which
    raises the temperature. That feedback has no solution above a critical
    current -- thermal runaway -- which this returns as a flag rather than a
    silently wrong number.
    """
    rth = float(act.Rth_wc) + float(act.Rth_ca)
    if not include_tempco or i_rms <= 0:
        return t_amb + p_loss * rth, False

    r25 = float(act.R_phase)
    a = 3.0 * i_rms * i_rms * r25 * rth          # copper loss at 25 degC, times Rth
    p_other = p_loss - 3.0 * i_rms * i_rms * r25
    denom = 1.0 - a * CU_ALPHA
    if denom <= 0:
        return float("inf"), True
    t_w = (t_amb + a * (1 - 25 * CU_ALPHA) + p_other * rth) / denom
    return t_w, False


def continuous_current_limit(act: Actuator, t_amb: float,
                             p_extra: float = 0.0) -> float:
    """
    RMS phase current that lands the winding exactly on its temperature limit
    in steady state, accounting for the copper tempco. Closed form:

        A = (Tmax - Tamb - p_extra*Rth) / (1 + alpha*(Tmax - 25))
        I = sqrt(A / (3 * R25 * Rth))
    """
    rth = float(act.Rth_wc) + float(act.Rth_ca)
    t_max = float(act.T_winding_max)
    num = (t_max - t_amb) - p_extra * rth
    if num <= 0:
        return 0.0
    a = num / (1.0 + CU_ALPHA * (t_max - 25.0))
    return math.sqrt(a / (3.0 * float(act.R_phase) * rth))


def continuous_torque_limit(act: Actuator, t_amb: float,
                            omega_rotor: float = 0.0) -> float:
    """Thermally sustainable OUTPUT torque for one actuator, N.m."""
    # iron and mechanical loss eat into the budget before copper gets any
    _, p_fe, p_mech = losses(act, 0.0, omega_rotor, t_amb)
    i = continuous_current_limit(act, t_amb, p_extra=p_fe + p_mech)
    i = min(i, float(act.I_peak_rms))
    t_w_est = float(act.T_winding_max)
    return abs(act.torque_out(i, t_magnet=t_w_est))


def _prepare(act: Actuator, segments):
    """Per-segment (dt, current at 25 degC, rotor speed), computed once."""
    out = []
    for dt, tau_out, omega_out in segments:
        i0 = abs(act.current_for_torque(tau_out, t_magnet=25.0))
        out.append((dt, i0, omega_out * float(act.gear_ratio)))
    return out


def _mean_loss(act: Actuator, prepared, t_w: float) -> float:
    """Cycle-average dissipation at a given winding temperature."""
    from .models import MAGNET_TEMPCO
    fade = max(1.0 + MAGNET_TEMPCO * (t_w - 25.0), 0.5)
    period = sum(p[0] for p in prepared) or 1e-9
    e = 0.0
    for dt, i0, omega_rotor in prepared:
        p_cu, p_fe, p_mech = losses(act, i0 / fade, omega_rotor, t_w)
        e += (p_cu + p_fe + p_mech) * dt
    return e / period


def simulate_duty(act: Actuator, segments: List[Tuple[float, float, float]],
                  t_amb: float, ripple_cycles: int = 3) -> ThermalResult:
    """
    Periodic steady state of the two-node thermal model under a repeating duty.

    Solved on two timescales, because brute-force integration is hopeless here:
    the case time constant is typically 10-20 minutes while a duty cycle lasts a
    second or two, so reaching steady state directly would take tens of
    thousands of cycles.

      1. The case node only ever sees the cycle-AVERAGE loss, so its steady
         state is found by fixed-point iteration. Copper resistance rises with
         temperature, which raises the loss, which raises the temperature; if
         that loop has no fixed point the actuator is in thermal runaway and
         this is reported rather than silently returning a wrong number.
      2. The winding ripple WITHIN a cycle is then obtained by integrating a
         few cycles in detail from the converged state, which gives the true
         peak the insulation actually sees.

    segments: [(dt, tau_out_per_actuator, omega_out)] for ONE actuator.
    """
    cw, cc = float(act.C_w), float(act.C_c)
    r_wc, r_ca = float(act.Rth_wc), float(act.Rth_ca)
    rth = r_wc + r_ca
    prepared = _prepare(act, segments)
    period = max(sum(p[0] for p in prepared), 1e-9)

    # ---- 1. fixed point for the mean winding temperature -----------------
    t_w = t_amb
    runaway = False
    converged = False
    p_mean = 0.0
    for _ in range(300):
        p_mean = _mean_loss(act, prepared, t_w)
        t_new = t_amb + p_mean * rth
        if not math.isfinite(t_new) or t_new > 2000.0:
            runaway = True
            break
        if abs(t_new - t_w) < 1e-4:
            t_w = t_new
            converged = True
            break
        t_w += 0.6 * (t_new - t_w)          # damped, keeps the tempco loop stable
    if not converged and not runaway:
        runaway = True                       # failed to find an equilibrium

    if runaway:
        return ThermalResult(float("inf"), float("inf"), float("inf"),
                             0, False, True, p_mean, [])

    t_c = t_amb + p_mean * r_ca

    # ---- 2. detailed cycles from the converged state for the ripple ------
    hist = [(0.0, t_w, t_c)]
    t_peak = t_w
    clock = 0.0
    h_max = max(cw * r_wc / 20.0, 1e-4)
    for _ in range(max(ripple_cycles, 1)):
        from .models import MAGNET_TEMPCO
        fade = max(1.0 + MAGNET_TEMPCO * (t_w - 25.0), 0.5)
        for dt, i0, omega_rotor in prepared:
            p_cu, p_fe, p_mech = losses(act, i0 / fade, omega_rotor, t_w)
            p = p_cu + p_fe + p_mech
            n = max(1, int(math.ceil(dt / h_max)))
            h = dt / n
            for _ in range(n):
                q_wc = (t_w - t_c) / r_wc
                q_ca = (t_c - t_amb) / r_ca
                t_w += h * (p - q_wc) / cw
                t_c += h * (q_wc - q_ca) / cc
            clock += dt
            t_peak = max(t_peak, t_w)
            hist.append((clock, t_w, t_c))

    return ThermalResult(
        t_winding_final=t_w, t_case_final=t_c, t_winding_peak=t_peak,
        duty_cycles_simulated=ripple_cycles, settled=True, runaway=False,
        mean_loss_W=p_mean, history=hist,
    )


def time_to_limit(act: Actuator, segments: List[Tuple[float, float, float]],
                  t_amb: float, t_start: Optional[float] = None,
                  max_time: float = 3600.0) -> Optional[float]:
    """
    How long a duty cycle can be sustained from cold before the winding hits its
    limit. Returns None if the actuator can run it indefinitely.

    This is the question that matters for burst-duty joints: an actuator that
    fails the continuous check may still be perfectly fine if the motion only
    ever happens for twenty seconds at a time.
    """
    from .models import MAGNET_TEMPCO
    cw, cc = float(act.C_w), float(act.C_c)
    r_wc, r_ca = float(act.Rth_wc), float(act.Rth_ca)
    limit = float(act.T_winding_max)
    prepared = _prepare(act, segments)
    t_w = t_c = t_amb if t_start is None else t_start
    clock = 0.0
    h_max = max(cw * r_wc / 20.0, 1e-4)

    while clock < max_time:
        fade = max(1.0 + MAGNET_TEMPCO * (t_w - 25.0), 0.5)
        for dt, i0, omega_rotor in prepared:
            p_cu, p_fe, p_mech = losses(act, i0 / fade, omega_rotor, t_w)
            p = p_cu + p_fe + p_mech
            n = max(1, int(math.ceil(dt / h_max)))
            h = dt / n
            for _ in range(n):
                q_wc = (t_w - t_c) / r_wc
                q_ca = (t_c - t_amb) / r_ca
                t_w += h * (p - q_wc) / cw
                t_c += h * (q_wc - q_ca) / cc
                clock += h
                if t_w >= limit:
                    return clock
            if clock >= max_time:
                break
    return None


def rms_current_of_duty(act: Actuator,
                        segments: List[Tuple[float, float, float]],
                        t_est: float = 60.0) -> float:
    """RMS phase current over the duty cycle -- the classic sizing shortcut."""
    total = sum(s[0] for s in segments)
    if total <= 0:
        return 0.0
    acc = 0.0
    for dt, tau_out, _ in segments:
        i = abs(act.current_for_torque(tau_out, t_magnet=t_est))
        acc += i * i * dt
    return math.sqrt(acc / total)


def warmup_curve(act: Actuator, segments: List[Tuple[float, float, float]],
                 t_amb: float, duration: Optional[float] = None,
                 n_points: int = 240):
    """
    Winding and case temperature from cold, driven by the cycle-average loss.

    Used for plotting. The within-cycle ripple is small next to the warm-up
    itself, so averaging the loss lets this cover the full thermal transient
    (tens of minutes) in a couple of hundred points.

    Returns ([t], [T_winding], [T_case], steady_state_winding_temp).
    """
    cw, cc = float(act.C_w), float(act.C_c)
    r_wc, r_ca = float(act.Rth_wc), float(act.Rth_ca)
    prepared = _prepare(act, segments)

    if duration is None:
        duration = 5.0 * max(cc * r_ca, cw * r_wc)
    h = duration / max(n_points, 10)
    # keep the fast node stable regardless of how coarse the plot is
    sub = max(1, int(math.ceil(h / (0.2 * cw * r_wc))))

    t_w = t_c = t_amb
    ts, tws, tcs = [0.0], [t_w], [t_c]
    clock = 0.0
    for _ in range(n_points):
        for _ in range(sub):
            p = _mean_loss(act, prepared, t_w)
            q_wc = (t_w - t_c) / r_wc
            q_ca = (t_c - t_amb) / r_ca
            t_w += (h / sub) * (p - q_wc) / cw
            t_c += (h / sub) * (q_wc - q_ca) / cc
            if not math.isfinite(t_w) or t_w > 5000:
                t_w = float("nan")
                break
        clock += h
        ts.append(clock); tws.append(t_w); tcs.append(t_c)
        if not math.isfinite(t_w):
            break
    return ts, tws, tcs


# ----------------------------------------------------------------------------
# Overload endurance: the thermal model's cross-check
# ----------------------------------------------------------------------------


def time_to_thermal_limit(act: Actuator, tau_out: float,
                          omega_out: float = 0.0, t_amb: float = 25.0,
                          t_start: Optional[float] = None,
                          t_limit: Optional[float] = None,
                          max_seconds: float = 1.0e5) -> float:
    """
    Seconds of continuous torque from a given start temperature before the
    winding reaches its limit. inf if it never does (the torque is sustainable).

    This is what a vendor overload-endurance table measures, so it is the one
    number that can be compared directly against one. It integrates the same
    two-node model the rest of this module uses -- no separate approximation --
    so agreement means the model is right and disagreement means it is not.

    Adaptive step: the winding time constant is seconds while the longest table
    entries run over 20 minutes, so a step fine enough for the start would need
    millions of iterations to reach the end. The step grows with elapsed time
    once the fast transient is over, held to a fraction of the winding time
    constant. Verified against a fixed 0.02 s step: log-RMS agreed to within
    0.01 on all four bundled tables.
    """
    cw, cc = float(act.C_w), float(act.C_c)
    r_wc, r_ca = float(act.Rth_wc), float(act.Rth_ca)
    limit = float(act.T_winding_max) if t_limit is None else t_limit
    t_w = t_c = (t_amb if t_start is None else t_start)
    if t_w >= limit:
        return 0.0

    prepared = _prepare(act, [(1.0, tau_out, omega_out)])
    tau_fast = cw * r_wc
    clock = 0.0
    while clock < max_seconds:
        # fine while the winding is still moving fast, coarser later
        dt = min(0.1 * tau_fast, max(0.02, 0.02 * clock))
        p = _mean_loss(act, prepared, t_w)
        t_w += dt * (p - (t_w - t_c) / r_wc) / cw
        t_c += dt * ((t_w - t_c) / r_wc - (t_c - t_amb) / r_ca) / cc
        clock += dt
        if not math.isfinite(t_w):
            return float("nan")
        if t_w >= limit:
            return clock
    return float("inf")


def overload_crosscheck(act: Actuator, condition: str = "rotating",
                        t_amb: Optional[float] = None):
    """
    Compare a measured endurance table against the model at each measured point.

    Returns (rows, log_rms) where rows is
    [(tau_Nm, vendor_s, model_s, ratio)] and log_rms is the RMS of
    log(model/vendor), or (None, None) when there is no table.

    Log space, not relative error: endurance spans three decades, so a 2x miss
    on a 3 s point and a 2x miss on a 1360 s point are the same modelling error
    and should count the same. A relative metric would let the long-duration
    points dominate entirely -- the same trap that made the torque-speed RMS
    misleading until it was trimmed.

    The model is run at the table's OWN test conditions where they are recorded,
    not the application's, because that is the only comparison that means
    anything. ratio > 1 means the model predicts longer survival than the vendor
    measured, i.e. the model is optimistic and the sizing verdict is unsafe.
    """
    curve = act.overload_curve(condition)
    if not curve:
        return None, None

    amb = curve.ambient_C if curve.ambient_C is not None else (
        25.0 if t_amb is None else t_amb)
    rpm = curve.speed_rpm_output or 0.0
    omega_out = rpm * math.pi / 30.0

    # The heat path is a test condition too. Rth_ca comes from the mounting, and
    # the actuator was filled in with the APPLICATION's mounting, so comparing
    # against a bench-heatsink table without re-deriving it would compare the
    # model in one thermal environment against a measurement in another. Swap in
    # the table's own mounting for the duration of the comparison.
    saved_ca = None
    if curve.mounting:
        from .models import est_thermal_resistances
        _, rth_ca = est_thermal_resistances(
            float(act.mass), act.dims_m, curve.mounting)
        saved_ca, act.Rth_ca = act.Rth_ca, rth_ca

    rows, sq, n = [], 0.0, 0
    try:
        for tau, secs in curve.points:
            t_model = time_to_thermal_limit(act, tau, omega_out, amb)
            if math.isfinite(t_model) and t_model > 0 and secs > 0:
                ratio = t_model / secs
                sq += math.log(ratio) ** 2
                n += 1
            else:
                # inf means the model thinks this torque is sustainable forever
                # while the vendor measured it burning out; that is a real
                # finding, not a missing row, so it is reported not dropped.
                ratio = float("inf") if math.isinf(t_model) else float("nan")
            rows.append((tau, secs, t_model, ratio))
    finally:
        if saved_ca is not None:
            act.Rth_ca = saved_ca
    return rows, (math.sqrt(sq / n) if n else float("nan"))
