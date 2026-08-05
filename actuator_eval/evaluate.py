"""
Evaluation engine.

Each criterion returns a margin (capability / demand) plus a confidence level
inherited from the weakest input it depended on. A criterion that rests on a
guessed thermal resistance is reported as such rather than being suppressed --
you still get the number, you just also get told not to trust it yet.
"""

from __future__ import annotations
import math
from dataclasses import dataclass, field, replace
from typing import List, Optional, Dict

from .params import P, assumed_among, worst_source, ESTIMATED, GUESS
from .models import Actuator, Joint
from . import physics as phys

PASS, MARGINAL, FAIL, UNKNOWN = "PASS", "MARGINAL", "FAIL", "UNKNOWN"
_ORDER = {FAIL: 0, UNKNOWN: 1, MARGINAL: 2, PASS: 3}

# margin thresholds
PASS_MARGIN = 1.15
MARGINAL_MARGIN = 1.00


@dataclass
class Criterion:
    name: str
    status: str
    demand: float
    capability: float
    units: str
    margin: float = 0.0
    confidence: str = ESTIMATED
    detail: str = ""
    depends_on: List[str] = field(default_factory=list)
    # Some criteria report a ratio whose "margin" is not a safety factor
    # (inertia ratio, bus voltage). Those must not be picked as the binding
    # constraint just because their number happens to be small.
    margin_meaningful: bool = True
    # Advisory criteria inform the reader but must not veto the overall
    # verdict. The inertia ratio is the case in point: "rotor inertia dominates
    # the load" is worth knowing, but on a light link it is a design
    # observation, not grounds for rejecting the actuator.
    advisory: bool = False

    @staticmethod
    def from_margin(name, demand, capability, units, confidence=ESTIMATED,
                    detail="", depends_on=None, higher_is_better=True):
        if demand <= 0:
            margin = float("inf")
        else:
            margin = capability / demand
        if margin >= PASS_MARGIN:
            st = PASS
        elif margin >= MARGINAL_MARGIN:
            st = MARGINAL
        else:
            st = FAIL
        return Criterion(name, st, demand, capability, units, margin,
                         confidence, detail, depends_on or [])


@dataclass
class Evaluation:
    actuator: Actuator
    joint: Joint
    criteria: List[Criterion] = field(default_factory=list)
    thermal: Optional[phys.ThermalResult] = None
    duty_segments: List = field(default_factory=list)      # per actuator output
    joint_segments: List = field(default_factory=list)     # at the joint
    motion_phases: List = field(default_factory=list)      # tag per joint_segment
    unit_audits: List = field(default_factory=list)
    assumptions: List[P] = field(default_factory=list)
    sensitivity: Dict[str, List[str]] = field(default_factory=dict)
    extras: Dict[str, object] = field(default_factory=dict)

    @property
    def verdict(self) -> str:
        pool = [c for c in self.criteria if not c.advisory]
        if not pool:
            return UNKNOWN
        return min(pool, key=lambda c: _ORDER[c.status]).status

    @property
    def binding(self) -> Criterion:
        pool = [c for c in self.criteria
                if c.margin_meaningful and not c.advisory] or self.criteria
        return min(pool, key=lambda c: (_ORDER[c.status], c.margin))


# ----------------------------------------------------------------------------

def evaluate(act: Actuator, joint: Joint, run_sensitivity: bool = True) -> Evaluation:
    joint.fill_defaults()
    act.fill_defaults(mounting=joint.mounting)

    ev = Evaluation(actuator=act, joint=joint)
    v_bus = joint.bus_v(act)          # what THIS application supplies
    ev.extras["bus_voltage"] = v_bus

    # ---- build the duty cycle, referred to one actuator's output ----------
    joint_segs = joint.required_torque_profile(act)
    segs = []
    for dt, tau_j, om_j in joint_segs:
        tau_a, om_a = joint.actuator_output_demand(tau_j, om_j)
        segs.append((dt, tau_a, om_a))
    ev.duty_segments = segs
    ev.joint_segments = list(joint_segs)
    ev.motion_phases = joint.motion_phases()

    env = joint.envelope if joint.active_motion_source() == "envelope" else None
    ev.extras["motion_source"] = joint.active_motion_source()
    ev.extras["motion_superseded"] = joint.superseded_motion_sources()
    if env is not None:
        ev.extras["envelope"] = env

    tau_peak_joint = max(abs(t) for _, t, _ in joint_segs) if joint_segs else 0.0
    om_peak_joint = max(abs(w) for _, _, w in joint_segs) if joint_segs else 0.0

    if env is not None:
        # A cell's torque is an average over its occupants, so the segment list
        # understates the peak by construction. The percentiles and the max come
        # from raw samples and are the only honest source for a peak demand.
        #
        # Sizing uses p99.9 rather than the max: one sensor glitch or a single
        # collision should not size the whole joint. The max is not discarded --
        # it drives the DRIVE CURRENT criterion below, where a fault threshold
        # really is a true maximum rather than a mechanical margin.
        tau_peak_joint = env.peak_torque_demand()
        om_peak_joint = env.peak_speed_demand()
        ev.extras["peak_percentile"] = "p99.9"
        ev.extras["tau_peak_max"] = env.extremes.t("max")
        ev.extras["om_peak_max"] = env.extremes.w("max")

    if joint.tau_peak_req is not None:
        tau_peak_joint = max(tau_peak_joint, joint.tau_peak_req)
    if joint.omega_max_req is not None:
        om_peak_joint = max(om_peak_joint, joint.omega_max_req)

    tau_peak_act = max(abs(t) for _, t, _ in segs) if segs else 0.0
    om_peak_act = max(abs(w) for _, _, w in segs) if segs else 0.0
    n = joint.n_actuators

    # ---- 1. peak torque ---------------------------------------------------
    # An envelope's provenance rides along with every criterion that reads it:
    # torque taken from commanded current is `derived` however careful the
    # capture was, and that correctly caps the confidence of the verdict.
    env_p = [env.provenance()] if env is not None else []
    env_dep = ["envelope"] if env is not None else []

    cap_peak_out = abs(act.torque_out(float(act.I_peak_rms)))
    cap_peak_joint = cap_peak_out * n * float(joint.ratio) * float(joint.ratio_eff)
    peak_detail = (f"{n} x {cap_peak_out:.2f} N.m at actuator output, "
                   f"x{float(joint.ratio):.2f} joint ratio")
    if env is not None:
        ec = env.event_counts.get("p99_9") or env.event_counts.get("p99") or {}
        peak_detail += (f"; demand is the p99.9 of {env.capture.duration/3600.0:.2f} hr "
                        f"measured, max was {env.extremes.t('max'):.2f} N.m")
        if ec:
            peak_detail += (f" ({ec.get('events', 0)} excursions totalling "
                            f"{ec.get('total_time', 0):.1f} s, longest "
                            f"{ec.get('max_duration', 0):.2f} s)")
        if env.tail_is_outlier_dominated():
            peak_detail += ("; TAIL IS OUTLIER-DOMINATED (max more than 2x "
                            "p99.9) -- inspect the log before trusting this")
    ev.criteria.append(Criterion.from_margin(
        "Peak torque", tau_peak_joint, cap_peak_joint, "N.m",
        confidence=worst_source(act.Kt_rotor, act.I_peak_rms, act.gear_eff,
                                act.kt_sat_derate, *env_p),
        detail=peak_detail,
        depends_on=["Kt_rotor", "I_peak_rms", "kt_sat_derate", "gear_eff"] + env_dep))

    # ---- 2. speed at the required torque ----------------------------------
    tau_at_peak_speed = 0.0
    for _, t, w in segs:
        if abs(w) > 0.95 * om_peak_act:
            tau_at_peak_speed = max(tau_at_peak_speed, abs(t))
    cap_speed = _max_speed_at_torque(act, tau_at_peak_speed, v_bus)
    cap_speed_joint = cap_speed / max(float(joint.ratio), 1e-9)
    ev.criteria.append(Criterion.from_margin(
        "Speed at load", om_peak_joint, cap_speed_joint, "rad/s",
        confidence=worst_source(act.Ke, act.R_phase, act.L_phase, act.V_bus_nom),
        detail=f"needs {om_peak_joint*30/math.pi:.0f} rpm at {tau_at_peak_speed:.2f} N.m/act; "
               f"bus {v_bus:.0f} V, no-load "
               f"{phys.no_load_speed_out(act, v_bus)*30/math.pi:.0f} rpm at output",
        depends_on=["Ke", "V_bus_nom", "R_phase", "L_phase"]))

    # ---- 3. every duty point inside the voltage envelope ------------------
    # Under an envelope the segment list is binned, so it cannot be the only
    # source here: the tightest real point may have been averaged away. The
    # boundary staircase and every stored outlier are raw-sample facts, so they
    # are tested individually as well -- a one-off excursion outside the
    # envelope must not be missed just because it was brief.
    probe = [(abs(t), abs(w)) for _, t, w in segs]
    n_outside = 0
    if env is not None:
        gain = n * float(joint.ratio) * max(float(joint.ratio_eff), 1e-9)
        for om_j, tau_j in env.boundary:
            probe.append((abs(tau_j) / gain, abs(om_j) * float(joint.ratio)))
        for o in env.outliers:
            probe.append((abs(o.tau_peak) / gain,
                          abs(o.omega_at_peak) * float(joint.ratio)))

    worst_env = float("inf")
    worst_pt = None
    for t, w in probe:
        avail = phys.max_torque_at_speed(act, w, v_bus)
        if t > 1e-9:
            m = avail / t
            if m < 1.0:
                n_outside += 1
            if m < worst_env:
                worst_env, worst_pt = m, (t, w)
    if worst_pt:
        ts_detail = (f"tightest point: {worst_pt[0]:.2f} N.m at "
                     f"{worst_pt[1]*30/math.pi:.0f} rpm (actuator output)")
        if env is not None:
            ts_detail += (f"; checked against the measured boundary and all "
                          f"{len(env.outliers)} stored excursions individually")
            if n_outside:
                ts_detail += f", {n_outside} of which fall outside the envelope"
            ev.extras["points_outside_envelope"] = n_outside
        ev.criteria.append(Criterion.from_margin(
            "Torque-speed envelope", worst_pt[0],
            worst_pt[0] * worst_env, "N.m",
            confidence=worst_source(act.Ke, act.R_phase, act.V_bus_nom, *env_p),
            detail=ts_detail,
            depends_on=["Ke", "V_bus_nom", "R_phase", "L_phase"] + env_dep))

    # ---- 3b. can the actuator actually FOLLOW the commanded trajectory? ----
    # The profile is the controller's own minimum-time S-curve under
    # max_velocity / max_accel / max_jerk -- it is commanded open-loop, with no
    # knowledge of this actuator. So the question is whether the torque that
    # motion demands is available at the speed it demands it, at every instant.
    # Two distinct ways to fail, reported separately because the fixes differ:
    # short of torque means lower max_accel, short of speed means lower
    # max_velocity.
    # Gated on the ACTIVE source, not on which blocks happen to be defined: a
    # file may carry a profile alongside an envelope, and asking whether the
    # actuator can follow a trajectory it is not being evaluated against would
    # answer a question nobody asked.
    if joint.active_motion_source() == "profile":
        phases = ev.motion_phases if len(ev.motion_phases) == len(segs) else []
        worst = float("inf")
        worst_at = None
        for i, (_, tau_a, om_a) in enumerate(segs):
            avail = phys.max_torque_at_speed(act, abs(om_a), v_bus)
            if abs(tau_a) <= 1e-9:
                continue
            m = avail / abs(tau_a)
            if m < worst:
                worst = m
                worst_at = (abs(tau_a), abs(om_a), avail,
                            phases[i] if phases else "move")
        prof = joint.profile
        no_load = phys.no_load_speed_out(act, v_bus)
        om_cmd_act = prof.peak_velocity * float(joint.ratio)
        if worst_at is not None:
            tau_need, om_need, avail, ph = worst_at
            if om_cmd_act > no_load:
                why = (f"commanded {om_cmd_act*30/math.pi:.0f} rpm at the "
                       f"actuator exceeds its {no_load*30/math.pi:.0f} rpm "
                       f"no-load speed on a {v_bus:.0f} V bus: reduce "
                       f"max_velocity")
            else:
                why = (f"tightest during {ph}: needs {tau_need:.2f} N.m at "
                       f"{om_need*30/math.pi:.0f} rpm, envelope gives "
                       f"{avail:.2f} N.m")
            ev.criteria.append(Criterion.from_margin(
                "Trajectory following", tau_need, tau_need * worst, "N.m",
                confidence=worst_source(act.Ke, act.R_phase, act.V_bus_nom,
                                        act.J_rotor),
                detail=(f"{prof.regime}-limited move, solved "
                        f"{prof.move_time*1e3:.0f} ms per traverse; {why}"),
                depends_on=["Ke", "V_bus_nom", "R_phase", "L_phase",
                            "J_rotor"]))
        ev.extras["move_time"] = prof.move_time
        ev.extras["regime"] = prof.regime
        ev.extras["peak_velocity_cmd"] = prof.peak_velocity
        ev.extras["peak_accel_cmd"] = prof.peak_accel

        # Inertial torque implied by the commanded acceleration, for controllers
        # that limit TORQUE rather than acceleration: set the controller's cap to
        # this and the two limits describe the same move. J_total carries the
        # reflected rotor inertia of the units actually fitted, so this is
        # per-candidate, not a property of the application alone -- both parts
        # are reported so the load-only term can be read off directly.
        total_ratio = float(act.gear_ratio) * float(joint.ratio)
        j_refl = joint.n_actuators * float(act.J_rotor) * total_ratio ** 2
        j_load = joint.load.inertia_total()
        ev.extras["J_load_joint"] = j_load
        ev.extras["J_reflected_joint"] = j_refl
        tau_in_joint = (j_load + j_refl) * prof.peak_accel
        ev.extras["tau_inertial_cmd"] = tau_in_joint
        # Referred to one actuator's output shaft. Accelerating is always the
        # motoring case, so efficiency ADDS to the demand -- the sign-dependent
        # branch in actuator_output_demand() has no ambiguity to resolve here.
        ev.extras["tau_inertial_cmd_per_actuator"] = tau_in_joint / (
            n * float(joint.ratio) * max(float(joint.ratio_eff), 1e-9))

    # ---- 4. thermal (the one that usually decides it) ---------------------
    th = phys.simulate_duty(act, segs, joint.T_ambient)

    if env is not None and not th.runaway:
        # Replace the node temperatures with the ones implied by the ORDER-FREE
        # mean loss, before anything downstream reads them.
        #
        # simulate_duty integrates its segment list in order, which is correct
        # for a real cycle and meaningless for an occupancy record: a cell is a
        # value bucket, not an interval, so a "35 s" cell is thousands of brief
        # moments the sort happens to have placed together. Every temperature it
        # integrates therefore inherits the sort order, while mean_loss_W does
        # not. Substituting the closed-form fixed point keeps ev.thermal honest
        # for the surface-temp criterion, the report and the charts alike,
        # instead of leaving a trap that each consumer has to know about.
        rca = float(act.Rth_ca)
        rth = float(act.Rth_wc) + rca
        th = replace(th,
                     t_winding_final=joint.T_ambient + th.mean_loss_W * rth,
                     t_case_final=joint.T_ambient + th.mean_loss_W * rca,
                     t_winding_peak=joint.T_ambient + th.mean_loss_W * rth)

    ev.thermal = th
    t_limit = float(act.T_winding_max)
    if th.runaway:
        ev.criteria.append(Criterion(
            "Thermal (winding)", FAIL, float("inf"), t_limit, "degC", 0.0,
            worst_source(act.R_phase, act.Rth_wc, act.Rth_ca),
            "thermal runaway: copper tempco feedback has no equilibrium",
            ["R_phase", "Rth_wc", "Rth_ca", "T_winding_max"]))
    elif env is not None:
        # Under an envelope the thermal question splits in two, because the two
        # halves come from different artifacts and neither can answer the other.
        #
        #   (a) the SESSION MEAN, from the occupancy record.
        #
        # ONLY mean_loss_W may be taken from simulate_duty here. Its node
        # temperatures may not, and this is not a subtlety -- it is the whole
        # reason the occupancy record is restricted to order-free quantities.
        #
        # An occupancy cell is not an interval of the session: a cell holding
        # 4.6 A for "35 s" is thousands of brief moments scattered across the
        # hour, aggregated by VALUE. simulate_duty integrates its segment list
        # in order, so it reads that cell as a sustained 35-second burst at
        # 35 W. Because as_duty_segments() sorts by ascending tau^2, every hot
        # cell then runs consecutively at the end -- the worst possible
        # arrangement. Measured on the bundled example, t_winding_final swings
        # from 43 degC to 155 degC purely with sort order while mean_loss_W
        # stays at 4.62 W in every ordering.
        #
        # th was rebuilt from that mean loss above, so its node temperatures are
        # now the closed-form fixed point and carry no ordering at all.
        t_w_settled = th.t_winding_final
        t_c_settled = th.t_case_final

        rise_limit = t_limit - joint.T_ambient
        rise_settled = t_w_settled - joint.T_ambient
        ev.criteria.append(Criterion.from_margin(
            "Thermal, session mean", max(rise_settled, 1e-9),
            max(rise_limit, 1e-9), "degC rise",
            confidence=worst_source(act.R_phase, act.Rth_wc, act.Rth_ca,
                                    act.T_winding_max, *env_p),
            detail=f"settled winding {t_w_settled:.0f} degC, case "
                   f"{t_c_settled:.0f} degC, limit {t_limit:.0f} degC at "
                   f"{joint.T_ambient:.0f} degC ambient; mean loss "
                   f"{th.mean_loss_W:.1f} W/actuator over the whole session. "
                   f"Both follow from that cycle-average loss, which is the "
                   f"only order-free quantity the occupancy record carries; "
                   f"any within-cycle peak read off it would be an artefact of "
                   f"the cell sort. See the worst-sequence check.",
            depends_on=["R_phase", "Rth_wc", "Rth_ca", "T_winding_max",
                        "iron_loss_ref_W"] + env_dep))
        ev.extras["t_case_session_mean"] = t_c_settled
        ev.extras["t_winding_session_mean"] = t_w_settled

        #   (b) the WORST SEQUENCE, from an ordered window. The composition is
        #       the point: the case node warms to the session mean first, and
        #       then the worst real stretch of the log happens on top of that.
        #       time_to_limit already takes a starting temperature, so this
        #       needs no new physics at all.
        tau_w_act = float(act.C_w) * float(act.Rth_wc)
        win = env.binding_window(tau_w_act)
        if win and win.segments:
            wsegs = []
            for dt, tau_j, om_j in win.segments:
                tau_a, om_a = joint.actuator_output_demand(tau_j, om_j)
                wsegs.append((dt, tau_a, om_a))
            ev.extras["binding_window"] = win
            ev.extras["actuator_tau_w"] = tau_w_act
            try:
                ttl = phys.time_to_limit(act, wsegs, joint.T_ambient,
                                         t_start=th.t_case_final,
                                         max_time=max(win.duration * 4, 60.0))
            except Exception:
                ttl = None
            ev.extras["window_time_to_limit_s"] = ttl

            # Repeating the window forever is a deliberate over-estimate -- a
            # burst is not a duty cycle -- but it bounds how much of the margin
            # comes from the burst being a burst, which is worth stating.
            th_w = phys.simulate_duty(act, wsegs, joint.T_ambient)
            ev.extras["window_settled_winding"] = th_w.t_winding_final

            sel = ", ".join(f"{t:g}" for t in win.selected_by_tau_w) or "-"
            if ttl is None:
                st, margin = PASS, 2.0
                verdict = (f"survives, and would be sustainable indefinitely "
                           f"even repeated")
            elif ttl >= win.duration:
                st = PASS
                margin = min(ttl / max(win.duration, 1e-9), 3.0)
                verdict = (f"survives it: reaches the winding limit only after "
                           f"{ttl:.0f} s, and the sequence lasts "
                           f"{win.duration:.0f} s")
            else:
                st, margin = FAIL, ttl / max(win.duration, 1e-9)
                verdict = (f"does NOT survive it: hits the winding limit "
                           f"{ttl:.0f} s in, {win.duration:.0f} s before the "
                           f"sequence ends")
            ev.criteria.append(Criterion(
                "Thermal, worst sequence", st, win.duration,
                ttl if ttl is not None else float("inf"), "s", margin,
                worst_source(act.R_phase, act.Rth_wc, act.C_w, act.T_winding_max,
                             *env_p),
                f"worst {win.duration:.0f} s of the session (at t={win.found_at:.0f} s, "
                f"RMS {win.rms_torque:.2f} N.m at the joint, selected by the "
                f"{sel} s kernel; this actuator's winding constant is "
                f"{tau_w_act:.1f} s), starting from the {th.t_case_final:.0f} degC "
                f"case the session mean produces: {verdict}. Repeated forever it "
                f"would settle at {th_w.t_winding_final:.0f} degC.",
                ["R_phase", "Rth_wc", "C_w", "T_winding_max"] + env_dep))
    else:
        rise_limit = t_limit - joint.T_ambient
        rise_actual = th.t_winding_peak - joint.T_ambient
        ev.criteria.append(Criterion.from_margin(
            "Thermal (winding)", max(rise_actual, 1e-9), max(rise_limit, 1e-9), "degC rise",
            confidence=worst_source(act.R_phase, act.Rth_wc, act.Rth_ca, act.T_winding_max),
            detail=f"peak winding {th.t_winding_peak:.0f} degC, case {th.t_case_final:.0f} degC, "
                   f"limit {t_limit:.0f} degC at {joint.T_ambient:.0f} degC ambient; "
                   f"mean loss {th.mean_loss_W:.1f} W/actuator"
                   + ("" if th.settled else "  [NOT SETTLED - still rising]"),
            depends_on=["R_phase", "Rth_wc", "Rth_ca", "T_winding_max", "iron_loss_ref_W"]))

    # ---- 4b. burst capability --------------------------------------------
    # An actuator that cannot sustain a duty forever may still be fine if the
    # motion only ever happens in short bursts, so quantify how long it lasts
    # from cold rather than just failing it.
    # Under an envelope the worst-sequence criterion above already answers this
    # against a real ordered excerpt, which is strictly better than repeating a
    # binned occupancy record, so this is skipped there.
    th_crit = next((c for c in ev.criteria
                    if c.name.startswith("Thermal")), ev.criteria[-1])
    if env is None and (th_crit.status != PASS or th_crit.margin < 1.5):
        try:
            ev.extras["time_to_limit_s"] = phys.time_to_limit(
                act, segs, joint.T_ambient, max_time=3600.0)
        except Exception:
            pass

    # ---- 5. continuous torque headroom -----------------------------------
    i_rms = phys.rms_current_of_duty(act, segs)
    om_mean = sum(abs(w) * dt for dt, _, w in segs) / max(sum(d for d, _, _ in segs), 1e-9)
    tau_cont_cap = phys.continuous_torque_limit(act, joint.T_ambient,
                                                om_mean * float(act.gear_ratio))
    tau_rms_out = abs(act.torque_out(i_rms))
    cont_detail = (f"duty RMS current {i_rms:.2f} A_rms vs continuous limit "
                   f"{phys.continuous_current_limit(act, joint.T_ambient):.2f} A_rms")
    if env is not None:
        # The RMS above is over the whole session. What actually sets a thermal
        # verdict is the worst sustained stretch on the timescale the case node
        # integrates over, which is what the duration curve reports.
        tau_c = float(act.C_c) * float(act.Rth_ca)
        sustained = env.sustained_torque_at(tau_c)
        if sustained is not None:
            cont_detail += (f"; over this actuator's {tau_c/60.0:.0f} min case "
                            f"constant the session's worst sustained joint "
                            f"torque was {sustained:.2f} N.m")
    ev.criteria.append(Criterion.from_margin(
        "Continuous torque (RMS)", tau_rms_out, tau_cont_cap, "N.m/actuator",
        confidence=worst_source(act.R_phase, act.Rth_wc, act.Rth_ca, *env_p),
        detail=cont_detail,
        depends_on=["R_phase", "Rth_wc", "Rth_ca", "T_winding_max"] + env_dep))
    ev.extras["i_rms_duty"] = i_rms
    ev.extras["tau_cont_cap_per_actuator"] = tau_cont_cap
    ev.extras["tau_cont_cap_joint"] = tau_cont_cap * n * float(joint.ratio) * float(joint.ratio_eff)

    # ---- 6. drive current headroom ---------------------------------------
    # The one criterion that uses the true MAX rather than a percentile. An
    # overcurrent fault is a threshold the silicon either crosses or does not,
    # so a 4 ms spike trips it exactly as surely as a sustained one; sizing this
    # off p99.9 would clear a drive that faults in service.
    i_peak_duty = max(abs(act.current_for_torque(t)) for _, t, _ in segs) if segs else 0.0
    dc_detail = "peak phase current demanded by the duty cycle vs the drive's rating"
    if env is not None:
        gain = n * float(joint.ratio) * max(float(joint.ratio_eff), 1e-9)
        i_peak_duty = abs(act.current_for_torque(env.extremes.t("max") / gain))
        dc_detail = (f"from the true measured peak {env.extremes.t('max'):.2f} N.m "
                     f"at the joint, not a percentile: an overcurrent fault is a "
                     f"threshold, so the briefest spike trips it")
        if env.extremes.t("max_raw") > env.extremes.t("max") * 1.05:
            dc_detail += (f". Before despiking the log peaked at "
                          f"{env.extremes.t('max_raw'):.2f} N.m")
    ev.criteria.append(Criterion.from_margin(
        "Drive current limit", i_peak_duty, float(act.I_peak_rms), "A_rms",
        confidence=worst_source(act.I_peak_rms, act.Kt_rotor, *env_p),
        detail=dc_detail,
        depends_on=["I_peak_rms", "Kt_rotor"] + env_dep))

    # ---- 7. inertia match -------------------------------------------------
    if env is not None:
        # An envelope is a record of what the joint ACTUALLY did, so nothing in
        # the application's `load` block is consulted on this path -- gravity,
        # friction and external torque were all present in the logged torque as
        # experienced. The mass properties are the last thing that could still
        # be read here, and reading them would make the rule "the load block is
        # unused with an envelope" false in exactly one place, from declared
        # numbers the envelope does not corroborate. Report it as unavailable
        # instead. Advisory, so this never reaches a verdict.
        ev.criteria.append(Criterion(
            "Inertia ratio", UNKNOWN, 0.0, 0.0, "kg.m^2", float("inf"),
            worst_source(act.J_rotor),
            "not available: sized against a measured envelope, so no declared "
            "load values are used (set motion_source to 'profile' to size "
            "against the load block instead)",
            ["J_rotor", "gear_ratio"], margin_meaningful=False, advisory=True))
    else:
        _inertia_match(ev, act, joint, n)

    # ---- 8. mass budget ---------------------------------------------------
    if joint.mass_budget is not None:
        ev.criteria.append(Criterion.from_margin(
            "Mass budget", n * float(act.mass), joint.mass_budget, "kg",
            confidence=worst_source(act.mass),
            detail=f"{n} x {float(act.mass)*1000:.0f} g = {n*float(act.mass)*1000:.0f} g",
            depends_on=["mass"]))

    # ---- 9. bus voltage compatibility ------------------------------------
    # The application supplies the voltage; the actuator declares what it will
    # accept. Running above the vendor's NOMINAL is routine and simply extends
    # the speed envelope. Running above their stated MAXIMUM is a different
    # thing entirely, and is deliberately not something this tool will clear on
    # thermal grounds -- see the note below.
    vmin = float(act.V_bus_min) if act.V_bus_min is not None else v_bus
    vmax = float(act.V_bus_max) if act.V_bus_max is not None else v_bus
    vnom = float(act.V_bus_nom)
    src = "application" if joint.bus_v_is_from_app() else "actuator nominal (no application value given)"

    if v_bus < vmin:
        st, detail = FAIL, (f"{v_bus:.1f} V is below the actuator's minimum "
                            f"{vmin:.0f} V; expect undervoltage lockout")
    elif v_bus <= vmax:
        over = f", {100*(v_bus/vnom - 1):+.0f}% vs the {vnom:.0f} V nominal" if abs(v_bus - vnom) > 0.5 else ""
        st, detail = PASS, (f"{v_bus:.1f} V from the {src}; actuator accepts "
                            f"{vmin:.0f}-{vmax:.0f} V{over}")
    else:
        st = MARGINAL if joint.accept_overvoltage else FAIL
        detail = (f"{v_bus:.1f} V EXCEEDS the actuator's stated maximum "
                  f"{vmax:.0f} V. This limit is set by the drive's transistor "
                  f"and DC-link capacitor voltage ratings, not by heat, so "
                  f"respecting the thermal limits does not make it safe. Note "
                  f"also that regenerative braking pushes the bus ABOVE its "
                  f"resting value, so the transient peak is what the silicon "
                  f"actually sees.")
        if joint.accept_overvoltage:
            detail += " Downgraded to a warning by accept_overvoltage in the application file."
    ev.criteria.append(Criterion(
        "Bus voltage", st, v_bus, vmax, "V",
        1.0 if st == PASS else (0.5 if st == MARGINAL else 0.0),
        worst_source(act.V_bus_max), detail,
        ["V_bus_max", "V_bus_min"], margin_meaningful=False))

    # ---- 9b. surface temperature (user safety) ---------------------------
    if joint.max_surface_temp is not None and ev.thermal and not ev.thermal.runaway:
        rise_actual = max(ev.thermal.t_case_final - joint.T_ambient, 1e-9)
        rise_allowed = max(joint.max_surface_temp - joint.T_ambient, 1e-9)
        c = Criterion.from_margin(
            "Surface temp (safety)", rise_actual, rise_allowed, "degC rise",
            confidence=worst_source(act.Rth_ca, act.R_phase),
            detail=f"housing settles at {ev.thermal.t_case_final:.0f} degC vs a "
                   f"{joint.max_surface_temp:.0f} degC touch limit "
                   f"({joint.T_ambient:.0f} degC ambient). Lumped housing "
                   f"temperature: real surfaces have hot spots near the stator, "
                   f"so treat this as optimistic and verify by measurement.",
            depends_on=["Rth_ca", "R_phase", "Rth_wc"])
        ev.criteria.append(c)

    # ---- 9c. supply current (only if the application declares a limit) ----
    if joint.supply_current_limit is not None:
        ev.criteria.append(Criterion.from_margin(
            "Supply current", i_peak_duty, float(joint.supply_current_limit),
            "A_rms/actuator", confidence=worst_source(act.Kt_rotor),
            detail="peak phase current demanded vs what this robot's supply can "
                   "deliver per actuator",
            depends_on=["Kt_rotor"]))

    # ---- 10. backdrivability ---------------------------------------------
    if joint.require_backdrivable:
        # torque at the joint needed to overcome reflected friction + cogging
        total_ratio = float(act.gear_ratio) * float(joint.ratio)
        tau_bd = (float(act.friction_torque_rotor) * total_ratio
                  / max(float(act.gear_eff) * float(joint.ratio_eff), 1e-6)) * n
        thresh = 0.10 * max(tau_peak_joint, 1e-9)
        ev.criteria.append(Criterion.from_margin(
            "Backdrive torque", tau_bd, thresh, "N.m",
            confidence=worst_source(act.friction_torque_rotor),
            detail=f"est. {tau_bd:.2f} N.m to backdrive at the joint; "
                   f"target under {thresh:.2f} N.m (10% of peak)",
            depends_on=["friction_torque_rotor", "gear_eff"]))

    # ---- assumptions and sensitivity -------------------------------------
    ev.assumptions = [p for p in vars(act).values()
                      if isinstance(p, P) and p.is_assumed]
    if run_sensitivity:
        ev.sensitivity = _sensitivity(act, joint, ev)
    return ev


def _inertia_match(ev: Evaluation, act: Actuator, joint: Joint, n: int) -> None:
    """
    Reflected rotor inertia against the load, from the declared mass properties.

    Only reachable on the predicted paths. An envelope supersedes the whole
    `load` block, so evaluate() reports this as unavailable there rather than
    computing it from numbers the measurement does not corroborate.
    """
    total_ratio = float(act.gear_ratio) * float(joint.ratio)
    j_refl = n * float(act.J_rotor) * total_ratio ** 2
    j_load = joint.load.inertia_total()
    ratio_im = j_refl / max(j_load, 1e-12)
    # Classic "inertia matching" (ratio ~ 1) maximises acceleration per amp, but
    # that is a machine-tool criterion. For a robot joint, reflected rotor
    # inertia that is SMALL next to the load is exactly what a quasi-direct-drive
    # is for: it keeps the joint backdrivable and force-transparent. Only a
    # rotor that dominates the load is a genuine problem, because then most of
    # the torque goes into accelerating the motor itself and impacts are
    # transmitted straight back through the gearbox.
    if ratio_im <= 2.0:
        im_status, verdict_txt = PASS, "rotor inertia is small next to the load (good for a QDD joint)"
    elif ratio_im <= 5.0:
        im_status, verdict_txt = MARGINAL, "rotor inertia is becoming significant next to the load"
    else:
        im_status, verdict_txt = FAIL, "rotor inertia DOMINATES the load: most torque accelerates the motor"
    ev.criteria.append(Criterion(
        "Inertia ratio", im_status, j_load, j_refl, "kg.m^2", ratio_im,
        worst_source(act.J_rotor),
        f"reflected/load = {ratio_im:.3f} (reflected {j_refl*1e3:.2f} g.m^2, "
        f"load {j_load*1e3:.2f} g.m^2); {verdict_txt}",
        ["J_rotor", "gear_ratio"], margin_meaningful=False, advisory=True))


def _max_speed_at_torque(act: Actuator, tau_out: float, v_bus: float) -> float:
    """Highest output speed at which the actuator can still make tau_out."""
    hi = phys.no_load_speed_out(act, v_bus)
    if tau_out <= 1e-9:
        return hi
    if phys.max_torque_at_speed(act, 0.0, v_bus) < tau_out:
        return 0.0
    lo = 0.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if phys.max_torque_at_speed(act, mid, v_bus) >= tau_out:
            lo = mid
        else:
            hi = mid
    return lo


def _sensitivity(act: Actuator, joint: Joint, base: Evaluation) -> Dict[str, List[str]]:
    """
    Push each assumed parameter to the edge of its uncertainty band and see
    which verdicts flip. This is what turns "we don't know Rth" from a blocker
    into a prioritised measurement list.
    """
    import copy
    result: Dict[str, List[str]] = {}
    base_status = {c.name: c.status for c in base.criteria}

    for p in base.assumptions:
        flips = set()
        for k in (1.0 - p.tol, 1.0 + p.tol):
            if k <= 0:
                continue
            a2 = copy.deepcopy(act)
            getattr(a2, p.name).value = p.value * k
            try:
                ev2 = evaluate(a2, copy.deepcopy(joint), run_sensitivity=False)
            except Exception:
                continue
            for c in ev2.criteria:
                if base_status.get(c.name) != c.status:
                    flips.add(f"{c.name}: {base_status.get(c.name)} -> {c.status} "
                              f"at {p.name} x{k:.2f}")
        if flips:
            result[p.name] = sorted(flips)
    return result
