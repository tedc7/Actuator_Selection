"""
Actuator / joint database, with unit validation on load.

Definitions live as JSON so they diff cleanly in git, can be reviewed, and can
be edited without touching code.

Every physical field is checked against the dimension it is supposed to have
and converted to canonical SI before the physics sees it (see units.py). A unit
from the wrong dimension is a hard error. A missing unit is recorded in the
unit audit that the report prints, or a hard error if the file sets
"strict_units": true.

Fields you leave out entirely are filled in by the estimators in models.py and
reported as assumptions.
"""

from __future__ import annotations
import json
import math
import os
from typing import Dict, List, Optional, Tuple

from .params import P, VENDOR_SPEC
from .models import (Actuator, Joint, Load, MotionProfile, OverloadCurve,
                     TNCurve)
from . import units as U

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)

# Actuators are vendor reference data: public, reusable, shipped with the repo.
ACTUATOR_DIR = os.path.join(_HERE, "db", "actuators")

# Applications describe one joint on one robot. They are not reusable reference
# data -- payload, geometry, bus voltage and internal ambient are facts about
# YOUR machine -- so they live outside the package and are gitignored. Set
# ACTUATOR_EVAL_APPS_DIR to keep them somewhere else entirely (another repo, a
# synced drive); it takes precedence over the default location.
APPLICATION_DIR = os.environ.get(
    "ACTUATOR_EVAL_APPS_DIR", os.path.join(_REPO, "applications"))

# The bundled examples are public so a fresh clone can run the quick start.
EXAMPLE_DIR = os.path.join(APPLICATION_DIR, "examples")

# Envelopes are summaries of real capture sessions. They get their own directory
# rather than living inside the application file because a joint is re-captured
# periodically without the mechanism changing: a recapture should produce a diff
# that touches only the envelope, and carry its own provenance (which log, which
# robot, which day) independently of the joint's physical definition.
ENVELOPE_DIR = os.environ.get(
    "ACTUATOR_EVAL_ENVELOPES_DIR", os.path.join(_REPO, "envelopes"))
EXAMPLE_ENVELOPE_DIR = os.path.join(ENVELOPE_DIR, "examples")

ENVELOPE_SCHEMA = "actuator_eval.envelope/1"


def application_search_path() -> List[str]:
    """Directories searched for an application named without a path."""
    return [APPLICATION_DIR, EXAMPLE_DIR]


def envelope_search_path() -> List[str]:
    """Directories searched for an envelope named without a path."""
    return [ENVELOPE_DIR, EXAMPLE_ENVELOPE_DIR]

_ACT_PLAIN = ["name", "vendor", "url", "price_usd", "gear_type",
              "pole_pairs", "modulation_k", "notes",
              "datasheet", "entry_revision"]


def _meta(container: dict, key: str) -> Tuple[str, Optional[float], str]:
    """Pull source / tol / note off a field written in object form."""
    d = container.get(key)
    if isinstance(d, dict):
        return d.get("source", VENDOR_SPEC), d.get("tol"), d.get("note", "")
    return VENDOR_SPEC, None, ""


def _as_param(container, key, dimension, audit, strict, name=None):
    """parse_value + wrap in a P carrying provenance and canonical units."""
    v = U.parse_value(container, key, dimension, audit=audit, strict=strict)
    if v is None:
        return None
    src, tol, note = _meta(container, key)
    return P(v, U.canonical_unit(dimension), src, tol, note, name or key)


# ---------------------------------------------------------------------------
# Actuator
# ---------------------------------------------------------------------------

def actuator_from_dict(d: Dict) -> Tuple[Actuator, U.UnitAudit]:
    audit = U.UnitAudit()
    strict = bool(d.get("strict_units", False))
    a = Actuator()

    for k in _ACT_PLAIN:
        if k in d and d[k] is not None:
            setattr(a, k, d[k])

    # dimensions: accept mm (conventional) or an explicit list in m
    if "dims_mm" in d:
        a.dims_m = tuple(x / 1000.0 for x in d["dims_mm"])
        audit.record("dims", a.dims_m[0], "m",
                     f"{d['dims_mm']} mm (from field name)", True)
    elif "dims_m" in d:
        a.dims_m = tuple(d["dims_m"])
        audit.record("dims", a.dims_m[0], "m", f"{d['dims_m']} m", True)

    simple = ["mass", "gear_ratio", "gear_eff", "J_rotor", "R_phase", "L_phase",
              "I_cont_rms", "I_peak_rms", "kt_sat_derate", "V_bus_nom",
              "V_bus_min", "V_bus_max", "tau_cont_out_spec", "tau_peak_out_spec",
              "Rth_wc", "Rth_ca", "C_w", "C_c", "T_winding_max",
              "iron_loss_ref_W", "friction_torque_rotor", "viscous_rotor",
              "Ke", "Kt_rotor"]
    for k in simple:
        p = _as_param(d, k, U.ACTUATOR_DIMENSIONS[k], audit, strict)
        if p is not None:
            setattr(a, k, p)

    if "backlash" in d or any(k.startswith("backlash_") for k in d):
        p = _as_param(d, "backlash", "angle", audit, strict, name="backlash_arcmin")
        if p is not None:
            a.backlash_arcmin = p

    # peak-amplitude currents convert to the RMS the model works in
    for src_key, dst in (("I_cont_peak_amp", "I_cont_rms"),
                         ("I_peak_peak_amp", "I_peak_rms")):
        if getattr(a, dst) is None:
            p = _as_param(d, src_key, "current_peak", audit, strict, name=dst)
            if p is not None:
                p.value /= math.sqrt(2.0)
                p.units = "A"
                p.note = (p.note + "; converted from peak amplitude to RMS").strip("; ")
                setattr(a, dst, p)

    # back-EMF in the conventional datasheet unit Vrms(LL) per 1000 rotor rpm
    if a.Ke is None and "Ke_Vrms_per_kRPM" in d:
        raw = d["Ke_Vrms_per_kRPM"]
        stated = raw.get("units") if isinstance(raw, dict) else None
        value = float(raw["value"]) if isinstance(raw, dict) else float(raw)
        unit = stated or "Vrms/kRPM"      # field name declares the unit
        v = U.convert(value, unit, "back_emf")
        src, tol, note = _meta(d, "Ke_Vrms_per_kRPM")
        if stated is None:
            note = (note + "; unit taken from field name (Vrms_LL per kRPM)").strip("; ")
        audit.record("Ke", v, "V_rms_LL/(rad/s)", f"{value} {unit}", True)
        a.Ke = P(v, "V_rms_LL/(rad/s)", src, tol, note, "Ke")

    # torque constant quoted at the output shaft
    if a.Kt_rotor is None and a.Ke is None:
        p = _as_param(d, "Kt_out_per_Arms", "torque_constant", audit, strict,
                      name="Kt_rotor")
        if p is not None:
            eff = float(a.gear_eff) if a.gear_eff is not None else 0.94
            p.value /= (float(a.gear_ratio) * eff)
            p.note = (p.note + "; converted from output-referred Kt").strip("; ")
            a.Kt_rotor = p

    a.tn_curve = _tn_curve_from_dict(d.get("tn_curve"), audit)
    a.overload_curves = _overload_curves_from_dict(d.get("overload_curves"), audit)

    return a, audit


def _overload_curves_from_dict(raw, audit) -> List[OverloadCurve]:
    """
    Parse measured overload-endurance tables. Absent means no thermal
    cross-check is possible, which is the normal case, not an error.

    Accepts a single table or a list of them: vendors publish separate tables
    for rotating and stalled operation and they are not interchangeable.
    """
    if not raw:
        return []
    entries = raw if isinstance(raw, list) else [raw]
    out = []
    for e in entries:
        if not e or e.get("source") in ("unavailable", "none") and "points" not in e:
            continue
        pts = e.get("points") or []
        if len(pts) < 2:
            continue

        # Stall heating is 1.414x rotating for the same current, because the
        # three phases heat unevenly when the rotor is not moving. Guessing
        # which condition a table describes would be a 41% error, so require it.
        cond = (e.get("condition") or "").lower()
        if cond not in ("rotating", "stalled"):
            raise U.UnitError(
                f"overload_curves.condition '{e.get('condition')}' not understood; "
                "use 'rotating' or 'stalled' -- stall heating is 1.414x rotating "
                "for the same current, so this cannot be assumed")

        tu = (e.get("torque_units") or "N.m").lower()
        if tu not in ("n.m", "nm", "n_m"):
            raise U.UnitError(
                f"overload_curves.torque_units '{e.get('torque_units')}' not "
                "understood; use 'N.m'")
        su = (e.get("time_units") or "s").lower()
        if su in ("s", "sec", "seconds"):
            tconv = 1.0
        elif su in ("min", "minutes"):
            tconv = 60.0
        else:
            raise U.UnitError(
                f"overload_curves.time_units '{e.get('time_units')}' not "
                "understood; use 's' or 'min'")

        c = OverloadCurve(
            points=[(float(t), float(s) * tconv) for t, s in pts],
            condition=cond,
            speed_rpm_output=(float(e["speed_rpm_output"])
                              if e.get("speed_rpm_output") is not None else None),
            ambient_C=(float(e["ambient_C"])
                       if e.get("ambient_C") is not None else None),
            mounting=e.get("mounting"),
            rated_torque=(float(e["rated_torque"])
                          if e.get("rated_torque") is not None else None),
            source=e.get("source", "vendor"),
            tol=float(e.get("tol", 0.15)),
            note=e.get("note", ""))
        lo, hi = c.torque_range
        audit.record(f"overload_curves[{cond}]", hi, "N.m",
                     f"{len(c.points)} pts, {lo:.1f}-{hi:.1f} N.m, "
                     f"{c.points[-1][1]:.0f}-{c.points[0][1]:.0f} s"
                     + (f" at {c.speed_rpm_output:.0f} rpm" if c.speed_rpm_output else "")
                     + (f", {c.ambient_C:.0f} degC" if c.ambient_C is not None else ""),
                     True)
        out.append(c)
    return out


def _tn_curve_from_dict(raw, audit) -> Optional[TNCurve]:
    """
    Parse a measured torque-speed envelope, or return None to fall back to the
    computed one.

    An entry may declare "source": "unavailable" to say explicitly that no
    trustworthy curve exists -- worth doing when a vendor prints one you have
    reason to distrust, because it records the judgement instead of looking
    like an oversight.
    """
    if not raw:
        return None
    if raw.get("source") in ("unavailable", "none", None) and "points" not in raw:
        return None

    pts = raw.get("points") or []
    if len(pts) < 2:
        return None

    # Speed may be given at the output (the usual way a datasheet plots it) or
    # at the rotor. Getting this wrong is a gear-ratio-sized error, so the unit
    # is explicit and anything unrecognised is refused rather than assumed.
    su = (raw.get("speed_units") or "rpm_output").lower()
    if su in ("rpm_output", "rpm", "rpm_out"):
        conv = 1.0
    elif su in ("rad_s_output", "rad/s", "rad_s"):
        conv = 30.0 / math.pi
    else:
        raise U.UnitError(
            f"tn_curve.speed_units '{raw.get('speed_units')}' not understood; "
            "use 'rpm_output' or 'rad_s_output'")

    tu = (raw.get("torque_units") or "N.m").lower()
    if tu not in ("n.m", "nm", "n_m"):
        raise U.UnitError(
            f"tn_curve.torque_units '{raw.get('torque_units')}' not understood; use 'N.m'")

    bus = raw.get("bus_voltage")
    if isinstance(bus, dict):
        bus = float(bus["value"])
    elif bus is not None:
        bus = float(bus)

    curve = TNCurve(points=[(float(s) * conv, float(t)) for s, t in pts],
                    bus_voltage=bus,
                    source=raw.get("source", "vendor"),
                    tol=float(raw.get("tol", 0.05)),
                    note=raw.get("note", ""))
    lo, hi = curve.speed_range_rpm
    audit.record("tn_curve", hi, "rpm_output",
                 f"{len(curve.points)} pts, {lo:.0f}-{hi:.0f} rpm"
                 + (f" at {bus:.0f} V" if bus else ""), True)
    return curve


def resolve_actuator_path(name_or_path: str) -> str:
    """Locate an actuator file by name or explicit path."""
    if os.path.exists(name_or_path):
        return os.path.abspath(name_or_path)
    path = os.path.join(ACTUATOR_DIR, name_or_path)
    if not path.endswith(".json"):
        path += ".json"
    if not os.path.exists(path):
        known = ", ".join(list_actuators()) or "none found"
        raise FileNotFoundError(
            f"no actuator '{name_or_path}'. Give a path to a json file, or one "
            f"of the names in the database: {known}")
    return os.path.abspath(path)


def load_actuator(name_or_path: str, with_audit: bool = False):
    path = resolve_actuator_path(name_or_path)
    with open(path) as f:
        d = json.load(f)
    try:
        a, audit = actuator_from_dict(d)
    except U.UnitError as e:
        raise U.UnitError(f"in actuator file '{path}': {e}") from None
    a.source_path = path
    return (a, audit) if with_audit else a


def list_actuators() -> List[str]:
    if not os.path.isdir(ACTUATOR_DIR):
        return []
    return sorted(f[:-5] for f in os.listdir(ACTUATOR_DIR)
                  if f.endswith(".json") and not f.startswith("_"))


# ---------------------------------------------------------------------------
# Application (one joint on one robot)
# ---------------------------------------------------------------------------

def joint_from_dict(d: Dict) -> Tuple[Joint, U.UnitAudit]:
    audit = U.UnitAudit()
    strict = bool(d.get("strict_units", False))
    j = Joint()

    for k in ("name", "kind", "n_actuators", "ratio_type",
              "require_backdrivable", "mounting", "duty_repeats_forever",
              "accept_overvoltage"):
        if k in d and d[k] is not None:
            setattr(j, k, d[k])

    for k in ("ratio", "ratio_eff"):
        p = _as_param(d, k, U.JOINT_DIMENSIONS[k], audit, strict)
        if p is not None:
            setattr(j, k, p)

    for k in ("tau_peak_req", "tau_cont_req", "omega_max_req", "mass_budget",
              "T_ambient", "lead", "bus_voltage", "supply_current_limit",
              "max_surface_temp"):
        v = U.parse_value(d, k, U.JOINT_DIMENSIONS[k], audit=audit, strict=strict)
        if v is not None:
            setattr(j, "lead_m" if k == "lead" else k, v)

    v = U.parse_value(d, "position_tolerance", "angle", audit=audit, strict=strict)
    if v is not None:
        j.position_tolerance_rad = v

    if "load" in d:
        ld = d["load"]
        # Renamed fields are a hard error, never a silent drop: the parser only
        # reads keys it knows, so an un-renamed file would otherwise lose its
        # payload distance and evaluate against zero gravity torque.
        for old, new in (("com_distance", "distance_joint_axis_to_CG"),
                         ("payload_distance", "distance_joint_axis_to_CG"),
                         ("payload_mass", "total_mass_at_CG"),
                         ("gravity_phase", "gravity_angle")):
            if any(k == old or k.startswith(old + "_") for k in ld):
                raise ValueError(
                    f"'{old}' was renamed to '{new}'. Update the load block of "
                    f"this application file; the value and units are unchanged.")
        # link_inertia is the one whose MEANING moved, not just its name: it was
        # documented as inertia about the joint axis, and is now stated about the
        # CG with this module doing the parallel-axis transfer. Passing an
        # axis-referred figure through silently would double-count m*d^2.
        if any(k == "link_inertia" or k.startswith("link_inertia_") for k in ld):
            raise ValueError(
                "'link_inertia' was replaced by 'moment_of_inertia_around_CG', "
                "which is referred to the CENTRE OF GRAVITY rather than to the "
                "joint axis -- the parallel-axis transfer "
                "total_mass_at_CG * distance_joint_axis_to_CG^2 is now added for "
                "you. Take the CAD figure about the CG directly; if all you have "
                "is an axis-referred inertia I_axis, use "
                "moment_of_inertia_around_CG = I_axis - m*d^2.")
        # gravity_factor -> joint_plane_tilt is not a rename: the value changes
        # meaning from cos(tilt) to the tilt angle itself.
        if any(k == "gravity_factor" or k.startswith("gravity_factor_")
               for k in ld):
            raise ValueError(
                "'gravity_factor' was replaced by 'joint_plane_tilt', an ANGLE "
                "(degrees by default) rather than a 0..1 scale: it is the tilt "
                "of the joint's plane of rotation out of vertical. "
                "gravity_factor 1.0 -> joint_plane_tilt 0 deg; "
                "0.0 -> 90 deg; otherwise joint_plane_tilt = acos(gravity_factor).")
        # The gravity_angle CONVENTION changed with the same release: it now
        # points along the gravity vector (where the payload hangs, zero
        # torque) instead of marking the torque peak. A file written for the
        # old convention parses cleanly but is wrong by 90 degrees, so the new
        # spelling of the tilt field is what distinguishes them -- hence the
        # check above must fire before any gravity_angle value is trusted.
        kw = {}
        for k, dim in U.LOAD_DIMENSIONS.items():
            val = U.parse_value(ld, k, dim, audit=audit, strict=strict,
                                default_unit=U.DEFAULT_UNITS.get(k))
            if val is not None:
                kw[k] = val
        j.load = Load(**kw)

    if "profile" in d:
        pd = d["profile"]
        # As with the load block: renamed keys must not be silently dropped.
        # 'stroke' + 'theta_start' described the same move as a start plus a
        # signed sweep; the two endpoints say it directly.
        # Match the old name and its unit-suffixed spellings (stroke_deg), but
        # not stroke_start / stroke_end, which share the 'stroke' prefix.
        def _is_old(key: str, old: str) -> bool:
            if any(key == n or key.startswith(n + "_")
                   for n in U.PROFILE_DIMENSIONS):
                return False
            return key == old or key.startswith(old + "_")

        for old, new in (("stroke", "stroke_start/stroke_end"),
                         ("theta_start", "stroke_start")):
            if any(_is_old(k, old) for k in pd):
                raise ValueError(
                    f"'{old}' was replaced by '{new}'. Give the two endpoints "
                    f"of the travel instead, e.g. stroke_start -45, stroke_end "
                    f"45 (degrees by default).")

        # move_time and accel_fraction described the move by prescribing its
        # DURATION and shape. The profile is now solved as the minimum-time
        # S-curve under the controller's own kinematic limits, so move_time is
        # an OUTPUT and the shape follows from which limit binds. A file
        # carrying the old keys would otherwise parse cleanly and silently
        # ignore them, reporting a move time the author never asked for.
        for old in ("move_time", "accel_fraction"):
            if any(_is_old(k, old) for k in pd):
                raise ValueError(
                    f"'{old}' is no longer an input. The trajectory is now the "
                    f"fastest move that respects max_velocity, max_accel and "
                    f"max_jerk, so the move time is SOLVED and reported rather "
                    f"than prescribed. State the three limits your motion "
                    f"controller enforces, e.g. max_velocity 260 rpm, "
                    f"max_accel 40 rad/s^2, max_jerk 800 rad/s^3. To reproduce "
                    f"a specific move time, raise or lower the limit that "
                    f"binds (the report names it).")
        kw = {}
        for k, dim in U.PROFILE_DIMENSIONS.items():
            val = U.parse_value(pd, k, dim, audit=audit, strict=strict,
                                default_unit=U.DEFAULT_UNITS.get(k))
            if val is not None:
                kw[k] = val
        if "samples" in pd:
            kw["samples"] = int(pd["samples"])
        j.profile = MotionProfile(**kw)

    if "duty_segments" in d:
        # [[dt, torque, speed], ...] with a units triple declared alongside
        du = d.get("duty_segments_units", ["s", "N.m", "rad/s"])
        segs = []
        for s in d["duty_segments"]:
            segs.append((
                U.convert(float(s[0]), du[0], "time"),
                U.convert(float(s[1]), du[1], "torque"),
                U.convert(float(s[2]), du[2], "angular_velocity"),
            ))
        j.duty_segments = segs
        audit.record("duty_segments", len(segs), "count",
                     f"{len(segs)} segments in ({', '.join(du)})", True)

    # An envelope is a record of what this joint actually did. It outranks a
    # profile (a guess at one simple move), which outranks duty_segments (a
    # hand-written list). Declaring several is fine -- the superseded blocks are
    # kept so that removing the envelope reference falls straight back.
    if "envelope" in d and d["envelope"] is not None:
        spec = d["envelope"]
        ref = spec.get("file") if isinstance(spec, dict) else spec
        if not ref:
            raise ValueError(
                "the 'envelope' block must name a file, either as a bare "
                "string or as {\"file\": \"name\"}.")
        env, _ = load_envelope_with_audit(ref)
        if isinstance(spec, dict) and spec.get("note"):
            env.note = env.note or spec["note"]

        # The envelope is joint-referred, so it is only valid for the drivetrain
        # it was captured on. A ratio mismatch means the capture came off a
        # different machine than this application describes, and every torque in
        # it is wrong by that factor.
        j_ratio = float(j.ratio) if j.ratio is not None else 1.0
        if abs(env.ratio - j_ratio) > 1e-6 * max(1.0, abs(j_ratio)):
            raise ValueError(
                f"envelope '{ref}' was captured at ratio {env.ratio:g} but this "
                f"application declares ratio {j_ratio:g}. Joint-referred torque "
                f"depends on that ratio, so the capture does not describe this "
                f"drivetrain. Recapture, or correct whichever value is wrong.")
        j.envelope = env
        audit.record("envelope", env.total_time, "s",
                     f"{env.name} ({len(env.cells)} cells, "
                     f"{len(env.windows)} windows)", True)

    if "motion_source" in d and d["motion_source"]:
        pin = str(d["motion_source"])
        if pin != "auto" and pin not in Joint.MOTION_SOURCES:
            raise ValueError(
                f"motion_source '{pin}' is not one of "
                f"{', '.join(Joint.MOTION_SOURCES)} or 'auto'.")
        j.motion_source = pin

    return j, audit


# ---------------------------------------------------------------------------
# Envelope
# ---------------------------------------------------------------------------

def _env_scalar(container: dict, key: str, audit, strict, default=None):
    """A dimensioned scalar inside an envelope file."""
    dim = U.ENVELOPE_DIMENSIONS.get(key)
    if dim is None or key not in container:
        return default
    v = U.parse_value(container, key, dim, audit=audit, strict=strict)
    return default if v is None else v


def envelope_from_dict(d: Dict) -> Tuple["Envelope", U.UnitAudit]:
    """
    Parse an envelope file into an Envelope.

    The whole file is version-gated rather than guarded field by field: this is
    a new format with no deployed history, so one check on `schema` is cheaper
    and clearer than a migration guard per key, and it gives a pointed error the
    day the format does change.
    """
    from .envelope import (Cell, Capture, Despike, Envelope, Extremes, Outlier,
                           Window)

    audit = U.UnitAudit()
    strict = bool(d.get("strict_units", False))

    schema = d.get("schema")
    if schema != ENVELOPE_SCHEMA:
        raise ValueError(
            f"envelope schema '{schema}' is not supported; this build reads "
            f"'{ENVELOPE_SCHEMA}'. Regenerate the envelope from its source log "
            f"with envelope_from_log.py rather than hand-editing the version.")

    env = Envelope(name=d.get("name", "envelope"))
    env.note = d.get("note", "")
    env.source = d.get("source")
    env.tol = d.get("tol")

    cap = d.get("capture", {}) or {}
    env.capture = Capture(
        captured_at=cap.get("captured_at", ""),
        robot_id=cap.get("robot_id", ""),
        joint_id=cap.get("joint_id", ""),
        firmware=cap.get("firmware", ""),
        duration=_env_scalar(cap, "duration", audit, strict, 0.0) or 0.0,
        sample_rate=_env_scalar(cap, "sample_rate", audit, strict),
        coverage=float(cap.get("coverage", 1.0)),
        gaps=[(float(g["t"]), float(g["duration"])) for g in cap.get("gaps", [])],
        source_logs=list(cap.get("source_logs", [])),
        tool_version=cap.get("tool_version", ""),
    )

    ref = d.get("referred_to", {}) or {}
    env.ratio = _env_scalar(ref, "ratio", audit, strict, 1.0) or 1.0
    env.n_actuators_at_capture = ref.get("n_actuators_at_capture")
    env.torque_source = ref.get("torque_source", "commanded_current")
    env.sign_convention = ref.get("sign_convention", "")
    ctrl = ref.get("controller", {}) or {}
    env.fixed_rate = bool(ctrl.get("fixed_rate", True))
    env.dt_nominal = _env_scalar(ctrl, "dt", audit, strict)
    dsp = ref.get("despike", {}) or {}
    env.despike = Despike(
        filter=dsp.get("filter", "median"),
        width_samples=int(dsp.get("width_samples", 0)),
        width_s=_env_scalar(dsp, "width", audit, strict),
        samples_changed=int(dsp.get("samples_changed", 0)),
        worst_change=_env_scalar(dsp, "worst_change", audit, strict, 0.0) or 0.0,
    )

    occ = d.get("occupancy", {}) or {}
    env.total_time = _env_scalar(occ, "total_time", audit, strict, 0.0) or 0.0
    env.binned_below = _env_scalar(occ, "binned_below", audit, strict)
    env.torque_edges = [float(x) for x in
                        (occ.get("torque_edges", {}) or {}).get("edges", [])]
    env.speed_edges = [float(x) for x in
                       (occ.get("speed_edges", {}) or {}).get("edges", [])]
    cu = occ.get("cells_units", ["-", "-", "s", "N.m", "rad/s", "N.m"])
    for row in occ.get("cells", []):
        tau_rms = U.convert(float(row[3]), cu[3], "torque")
        env.cells.append(Cell(
            ti=int(row[0]), si=int(row[1]),
            seconds=U.convert(float(row[2]), cu[2], "time"),
            tau=tau_rms,
            omega=U.convert(float(row[4]), cu[4], "angular_velocity"),
            tau_mean=(U.convert(float(row[5]), cu[5] if len(cu) > 5 else cu[3],
                                "torque") if len(row) > 5 else tau_rms),
        ))
    if env.cells and not env.total_time:
        env.total_time = sum(c.seconds for c in env.cells)

    for key, target in (("boundary", "boundary"), ("boundary_raw", "boundary_raw")):
        b = d.get(key, {}) or {}
        bu = b.get("units", ["rad/s", "N.m"])
        setattr(env, target, [
            (U.convert(float(p[0]), bu[0], "angular_velocity"),
             U.convert(float(p[1]), bu[1], "torque")) for p in b.get("points", [])])

    ol = d.get("outliers", {}) or {}
    ou = ol.get("rows_units", ["s", "s", "N.m", "rad/s", "N.m", "N.m"])
    for r in ol.get("rows", []):
        mean = U.convert(float(r[4]), ou[4], "torque")
        env.outliers.append(Outlier(
            t_start=U.convert(float(r[0]), ou[0], "time"),
            duration=U.convert(float(r[1]), ou[1], "time"),
            tau_peak=U.convert(float(r[2]), ou[2], "torque"),
            omega_at_peak=U.convert(float(r[3]), ou[3], "angular_velocity"),
            tau_mean=mean,
            tau_rms=(U.convert(float(r[5]), ou[5] if len(ou) > 5 else ou[4],
                               "torque") if len(r) > 5 else mean),
        ))

    ex = d.get("extremes", {}) or {}
    tq = ex.get("torque_abs", {}) or {}
    sp = ex.get("speed_abs", {}) or {}
    tqu, spu = tq.get("units", "N.m"), sp.get("units", "rad/s")
    env.extremes = Extremes(
        torque={k: U.convert(float(v), tqu, "torque")
                for k, v in tq.items() if k != "units"},
        speed={k: U.convert(float(v), spu, "angular_velocity")
               for k, v in sp.items() if k != "units"},
    )

    dc = d.get("duration_curve", {}) or {}
    dcu = dc.get("units", "N.m")
    env.duration_curve = [(float(p[0]), U.convert(float(p[1]), dcu, "torque"))
                          for p in dc.get("points", [])]

    env.event_counts = dict(d.get("event_counts", {}) or {})

    for w in d.get("windows", []):
        su = w.get("segments_units", ["s", "N.m", "rad/s"])
        sel = (w.get("selected_by_tau_w", {}) or {})
        env.windows.append(Window(
            duration=_env_scalar(w, "duration", audit, strict, 0.0) or 0.0,
            found_at=_env_scalar(w, "found_at", audit, strict, 0.0) or 0.0,
            rms_torque=_env_scalar(w, "rms_torque", audit, strict, 0.0) or 0.0,
            segments=[(U.convert(float(s[0]), su[0], "time"),
                       U.convert(float(s[1]), su[1], "torque"),
                       U.convert(float(s[2]), su[2], "angular_velocity"))
                      for s in w.get("segments", [])],
            selected_by_tau_w=[float(x) for x in (sel.get("value", []) if
                                                  isinstance(sel, dict) else sel)],
            selection_metric=w.get("selection_metric", "tau_squared_single_pole"),
        ))
    env.windows.sort(key=lambda w: w.duration)

    return env, audit


def resolve_envelope_path(name_or_path: str) -> str:
    """Locate an envelope file by name or explicit path; private shadows example."""
    if os.path.exists(name_or_path):
        return os.path.abspath(name_or_path)
    fname = name_or_path if name_or_path.endswith(".json") else name_or_path + ".json"
    for d in envelope_search_path():
        cand = os.path.join(d, fname)
        if os.path.exists(cand):
            return os.path.abspath(cand)
    known = ", ".join(n for n, _ in list_envelopes()) or "none found"
    raise FileNotFoundError(
        f"no envelope '{name_or_path}'. Give a path to a json file, or one "
        f"of: {known}")


def load_envelope_with_audit(name_or_path: str) -> Tuple["Envelope", U.UnitAudit]:
    """Load one captured operating envelope, keeping its unit audit."""
    path = resolve_envelope_path(name_or_path)
    with open(path) as f:
        d = json.load(f)
    try:
        env, audit = envelope_from_dict(d)
    except U.UnitError as e:
        raise U.UnitError(f"in envelope file '{path}': {e}") from None
    env.source_path = path
    return env, audit


def load_envelope(name_or_path: str, with_audit: bool = False):
    """Load one captured operating envelope."""
    env, audit = load_envelope_with_audit(name_or_path)
    return (env, audit) if with_audit else env


def list_envelopes() -> List[Tuple[str, str]]:
    """[(name, kind)] for every envelope found, kind 'private' or 'example'."""
    out: List[Tuple[str, str]] = []
    seen = set()
    for d, kind in ((ENVELOPE_DIR, "private"), (EXAMPLE_ENVELOPE_DIR, "example")):
        if not os.path.isdir(d):
            continue
        for f in sorted(os.listdir(d)):
            if not f.endswith(".json") or f.startswith("_"):
                continue
            name = f[:-5]
            if name in seen:
                continue
            seen.add(name)
            out.append((name, kind))
    return out


def resolve_application_path(name_or_path: str) -> str:
    """
    Locate an application file by name or explicit path.

    A bare name is searched for in applications/ first, then in the bundled
    applications/examples/, so a private file always shadows an example of the
    same name rather than the other way round.
    """
    if os.path.exists(name_or_path):
        return os.path.abspath(name_or_path)
    fname = name_or_path if name_or_path.endswith(".json") else name_or_path + ".json"
    for d in application_search_path():
        cand = os.path.join(d, fname)
        if os.path.exists(cand):
            return os.path.abspath(cand)
    known = ", ".join(n for n, _ in list_applications()) or "none found"
    raise FileNotFoundError(
        f"no application '{name_or_path}'. Give a path to a json file, or one "
        f"of: {known}")


def load_application(name_or_path: str, with_audit: bool = False):
    """
    Load one joint-on-a-robot definition.

    The resolved path is recorded on the Joint as `source_path` so that reports
    and charts can be written into the same directory as the file that produced
    them.
    """
    path = resolve_application_path(name_or_path)
    with open(path) as f:
        d = json.load(f)
    try:
        j, audit = joint_from_dict(d)
    except U.UnitError as e:
        raise U.UnitError(f"in application file '{path}': {e}") from None
    j.source_path = path
    return (j, audit) if with_audit else j


def list_applications() -> List[Tuple[str, str]]:
    """
    [(name, kind)] for every application found, kind being 'private' or
    'example'. Names are de-duplicated with private files winning, matching
    what resolve_application_path would pick.
    """
    out: List[Tuple[str, str]] = []
    seen = set()
    for d, kind in ((APPLICATION_DIR, "private"), (EXAMPLE_DIR, "example")):
        if not os.path.isdir(d):
            continue
        for f in sorted(os.listdir(d)):
            if not f.endswith(".json") or f.startswith("_"):
                continue
            name = f[:-5]
            if name in seen:
                continue
            seen.add(name)
            out.append((name, kind))
    return out


# Applications were called "joints" before it became clear that a joint file and
# an application file were the same thing. Keep the old names working.
load_joint = load_application


def list_joints() -> List[str]:
    return [n for n, _ in list_applications()]
