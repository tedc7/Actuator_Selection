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
from .models import Actuator, Joint, Load, MotionProfile
from . import units as U

_HERE = os.path.dirname(os.path.abspath(__file__))
ACTUATOR_DIR = os.path.join(_HERE, "db", "actuators")
JOINT_DIR = os.path.join(_HERE, "db", "joints")

_ACT_PLAIN = ["name", "vendor", "url", "price_usd", "gear_type",
              "pole_pairs", "modulation_k", "notes"]


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

    return a, audit


def load_actuator(name_or_path: str, with_audit: bool = False):
    path = name_or_path
    if not os.path.exists(path):
        path = os.path.join(ACTUATOR_DIR, name_or_path)
        if not path.endswith(".json"):
            path += ".json"
    with open(path) as f:
        d = json.load(f)
    try:
        a, audit = actuator_from_dict(d)
    except U.UnitError as e:
        raise U.UnitError(f"in actuator file '{path}': {e}") from None
    return (a, audit) if with_audit else a


def list_actuators() -> List[str]:
    if not os.path.isdir(ACTUATOR_DIR):
        return []
    return sorted(f[:-5] for f in os.listdir(ACTUATOR_DIR)
                  if f.endswith(".json") and not f.startswith("_"))


# ---------------------------------------------------------------------------
# Joint
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
        kw = {}
        for k, dim in U.LOAD_DIMENSIONS.items():
            val = U.parse_value(ld, k, dim, audit=audit, strict=strict)
            if val is not None:
                kw[k] = val
        j.load = Load(**kw)

    if "profile" in d:
        pd = d["profile"]
        kw = {}
        for k, dim in U.PROFILE_DIMENSIONS.items():
            val = U.parse_value(pd, k, dim, audit=audit, strict=strict)
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

    return j, audit


def load_joint(name_or_path: str, with_audit: bool = False):
    path = name_or_path
    if not os.path.exists(path):
        path = os.path.join(JOINT_DIR, name_or_path)
        if not path.endswith(".json"):
            path += ".json"
    with open(path) as f:
        d = json.load(f)
    try:
        j, audit = joint_from_dict(d)
    except U.UnitError as e:
        raise U.UnitError(f"in joint file '{path}': {e}") from None
    return (j, audit) if with_audit else j


def list_joints() -> List[str]:
    if not os.path.isdir(JOINT_DIR):
        return []
    return sorted(f[:-5] for f in os.listdir(JOINT_DIR)
                  if f.endswith(".json") and not f.startswith("_"))
