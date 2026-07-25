"""
Units: declaration, validation and conversion.

Getting a unit wrong is the easiest way to produce a confident, wrong answer,
and it is completely invisible in a bare JSON number. So every physical input
is checked against the DIMENSION its field expects, and converted to a single
canonical SI unit before any physics touches it.

Three ways to give a value, in order of preference:

  1. Explicit object   "com_distance": {"value": 11, "units": "in"}
  2. Suffixed name     "com_distance_in": 11
  3. Bare number       "com_distance": 0.28        -> assumed canonical (m)

Form 3 is accepted so quick edits stay quick, but every bare number is recorded
and listed in the report's UNIT AUDIT so an assumption never passes silently.
Set "strict_units": true in a file to make form 3 a hard error.

A unit that does not belong to the field's dimension is ALWAYS an error, never
a warning: writing "com_distance": {"value": 280, "units": "kg"} is a mistake
no default can rescue.
"""

from __future__ import annotations
from typing import Dict, Optional, Tuple, List

import math


class UnitError(ValueError):
    """Raised when a unit is unknown, or belongs to the wrong dimension."""


# ---------------------------------------------------------------------------
# Registry: dimension -> {unit: (factor, offset)}
# value_canonical = value * factor + offset
# The FIRST entry of each dimension is the canonical unit.
# ---------------------------------------------------------------------------

DEG = math.pi / 180.0
RPM = 2 * math.pi / 60.0
LBF = 4.4482216152605
IN = 0.0254

REGISTRY: Dict[str, Dict[str, Tuple[float, float]]] = {
    "dimensionless": {"-": (1.0, 0.0), "": (1.0, 0.0), "ratio": (1.0, 0.0),
                      "%": (0.01, 0.0)},
    "mass": {"kg": (1.0, 0.0), "g": (1e-3, 0.0), "lb": (0.45359237, 0.0),
             "oz": (0.028349523125, 0.0)},
    "length": {"m": (1.0, 0.0), "mm": (1e-3, 0.0), "cm": (1e-2, 0.0),
               "in": (IN, 0.0), "ft": (0.3048, 0.0)},
    "inertia": {"kg.m^2": (1.0, 0.0), "kg.m2": (1.0, 0.0),
                "kg.cm^2": (1e-4, 0.0), "kg.mm^2": (1e-6, 0.0),
                "g.cm^2": (1e-7, 0.0), "g.mm^2": (1e-9, 0.0),
                "lb.in^2": (0.45359237 * IN * IN, 0.0),
                "lb.ft^2": (0.45359237 * 0.3048 ** 2, 0.0)},
    "angle": {"rad": (1.0, 0.0), "deg": (DEG, 0.0), "degree": (DEG, 0.0),
              "rev": (2 * math.pi, 0.0), "arcmin": (DEG / 60.0, 0.0),
              "arcsec": (DEG / 3600.0, 0.0)},
    "time": {"s": (1.0, 0.0), "ms": (1e-3, 0.0), "min": (60.0, 0.0),
             "hr": (3600.0, 0.0), "h": (3600.0, 0.0)},
    "torque": {"N.m": (1.0, 0.0), "Nm": (1.0, 0.0), "N.cm": (1e-2, 0.0),
               "mN.m": (1e-3, 0.0), "kgf.cm": (9.80665e-2, 0.0),
               "kgf.m": (9.80665, 0.0), "lbf.in": (LBF * IN, 0.0),
               "lbf.ft": (LBF * 0.3048, 0.0), "oz.in": (LBF / 16.0 * IN, 0.0)},
    "angular_velocity": {"rad/s": (1.0, 0.0), "rpm": (RPM, 0.0),
                         "deg/s": (DEG, 0.0), "rev/s": (2 * math.pi, 0.0),
                         "Hz": (2 * math.pi, 0.0)},
    "temperature": {"degC": (1.0, 0.0), "C": (1.0, 0.0), "degF": (5.0 / 9.0, -32 * 5.0 / 9.0),
                    "F": (5.0 / 9.0, -32 * 5.0 / 9.0), "K": (1.0, -273.15)},
    "temperature_delta": {"degC": (1.0, 0.0), "K": (1.0, 0.0),
                          "degF": (5.0 / 9.0, 0.0)},
    "voltage": {"V": (1.0, 0.0), "mV": (1e-3, 0.0)},
    "current": {"A": (1.0, 0.0), "A_rms": (1.0, 0.0), "Arms": (1.0, 0.0),
                "mA": (1e-3, 0.0)},
    "current_peak": {"A_pk": (1.0, 0.0), "Apk": (1.0, 0.0), "A": (1.0, 0.0)},
    "resistance": {"ohm": (1.0, 0.0), "Ohm": (1.0, 0.0), "mohm": (1e-3, 0.0)},
    "inductance": {"H": (1.0, 0.0), "mH": (1e-3, 0.0), "uH": (1e-6, 0.0)},
    "power": {"W": (1.0, 0.0), "mW": (1e-3, 0.0), "kW": (1e3, 0.0)},
    "thermal_resistance": {"degC/W": (1.0, 0.0), "K/W": (1.0, 0.0),
                           "C/W": (1.0, 0.0)},
    "heat_capacity": {"J/K": (1.0, 0.0), "J/degC": (1.0, 0.0)},
    "torque_constant": {"N.m/A_rms": (1.0, 0.0), "Nm/Arms": (1.0, 0.0),
                        "N.m/A": (1.0, 0.0)},
    "back_emf": {"V_rms_LL/(rad/s)": (1.0, 0.0),
                 "Vrms/kRPM": (1.0 / (1000 * RPM), 0.0),
                 "V_rms_LL/kRPM": (1.0 / (1000 * RPM), 0.0)},
    "viscous": {"N.m/(rad/s)": (1.0, 0.0)},
    "count": {"-": (1.0, 0.0), "": (1.0, 0.0)},
    "currency": {"USD": (1.0, 0.0), "$": (1.0, 0.0)},
}


def canonical_unit(dimension: str) -> str:
    try:
        return next(iter(REGISTRY[dimension]))
    except KeyError:
        raise UnitError(f"unknown dimension '{dimension}'")


def valid_units(dimension: str) -> List[str]:
    return list(REGISTRY[dimension].keys())


def convert(value: float, unit: str, dimension: str) -> float:
    """Convert `value` expressed in `unit` into the canonical unit."""
    table = REGISTRY.get(dimension)
    if table is None:
        raise UnitError(f"unknown dimension '{dimension}'")
    if unit not in table:
        # Is it a real unit, just of the wrong dimension? Say so explicitly --
        # that is almost always a copy-paste slip and deserves a pointed error.
        for dim, tbl in REGISTRY.items():
            if unit in tbl and dim != dimension:
                raise UnitError(
                    f"unit '{unit}' is a {dim} unit, but this field expects "
                    f"{dimension}. Valid units here: {', '.join(valid_units(dimension))}")
        raise UnitError(
            f"unrecognised unit '{unit}' for a {dimension} field. "
            f"Valid units: {', '.join(valid_units(dimension))}")
    factor, offset = table[unit]
    return value * factor + offset


# ---------------------------------------------------------------------------
# Field parsing
# ---------------------------------------------------------------------------

class UnitAudit:
    """Records how every value in a file got its units, for the report."""

    def __init__(self):
        self.entries: List[Tuple[str, float, str, str, bool]] = []
        # (field, canonical_value, canonical_unit, as_written, was_explicit)

    def record(self, field, value, unit, as_written, explicit):
        self.entries.append((field, value, unit, as_written, explicit))

    @property
    def assumed(self):
        return [e for e in self.entries if not e[4]]

    def render(self) -> str:
        if not self.entries:
            return ""
        w = max(len(e[0]) for e in self.entries)
        lines = []
        for f, v, u, aw, ex in self.entries:
            mark = "   " if ex else " ! "
            lines.append(f"  {mark}{f:<{w}}  = {v:>12.6g} {u:<14} <- {aw}")
        if self.assumed:
            lines.append("")
            lines.append(f"  ! = units were not stated; canonical SI assumed "
                         f"({len(self.assumed)} field(s))")
        return "\n".join(lines)


def parse_value(container: dict, key: str, dimension: str,
                audit: Optional[UnitAudit] = None,
                strict: bool = False,
                required: bool = False,
                default: Optional[float] = None):
    """
    Pull `key` out of `container`, in whatever form it was written, and return
    the value converted to the canonical unit for `dimension`.

    Returns None if absent and not required.
    """
    canon = canonical_unit(dimension)

    # form 1: explicit object
    if key in container and isinstance(container[key], dict):
        d = container[key]
        if "value" not in d:
            raise UnitError(f"field '{key}' is an object but has no 'value'")
        if "units" not in d:
            if strict:
                raise UnitError(
                    f"field '{key}' has no 'units' and strict_units is on. "
                    f"Valid units: {', '.join(valid_units(dimension))}")
            v = float(d["value"])
            if audit:
                audit.record(key, v, canon, f"{d['value']} (no units given)", False)
            return v
        unit = d["units"]
        v = convert(float(d["value"]), unit, dimension)
        if audit:
            audit.record(key, v, canon, f"{d['value']} {unit}", True)
        return v

    # form 2: suffixed field name, e.g. com_distance_mm or Ke_Vrms_per_kRPM
    for unit in sorted(valid_units(dimension), key=len, reverse=True):
        if not unit or unit == "-":
            continue
        suffixed = f"{key}_{unit.replace('/', '_per_').replace('.', '').replace('^', '')}"
        if suffixed in container:
            entry = container[suffixed]
            # the suffix declares the unit, but the field may still be written
            # in object form to carry source/tol/note (or override the unit)
            if isinstance(entry, dict):
                if "value" not in entry:
                    raise UnitError(
                        f"field '{suffixed}' is an object but has no 'value'")
                raw = float(entry["value"])
                unit_used = entry.get("units", unit)
            else:
                raw = float(entry)
                unit_used = unit
            v = convert(raw, unit_used, dimension)
            if audit:
                audit.record(key, v, canon,
                             f"{raw} {unit_used} (from field name)", True)
            return v

    # form 3: bare number
    if key in container and container[key] is not None:
        if strict:
            raise UnitError(
                f"field '{key}' is a bare number and strict_units is on. "
                f"Write it as {{\"value\": X, \"units\": \"...\"}}. "
                f"Valid units: {', '.join(valid_units(dimension))}")
        v = float(container[key])
        if audit:
            audit.record(key, v, canon, f"{container[key]} (bare number)", False)
        return v

    if required:
        raise UnitError(f"required field '{key}' is missing")
    if default is not None and audit:
        audit.record(key, default, canon, "default", False)
    return default


# ---------------------------------------------------------------------------
# Which dimension does each field belong to?
# ---------------------------------------------------------------------------

JOINT_DIMENSIONS = {
    "ratio": "dimensionless",
    "ratio_eff": "dimensionless",
    "lead": "length",
    "tau_peak_req": "torque",
    "tau_cont_req": "torque",
    "omega_max_req": "angular_velocity",
    "mass_budget": "mass",
    "T_ambient": "temperature",
    "max_surface_temp": "temperature",
    "bus_voltage": "voltage",
    "supply_current_limit": "current",
    "position_tolerance": "angle",
}

LOAD_DIMENSIONS = {
    "payload_mass": "mass",
    "com_distance": "length",
    "link_inertia": "inertia",
    "gravity_factor": "dimensionless",
    "gravity_phase": "angle",
    "external_torque": "torque",
    "joint_friction": "torque",
}

PROFILE_DIMENSIONS = {
    "stroke": "angle",
    "move_time": "time",
    "dwell_time": "time",
    "accel_fraction": "dimensionless",
    "theta_start": "angle",
}

ACTUATOR_DIMENSIONS = {
    "mass": "mass",
    "gear_ratio": "dimensionless",
    "gear_eff": "dimensionless",
    "J_rotor": "inertia",
    "backlash": "angle",
    "Ke": "back_emf",
    "Kt_rotor": "torque_constant",
    "Kt_out": "torque_constant",
    "R_phase": "resistance",
    "L_phase": "inductance",
    "I_cont_rms": "current",
    "I_peak_rms": "current",
    "I_cont_peak_amp": "current_peak",
    "I_peak_peak_amp": "current_peak",
    "kt_sat_derate": "dimensionless",
    "V_bus_nom": "voltage",
    "V_bus_min": "voltage",
    "V_bus_max": "voltage",
    "tau_cont_out_spec": "torque",
    "tau_peak_out_spec": "torque",
    "Rth_wc": "thermal_resistance",
    "Rth_ca": "thermal_resistance",
    "C_w": "heat_capacity",
    "C_c": "heat_capacity",
    "T_winding_max": "temperature",
    "iron_loss_ref_W": "power",
    "friction_torque_rotor": "torque",
    "viscous_rotor": "viscous",
}
