"""
Domain models: actuator, joint, load, duty cycle.

Sign / unit conventions (stated once, used everywhere):
  * Motor electrical quantities are referenced to the ROTOR unless the name
    ends in `_out` (after the actuator's own gearbox).
  * Currents are RMS phase current (q-axis). Datasheets often quote peak
    amplitude; divide by sqrt(2). Helpers below do this for you.
  * Kt_rotor [N.m per A_rms] relates to Ke [V_rms line-line per rad/s] by
        Kt_rotor = sqrt(3) * Ke
    which is exact for a sinusoidal PMSM under id=0 field-oriented control.
  * Copper loss uses per-phase (line-to-neutral) resistance:
        P_cu = 3 * I_rms^2 * R_phase
    A datasheet "line-to-line resistance" is 2 * R_phase.
  * Angles in rad, speeds in rad/s, torque in N.m, temperature in degC.
"""

from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from .params import P, as_P, VENDOR_SPEC, VENDOR_DERIVED, ESTIMATED, GUESS

G = 9.80665
SQRT2 = math.sqrt(2.0)
SQRT3 = math.sqrt(3.0)
CU_ALPHA = 0.00393        # copper resistivity tempco, 1/degC
MAGNET_TEMPCO = -0.0011   # NdFeB remanence drift, 1/degC (Kt fades when hot)


# ----------------------------------------------------------------------------
# Default estimators. Each returns a P tagged ESTIMATED with a note explaining
# the reasoning, so the report can tell the user exactly what was assumed.
# ----------------------------------------------------------------------------

def est_phase_resistance(mass_kg: float, kt_rotor: float,
                         i_rated_rms: float, p_mech_rated: float) -> P:
    """
    Estimate per-phase resistance from a rated-point loss budget.

    Small QDD actuators run roughly 80-88% efficient at their rated point, and
    copper loss is typically ~60% of the total loss there (the rest is iron,
    gearbox and bearing drag). Invert that to get R.
    """
    eff = 0.84
    p_loss_total = p_mech_rated * (1.0 - eff) / eff
    p_cu = 0.60 * p_loss_total
    r = p_cu / (3.0 * max(i_rated_rms, 1e-6) ** 2)
    return P(r, "ohm", ESTIMATED, tol=0.45,
             note="back-solved from an assumed 84% rated-point efficiency with "
                  "60% of loss in copper; MEASURE THIS (4-wire, line-to-line/2)")


def est_thermal_resistances(mass_kg: float, dims_m: Tuple[float, float, float],
                            mounting: str = "bolted_metal") -> Tuple[P, P]:
    """
    Estimate winding->case and case->ambient thermal resistance.

    case->ambient is natural convection over the housing area, h ~ 10 W/m^2K,
    improved by conduction into the mounting structure. winding->case is scaled
    from small-frame BLDC practice (~1.5 degC/W at this size).
    """
    a, b, c = dims_m
    area = 2 * (a * b) + 2 * (a * c) + 2 * (b * c)
    h = 10.0
    rth_ca_freeair = 1.0 / (h * max(area, 1e-6))

    mount_factor = {
        "free_air": 1.00,       # hanging in still air, no heat path out
        "bolted_plastic": 0.85,
        "bolted_metal": 0.55,   # typical: bolted to an aluminium bracket
        "heatsunk": 0.35,       # deliberate heatsink or forced air
    }.get(mounting, 0.55)

    rth_ca = P(rth_ca_freeair * mount_factor, "degC/W", ESTIMATED, tol=0.50,
               note=f"natural convection over {area*1e4:.0f} cm^2 housing, "
                    f"h=10 W/m^2K, mounting='{mounting}' factor {mount_factor}")

    # winding->case scales roughly with 1/linear-dimension for similar builds
    ref_mass = 0.31
    rth_wc = P(1.5 * (ref_mass / max(mass_kg, 1e-3)) ** (1 / 3), "degC/W",
               ESTIMATED, tol=0.50,
               note="small-frame BLDC slot-to-housing typical, mass-scaled")
    return rth_wc, rth_ca


def est_thermal_capacitances(mass_kg: float) -> Tuple[P, P]:
    """Winding and case heat capacity. Copper ~15% of module mass, rest ~Al."""
    m_cu = 0.15 * mass_kg
    m_case = mass_kg - m_cu
    cw = P(m_cu * 385.0, "J/K", ESTIMATED, tol=0.40,
           note="copper taken as 15% of module mass, c=385 J/kgK")
    cc = P(m_case * 900.0, "J/K", ESTIMATED, tol=0.40,
           note="remaining mass as aluminium, c=900 J/kgK")
    return cw, cc


def est_rotor_inertia(mass_kg: float, outer_dia_m: float) -> P:
    """
    Rotor inertia for an inner-rotor BLDC. Rotor is ~30% of module mass at a
    radius of gyration ~35% of the housing width.
    """
    m_rot = 0.30 * mass_kg
    k = 0.35 * outer_dia_m / 2.0
    return P(m_rot * k * k, "kg.m^2", ESTIMATED, tol=0.60,
             note="rotor ~30% of module mass, radius of gyration ~0.35*R_housing")


def est_inductance(kt_rotor: float, pole_pairs: int) -> P:
    """Phase inductance from a typical 1 ms electrical time constant."""
    return P(200e-6, "H", GUESS, tol=1.0,
             note="placeholder; only matters near the high-speed voltage limit")


GEAR_EFFICIENCY = {          # per stage, typical, warm
    "direct": (1.00, VENDOR_DERIVED),
    "planetary_1stage": (0.94, ESTIMATED),
    "planetary_2stage": (0.88, ESTIMATED),
    "cycloidal": (0.82, ESTIMATED),
    "harmonic": (0.75, ESTIMATED),
    "belt": (0.96, ESTIMATED),
    "worm": (0.55, ESTIMATED),
}


# ----------------------------------------------------------------------------
# Actuator
# ----------------------------------------------------------------------------

@dataclass
class Actuator:
    name: str = "unnamed"
    vendor: str = ""
    url: str = ""
    price_usd: Optional[float] = None

    # mechanical
    mass: P = None                     # kg
    dims_m: Tuple[float, float, float] = (0.06, 0.06, 0.06)
    gear_ratio: P = None               # rotor:output
    gear_type: str = "planetary_1stage"
    gear_eff: P = None
    pole_pairs: int = 14
    J_rotor: P = None                  # kg.m^2, rotor side
    backlash_arcmin: Optional[P] = None

    # electrical
    Ke: P = None                       # V_rms line-line per rad/s (rotor)
    Kt_rotor: P = None                 # N.m per A_rms (rotor)
    R_phase: P = None                  # ohm, line-to-neutral, at 25 degC
    L_phase: P = None                  # H
    I_cont_rms: P = None               # rotor phase current, continuous rating
    I_peak_rms: P = None               # rotor phase current, peak rating
    kt_sat_derate: P = None            # Kt at I_peak / Kt at low current
    V_bus_nom: P = None
    V_bus_min: P = None
    V_bus_max: P = None
    modulation_k: float = 0.95         # SVM utilisation after deadtime/margin

    # vendor headline numbers (used for cross-checks, not as primary inputs)
    tau_cont_out_spec: Optional[P] = None
    tau_peak_out_spec: Optional[P] = None

    # thermal
    Rth_wc: P = None                   # winding -> case, degC/W
    Rth_ca: P = None                   # case -> ambient, degC/W
    C_w: P = None                      # J/K
    C_c: P = None                      # J/K
    T_winding_max: P = None            # degC
    iron_loss_ref_W: P = None          # iron loss at rated speed
    friction_torque_rotor: P = None    # N.m, rotor-side Coulomb drag
    viscous_rotor: P = None            # N.m/(rad/s)

    notes: List[str] = field(default_factory=list)
    source_path: Optional[str] = None   # file this was loaded from, if any

    # ------------------------------------------------------------------
    def fill_defaults(self, mounting: str = "bolted_metal"):
        """Populate anything left as None with a documented estimate."""
        m = float(self.mass)

        if self.gear_eff is None:
            v, s = GEAR_EFFICIENCY.get(self.gear_type, (0.90, ESTIMATED))
            self.gear_eff = P(v, "-", s, note=f"class default for '{self.gear_type}'")

        # Kt <-> Ke: whichever we have, derive the other
        if self.Kt_rotor is None and self.Ke is not None:
            self.Kt_rotor = P(SQRT3 * float(self.Ke), "N.m/A_rms", VENDOR_DERIVED,
                              tol=self.Ke.tol, note="Kt_rotor = sqrt(3) * Ke")
        if self.Ke is None and self.Kt_rotor is not None:
            self.Ke = P(float(self.Kt_rotor) / SQRT3, "V_rms_LL/(rad/s)",
                        VENDOR_DERIVED, tol=self.Kt_rotor.tol,
                        note="Ke = Kt_rotor / sqrt(3)")

        if self.J_rotor is None:
            self.J_rotor = est_rotor_inertia(m, max(self.dims_m[0], self.dims_m[1]))

        if self.kt_sat_derate is None:
            self.kt_sat_derate = P(0.88, "-", ESTIMATED, tol=0.10,
                                   note="typical Kt fade at peak current")

        if self.L_phase is None:
            self.L_phase = est_inductance(float(self.Kt_rotor), self.pole_pairs)

        if self.R_phase is None:
            w_rated = self._rated_rotor_speed()
            p_mech = float(self.tau_cont_out_spec or 1.0) * (w_rated / float(self.gear_ratio))
            self.R_phase = est_phase_resistance(m, float(self.Kt_rotor),
                                                float(self.I_cont_rms), max(p_mech, 1.0))

        if self.Rth_wc is None or self.Rth_ca is None:
            wc, ca = est_thermal_resistances(m, self.dims_m, mounting)
            self.Rth_wc = self.Rth_wc or wc
            self.Rth_ca = self.Rth_ca or ca

        if self.C_w is None or self.C_c is None:
            cw, cc = est_thermal_capacitances(m)
            self.C_w = self.C_w or cw
            self.C_c = self.C_c or cc

        if self.T_winding_max is None:
            self.T_winding_max = P(120.0, "degC", ESTIMATED, tol=0.15,
                                   note="Class B insulation (130 degC) less 10 degC margin")

        if self.iron_loss_ref_W is None:
            self.iron_loss_ref_W = P(0.25 * 3 * float(self.I_cont_rms) ** 2
                                     * float(self.R_phase), "W", ESTIMATED, tol=0.7,
                                     note="iron loss at rated speed taken as 25% of rated copper loss")

        if self.friction_torque_rotor is None:
            self.friction_torque_rotor = P(0.02 * float(self.Kt_rotor)
                                           * float(self.I_cont_rms), "N.m", ESTIMATED,
                                           tol=0.7, note="2% of rated rotor torque")
        if self.viscous_rotor is None:
            self.viscous_rotor = P(1e-5, "N.m/(rad/s)", GUESS, tol=1.0)

        for k, v in vars(self).items():
            if isinstance(v, P) and not v.name:
                v.name = k
        return self

    def _rated_rotor_speed(self) -> float:
        """Rotor speed at the rated point, from the voltage envelope."""
        e_max = float(self.V_bus_nom) * self.modulation_k / SQRT2   # V_rms LL
        return 0.8 * e_max / max(float(self.Ke), 1e-9)

    # ------------------------------------------------------------------
    @property
    def Kt_out(self) -> float:
        """Output-referred torque constant, N.m per A_rms."""
        return float(self.Kt_rotor) * float(self.gear_ratio) * float(self.gear_eff)

    def torque_out(self, i_rms: float, t_magnet: float = 25.0) -> float:
        """Output torque for a given rotor phase current, with saturation + heat fade."""
        frac = min(abs(i_rms) / max(float(self.I_peak_rms), 1e-9), 1.0)
        # Saturation onset is progressive, not linear: Kt is essentially flat at
        # low current and falls away near the peak. A quadratic in the current
        # fraction reproduces the rated and peak points of real datasheets well.
        sat = 1.0 - (1.0 - float(self.kt_sat_derate)) * frac ** 2
        thermal = 1.0 + MAGNET_TEMPCO * (t_magnet - 25.0)
        return math.copysign(self.Kt_out * abs(i_rms) * sat * thermal, i_rms)

    def current_for_torque(self, tau_out: float, t_magnet: float = 25.0) -> float:
        """
        Invert torque_out(). Returns RMS rotor phase current.

        Below the peak-current rating the model is the cubic
            tau = k*i - k*s*i^3,   k = Kt_out*thermal,  s = (1-derate)/Ipk^2
        which Newton solves in a handful of steps. Above the rating the Kt fade
        is clamped, so the relationship is linear again.
        """
        target = abs(tau_out)
        if target < 1e-12:
            return 0.0
        thermal = 1.0 + MAGNET_TEMPCO * (t_magnet - 25.0)
        k = self.Kt_out * thermal
        ipk = max(float(self.I_peak_rms), 1e-9)
        s = (1.0 - float(self.kt_sat_derate)) / (ipk * ipk)

        tau_at_rating = k * ipk * float(self.kt_sat_derate)
        if target >= tau_at_rating:                     # clamped region
            i = target / max(k * float(self.kt_sat_derate), 1e-12)
            return math.copysign(i, tau_out)

        i = target / max(k, 1e-12)                      # ignore saturation to start
        for _ in range(12):
            f = k * i - k * s * i ** 3 - target
            fp = k - 3.0 * k * s * i * i
            if abs(fp) < 1e-12:
                break
            step = f / fp
            i -= step
            if i < 0.0:
                i = 0.0
            if abs(step) < 1e-12:
                break
        return math.copysign(i, tau_out)

    def consistency_report(self) -> List[str]:
        """Cross-check vendor headline numbers against the electrical model."""
        out = []
        if self.tau_cont_out_spec is not None:
            pred = abs(self.torque_out(float(self.I_cont_rms)))
            spec = float(self.tau_cont_out_spec)
            out.append(f"rated torque: model {pred:.2f} N.m vs spec {spec:.2f} N.m "
                       f"({100*(pred-spec)/spec:+.0f}%)")
        if self.tau_peak_out_spec is not None:
            pred = abs(self.torque_out(float(self.I_peak_rms)))
            spec = float(self.tau_peak_out_spec)
            out.append(f"peak torque:  model {pred:.2f} N.m vs spec {spec:.2f} N.m "
                       f"({100*(pred-spec)/spec:+.0f}%)")
        return out


# ----------------------------------------------------------------------------
# Joint: load, transmission, requirements
# ----------------------------------------------------------------------------

@dataclass
class Load:
    """What the joint has to move. All referred to the JOINT output."""
    payload_mass: float = 0.0        # kg at the COM distance
    com_distance: float = 0.0        # m from joint axis to payload COM
    link_inertia: float = 0.0        # kg.m^2 about the joint axis (link + payload)
    gravity_factor: float = 1.0      # 1 = axis horizontal (full gravity), 0 = vertical axis
    gravity_phase: float = 0.0       # rad; joint angle at which gravity torque peaks
    external_torque: float = 0.0     # N.m constant disturbance
    joint_friction: float = 0.0      # N.m Coulomb at the joint

    def inertia_total(self) -> float:
        return self.link_inertia + self.payload_mass * self.com_distance ** 2

    def gravity_torque(self, theta: float) -> float:
        return (self.payload_mass * G * self.com_distance
                * self.gravity_factor * math.cos(theta - self.gravity_phase))


@dataclass
class MotionProfile:
    """
    Trapezoidal point-to-point move, repeated. This is the "few inputs, lots of
    output" path: give a stroke, a move time and a duty, and the tool builds the
    whole torque/speed/time history for you.
    """
    stroke: float = math.radians(90)   # rad, peak-to-peak
    move_time: float = 0.5             # s for one traverse
    dwell_time: float = 0.5            # s at rest between traverses
    accel_fraction: float = 0.25       # of move_time spent accelerating
    theta_start: float = 0.0           # rad
    samples: int = 200

    def trajectory(self) -> List[Tuple[float, float, float, float]]:
        """Return [(t, theta, omega, alpha)] over one out-and-back cycle."""
        a = min(max(self.accel_fraction, 0.01), 0.49)
        T = self.move_time
        ta = a * T
        v = self.stroke / (T * (1.0 - a))
        acc = v / ta
        pts = []

        def leg(theta0, sign, t0):
            n = max(self.samples // 2, 20)
            for i in range(n + 1):
                t = T * i / n
                if t < ta:
                    om, al = acc * t, acc
                    th = 0.5 * acc * t * t
                elif t < T - ta:
                    om, al = v, 0.0
                    th = 0.5 * acc * ta * ta + v * (t - ta)
                else:
                    td = T - t
                    om, al = acc * td, -acc
                    th = self.stroke - 0.5 * acc * td * td
                pts.append((t0 + t, theta0 + sign * th, sign * om, sign * al))

        leg(self.theta_start, +1, 0.0)
        if self.dwell_time > 0:
            pts.append((T, self.theta_start + self.stroke, 0.0, 0.0))
            pts.append((T + self.dwell_time, self.theta_start + self.stroke, 0.0, 0.0))
        t0 = T + self.dwell_time
        leg(self.theta_start + self.stroke, -1, t0)
        if self.dwell_time > 0:
            pts.append((t0 + T, self.theta_start, 0.0, 0.0))
            pts.append((t0 + T + self.dwell_time, self.theta_start, 0.0, 0.0))
        return pts

    @property
    def period(self) -> float:
        return 2 * (self.move_time + self.dwell_time)


@dataclass
class Joint:
    name: str = "joint"
    kind: str = "revolute"             # revolute | prismatic | continuous
    n_actuators: int = 1               # actuators sharing this joint in parallel
    ratio: P = None                    # extra reduction between actuator output and joint
    ratio_type: str = "direct"
    ratio_eff: P = None
    lead_m: Optional[float] = None     # for prismatic: metres per output revolution

    load: Load = field(default_factory=Load)
    profile: Optional[MotionProfile] = None
    duty_segments: Optional[List[Tuple[float, float, float]]] = None  # (dt, tau_joint, omega_joint)

    # --- power system: a property of THIS application, not of the actuator ---
    # The vendor's "rated voltage" is what they characterised the unit at. What
    # your robot actually supplies is a separate fact, and is often different.
    # Leave as None to fall back to the actuator's nominal.
    bus_voltage: Optional[float] = None       # V, actually available in this robot
    supply_current_limit: Optional[float] = None   # A_rms per actuator; None = not a constraint
    accept_overvoltage: bool = False          # acknowledge running above the vendor's max

    # requirements
    tau_peak_req: Optional[float] = None      # N.m at the joint; None = derive from motion
    tau_cont_req: Optional[float] = None
    omega_max_req: Optional[float] = None     # rad/s at the joint
    mass_budget: Optional[float] = None       # kg for the whole joint's actuation
    T_ambient: float = 40.0                   # degC inside the robot, not room temp
    max_surface_temp: Optional[float] = None  # degC on the housing; user-safety limit
    duty_repeats_forever: bool = True
    position_tolerance_rad: Optional[float] = None
    require_backdrivable: bool = False
    mounting: str = "bolted_metal"

    source_path: Optional[str] = None   # file this was loaded from, if any

    def output_dir(self) -> str:
        """
        Where results for this application belong: beside the file that defined
        it, so a report never gets separated from its inputs. Falls back to the
        working directory for a Joint built in code rather than loaded.
        """
        import os
        if self.source_path:
            return os.path.dirname(os.path.abspath(self.source_path))
        return os.getcwd()

    def slug(self) -> str:
        """Filename-safe stem for this application's output files."""
        import os
        import re
        if self.source_path:
            return os.path.splitext(os.path.basename(self.source_path))[0]
        return re.sub(r"[^A-Za-z0-9]+", "_", self.name).strip("_").lower() or "joint"

    def bus_v(self, act: "Actuator") -> float:
        """
        Bus voltage to evaluate against: what this application supplies, or the
        actuator's nominal if the application did not say.
        """
        if self.bus_voltage is not None:
            return float(self.bus_voltage)
        return float(act.V_bus_nom)

    def bus_v_is_from_app(self) -> bool:
        return self.bus_voltage is not None

    def fill_defaults(self):
        if self.ratio is None:
            self.ratio = P(1.0, "-", VENDOR_SPEC, tol=0.0, note="direct drive from actuator output")
        if self.ratio_eff is None:
            v, s = GEAR_EFFICIENCY.get(self.ratio_type, (1.0, ESTIMATED))
            self.ratio_eff = P(v, "-", s, note=f"class default for '{self.ratio_type}'")
        return self

    # ------------------------------------------------------------------
    def required_torque_profile(self, act: "Actuator") -> List[Tuple[float, float, float]]:
        """
        Build [(dt, tau_joint, omega_joint)] including inertia, gravity and the
        reflected rotor inertia of the actuators actually fitted.
        """
        if self.duty_segments is not None:
            return list(self.duty_segments)
        if self.profile is None:
            raise ValueError(f"joint '{self.name}' has neither a profile nor duty_segments")

        # rotor inertia reflected to the joint, for all n actuators
        total_ratio = float(act.gear_ratio) * float(self.ratio)
        J_reflected = self.n_actuators * float(act.J_rotor) * total_ratio ** 2
        J_total = self.load.inertia_total() + J_reflected

        traj = self.profile.trajectory()
        segs = []
        for i in range(1, len(traj)):
            t0, th0, om0, al0 = traj[i - 1]
            t1, th1, om1, al1 = traj[i]
            dt = t1 - t0
            if dt <= 0:
                continue
            th, om, al = 0.5 * (th0 + th1), 0.5 * (om0 + om1), 0.5 * (al0 + al1)
            tau = (J_total * al
                   + self.load.gravity_torque(th)
                   + self.load.external_torque)
            if abs(om) > 1e-9:
                tau += math.copysign(self.load.joint_friction, om)
            segs.append((dt, tau, om))
        return segs

    def actuator_output_demand(self, tau_joint: float, omega_joint: float
                               ) -> Tuple[float, float]:
        """
        Convert a joint-level (torque, speed) into what ONE actuator must produce
        at its own output shaft. Efficiency helps rather than hurts when the load
        is driving the motor.
        """
        r = float(self.ratio)
        eff = float(self.ratio_eff)
        power = tau_joint * omega_joint
        tau_per = tau_joint / (self.n_actuators * r)
        if power >= 0:
            tau_per /= eff            # motor pushes, losses add to the demand
        else:
            tau_per *= eff            # load pushes back, losses absorb some
        return tau_per, omega_joint * r
