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
    """
    What the joint has to move. All referred to the JOINT output.

    The three mass properties are exactly what a CAD system reports for the
    moving assembly, so they can be transcribed rather than pre-processed:
    total mass, distance from the joint axis to the CG, and moment of inertia
    about the CG. This module does the parallel-axis transfer to the joint axis
    itself, which is what makes double-counting impossible -- nothing the user
    states is referred to the joint axis, so nothing can be added to it twice.
    """
    total_mass_at_CG: float = 0.0        # kg, whole moving assembly
    distance_joint_axis_to_CG: float = 0.0   # m, joint axis to the CG
    moment_of_inertia_around_CG: float = 0.0  # kg.m^2 about the CG, NOT the axis
    joint_plane_tilt: float = 0.0    # rad; tilt of the joint's plane of rotation
                                     # out of vertical. 0 = plane contains gravity
                                     # (full effect), 90 deg = plane horizontal
                                     # (axis vertical, gravity does nothing)
    gravity_angle: float = -math.pi / 2   # rad; joint angle at which the CG
                                     # points ALONG the gravity vector (straight
                                     # down), in the same frame as
                                     # profile.stroke_start. Torque is zero there
                                     # and peaks 90 deg away.
    external_torque: float = 0.0     # N.m constant disturbance
    joint_friction: float = 0.0      # N.m Coulomb at the joint

    def inertia_total(self) -> float:
        """
        Inertia about the JOINT AXIS, by the parallel axis theorem.

        The transfer term m*d^2 is added here and nowhere else, so a CAD figure
        can be copied in as-is: state the inertia about the CG, never about the
        axis. Quoting an axis-referred inertia would double-count the transfer.
        """
        return (self.moment_of_inertia_around_CG
                + self.total_mass_at_CG * self.distance_joint_axis_to_CG ** 2)

    def gravity_torque(self, theta: float) -> float:
        # Angles are referenced to the GRAVITY VECTOR, as in standard mechanics:
        # gravity_angle points the way gravity does. An assembly whose CG hangs
        # straight below the axis exerts no torque, and the moment arm grows as
        # it swings away -- hence sin() of the offset, not cos().
        # joint_plane_tilt then scales by how much of gravity this plane sees.
        # Only the CG offset produces gravity torque; inertia about the CG does
        # not, which is why the same three inputs serve both terms without any
        # of them being a "payload" as distinct from the structure.
        return (self.total_mass_at_CG * G * self.distance_joint_axis_to_CG
                * math.cos(self.joint_plane_tilt)
                * math.sin(theta - self.gravity_angle))


@dataclass
class MotionProfile:
    """
    Trapezoidal point-to-point move, repeated. This is the "few inputs, lots of
    output" path: give the two endpoints, a move time and a dwell, and the tool
    builds the whole torque/speed/time history for you.

    The endpoints are joint angles in the same frame as Load.gravity_angle, so
    where the move sits relative to gravity is explicit rather than implied by
    a start plus a signed sweep.
    """
    stroke_start: float = math.radians(-45)   # rad, one end of the travel
    stroke_end: float = math.radians(45)      # rad, the other end
    max_velocity: float = 27.2         # rad/s  (260 rpm)
    max_accel: float = 40.0            # rad/s^2
    max_jerk: float = 800.0            # rad/s^3
    dwell_time: float = 0.5            # s at rest between traverses
    samples: int = 200

    @property
    def stroke(self) -> float:
        """Signed travel from stroke_start to stroke_end (rad)."""
        return self.stroke_end - self.stroke_start

    # ------------------------------------------------------------------
    def _solve(self) -> Tuple[float, float, float, float, float, str]:
        """
        Minimum-time S-curve for one traverse, honouring all three limits.

        Returns (j, a, v, t_j, t_a, t_v, regime) where j/a/v are the jerk,
        acceleration and velocity actually USED -- each is its limit or less --
        and t_j/t_a/t_v are the durations of the jerk, constant-accel and
        cruise phases. The move is always limited by at least one of the three,
        so this is the fastest traverse the controller would command.

        Three regimes, tested in order of how much of the profile survives:
          velocity-limited  full 7 phases, a cruise exists
          accel-limited     6 phases, reaches max_accel but never max_velocity
          jerk-limited      4 phases, never even reaches max_accel
        """
        d = abs(self.stroke)
        j = max(self.max_jerk, 1e-9)
        a = max(self.max_accel, 1e-9)
        v = max(self.max_velocity, 1e-9)

        if d <= 1e-12:
            return j, a, v, 0.0, 0.0, 0.0, "degenerate"

        # Can acceleration reach `a` before the jerk phase alone would carry
        # velocity past the point of no return? t_j = a/j is the ramp time.
        t_j = a / j
        # Velocity gained by one jerk-up plus one jerk-down pair at full `a`
        # is a*t_j; reaching cruise speed `v` needs the constant-accel phase
        # to supply the rest.
        if v < a * t_j:
            # Velocity limit bites before acceleration saturates: the accel
            # phase is itself jerk-limited (triangular in alpha).
            t_j = math.sqrt(v / j)
            a_used = j * t_j
            t_a = 0.0
        else:
            a_used = a
            t_a = (v - a * t_j) / a

        # Distance consumed getting up to speed and back down again.
        d_ramp = 2.0 * self._ramp_distance(j, a_used, t_j, t_a)
        if d_ramp <= d:
            # Velocity-limited: there is distance left over for a cruise.
            t_v = (d - d_ramp) / v
            return j, a_used, v, t_j, t_a, t_v, "velocity"

        # No cruise. Solve for the peak velocity that exactly fits `d`,
        # first assuming acceleration still saturates at `a`.
        # With t_j = a/j fixed, distance over accel+decel as a function of
        # t_a is quadratic: d = a*(t_a + t_j)*(t_a + 2*t_j)
        t_j = a / j
        # a*t_a^2 + 3*a*t_j*t_a + (2*a*t_j^2 - d) = 0
        disc = (3 * a * t_j) ** 2 - 4 * a * (2 * a * t_j * t_j - d)
        t_a = (-3 * a * t_j + math.sqrt(max(disc, 0.0))) / (2 * a)
        if t_a >= 0.0:
            v_pk = a * (t_a + t_j)
            return j, a, v_pk, t_j, t_a, 0.0, "accel"

        # Jerk-limited: acceleration never reaches `a`. Four phases, and
        # distance over a symmetric jerk-only accel/decel pair is 2*j*t_j^3.
        t_j = (d / (2.0 * j)) ** (1.0 / 3.0)
        return j, j * t_j, j * t_j * t_j, t_j, 0.0, 0.0, "jerk"

    @staticmethod
    def _ramp_distance(j: float, a: float, t_j: float, t_a: float) -> float:
        """Distance covered accelerating from rest to cruise speed."""
        # jerk up: theta = j*t_j^3/6, ends at omega = j*t_j^2/2
        om1 = 0.5 * j * t_j * t_j
        d1 = j * t_j ** 3 / 6.0
        # constant accel
        d2 = om1 * t_a + 0.5 * a * t_a * t_a
        om2 = om1 + a * t_a
        # jerk down to zero accel
        d3 = om2 * t_j + 0.5 * a * t_j * t_j - j * t_j ** 3 / 6.0
        return d1 + d2 + d3

    @property
    def move_time(self) -> float:
        """Solved minimum time for ONE traverse, s. An OUTPUT, not an input."""
        _, _, _, t_j, t_a, t_v, _ = self._solve()
        return 4 * t_j + 2 * t_a + t_v

    @property
    def regime(self) -> str:
        """Which limit binds: 'velocity', 'accel' or 'jerk'."""
        return self._solve()[6]

    @property
    def peak_velocity(self) -> float:
        return self._solve()[2]

    @property
    def peak_accel(self) -> float:
        return self._solve()[1]

    def trajectory(self) -> List[Tuple[float, float, float, float]]:
        """Return [(t, theta, omega, alpha)] over one out-and-back cycle."""
        j, a, v, t_j, t_a, t_v, regime = self._solve()
        pts: List[Tuple[float, float, float, float]] = []
        if regime == "degenerate":
            return [(0.0, self.stroke_start, 0.0, 0.0),
                    (max(self.dwell_time, 1e-3), self.stroke_start, 0.0, 0.0)]

        # The seven phases, as (duration, jerk). Zero-length ones are skipped
        # when sampling, which is what collapses this to 6 or 4 phases.
        phases = ((t_j, +j), (t_a, 0.0), (t_j, -j),
                  (t_v, 0.0),
                  (t_j, -j), (t_a, 0.0), (t_j, +j))
        T = 4 * t_j + 2 * t_a + t_v
        dist = abs(self.stroke)

        def leg(theta0, sign, t0):
            # Sample each phase on its OWN grid so every breakpoint in jerk
            # lands exactly on a sample. Alpha is piecewise LINEAR here rather
            # than piecewise constant, so a segment that straddled a breakpoint
            # would misstate the torque over its whole span.
            n = max(self.samples // 2, 28)
            span = max(T, 1e-12)
            th = om = al = 0.0
            t_at = 0.0
            pts.append((t0, theta0, 0.0, 0.0))
            for dur, jk in phases:
                if dur <= 1e-12:
                    continue
                th0, om0, al0 = th, om, al
                k = max(int(round(n * dur / span)), 1)
                for i in range(1, k + 1):
                    dt = dur * i / k
                    al = al0 + jk * dt
                    om = om0 + al0 * dt + 0.5 * jk * dt * dt
                    th = th0 + om0 * dt + 0.5 * al0 * dt * dt + jk * dt ** 3 / 6.0
                    pts.append((t0 + t_at + dt, theta0 + sign * th,
                                sign * om, sign * al))
                t_at += dur
                th, om, al = th, om, al
            # Close out any accumulated rounding so the leg lands exactly on
            # its endpoint: the consumer integrates torque, not position, but a
            # drifting endpoint would make the return leg start in the wrong
            # place and shift the gravity term.
            t_end, _, _, _ = pts[-1]
            pts[-1] = (t_end, theta0 + sign * dist, 0.0, 0.0)

        # `stroke` is signed, so a move defined end-to-start simply runs the
        # other way: sign carries through position, velocity and acceleration.
        s = 1.0 if self.stroke >= 0 else -1.0
        leg(self.stroke_start, s, 0.0)
        if self.dwell_time > 0:
            pts.append((T + self.dwell_time, self.stroke_end, 0.0, 0.0))
        t0 = T + self.dwell_time
        leg(self.stroke_end, -s, t0)
        if self.dwell_time > 0:
            pts.append((t0 + T + self.dwell_time, self.stroke_start, 0.0, 0.0))
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
            th, om = 0.5 * (th0 + th1), 0.5 * (om0 + om1)
            # With a jerk limit alpha is piecewise LINEAR and continuous, so the
            # midpoint is the correct representative value over the interval --
            # unlike the old jerk-free trapezoid, where alpha stepped and the
            # start-of-interval value was the only honest choice. The sampler
            # puts every jerk breakpoint on a sample, so no interval straddles
            # a slope change and this average is exact.
            al = 0.5 * (al0 + al1)
            tau = (J_total * al
                   + self.load.gravity_torque(th)
                   + self.load.external_torque)
            if abs(om) > 1e-9:
                tau += math.copysign(self.load.joint_friction, om)
            segs.append((dt, tau, om))
        return segs

    def motion_phases(self) -> List[str]:
        """
        Phase tag per segment of required_torque_profile(), aligned 1:1 with it.

        Trapezoidal profiles only: an explicit duty_segments list carries no
        trajectory to classify, so the caller gets an empty list and should fall
        back to a single unlabelled series.
        """
        if self.duty_segments is not None or self.profile is None:
            return []
        traj = self.profile.trajectory()
        al_pk = max((abs(a) for _, _, _, a in traj), default=0.0)
        out = []
        for i in range(1, len(traj)):
            t0, _, om0, al0 = traj[i - 1]
            t1, _, om1, al1 = traj[i]
            if t1 - t0 <= 0:
                continue
            om = 0.5 * (om0 + om1)
            al = 0.5 * (al0 + al1)
            if abs(om) < 1e-9 and abs(al) < 1e-9:
                out.append("Dwell")             # stationary, the dwell_time rest
            elif abs(al) < 1e-9:
                out.append("Cruise")            # constant velocity, v-limited
            else:
                # Accel and decel are split because they, not the direction of
                # travel, are what separate the torque bands whenever inertia
                # dominates gravity: J*alpha reverses sign between them while
                # the gravity term only tracks angle. alpha*omega > 0 is
                # speeding up, whichever way the joint is going.
                direction = "Pos" if om > 0 else "Neg"
                sense = "Accel" if al * om > 0 else "Decel"
                out.append(f"Move_{direction}_{sense}")
        return out

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
