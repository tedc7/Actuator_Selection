"""
Smoke tests.

These exist because the package was once committed with its modules flattened to
the repo root, which broke every relative import and made the whole tool
unrunnable while looking perfectly fine in a file listing. The first test here
would have caught that instantly.

Run with either:
    python3 -m pytest tests/
    python3 tests/test_smoke.py
"""

import re
import math
import os
import subprocess
import sys
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from actuator_eval import charts, db, evaluate, report          # noqa: E402
from actuator_eval.models import Actuator, Joint, Load, MotionProfile  # noqa: E402


def test_package_imports():
    """Every module imports as a package. Guards the flattening regression."""
    import actuator_eval
    for m in ("params", "units", "models", "physics", "db", "evaluate",
              "report", "charts"):
        assert hasattr(actuator_eval, m), f"actuator_eval.{m} missing"


def test_database_is_discoverable():
    """The bundled actuators and the example application are found by name."""
    acts = db.list_actuators()
    assert "robstride_00" in acts and "robstride_06" in acts, acts
    assert "_TEMPLATE" not in acts, "templates must not appear as selectable"

    names = [n for n, _ in db.list_applications()]
    assert "elbow_example" in names, names
    assert "_TEMPLATE" not in names


def test_templates_parse():
    """Both templates must stay loadable, since they are the documented starting point."""
    import json
    a_tpl = os.path.join(db.ACTUATOR_DIR, "_TEMPLATE.json")
    with open(a_tpl) as f:
        act, _ = db.actuator_from_dict(json.load(f))
    assert act is not None

    j_tpl = os.path.join(db.EXAMPLE_DIR, "_TEMPLATE.json")
    with open(j_tpl) as f:
        joint, _ = db.joint_from_dict(json.load(f))
    assert joint.profile is not None, "template should carry a motion profile"
    assert joint.load.total_mass_at_CG > 0


def test_end_to_end_example():
    """The quick-start command produces a complete, self-consistent report."""
    act, aa = db.load_actuator("robstride_00", with_audit=True)
    joint, ja = db.load_application("elbow_example", with_audit=True)
    ev = evaluate.evaluate(act, joint)
    ev.unit_audits = [aa, ja]

    assert ev.verdict in ("PASS", "MARGINAL", "FAIL", "UNKNOWN")
    assert ev.criteria, "no criteria evaluated"
    assert ev.thermal is not None and not ev.thermal.runaway
    assert math.isfinite(ev.thermal.t_winding_peak)
    # two actuators in the example file, each carrying half the load
    assert joint.n_actuators == 2

    text = report.render(ev)
    for section in ("CONFIGURATION", "CRITERIA", "VERDICT", "UNIT AUDIT",
                    "ASSUMED PARAMETERS"):
        assert section in text, f"{section} missing from report"


def test_units_wrong_dimension_is_hard_error():
    """A unit from the wrong dimension must never pass silently."""
    from actuator_eval import units as U
    try:
        U.convert(1.0, "kg", "length")
    except U.UnitError as e:
        assert "mass unit" in str(e) and "length" in str(e), str(e)
    else:
        raise AssertionError("expected a UnitError for kg as a length")


def test_source_path_and_output_dir():
    """Reports must be routed to the directory holding the application file."""
    joint = db.load_application("elbow_example")
    assert joint.source_path and joint.source_path.endswith("elbow_example.json")
    assert joint.output_dir() == os.path.realpath(db.EXAMPLE_DIR) or \
           joint.output_dir() == db.EXAMPLE_DIR, joint.output_dir()
    assert joint.slug() == "elbow_example"

    # a Joint built in code, not loaded, must still give a usable destination
    bare = Joint(name="Hand Built")
    assert bare.output_dir() == os.getcwd()
    assert bare.slug() == "hand_built"


def test_charts_write_self_contained_html():
    act = db.load_actuator("robstride_00")
    joint = db.load_application("elbow_example")
    ev = evaluate.evaluate(act, joint)
    with tempfile.TemporaryDirectory() as d:
        out = os.path.join(d, "c.html")
        charts.write_html(ev, out, report.render(ev))
        html = open(out).read()
    assert "<svg" in html and "</html>" in html
    # No CDN, no JS: the README promises a self-contained file. The SVG xmlns
    # is a bare identifier rather than a fetch, so exempt it.
    stripped = html.replace('xmlns="http://www.w3.org/2000/svg"', "")
    for bad in ("http://", "https://", "<script", "<link", "@import"):
        assert bad not in stripped, f"chart html should not contain {bad}"


def test_cli_runs():
    """The documented command line actually executes."""
    for args in (["--list"],
                 ["-a", "robstride_00", "-j", "elbow_example", "--brief"],
                 ["-a", "robstride_00", "-j", "elbow_example", "-n", "1,2"]):
        r = subprocess.run([sys.executable, "eval_actuator.py"] + args,
                           cwd=REPO, capture_output=True, text=True)
        assert r.returncode == 0, f"{args} failed:\n{r.stderr}"
        assert r.stdout.strip(), f"{args} produced no output"


def test_cli_writes_beside_application(tmp_path=None):
    """--save puts the report next to the application, not in the cwd."""
    with tempfile.TemporaryDirectory() as d:
        app = os.path.join(d, "probe.json")
        with open(os.path.join(db.EXAMPLE_DIR, "elbow_example.json")) as f:
            src = f.read()
        with open(app, "w") as f:
            f.write(src)
        r = subprocess.run(
            [sys.executable, "eval_actuator.py", "-a", "robstride_00",
             "-j", app, "--save", "--charts"],
            cwd=REPO, capture_output=True, text=True)
        assert r.returncode == 0, r.stderr
        produced = sorted(os.listdir(d))
        assert any(f.endswith(".txt") for f in produced), produced
        assert any(f.endswith(".html") for f in produced), produced


def test_every_actuator_declares_its_datasheet():
    """
    Every db entry must say which vendor document it came from, and the report
    must repeat it. Vendors revise datasheets in place, so an undeclared entry
    is one whose numbers can never be re-checked.
    """
    from actuator_eval import db

    for name in db.list_actuators():
        act = db.load_actuator(name)
        d = act.datasheet
        assert d, f"{name} declares no datasheet"
        for k in ("title", "file", "sha256"):
            assert d.get(k), f"{name} datasheet is missing '{k}'"
        assert len(d["sha256"]) == 64, f"{name} sha256 is not a full digest"
        assert act.datasheet_id(), f"{name} datasheet_id() is empty"
        assert act.entry_revision, f"{name} declares no entry_revision"


def test_report_names_the_datasheet():
    from actuator_eval import db, evaluate, report

    act = db.load_actuator("robstride_00")
    joint = db.load_application("elbow_example")
    text = report.render(evaluate.evaluate(act, joint), verbose=False)
    assert "RS00 User Manual" in text, "report does not name the datasheet"

    # ...and says so loudly when an entry does not declare one.
    act.datasheet = None
    bare = report.render(evaluate.evaluate(act, joint), verbose=False)
    assert "NOT DECLARED" in bare


def test_tn_curve_is_used_when_present_and_optional_when_not():
    """
    The whole point of the T-N curve support: a measured envelope wins when it
    exists, and its absence must not break anything.
    """
    from actuator_eval import db, physics
    from actuator_eval.models import TNCurve

    # robstride_01 deliberately records no curve (the vendor's is mislabelled).
    bare = db.load_actuator("robstride_01")
    bare.fill_defaults()
    assert bare.tn_curve is None
    assert physics.max_torque_at_speed(bare, 5.0, 48.0) > 0, "fallback broke"
    assert physics.tn_crosscheck(bare, 48.0) == (None, None)

    # robstride_00 has one; inside its span the envelope must BE the curve.
    act = db.load_actuator("robstride_00")
    act.fill_defaults()
    curve = act.tn_curve
    assert curve and len(curve.points) >= 2

    for rpm, tau in curve.points:
        got = physics.max_torque_at_speed(act, rpm * math.pi / 30.0, 48.0)
        assert abs(got - tau) < 1e-6, f"at {rpm} rpm expected {tau}, got {got}"

    # No-load speed comes off the end of the curve, not from Ke.
    assert abs(physics.no_load_speed_out(act, 48.0)
               - curve.speed_range_rpm[1] * math.pi / 30.0) < 1e-6

    # Interpolation between points, flat below the first, zero above the last.
    assert curve.torque_at_rpm(125) == 13.0            # midway 100->150
    assert curve.torque_at_rpm(0) == curve.points[0][1]
    assert curve.torque_at_rpm(10_000) == 0.0

    # The cross-check runs and reports a real disagreement for this actuator.
    rows, rms = physics.tn_crosscheck(act, 48.0)
    assert rows and len(rows) == len(curve.points)
    assert rms > 0.0


def test_tn_curve_bus_rescale_and_bad_units():
    from actuator_eval import db
    from actuator_eval.models import TNCurve
    from actuator_eval import units as U

    c = TNCurve([(100, 10.0), (200, 0.0)], bus_voltage=24.0)
    hot = c.scaled_to_bus(48.0)
    assert hot.speed_range_rpm == (200.0, 400.0), "speed axis should double"
    assert hot.torque_at_rpm(200) == 10.0, "torque axis must not scale"
    assert c.speed_range_rpm == (100.0, 200.0), "original must be unchanged"

    # A speed unit we do not understand is a gear-ratio-sized error: refuse it.
    try:
        db._tn_curve_from_dict(
            {"points": [[1, 1], [2, 2]], "speed_units": "rpm_rotor"},
            U.UnitAudit())
    except U.UnitError:
        pass
    else:
        assert False, "unrecognised speed_units must raise"


def test_inductance_estimate_tracks_the_vendor_curves():
    """
    L_phase used to be a flat 200 uH for every actuator, which put the modelled
    envelope 15-58% off the published curves -- too strong at speed on the RS00,
    far too weak on the RS06. It now scales as K*lambda/I_peak^2. Lock in that
    the modelled envelope stays close to every curve we have, so a change to the
    estimator cannot silently undo it.
    """
    from actuator_eval import db, physics

    for name in ("robstride_00", "robstride_02", "robstride_06"):
        act = db.load_actuator(name)
        act.fill_defaults()
        assert act.L_phase.source != "guess", f"{name}: L_phase fell back to a guess"

        rows, rms = physics.tn_crosscheck(act, 48.0)
        assert rows, f"{name}: no cross-check rows"
        assert rms < 0.15, f"{name}: modelled envelope off the curve by {rms:.0%}"

    # The estimator must actually differentiate between actuators: these two
    # differ ~17x in inductance, and a flat guess would make them equal.
    lo = db.load_actuator("robstride_06"); lo.fill_defaults()
    hi = db.load_actuator("robstride_00"); hi.fill_defaults()
    assert float(hi.L_phase) > 5 * float(lo.L_phase)

    # Without a current rating there is nothing to scale from; say so honestly
    # rather than returning a confident-looking number.
    from actuator_eval.models import est_inductance
    assert est_inductance(0.15, 14, 0.0).source == "guess"
    assert est_inductance(0.15, 14, 20.0).source == "estimated"


def test_chart_vendor_points_land_on_the_measured_curve():
    """
    The vendor scatter and the measured-curve line must occupy the same x-axis.

    They did not: the lines are built from w_out in rad/s and multiply by the
    rad/s->rpm factor, but TNCurve.points are already in rpm, so applying it
    again put the markers at 9.55x the right speed -- off the plot, and dragging
    the autoscaled axis with them so the curves themselves squashed into the
    left tenth of the chart. Pixel geometry is the only thing that catches this;
    every value involved is individually correct.
    """
    import re
    from actuator_eval import db, evaluate, charts

    act = db.load_actuator("robstride_00")
    joint = db.load_application("elbow_example")
    ev = evaluate.evaluate(act, joint)
    svg = charts.torque_speed(ev)

    # Data markers are r=3.4; the legend swatch is a different radius.
    pts = sorted((float(x), float(y)) for x, y in re.findall(
        r'<circle cx="([\d.]+)" cy="([\d.]+)" r="3\.4"[^>]*fill="#0072B2"', svg))
    assert len(pts) == len(act.tn_curve.points), "expected one marker per point"

    line = None
    for m in re.finditer(r'<polyline[^>]*/>', svg):
        if 'stroke="#0072B2"' in m.group(0):
            line = [tuple(map(float, p.split(',')))
                    for p in re.search(r'points="([^"]+)"', m.group(0)).group(1).split()]
    assert line, "no measured-curve polyline found"

    lo = min(x for x, _ in line)
    hi = max(x for x, _ in line)
    for x, _ in pts:
        assert lo - 1 <= x <= hi + 1, (
            f"vendor marker at x={x:.0f} is outside the curve's span "
            f"{lo:.0f}..{hi:.0f} -- unit mismatch between scatter and line")

    # Each marker should sit ON the line it was measured from.
    def interp(x):
        for (x0, y0), (x1, y1) in zip(line, line[1:]):
            if x0 <= x <= x1:
                return y0 if x1 == x0 else y0 + (y1 - y0) * (x - x0) / (x1 - x0)
        return None

    for x, y in pts:
        ly = interp(x)
        if ly is not None:
            assert abs(ly - y) < 2.0, (
                f"marker at x={x:.0f} sits {abs(ly-y):.1f} px off the curve")


def test_report_and_chart_show_both_envelopes():
    from actuator_eval import db, evaluate, report, charts

    joint = db.load_application("elbow_example")
    act = db.load_actuator("robstride_06")
    ev = evaluate.evaluate(act, joint)

    text = report.render(ev, verbose=False)
    assert "TORQUE-SPEED ENVELOPE" in text
    assert "measured curve" in text
    assert "RMS disagreement" in text

    svg = charts.torque_speed(ev)
    assert "vendor data points" in svg
    assert "our model (cross-check)" in svg

    # ...and the no-curve case still renders, without the cross-check furniture.
    ev2 = evaluate.evaluate(db.load_actuator("robstride_01"), joint)
    t2 = report.render(ev2, verbose=False)
    assert "computed from Ke" in t2
    assert "our model (cross-check)" not in charts.torque_speed(ev2)


def test_overload_curve_loads_and_requires_its_test_condition():
    """
    Endurance tables are optional, but a table whose test condition is unstated
    is refused: stall heating is 1.414x rotating for the same current, so
    assuming a condition would be a 41% error silently applied.
    """
    from actuator_eval import db, units as U

    act = db.load_actuator("robstride_06")
    act.fill_defaults()
    rot = act.overload_curve("rotating")
    stall = act.overload_curve("stalled")
    assert rot and stall, "RS06 publishes both tables"
    assert rot.speed_rpm_output == 100.0 and rot.ambient_C == 25.0
    assert act.overload_curve("nonsense") is None

    # log-linear interpolation, and None rather than extrapolation off the ends
    mid = rot.seconds_at_torque(33.0)
    assert 5.0 < mid < 8.0, mid
    assert rot.seconds_at_torque(100.0) is None
    assert rot.seconds_at_torque(1.0) is None

    for bad in ({"condition": "spinning"}, {}):
        raw = dict(bad, points=[[10, 5], [8, 20]])
        try:
            db._overload_curves_from_dict(raw, U.UnitAudit())
        except U.UnitError:
            pass
        else:
            assert False, f"unstated/unknown condition must raise: {bad}"

    # an actuator with no table is the normal case, not an error
    plain = db.load_actuator("robstride_00")
    plain.fill_defaults()
    assert plain.overload_curve("stalled") is None


def test_overload_crosscheck_measures_the_thermal_model():
    """
    The thermal model is four estimates in series (Rth_wc, Rth_ca, C_w, C_c).
    This is the only check that compares them against something measured, so
    lock in that it runs, that it reports in log space, and that it reports the
    direction of the error -- ratio > 1 means the model outlasts the vendor,
    i.e. it is optimistic and a sizing verdict built on it is unsafe.
    """
    from actuator_eval import db, physics

    act = db.load_actuator("robstride_06")
    act.fill_defaults(mounting="heatsunk")
    rows, log_rms = physics.overload_crosscheck(act, "rotating")
    assert rows and len(rows) == 8
    assert log_rms == log_rms and log_rms > 0        # not NaN

    for tau, vendor_s, model_s, ratio in rows:
        assert tau > 0 and vendor_s > 0
        assert model_s > 0
        assert abs(ratio - model_s / vendor_s) < 1e-9

    # Endurance must fall as torque rises -- if it does not, the integrator or
    # the loss model is wrong, and every verdict downstream of it is too.
    times = [physics.time_to_thermal_limit(act, t, 100 * math.pi / 30, 25.0)
             for t in (12.0, 18.0, 28.0, 36.0)]
    assert all(a > b for a, b in zip(times, times[1:])), times

    # A torque the actuator can hold forever must not report a finite time.
    assert math.isinf(physics.time_to_thermal_limit(act, 0.5, 0.0, 25.0))

    # Stall is where the two-node model is known to break down: it assumes even
    # three-phase heating, so it should read optimistic against a stall table.
    # Asserting the direction keeps that honest if the model changes.
    srows, slog = physics.overload_crosscheck(act, "stalled")
    assert srows and slog > 0
    finite = [r[3] for r in srows if math.isfinite(r[3])]
    assert sum(finite) / len(finite) > 1.0, "stall model should be optimistic"

    # No table -> no cross-check, cleanly.
    bare = db.load_actuator("robstride_00")
    bare.fill_defaults()
    assert physics.overload_crosscheck(bare, "stalled") == (None, None)

    # The comparison must be against the VENDOR's test conditions, including the
    # heat path: the same table judged from three different application
    # mountings has to give the same answer, or the report's claim that it runs
    # at the table's conditions is false. And Rth_ca must be put back.
    seen = []
    for mounting in ("heatsunk", "bolted_metal", "free_air"):
        a = db.load_actuator("robstride_00")
        a.fill_defaults(mounting=mounting)
        before = float(a.Rth_ca)
        _, lr = physics.overload_crosscheck(a, "rotating")
        seen.append(round(lr, 6))
        assert float(a.Rth_ca) == before, "Rth_ca must be restored"
    assert len(set(seen)) == 1, f"cross-check leaked the app's mounting: {seen}"


def test_endurance_chart_draws_with_and_without_vendor_data():
    """
    The endurance chart must render for EVERY actuator, not only the ones with
    a published table -- an actuator with no measurement is where the thermal
    model is least checked, so hiding the chart there is backwards. Without a
    table it must say so, loudly, instead of implying the model was verified.
    """
    from actuator_eval import db, evaluate, charts

    joint = db.load_joint("elbow_example")

    act = db.load_actuator("robstride_00")
    ev = evaluate.evaluate(act, joint)
    svg = charts.overload_endurance(ev, "rotating")
    assert svg and "vendor measured" in svg
    assert "UNVERIFIED" not in svg
    # all three layers present: model line, vendor dots, application squares
    assert svg.count("<polyline") >= 1          # the modelled endurance curve
    assert "<circle" in svg                      # vendor measured points
    assert 'width="9.00"' in svg                 # application profile squares

    # The rated torque is the ASYMPTOTE of the measured series -- the vendor's
    # table ends "...6 N.m for 285 s, 5 rated" -- not an independent level. It
    # must sit at the far right of the time axis, not span it as a horizontal
    # line, which would claim 5 N.m is also the limit at one second.
    import re as _re2
    _gain = joint.n_actuators * float(joint.ratio) * float(joint.ratio_eff)
    _rated = act.overload_curve("rotating").rated_torque * _gain
    _big = [(float(x), float(y)) for x, y in _re2.findall(
        r'<circle cx="([\d.]+)" cy="([\d.]+)" r="4\.6"', svg)]
    assert len(_big) == 1, "exactly one rated-continuous point"
    _plot_right = 1020 - 20          # width - right margin
    assert _big[0][0] > _plot_right - 60, \
        f"rated point must sit at the right edge, got x={_big[0][0]}"
    assert "indefinite" in svg, "the asymptote tick must not read as a duration"
    assert f"{_rated:.1f} N.m" in svg

    # This chart is JOINT-referred, like charts 1 and 2. It used to plot the
    # vendor's per-actuator table on per-actuator axes while charts 1 and 2
    # showed joint torque, so the same 2x RS00 elbow read 9.2 N.m continuous
    # there and 5.0 N.m here with nothing to explain the factor of two.
    assert joint.n_actuators > 1, "fixture must exercise the n>1 scaling"
    assert "joint torque (N.m)" in svg
    assert "actuators assumed to be loaded identically" in svg, \
        "n x a single-unit measurement is an assumption and must be declared"
    assert f"x{joint.n_actuators}" in svg, "legend must mark the scaled points"

    # The scaling must be applied to the model's OUTPUT torque, never fed into
    # the thermal integration: n actuators sharing a load each carry 1/n of it,
    # so the endurance TIME at a given joint torque is unchanged by n. If a
    # refactor ever multiplies the torque going in, times collapse and this
    # catches it.
    _solo = db.load_joint("elbow_example")
    _solo.n_actuators = 1
    _svg_solo = charts.overload_endurance(
        evaluate.evaluate(db.load_actuator("robstride_00"), _solo), "rotating")
    assert "per actuator" in _svg_solo, "n=1 needs no shared-load caveat"
    assert "loaded identically" not in _svg_solo
    # Check this on the MODEL CURVE, whose x positions are times produced by the
    # integration. Checking it on the vendor points instead proves nothing: they
    # are raw table data that no scaling path touches, so that assertion passes
    # even when the gain is wrongly fed into time_to_thermal_limit.
    def _model_xs(s):
        pts = _re2.findall(r'<polyline points="([^"]+)"', s)
        assert pts, "model curve must be drawn"
        return [q.split(",")[0] for q in pts[0].split()]

    assert _model_xs(svg) == _model_xs(_svg_solo), \
        "endurance times must not shift with actuator count"
    # ...while the curve itself must move, or nothing was scaled at all.
    assert _re2.findall(r'<polyline points="([^"]+)"', svg)[0] != \
        _re2.findall(r'<polyline points="([^"]+)"', _svg_solo)[0], \
        "the model curve must be joint-referred"

    # Same actuator with the table taken away: still draws, now flagged.
    bare = db.load_actuator("robstride_00")
    bare.overload_curves = []
    ev2 = evaluate.evaluate(bare, joint)
    svg2 = charts.overload_endurance(ev2, "rotating")
    assert svg2, "chart must still render without vendor data"
    assert "UNVERIFIED" in svg2
    assert "no vendor endurance data" in svg2
    assert "vendor measured" not in svg2
    # stall has no meaning without a table -- the model cannot represent it
    assert charts.overload_endurance(ev2, "stalled") == ""

    # The application layer is one point per duty variant at its RMS-EQUIVALENT
    # torque, not its instantaneous peak. Both reference lines on this chart
    # (vendor rated, and the endurance curve) describe sustained operation, so
    # plotting the peak against them would invite a comparison that does not
    # hold: the elbow touches 5.1 N.m for 23 ms of every 1.6 s, which sits on
    # the 5.0 N.m rated line and reads as marginal, while its RMS-equivalent is
    # ~2.5 N.m against a 4.6 N.m continuous limit.
    from actuator_eval import physics
    n_loaded = len([s for s in ev.duty_segments if abs(s[1]) > 1e-6])
    assert n_loaded > 20, "elbow duty should be finely sliced"
    n_variants = len(charts.duty_variants(ev))
    assert svg.count('width="9.00"') == n_variants, \
        "one square per duty variant, not per segment"

    # duty_segments and torque_out are both per-actuator; the chart is
    # joint-referred, so the labelled figures carry the same gain.
    peak = max(abs(s[1]) for s in ev.duty_segments)
    nominal = [v for v in charts.duty_variants(ev) if v[3]][0]
    i_rms = physics.rms_current_of_duty(act, nominal[1],
                                        t_est=float(act.T_winding_max))
    tau_rms = abs(act.torque_out(i_rms, t_magnet=float(act.T_winding_max)))
    assert tau_rms < 0.7 * peak, "RMS should be well below peak for this duty"
    assert f"{tau_rms * _gain:.2f} N.m RMS" in svg, \
        "the labelled figure must be the RMS, joint-referred"
    assert f"peaks at {peak * _gain:.1f} N.m" in svg
    # and the peak is named as a different question, not drawn on this axis
    assert "instantaneously" in svg

    # The footnote must not land on top of the x-axis label. Both used to be
    # pinned to the bottom edge 6 px apart, so the interpretation line was drawn
    # over the axis caption. The footnote now reserves bottom margin and wraps.
    import re as _re
    _xlab = [float(y) for y, t in _re.findall(
        r'<text x="[\d.]+" y="([\d.]+)" font-size="12"[^>]*'
        r'text-anchor="middle">([^<]*)', svg) if "time" in t]
    _foot = [float(y) for y in _re.findall(
        r'<text x="\d+" y="([\d.]+)" font-size="11" fill="#6b6b6b">', svg)]
    assert _xlab and _foot, "chart should have both an x label and a footnote"
    assert min(_foot) > max(_xlab) + 10, \
        f"footnote at {min(_foot)} collides with x label at {max(_xlab)}"
    _svg_h = float(_re.findall(r'viewBox="0 0 \d+ (\d+)"', svg)[0])
    assert max(_foot) <= _svg_h - 4, "footnote must stay inside the frame"
    # long footnotes wrap rather than running off the right edge
    assert len(_foot) >= 2, "this chart's footnote is long enough to wrap"

    # Charts without a footnote must be unaffected by that reserved space.
    plain = charts.thermal(ev)
    assert 'y="618" font-size="12"' in plain, \
        "footnote spacing must not shift charts that have no footnote"

    # Sections stay contiguously numbered whether or not the figure is present.
    import re as _re
    for e in (ev, ev2):
        html = charts.write_html(e, os.path.join(
            tempfile.mkdtemp(), "c.html"))
        heads = _re.findall(r"<h2>(\d+)\.", open(html).read())
        assert heads == [str(i + 1) for i in range(len(heads))], heads


def test_thermal_chart_reflects_the_load_split_between_actuators():
    """
    Chart 2 plots winding TEMPERATURE, so unlike chart 3 there is no torque to
    scale to the joint -- but it must still see the 1/n load split, or a pair of
    actuators would be shown heating like one carrying everything.

    The split enters upstream: Joint.actuator_output_demand divides joint torque
    by n_actuators, so ev.duty_segments is already per-actuator and warmup_curve
    integrates one winding carrying its own share. This test pins that chain,
    because it is invisible at the chart and easy to "fix" by multiplying.
    """
    from actuator_eval import db, evaluate, physics, charts

    act = db.load_actuator("robstride_00")
    rises, peaks = [], []
    for n in (1, 2, 4):
        joint = db.load_joint("elbow_example")
        joint.n_actuators = n
        ev = evaluate.evaluate(act, joint)
        peaks.append(max(abs(s[1]) for s in ev.duty_segments))
        _, tw, _ = physics.warmup_curve(act, ev.duty_segments, joint.T_ambient)
        rises.append(max(tw) - joint.T_ambient)

    # Torque per actuator splits as 1/n -- but NOT exactly. Each added actuator
    # also adds its own reflected rotor inertia, which the joint then has to
    # accelerate, so the total joint torque creeps up with n (10.16 -> 10.27
    # N.m here) and the per-actuator share falls slightly slower than 1/n.
    # Assert the split to a few percent and the inertia direction separately;
    # demanding exactly 1/n would be asserting a bug.
    assert abs(peaks[0] / peaks[1] - 2.0) < 0.05, peaks
    assert abs(peaks[0] / peaks[2] - 4.0) < 0.10, peaks
    totals = [p * n for p, n in zip(peaks, (1, 2, 4))]
    assert totals[0] < totals[1] < totals[2], \
        f"added rotor inertia should raise total joint torque: {totals}"

    # Temperature rise therefore falls at least as fast as 1/n^2 (I^2*R). It
    # falls FASTER than the square here, which is not an error: at n=1 this
    # duty drives the winding to ~200 degC, where copper resistance is ~1.6x
    # cold, so the n=1 case is superlinearly penalised. Assert the bound, not
    # an exact power, so the copper-feedback term stays free to change.
    assert rises[1] <= rises[0] / 4.0 + 1e-6, rises
    assert rises[2] <= rises[0] / 16.0 + 1e-6, rises

    # And the chart itself must move with n, not just the physics behind it.
    def _svg_for(n):
        jt = db.load_joint("elbow_example")
        jt.n_actuators = n
        return charts.thermal(evaluate.evaluate(act, jt))

    assert _svg_for(1) != _svg_for(2), \
        "chart 2 must respond to actuator count"


# ---------------------------------------------------------------------------
# Application envelopes
# ---------------------------------------------------------------------------

def _synthetic_duty(n=20000, seed=5):
    """A plausible joint history: PI jitter, one sustained burst, one spike."""
    import random
    rng = random.Random(seed)
    out = []
    for i in range(n):
        tau, om = 1.2 + rng.gauss(0, 0.4), rng.gauss(0, 3.0)
        if 5000 < i < 5600:                       # sustained move
            tau, om = 4.5 + rng.gauss(0, 0.3), 12.0
        if 12000 < i < 12050:                     # brief high-torque event
            tau, om = 9.0, 2.0
        out.append((0.001, tau, om))
    return out


def test_envelope_preserves_the_order_free_quantities():
    """
    The claim the whole feature rests on: an occupancy histogram carries
    everything the order-free quantities depend on.

    Cycle-mean loss and RMS current are pure functions of the multiset of
    (dt, tau, omega) triples -- no ordering anywhere -- so binning can only cost
    the width of a cell. Total time must come back exactly; the two derived
    quantities land far inside any actuator tolerance. Bitwise equality is
    impossible because current_for_torque inverts a cubic and the loss model
    also depends on the binned omega.
    """
    from actuator_eval import physics as phys
    from actuator_eval.envelope import from_samples

    act = db.load_actuator("robstride_00").fill_defaults()
    raw = _synthetic_duty()
    segs = from_samples(raw, name="t").as_duty_segments()

    assert abs(sum(s[0] for s in segs) - sum(s[0] for s in raw)) < 1e-9, \
        "time must survive summarisation exactly"

    r_raw = phys.rms_current_of_duty(act, raw)
    r_env = phys.rms_current_of_duty(act, segs)
    assert abs(r_env - r_raw) / r_raw < 1e-3, (r_raw, r_env)

    m_raw = phys.simulate_duty(act, raw, 40.0).mean_loss_W
    m_env = phys.simulate_duty(act, segs, 40.0).mean_loss_W
    assert abs(m_env - m_raw) / m_raw < 1e-3, (m_raw, m_env)


def test_envelope_peak_winding_temp_is_not_taken_from_the_histogram():
    """
    The counterpart of the test above, and the reason windows exist.

    t_winding_peak integrates in order, and the histogram has no order, so the
    number simulate_duty returns for it is an artefact of however the cells got
    sorted. It must therefore differ from the truth -- if it ever stopped
    differing, that would mean the ordering had become meaningful and this whole
    argument would need revisiting.
    """
    from actuator_eval import physics as phys
    from actuator_eval.envelope import from_samples

    act = db.load_actuator("robstride_00").fill_defaults()
    raw = _synthetic_duty()
    segs = from_samples(raw, name="t").as_duty_segments()

    peak_raw = phys.simulate_duty(act, raw, 40.0).t_winding_peak
    peak_env = phys.simulate_duty(act, segs, 40.0).t_winding_peak
    assert abs(peak_env - peak_raw) > 1e-6, \
        "if these agree, the ascending-tau^2 ordering has stopped being arbitrary"


def test_envelope_does_not_hide_a_brief_extreme_event():
    """
    A histogram must never round away a critical point.

    A log that is almost entirely benign plus one 4 ms spike at 3x torque: the
    spike has to survive as ITSELF, with its duration intact, because duration
    is what decides whether it is a drive-current problem or a thermal one.
    """
    from actuator_eval.envelope import from_samples

    raw = [(0.001, 1.0, 1.0) for _ in range(20000)]
    for i in range(4):                             # 4 ms at 3x
        raw[9000 + i] = (0.001, 30.0, 1.0)

    env = from_samples(raw, name="spike")

    assert abs(env.extremes.t("max") - 30.0) < 1e-9, \
        "the max is a raw-sample fact and must not be binned"

    hits = [o for o in env.outliers if abs(o.tau_peak - 30.0) < 1e-9]
    assert len(hits) == 1, [(o.t_start, o.duration, o.tau_peak) for o in env.outliers]
    assert abs(hits[0].duration - 0.004) < 1e-9, hits[0].duration

    assert env.tail_is_outlier_dominated(), \
        "max 30x p99.9 must be flagged as an outlier-dominated tail"


def test_envelope_distinguishes_a_brief_spike_from_a_sustained_one():
    """
    Same peak torque, different duration, different physics.

    This is what carrying duration end to end buys: 4 ms at 30 N.m and 1.9 s at
    30 N.m have identical tau_peak and land in completely different places
    thermally. A summary that kept only the peak could not tell them apart.
    """
    from actuator_eval import physics as phys
    from actuator_eval.envelope import from_samples

    act = db.load_actuator("robstride_00").fill_defaults()

    def build(n_samples):
        raw = [(0.001, 1.0, 1.0) for _ in range(20000)]
        for i in range(n_samples):
            raw[9000 + i] = (0.001, 30.0, 1.0)
        return from_samples(raw, name="x")

    brief, sustained = build(4), build(1900)

    assert abs(brief.extremes.t("max") - sustained.extremes.t("max")) < 1e-9, \
        "the peak alone cannot separate these two cases"

    b = [o for o in brief.outliers if abs(o.tau_peak - 30.0) < 1e-9][0]
    s = [o for o in sustained.outliers if abs(o.tau_peak - 30.0) < 1e-9][0]
    assert s.duration > 100 * b.duration, (b.duration, s.duration)

    # The thermal consequence follows the duration, not the peak.
    loss_b = phys.simulate_duty(act, brief.as_duty_segments(), 40.0).mean_loss_W
    loss_s = phys.simulate_duty(act, sustained.as_duty_segments(), 40.0).mean_loss_W
    assert loss_s > 1.5 * loss_b, (loss_b, loss_s)


def test_envelope_keeps_a_dominant_sustained_state():
    """
    Regression: a busy joint must not lose its work to the percentile cut.

    When one high-torque state occupies more than 1% of the session the p99 cut
    lands ON it, so nothing is strictly above the threshold and the excursion
    fell out of the outlier list entirely. At the same time the quantile edges
    collapsed -- a two-spike distribution puts nearly every quantile on the same
    value -- leaving one enormous cell that averaged the busy state together
    with the idle one. The event disappeared from both artifacts at once, which
    is precisely the failure mode the whole design exists to prevent.
    """
    from actuator_eval.envelope import from_samples

    for n_hot in (4, 100, 1900, 6000):
        raw = [(0.001, 1.0, 1.0) for _ in range(20000)]
        for i in range(n_hot):
            raw[9000 + i] = (0.001, 30.0, 1.0)
        env = from_samples(raw, name="busy")
        segs = env.as_duty_segments()

        assert abs(sum(s[0] for s in segs) - 20.0) < 1e-9, \
            f"n_hot={n_hot}: session time must be conserved"

        hot = sum(dt for dt, tau, _ in segs if abs(tau) > 20.0)
        assert abs(hot - n_hot * 0.001) < 1e-9, \
            f"n_hot={n_hot}: {hot} s at 30 N.m, expected {n_hot * 0.001}"

        assert env.binned_below < env.extremes.t("max"), \
            f"n_hot={n_hot}: the cut must sit strictly below the peak"


def test_envelope_conserves_time_when_the_outlier_cap_trims():
    """
    Regression: capping the outlier LIST must never drop time or energy.

    With more excursions than the cap keeps, the trimmed ones fall back into the
    binned population instead of vanishing. Two earlier bugs lived here: the
    dropped runs were excluded from binning as well as from the list, and the
    span test walked an accumulated dt clock whose drift misclassified samples
    at the boundaries. Either way the session quietly lost its busiest moments,
    which is the exact failure this design exists to prevent.
    """
    import random
    from actuator_eval import physics as phys
    from actuator_eval.envelope import from_samples

    for n_runs in (50, 400):                 # under and over the 200 cap
        rng = random.Random(3)
        raw = [(0.001, 1.0 + rng.gauss(0, 0.05), 0.5) for _ in range(60000)]
        for j in range(n_runs):
            for i in range(5):
                raw[100 + j * 140 + i] = (0.001, 25.0, 0.5)

        env = from_samples(raw, name="capped")
        segs = env.as_duty_segments()

        assert abs(sum(s[0] for s in segs) - sum(s[0] for s in raw)) < 1e-9, \
            f"n_runs={n_runs}: session time must survive the cap"

        # tau^2 is what copper loss integrates, so it is the quantity that must
        # not move even when the outlier LIST is trimmed.
        e_raw = sum(t * t * dt for dt, t, _ in raw)
        e_env = sum(t * t * dt for dt, t, _ in segs)
        assert abs(e_env - e_raw) / e_raw < 1e-9, \
            f"n_runs={n_runs}: thermal energy must survive the cap"

        assert abs(env.extremes.t("max") - 25.0) < 1e-9, \
            f"n_runs={n_runs}: the peak is a raw-sample fact, never binned"

        act = db.load_actuator("robstride_00").fill_defaults()
        r_raw = phys.rms_current_of_duty(act, raw)
        r_env = phys.rms_current_of_duty(act, segs)
        assert abs(r_env - r_raw) / r_raw < 1e-3, (n_runs, r_raw, r_env)


def test_envelope_boundary_covers_the_whole_speed_range():
    """
    Regression: the outline must be free to rise as well as fall.

    The first version walked samples in descending torque and kept one only if
    its speed exceeded every sample kept so far, which forces a monotonically
    DECREASING staircase. On a real capture the peak torque occurs at LOW speed
    -- that is where a gravity-loaded joint works hardest -- so everything below
    the peak's speed was discarded by construction. Chart 1 showed nothing under
    the peak point while excursions and window samples were plainly visible
    there.

    Monotonicity is the same global-max fabrication a convex hull commits, and
    the outline exists precisely to avoid that.
    """
    from actuator_eval.envelope import boundary_scan

    import random
    rng = random.Random(4)
    pts = []
    for _ in range(20000):
        w = rng.uniform(0.0, 30.0)
        # A real joint accelerates INTO its peak torque, so the ceiling rises
        # off standstill before tapering with speed. That interior peak is what
        # a monotone staircase cannot represent.
        ceiling = 20.0 - 0.45 * abs(w - 6.0)
        pts.append((w, rng.uniform(0.0, max(ceiling, 1.0))))
    bound = boundary_scan(pts)

    assert bound == sorted(bound), "the outline is ordered by speed"
    rises = sum(1 for i in range(1, len(bound)) if bound[i][1] > bound[i - 1][1])
    assert rises > 0, \
        "a forced-monotone staircase is back: the outline can only fall"

    peak = max(bound, key=lambda p: p[1])
    assert peak[0] > 1.0, f"the interior peak was not found: {peak}"
    below = [p for p in bound if p[0] < peak[0]]
    assert len(below) > 5, (
        f"only {len(below)} outline points below the peak speed -- the "
        f"low-speed region is being discarded again")

    # And it must still refuse to invent the corner a convex closure would add.
    for om, tau in bound:
        assert om < 25.0 or tau < 12.0, \
            f"({om}, {tau}) is outside the region the samples actually covered"


def test_envelope_boundary_is_not_convex_and_invents_no_corner():
    """
    The outline follows what the joint did. A convex hull would fabricate a
    high-torque-at-high-speed corner that was never visited, and sizing against
    an invented corner is over-conservative in a way the reader cannot see.
    """
    from actuator_eval.envelope import boundary_scan

    # An L-shaped duty: high torque only when slow, high speed only when light.
    pts = [(0.5, 20.0), (1.0, 19.0), (18.0, 3.0), (20.0, 2.5)]
    bound = boundary_scan(pts)

    assert bound == sorted(bound), "boundary must be ordered by speed"
    taus = [t for _, t in bound]
    assert taus == sorted(taus, reverse=True), \
        "an outer staircase falls monotonically with speed"
    for om, tau in bound:
        assert om > 10.0 or tau > 10.0, \
            f"({om}, {tau}) is the fabricated corner a convex hull would add"


def test_motion_source_precedence_and_pinning():
    """
    envelope > profile > duty_segments, with every block allowed to coexist so
    that dropping one falls back to the next rather than erroring.
    """
    from actuator_eval.envelope import from_samples

    jt = db.load_application("elbow_example")
    assert jt.active_motion_source() == "profile"

    jt.duty_segments = [(0.5, 2.0, 1.0), (0.5, 1.0, 0.0)]
    assert jt.active_motion_source() == "profile", "a profile outranks duty_segments"
    assert jt.superseded_motion_sources() == ["duty_segments"]

    jt.envelope = from_samples([(0.001, 1.5, 2.0)] * 500, name="e")
    assert jt.active_motion_source() == "envelope"
    assert jt.superseded_motion_sources() == ["profile", "duty_segments"]

    jt.motion_source = "profile"
    assert jt.active_motion_source() == "profile", "an explicit pin wins"

    jt.motion_source = "auto"
    jt.envelope = None
    assert jt.active_motion_source() == "profile", "dropping a block falls back"

    for bad, why in (("envelope", "pinning an undefined block"),
                     ("nonsense", "an unknown source name")):
        jt.motion_source = bad
        try:
            jt.active_motion_source()
            assert False, f"{why} must raise"
        except ValueError:
            pass


def _example_envelope_dict():
    import json
    path = db.resolve_envelope_path("elbow_pick_place_example")
    with open(path) as f:
        return json.load(f)


def test_envelope_file_round_trips():
    """The committed example loads, and its artifacts survive the file format."""
    env = db.load_envelope("elbow_pick_place_example")
    assert env.cells and env.windows and env.outliers, env.summary()
    assert env.boundary and env.boundary_raw
    assert env.extremes.t("max") > env.extremes.t("p99_9") > 0
    assert env.extremes.t("max_raw") >= env.extremes.t("max"), \
        "the unfiltered peak must be kept alongside the despiked one"
    assert env.provenance().source == "derived", \
        "a commanded-current capture is derived, however careful it was"

    segs = env.as_duty_segments()
    assert abs(sum(s[0] for s in segs) - env.total_time) < 1e-6

    # windows are distinct objects per timescale, never merged across durations
    durations = [w.duration for w in env.windows]
    assert len(set(durations)) == len(durations), durations
    assert durations == sorted(durations)


def test_envelope_schema_version_is_gated():
    """A future format must fail loudly rather than parse into a wrong shape."""
    d = _example_envelope_dict()
    d["schema"] = "actuator_eval.envelope/2"
    try:
        db.envelope_from_dict(d)
        assert False, "an unknown schema version must raise"
    except ValueError as e:
        assert "schema" in str(e)


def test_envelope_ratio_must_match_the_application():
    """
    A joint-referred envelope is only valid for the drivetrain it came off.

    A ratio mismatch means the capture describes a different machine, and every
    torque in it is wrong by that factor -- exactly the kind of silent,
    confident error the migration guards elsewhere in db.py exist to prevent.
    """
    import json
    with open(db.resolve_application_path("elbow_envelope_example")) as f:
        app = json.load(f)
    app["ratio"] = {"value": 2.5, "units": "-"}
    try:
        db.joint_from_dict(app)
        assert False, "a ratio mismatch must raise"
    except ValueError as e:
        assert "ratio" in str(e).lower()


def test_envelope_rejects_a_wrong_dimension_unit():
    """
    A sample rate is a frequency, not an angular velocity.

    They share the symbol Hz and angular_velocity maps it to 2*pi rad/s, which
    is right for a shaft and badly wrong for a logger, so the two dimensions are
    kept separate and convert() catches the confusion.
    """
    from actuator_eval import units as U

    d = _example_envelope_dict()
    d["capture"]["sample_rate"] = {"value": 360, "units": "rpm"}
    try:
        db.envelope_from_dict(d)
        assert False, "rpm is not a sample rate"
    except U.UnitError:
        pass

    assert abs(U.convert(1.0, "Hz", "frequency") - 1.0) < 1e-12
    assert abs(U.convert(1.0, "Hz", "angular_velocity") - 2 * math.pi) < 1e-9, \
        "the angular reading of Hz must stay, it is correct for a shaft"


def test_application_without_an_envelope_is_unchanged():
    """
    The fallback requirement: adding envelopes must not perturb the existing
    profile path by even one number.
    """
    act = db.load_actuator("robstride_00")
    jt = db.load_application("elbow_example")
    assert jt.envelope is None and jt.active_motion_source() == "profile"

    ev = evaluate.evaluate(act, jt)
    assert ev.verdict in ("PASS", "MARGINAL", "FAIL")
    assert any(c.name == "Trajectory following" for c in ev.criteria), \
        "a commanded profile still gets its trajectory-following check"


def test_envelope_evaluation_splits_the_thermal_question():
    """
    The headline result: session mean and worst sequence are separate verdicts,
    because they come from different artifacts and neither can answer the other.
    """
    act = db.load_actuator("robstride_00")
    jt = db.load_application("elbow_envelope_example")
    ev = evaluate.evaluate(act, jt)

    names = [c.name for c in ev.criteria]
    assert "Thermal, session mean" in names, names
    assert "Thermal, worst sequence" in names, names
    assert "Thermal (winding)" not in names, \
        "the single thermal criterion is replaced, not duplicated"
    assert "Trajectory following" not in names, \
        "there is no commanded trajectory to follow under an envelope"

    # The composition that needs both artifacts.
    assert "binding_window" in ev.extras
    assert ev.extras["t_case_session_mean"] > jt.T_ambient
    win = ev.extras["binding_window"]
    tau_w = ev.extras["actuator_tau_w"]
    assert 0.2 < win.duration / max(tau_w, 1e-9) < 50, \
        (win.duration, tau_w, "the binding window should suit the winding constant")

    # Every envelope-fed criterion carries the capture's provenance.
    for name in ("Peak torque", "Drive current limit"):
        c = next(c for c in ev.criteria if c.name == name)
        assert "envelope" in c.depends_on, (name, c.depends_on)
        assert c.confidence != "measured", \
            "a commanded-current capture cannot yield a measured verdict"


def test_envelope_thermal_verdict_is_independent_of_cell_order():
    """
    Regression: the session-mean temperatures must not depend on the sort.

    An occupancy cell is a VALUE bucket, not an interval of the session -- a
    cell holding 4.6 A for "35 s" is thousands of brief moments scattered across
    the hour. simulate_duty integrates its list in order, so it read that as a
    sustained 35 s burst, and since as_duty_segments() sorts by ascending tau^2
    every hot cell ran consecutively at the end. That reported 154 degC against
    a 145 degC limit and FAILED the actuator, while the RMS criterion, the
    duration curve and both charts agreed the duty had 2x margin.

    mean_loss_W was order-free throughout; only the integrated node
    temperatures were not. So the settled values come from the mean loss.
    """
    import random

    act = db.load_actuator("robstride_00")
    base = evaluate.evaluate(act, db.load_application("elbow_envelope_example"))

    for seed in (1, 2, 3):
        jt = db.load_application("elbow_envelope_example")
        random.Random(seed).shuffle(jt.envelope.cells)
        ev = evaluate.evaluate(act, jt)
        assert abs(ev.thermal.t_winding_final - base.thermal.t_winding_final) < 1e-6, \
            (seed, ev.thermal.t_winding_final, base.thermal.t_winding_final)
        assert abs(ev.thermal.t_case_final - base.thermal.t_case_final) < 1e-6
        assert ev.verdict == base.verdict

    # And it must agree with the closed form the RMS criterion is built on.
    rth = float(act.Rth_wc) + float(act.Rth_ca)
    expect = base.joint.T_ambient + base.thermal.mean_loss_W * rth
    assert abs(base.thermal.t_winding_final - expect) < 1e-6, \
        (base.thermal.t_winding_final, expect)

    # The two thermal criteria must not contradict the continuous-torque one:
    # a duty with RMS headroom cannot settle above the insulation limit.
    rms = next(c for c in base.criteria if c.name == "Continuous torque (RMS)")
    mean = next(c for c in base.criteria if c.name == "Thermal, session mean")
    if rms.margin > 1.15:
        assert mean.status == "PASS", \
            (f"RMS says {rms.margin:.2f}x headroom but the session mean says "
             f"{mean.status} -- the two describe the same steady state")


def test_envelope_peak_uses_percentile_but_drive_current_uses_max():
    """
    A mechanical margin and a fault threshold are different questions.

    Sizing peak torque off the max lets one sensor glitch drive the whole
    verdict; sizing drive current off a percentile clears a drive that faults on
    a real spike the percentile discarded. So they read different numbers, and
    the report has to say which is which.
    """
    act = db.load_actuator("robstride_00")
    jt = db.load_application("elbow_envelope_example")
    ev = evaluate.evaluate(act, jt)
    env = jt.envelope

    peak = next(c for c in ev.criteria if c.name == "Peak torque")
    assert abs(peak.demand - env.peak_torque_demand()) < 1e-6
    assert peak.demand < env.extremes.t("max"), \
        "the percentile must sit below the max, or it is not doing anything"
    assert f"{env.extremes.t('max'):.2f}" in peak.detail, \
        "the max must still be reported even though it does not size this"

    drive = next(c for c in ev.criteria if c.name == "Drive current limit")
    assert "not a percentile" in drive.detail


def test_envelope_report_states_what_each_artifact_can_answer():
    """
    The authority table and the missing ripple line are load-bearing prose: a
    reader who does not know the occupancy record has no time ordering will
    read a peak temperature off it that means nothing.
    """
    act = db.load_actuator("robstride_00")
    jt = db.load_application("elbow_envelope_example")
    txt = report.render(evaluate.evaluate(act, jt))

    for chunk in ("APPLICATION ENVELOPE", "WHAT EACH ARTIFACT IS AUTHORITATIVE FOR",
                  "TIME IN REGIME", "SUSTAINED JOINT TORQUE BY AVERAGING WINDOW",
                  "EXTREMES AND EXCURSIONS", "worst excursions",
                  "binding sequence", "if it repeated forever"):
        assert chunk in txt, f"missing report section: {chunk}"

    assert "no time ordering" in txt
    assert "supersedes" in txt, "a superseded profile must be named, not ignored"

    # The prose may DISCUSS peak winding temperature; what must not appear is
    # the THERMAL DETAIL data line reporting one, since under an envelope that
    # number would be an artefact of how the cells were sorted.
    thermal = txt.split("THERMAL DETAIL")[1].split("UNIT AUDIT")[0]
    assert "peak winding temp     :" not in thermal, thermal
    assert "within-cycle ripple" in thermal and "n/a" in thermal


def test_motion_source_flag_falls_back_to_the_profile():
    """--motion-source profile evaluates the commanded path on the same file."""
    out = subprocess.run(
        [sys.executable, "eval_actuator.py", "-a", "robstride_00",
         "-j", "elbow_envelope_example", "--motion-source", "profile"],
        cwd=REPO, capture_output=True, text=True, timeout=300)
    assert out.returncode == 0, out.stderr
    assert "Trajectory following" in out.stdout
    assert "APPLICATION ENVELOPE" not in out.stdout


def test_envelope_phase_tags_align_with_its_segments():
    """
    Charts key off len(phases) == len(segs) to decide whether to colour the
    series by regime. A misaligned list silently degrades every envelope chart
    to undifferentiated grey.
    """
    from actuator_eval.envelope import from_samples

    env = from_samples(_synthetic_duty(), name="t")
    segs, phases = env.as_duty_segments(), env.motion_phases()
    assert len(segs) == len(phases), (len(segs), len(phases))
    assert set(phases) <= {"Env_Hold", "Env_Drive_Pos", "Env_Drive_Neg",
                           "Env_Regen_Pos", "Env_Regen_Neg"}, set(phases)

    fractions = env.time_in_regime()
    assert abs(sum(f for f, _ in fractions.values()) - 1.0) < 1e-9, fractions


# ---------------------------------------------------------------------------
# Log post-processing
# ---------------------------------------------------------------------------

def _write_log(path, rows, header=("t_s", "q_rad", "tau_Nm", "qd_rad_s")):
    import csv as _csv
    with open(path, "w", newline="") as f:
        w = _csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    return path


def _pi_log(n=120000, dt=0.001, seed=11, bursts=(), glitch_at=(),
            glitch=40.0, base=1.4, jitter=0.12):
    """A fixed-rate PI log: jitter, optional sustained bursts, optional spikes."""
    import random
    rng = random.Random(seed)
    rows, q = [], 0.0
    for i in range(n):
        t = i * dt
        tau, om = base + rng.gauss(0, jitter), rng.gauss(0, 0.01)
        for t0, t1, amp, spd in bursts:
            if t0 <= t < t1:
                tau, om = amp + rng.gauss(0, 0.2), spd
        if i in glitch_at:
            tau = glitch
        q += om * dt
        rows.append((round(t, 6), round(q, 6), round(tau, 5), round(om, 5)))
    return rows


def _read(paths, **kw):
    from actuator_eval import logread as LR
    cmap = LR.ColumnMap(
        columns={"time": "t_s", "speed": "qd_rad_s", "torque": "tau_Nm"},
        units={"time": "s", "speed": "rad/s", "torque": "N.m"})
    p1 = LR.read_pass1(paths, cmap, **kw)
    return LR, cmap, p1


def test_despiker_rejects_isolated_spikes_but_keeps_short_real_events():
    """
    The distinction the whole filter choice rests on, pinned so nobody
    "simplifies" the median into a moving average later.

    A median removes any excursion narrower than half its window and passes
    anything wider through undistorted. A moving average attenuates by duration
    instead, so it cannot tell an isolated glitch from a genuine 8 ms collision
    -- it destroys both.
    """
    from actuator_eval.logread import MedianFilter

    import random
    rng = random.Random(11)
    base = [6.0 + rng.gauss(0, 1.2) for _ in range(6000)]
    real = list(base)
    for i in range(3000, 3008):          # a genuine 8 ms event at 1 kHz
        real[i] = 24.0
    noisy = list(real)
    for i in (500, 1500, 2500, 4500, 5500):
        noisy[i] = 40.0                  # isolated single-sample glitches

    med = MedianFilter(5)
    filtered = [med.push(v) for v in noisy]

    assert max(filtered) < 26.0, \
        f"the 40 N.m glitches survived the median: {max(filtered)}"
    assert abs(max(filtered[2995:3015]) - 24.0) < 1e-9, \
        "the genuine 8 ms event must pass through undistorted"

    # The counterexample, asserted so the trade-off stays visible.
    k = 50
    boxcar = max(sum(noisy[i:i + k]) / k for i in range(len(noisy) - k))
    assert boxcar < 16.0, \
        "a 50 ms moving average would flatten the real event -- that is why " \
        "despiking uses a median"

    changed, worst = med.summary()
    assert 0 < changed < 200, \
        f"{changed} 'changed' samples: the count must be of MATERIAL changes, " \
        f"not of every sample a median nudges by a fraction of the jitter"
    assert worst > 20.0


def test_window_rms_agrees_with_the_duration_curve():
    """
    The two numbers describe the same fact and must not disagree.

    A stored window's RMS is the worst sustained torque over its own length,
    which is exactly what the duration curve reports at that length. Earlier
    versions disagreed three separate ways -- the window trailed the kernel's
    argmax instead of maximising over its own length, the decimated series
    accumulated dt until it crossed a threshold and drifted 1% per bucket, and
    its warm-up emitted short elements that slid every later index. Each showed
    up here as a report contradicting itself.
    """
    with tempfile.TemporaryDirectory() as d:
        path = _write_log(os.path.join(d, "log.csv"), _pi_log(
            n=120000, bursts=((20.0, 20.2, 24.0, 2.0),      # brief, intense
                              (40.0, 62.0, 6.5, 8.0),       # long, moderate
                              (80.0, 88.0, 12.5, 5.0))))    # medium, medium
        LR, cmap, p1 = _read([path], despike=5)
        env = LR.build_envelope([path], cmap, p1, name="t", despike=5)

    curve = dict(env.duration_curve)
    assert env.windows, "the ladder must find at least one window"
    for w in env.windows:
        want = curve.get(round(w.duration, 6))
        if want is None:
            continue
        assert abs(w.rms_torque - want) / max(want, 1e-9) < 0.02, \
            (f"window of {w.duration:g} s reports RMS {w.rms_torque:.2f} but "
             f"the duration curve says {want:.2f} for the same length")


def test_window_ladder_resolves_distinct_timescales():
    """
    Different thermal masses are worst-hit by different events, and each window
    is stored separately even when two overlap the same busy stretch: a 3 s and
    a 300 s window are different lengths answering different questions.
    """
    with tempfile.TemporaryDirectory() as d:
        path = _write_log(os.path.join(d, "log.csv"), _pi_log(
            n=300000, bursts=((20.0, 20.2, 26.0, 2.0),
                              (100.0, 190.0, 7.0, 8.0),
                              (250.0, 258.0, 13.0, 5.0))))
        LR, cmap, p1 = _read([path], despike=5)
        env = LR.build_envelope([path], cmap, p1, name="t", despike=5)

    durations = [w.duration for w in env.windows]
    assert len(set(durations)) == len(durations), \
        f"windows of different lengths must never be merged: {durations}"
    assert len(env.windows) >= 3, durations
    # The short and the long window cannot both be worst at the same moment.
    short = min(env.windows, key=lambda w: w.duration)
    long_ = max(env.windows, key=lambda w: w.duration)
    assert short.rms_torque > long_.rms_torque, \
        "a shorter window must carry at least as much RMS as a longer one"


def test_log_reader_rejects_bad_input_rather_than_guessing():
    from actuator_eval import logread as LR
    from actuator_eval import units as U

    # No units is a hard error: a torque read as N.m when it was kgf.cm is
    # wrong by 100x and still evaluates.
    cm = LR.ColumnMap(columns={"time": "t", "torque": "tau", "speed": "w"},
                      units={"time": "s", "torque": "N.m"})
    try:
        cm.validate()
        assert False, "a column without declared units must raise"
    except LR.LogError as e:
        assert "units" in str(e)

    # Wrong dimension, caught before reading a single row.
    cm = LR.ColumnMap(columns={"time": "t", "torque": "tau", "speed": "w"},
                      units={"time": "s", "torque": "kg", "speed": "rad/s"})
    try:
        cm.validate()
        assert False, "kg is not a torque"
    except U.UnitError:
        pass

    # Neither torque nor current means there is nothing to size against.
    cm = LR.ColumnMap(columns={"time": "t", "speed": "w"},
                      units={"time": "s", "speed": "rad/s"})
    try:
        cm.validate()
        assert False, "a log with no torque or current must raise"
    except LR.LogError:
        pass


def test_log_reader_skips_gaps_and_out_of_order_rows_consistently():
    """
    Both passes must accept exactly the same rows: window spans are sample
    indices into that shared sequence, so a row one pass keeps and the other
    drops slides every stored window off its event.
    """
    rows = _pi_log(n=20000)
    rows.insert(9000, (rows[9000][0] - 5.0, 0.0, 99.0, 0.0))   # backwards jump
    rows.insert(5000, (rows[5000][0], "", "", ""))              # blank fields

    with tempfile.TemporaryDirectory() as d:
        path = _write_log(os.path.join(d, "log.csv"), rows)
        LR, cmap, p1 = _read([path], despike=5)
        assert p1.stats.out_of_order >= 1
        assert p1.stats.dropped_nan >= 1
        n_accepted = sum(1 for _ in LR.accepted_rows([path], cmap))
        assert n_accepted == p1.stats.rows, (n_accepted, p1.stats.rows)
        # the 99 N.m row was on the rejected backwards jump
        assert p1.tau_q.max_seen < 50.0


def test_envelope_from_log_cli_end_to_end():
    """The workflow a user actually runs: log in, committable envelope out."""
    with tempfile.TemporaryDirectory() as d:
        path = _write_log(os.path.join(d, "log.csv"), _pi_log(
            n=60000, bursts=((10.0, 30.0, 8.0, 6.0),), glitch_at={12345}))
        out = os.path.join(d, "env.json")
        proc = subprocess.run(
            [sys.executable, "envelope_from_log.py", "--log", path,
             "--map", "time=t_s,speed=qd_rad_s,torque=tau_Nm",
             "--units", "time=s,speed=rad/s,torque=N.m",
             "--name", "cli_test", "--torque-source", "torque_sensor",
             "--out", out],
            cwd=REPO, capture_output=True, text=True, timeout=600)
        assert proc.returncode == 0, proc.stdout + proc.stderr
        assert os.path.exists(out)

        for chunk in ("session length", "coverage", "despike",
                      "sustained joint torque", "worst sequences found"):
            assert chunk in proc.stdout, f"missing from the summary: {chunk}"

        env = db.load_envelope(out)
        assert env.cells and env.windows
        assert env.provenance().source == "measured", \
            "a torque-sensor capture earns 'measured'"
        assert abs(sum(s[0] for s in env.as_duty_segments())
                   - env.total_time) < 1e-6


def test_envelope_charts_layer_density_under_the_raw_sample_facts():
    """
    The density map must not be able to hide a critical point.

    Occupancy rects are drawn first so they sit underneath; the outline and
    every excursion are drawn from raw samples on top. A 4 ms spike is a cell of
    negligible occupancy and would be invisible in the shading alone, but it is
    exactly the point that trips a drive, so it gets its own marker.
    """
    act = db.load_actuator("robstride_00")
    jt = db.load_application("elbow_envelope_example")
    ev = evaluate.evaluate(act, jt)
    svg = charts.torque_speed(ev)

    assert "time in cell" in svg, "a density map without a scale is decorative"
    assert "measured envelope" in svg
    assert "excursions above p99" in svg

    # Rects (the density) must be emitted before the envelope polyline, or the
    # shading would paint over the capability curve it sits beneath.
    first_rect = svg.find("<rect")
    first_poly = svg.find("<polyline")
    assert 0 <= first_rect < first_poly, (first_rect, first_poly)

    # Opacity must span a visible range: occupancy covers 4+ decades, and a
    # linear ramp would render everything except the hold cell invisible.
    ops = sorted(float(m) for m in
                 re.findall(r'<rect[^>]*opacity="([0-9.]+)"', svg))
    assert len(ops) > 50 and ops[0] < 0.15, (ops[:3], ops[-3:])
    assert ops[-1] - ops[0] > 0.25, \
        f"the density ramp is too flat to read: {ops[0]} .. {ops[-1]}"
    # ...but it must stay light enough to read the lines and markers that carry
    # the verdict THROUGH it. A dark ramp swallowed the curves crossing the
    # busiest cells, which is the opposite of what a background layer is for.
    assert ops[-1] <= 0.5, \
        f"the busiest cells at opacity {ops[-1]} will hide the series over them"


def test_envelope_density_upper_edge_follows_the_outline():
    """
    Regression: the shaded region must be a region under a curve, not a box.

    Two things made it read as a flat-topped rectangle. The cells are
    quantile-spaced, so the sparse high-torque tail gets ONE very wide bin
    (10.2-17.5 N.m on the bundled example) that tiles across every speed slice;
    and the unbinned outlier cells were drawn in the density colour as well as
    in the excursion layer, scattering shading above the outline.

    So each rect is clipped to the measured outline column by column, and the
    outlier cells are left to the layer that can show their duration.
    """
    act = db.load_actuator("robstride_00")
    jt = db.load_application("elbow_envelope_example")
    svg = charts.torque_speed(evaluate.evaluate(act, jt))

    # Density rects only: drop the legend swatch (7.2 square) and the gradient
    # strip (6.2 wide), neither of which lives in the plot area.
    rects = [(float(x), float(y), float(w), float(h)) for x, y, w, h in
             re.findall(r'<rect x="([0-9.]+)" y="([0-9.]+)" width="([0-9.]+)" '
                        r'height="([0-9.]+)" fill="#2E8B57"', svg)
             if abs(float(w) - 6.2) > 0.01
             and not (abs(float(w) - 7.2) < 0.01 and abs(float(h) - 7.2) < 0.01)]
    assert len(rects) > 100, len(rects)

    # Highest drawn point per x column; smaller y is higher on screen.
    cols = {}
    for x, y, w, h in rects:
        k = round(x / 10) * 10
        cols[k] = min(cols.get(k, 1e9), y)
    tops = [cols[k] for k in sorted(cols)]

    assert len(set(round(t) for t in tops)) >= 4, \
        f"the density has a flat top across {len(tops)} columns: {sorted(set(tops))}"
    assert max(tops) - min(tops) > 40, \
        f"the upper edge barely moves ({max(tops) - min(tops):.0f} px) -- it is " \
        f"not following the measured outline"


def test_envelope_chart3_plots_the_measured_window_ladder():
    """
    Chart 3's axes already suit the window ladder exactly: log duration against
    RMS-equivalent torque, read against the vendor endurance curve. Real
    measured excerpts at real durations beat a rest-scaled hypothetical.
    """
    act = db.load_actuator("robstride_00")
    jt = db.load_application("elbow_envelope_example")
    ev = evaluate.evaluate(act, jt)
    svg = charts.overload_endurance(ev)

    assert "measured session" in svg
    assert "binding:" in svg, "the timescale that binds must be called out"
    assert "as specified" not in svg, \
        "the synthetic rest-scaled family is replaced, not drawn alongside"


def test_envelope_duty_variants_sweep_throughput_not_rest():
    """
    You cannot scale rest you did not author. The captured work done more or
    less often per hour is the knob that does exist, and 'stationary' has to be
    a small physical threshold because a servo holding position still dithers.
    """
    act = db.load_actuator("robstride_00")
    env_variants = charts.duty_variants(evaluate.evaluate(
        act, db.load_application("elbow_envelope_example")))
    assert env_variants
    labels = " ".join(v[0] for v in env_variants)
    assert "throughput" in labels and "as captured" in labels, labels
    dfs = [v[2] for v in env_variants]
    assert max(dfs) - min(dfs) > 0.2, \
        f"the family collapsed to one duty factor: {dfs}. |w| < 1e-6 never " \
        f"matches measured stationary, so every cell read as 'moving'"

    # The authored path keeps its own wording.
    prof = charts.duty_variants(evaluate.evaluate(
        act, db.load_application("elbow_example")))
    assert "as specified" in " ".join(v[0] for v in prof)


def test_envelope_html_is_self_contained():
    """Same promise as the profile path: one file, no network."""
    act = db.load_actuator("robstride_00")
    jt = db.load_application("elbow_envelope_example")
    ev = evaluate.evaluate(act, jt)
    with tempfile.TemporaryDirectory() as d:
        path = charts.write_html(ev, os.path.join(d, "e.html"),
                                 report.render(ev))
        html = open(path).read()
    assert "<svg" in html and "</html>" in html
    stripped = html.replace('xmlns="http://www.w3.org/2000/svg"', "")
    for bad in ("http://", "https://", "<script", "<link", "@import"):
        assert bad not in stripped, f"chart html should not contain {bad}"


def test_envelope_from_log_refuses_a_flipped_torque_sign():
    """
    Gravity is a free independent check on the sign convention, and a flipped
    sign yields a plausible envelope that is wrong in every downstream number.
    """
    import math as _m
    import random
    from envelope_from_log import gravity_sign_check

    rng = random.Random(5)

    class _P1:
        grav_pos = []
        grav_tau = []

    # Holding torque opposing gravity across a swept angle: tau = -k*sin(theta)
    for _ in range(4000):
        th = rng.uniform(-1.2, 1.2)
        _P1.grav_pos.append(th)
        _P1.grav_tau.append(-4.0 * _m.sin(th) + rng.gauss(0, 0.05))
    verdict, r, n = gravity_sign_check(_P1, 0.0)
    assert verdict == "ok", (verdict, r)

    _P1.grav_tau = [-t for t in _P1.grav_tau]
    verdict, r, n = gravity_sign_check(_P1, 0.0)
    assert verdict == "flipped", (verdict, r)


if __name__ == "__main__":
    fns = [(n, f) for n, f in sorted(globals().items())
           if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in fns:
        try:
            fn()
            print(f"  ok    {name}")
        except Exception as e:
            failed += 1
            print(f"  FAIL  {name}: {type(e).__name__}: {e}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
