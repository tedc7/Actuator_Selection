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
