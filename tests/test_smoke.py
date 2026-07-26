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
