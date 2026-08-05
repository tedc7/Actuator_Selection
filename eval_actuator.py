#!/usr/bin/env python3
"""
Command line front end.

    ./eval_actuator.py --list
    ./eval_actuator.py -a robstride_00 -j elbow_profile_example
    ./eval_actuator.py -a robstride_00 -j elbow_profile_example -n 1,2,3
    ./eval_actuator.py -a robstride_00 -j elbow_profile_example --set R_phase=0.42 --set Rth_ca=2.1
    ./eval_actuator.py -a robstride_00 -j elbow_profile_example --ambient 55 --bus 36

Results are written next to the application file that produced them, so a
report never gets separated from its inputs. Use --save to write the text
report (it always goes to stdout as well), --charts for the HTML charts, and
--outdir to send both somewhere else.
"""

from __future__ import annotations
import argparse
import copy
import hashlib
import os
import sys

from actuator_eval import db, evaluate, report, charts
from actuator_eval.params import P, VENDOR_MEASURED


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _check_datasheets() -> int:
    """
    Verify each db entry against the datasheet it claims to be based on.

    Vendors reissue manuals in place, keeping the same title and often the same
    stated version, so a filename is not an identity. The sha256 recorded in the
    entry is, which makes 'the vendor changed this document underneath us' a
    detectable event rather than a silent one.

    Exit status is non-zero if any entry is stale or unverifiable, so this can
    gate CI.
    """
    ds_dir = os.path.join(os.path.dirname(db.ACTUATOR_DIR), "datasheets")
    worst = 0
    for name in db.list_actuators():
        act = db.load_actuator(name)
        d = act.datasheet or {}
        if not d:
            print(f"  {name:16s} NO DATASHEET DECLARED")
            worst = max(worst, 1)
            continue

        recorded = d.get("sha256")
        fname = d.get("file")
        label = act.datasheet_id() or "(unnamed document)"
        if not fname or not recorded:
            print(f"  {name:16s} INCOMPLETE   {label}")
            print(f"  {'':16s}              declares no {'file' if not fname else 'sha256'}")
            worst = max(worst, 1)
            continue

        path = os.path.join(ds_dir, fname)
        if not os.path.exists(path):
            # Not an error: the PDFs are deliberately not distributed.
            print(f"  {name:16s} NOT LOCAL    {label}")
            print(f"  {'':16s}              {fname} absent; fetch it to verify")
            continue

        actual = _sha256(path)
        if actual == recorded:
            print(f"  {name:16s} OK           {label}")
        else:
            print(f"  {name:16s} MISMATCH     {label}")
            print(f"  {'':16s}              recorded {recorded[:16]}...")
            print(f"  {'':16s}              on disk  {actual[:16]}...")
            print(f"  {'':16s}              the vendor document has changed; "
                  "re-check the entry against it")
            worst = 2
    return worst


def _actuator_slug(act) -> str:
    """Short actuator tag for output filenames."""
    if act.source_path:
        return os.path.splitext(os.path.basename(act.source_path))[0]
    return act.name.replace(" ", "_").lower()


def _out_path(joint, act, outdir, suffix, ext):
    """
    Default output path: beside the application file unless overridden.

    Named application-first then actuator -- 'middle_joint_robstride_00.html' --
    so results for one application sort together and the pair that produced them
    is readable off the filename. Both slugs prefer the source FILENAME and fall
    back to the internal 'name' field for objects built in code. Any qualifier
    (an actuator count, or 'report') is appended after that stem rather than
    buried in the middle, so the leading application_actuator part is stable.
    """
    base = outdir or joint.output_dir()
    os.makedirs(base, exist_ok=True)
    stem = f"{joint.slug()}_{_actuator_slug(act)}"
    if suffix:
        stem = f"{stem}_{suffix}"
    return os.path.join(base, f"{stem}{ext}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-a", "--actuator", help="name in the db, or a path to a json file")
    ap.add_argument("-j", "--joint", "--application", dest="joint",
                    help="application name, or a path to a json file")
    ap.add_argument("-n", "--counts", default=None,
                    help="comma separated actuator counts to compare, e.g. 1,2,3")
    ap.add_argument("--set", action="append", default=[], metavar="FIELD=VALUE",
                    help="override an actuator parameter and mark it as measured")
    ap.add_argument("--ambient", type=float, default=None, help="ambient degC")
    ap.add_argument("--bus", type=float, default=None,
                    help="bus voltage actually available in this application (V)")
    ap.add_argument("--surface-limit", type=float, default=None,
                    help="max allowed housing surface temperature (degC)")
    ap.add_argument("--mounting", default=None,
                    help="free_air | bolted_plastic | bolted_metal | heatsunk")
    ap.add_argument("--envelope", default=None, metavar="NAME",
                    help="evaluate against this measured envelope instead of "
                         "the one the application names; use to A/B two captures "
                         "of the same joint")
    ap.add_argument("--motion-source", default=None,
                    choices=["envelope", "profile", "duty_segments", "auto"],
                    help="pin which block drives the evaluation. Default is the "
                         "highest one defined: envelope > profile > duty_segments")
    ap.add_argument("--list", action="store_true", help="list the database and exit")
    ap.add_argument("--check-datasheets", action="store_true",
                    help="verify each db entry against the datasheet it cites, and exit")
    ap.add_argument("--brief", action="store_true", help="omit per-criterion detail lines")
    ap.add_argument("--save", action="store_true",
                    help="also write the text report beside the application file")
    ap.add_argument("--charts", nargs="?", const=True, default=None,
                    metavar="FILE.html",
                    help="also write HTML charts; bare flag names the file "
                         "automatically and puts it beside the application file")
    ap.add_argument("--outdir", default=None,
                    help="write results here instead of beside the application file")
    args = ap.parse_args(argv)

    if args.list:
        print("actuators:")
        for a in db.list_actuators():
            print("   ", a)
        apps = db.list_applications()
        print("applications:")
        if not apps:
            print("    none found in", db.APPLICATION_DIR)
        for name, kind in apps:
            print(f"     {name}" + ("  (example)" if kind == "example" else ""))
        envs = db.list_envelopes()
        print("envelopes:")
        if not envs:
            print("    none found in", db.ENVELOPE_DIR)
        for name, kind in envs:
            print(f"     {name}" + ("  (example)" if kind == "example" else ""))
        return 0

    if args.check_datasheets:
        print("datasheet provenance:")
        rc = _check_datasheets()
        if rc:
            print("\nre-check any MISMATCH entry against the new document "
                  "before trusting its numbers.")
        return rc

    if not args.actuator or not args.joint:
        ap.error("need both --actuator and --joint (or use --list)")

    try:
        act, act_audit = db.load_actuator(args.actuator, with_audit=True)
        joint, joint_audit = db.load_application(args.joint, with_audit=True)
    except FileNotFoundError as e:
        ap.error(str(e))

    if args.ambient is not None:
        joint.T_ambient = args.ambient
    if args.mounting is not None:
        joint.mounting = args.mounting
    if args.bus is not None:
        joint.bus_voltage = args.bus
    if args.surface_limit is not None:
        joint.max_surface_temp = args.surface_limit

    if args.envelope is not None:
        try:
            env, _ = db.load_envelope_with_audit(args.envelope)
        except FileNotFoundError as e:
            ap.error(str(e))
        # Same cross-check the application loader applies: a joint-referred
        # envelope only describes the drivetrain it was captured on.
        j_ratio = float(joint.ratio) if joint.ratio is not None else 1.0
        if abs(env.ratio - j_ratio) > 1e-6 * max(1.0, abs(j_ratio)):
            ap.error(f"envelope '{args.envelope}' was captured at ratio "
                     f"{env.ratio:g} but application '{args.joint}' declares "
                     f"{j_ratio:g}; the capture does not describe this drivetrain")
        joint.envelope = env
    if args.motion_source is not None:
        joint.motion_source = args.motion_source
    try:
        joint.active_motion_source()
    except ValueError as e:
        ap.error(str(e))

    for s in args.set:
        if "=" not in s:
            ap.error(f"--set expects FIELD=VALUE, got {s!r}")
        k, v = s.split("=", 1)
        k = k.strip()
        if not hasattr(act, k):
            ap.error(f"unknown actuator field {k!r}")
        setattr(act, k, P(float(v), source=VENDOR_MEASURED, tol=0.05, name=k,
                          note="supplied on the command line"))

    # An explicit --charts path is honoured as given; the bare flag derives one.
    charts_path = None if args.charts in (None, True) else args.charts
    want_charts = args.charts is not None

    if args.counts:
        counts = [int(c) for c in args.counts.split(",")]
        evs = []
        for n in counts:
            j2 = copy.deepcopy(joint)
            j2.n_actuators = n
            e = evaluate.evaluate(copy.deepcopy(act), j2)
            e.unit_audits = [act_audit, joint_audit]
            evs.append(e)
        texts = [report.render(e, verbose=not args.brief) for e in evs]
        summary = report.compare(evs)
        print(summary)
        for t in texts:
            print(t)
        if args.save:
            out = _out_path(joint, act, args.outdir, "report", ".txt")
            with open(out, "w") as f:
                f.write(summary + "\n" + "\n".join(texts))
            print(f"report written to {out}")
        if want_charts:
            # One file per configuration: the charts describe a single actuator
            # configuration so that they always match the report beside them.
            for e, t in zip(evs, texts):
                if charts_path:
                    base, ext = os.path.splitext(charts_path)
                    out = f"{base}_{e.joint.n_actuators}x{ext or '.html'}"
                else:
                    out = _out_path(joint, act, args.outdir,
                                    f"{e.joint.n_actuators}x", ".html")
                charts.write_html(e, out, t)
                print(f"charts for {e.joint.n_actuators}x written to {out}")
    else:
        ev = evaluate.evaluate(act, joint)
        ev.unit_audits = [act_audit, joint_audit]
        text = report.render(ev, verbose=not args.brief)
        print(text)
        if args.save:
            out = _out_path(joint, act, args.outdir, "report", ".txt")
            with open(out, "w") as f:
                f.write(text)
            print(f"report written to {out}")
        if want_charts:
            out = charts_path or _out_path(joint, act, args.outdir,
                                           "", ".html")
            charts.write_html(ev, out, text)
            print(f"charts written to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
