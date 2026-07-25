#!/usr/bin/env python3
"""
Command line front end.

    ./eval_actuator.py --list
    ./eval_actuator.py -a robstride_00 -j elbow_example
    ./eval_actuator.py -a robstride_00 -j elbow_example -n 1,2,3
    ./eval_actuator.py -a robstride_00 -j elbow_example --set R_phase=0.42 --set Rth_ca=2.1
    ./eval_actuator.py -a robstride_00 -j elbow_example --ambient 55 --bus 36

Results are written next to the application file that produced them, so a
report never gets separated from its inputs. Use --save to write the text
report (it always goes to stdout as well), --charts for the HTML charts, and
--outdir to send both somewhere else.
"""

from __future__ import annotations
import argparse
import copy
import os
import sys

from actuator_eval import db, evaluate, report, charts
from actuator_eval.params import P, VENDOR_MEASURED


def _actuator_slug(act) -> str:
    """Short actuator tag for output filenames."""
    if act.source_path:
        return os.path.splitext(os.path.basename(act.source_path))[0]
    return act.name.replace(" ", "_").lower()


def _out_path(joint, act, outdir, suffix, ext):
    """Default output path: beside the application file unless overridden."""
    base = outdir or joint.output_dir()
    os.makedirs(base, exist_ok=True)
    return os.path.join(base, f"{joint.slug()}_{suffix}_{_actuator_slug(act)}{ext}")


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
    ap.add_argument("--list", action="store_true", help="list the database and exit")
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
        return 0

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
                                    f"charts_{e.joint.n_actuators}x", ".html")
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
                                           f"charts_{joint.n_actuators}x", ".html")
            charts.write_html(ev, out, text)
            print(f"charts written to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
