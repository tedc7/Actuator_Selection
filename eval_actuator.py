#!/usr/bin/env python3
"""
Command line front end.

    ./eval_actuator.py --list
    ./eval_actuator.py -a robstride_00 -j elbow_example
    ./eval_actuator.py -a robstride_00 -j elbow_example -n 1,2,3
    ./eval_actuator.py -a robstride_00 -j elbow_example --set R_phase=0.42 --set Rth_ca=2.1
    ./eval_actuator.py -a robstride_00 -j elbow_example --ambient 55 --bus 36
"""

from __future__ import annotations
import argparse
import copy
import sys

from actuator_eval import db, evaluate, report, charts
from actuator_eval.params import P, VENDOR_MEASURED


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-a", "--actuator", help="name in the db, or a path to a json file")
    ap.add_argument("-j", "--joint", help="name in the db, or a path to a json file")
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
    ap.add_argument("--charts", metavar="FILE.html", default=None,
                    help="also write an HTML file with torque-speed, thermal "
                         "and margin charts")
    args = ap.parse_args(argv)

    if args.list:
        print("actuators:")
        for a in db.list_actuators():
            print("   ", a)
        print("joints:")
        for j in db.list_joints():
            print("   ", j)
        return 0

    if not args.actuator or not args.joint:
        ap.error("need both --actuator and --joint (or use --list)")

    act, act_audit = db.load_actuator(args.actuator, with_audit=True)
    joint, joint_audit = db.load_joint(args.joint, with_audit=True)

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
        print(report.compare(evs))
        for t in texts:
            print(t)
        if args.charts:
            # One file per configuration: the charts describe a single actuator
            # configuration so that they always match the report beside them.
            import os
            base, ext = os.path.splitext(args.charts)
            for e, t in zip(evs, texts):
                out = f"{base}_{e.joint.n_actuators}x{ext or '.html'}"
                charts.write_html(e, out, t)
                print(f"charts for {e.joint.n_actuators}x written to {out}")
    else:
        ev = evaluate.evaluate(act, joint)
        ev.unit_audits = [act_audit, joint_audit]
        text = report.render(ev, verbose=not args.brief)
        print(text)
        if args.charts:
            charts.write_html(ev, args.charts, text)
            print(f"charts written to {args.charts}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
