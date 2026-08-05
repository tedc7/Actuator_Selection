"""
actuator_eval -- evaluate candidate vendor actuators against a robot joint.

    from actuator_eval import db, evaluate, report

    act   = db.load_actuator("robstride_00")
    joint = db.load_application("elbow_profile_example")
    ev    = evaluate.evaluate(act, joint)
    print(report.render(ev))

Two kinds of input file, and the split matters:

  * ACTUATORS are vendor reference data -- public, reviewable, reusable across
    every robot. They live in the package at db/actuators/ and ship with the
    repo so they can be improved by pull request.

  * APPLICATIONS describe one joint on YOUR robot: payload, geometry, the bus
    voltage your robot actually supplies, the ambient temperature inside your
    shell. None of that is reusable and much of it is proprietary, so it lives
    outside the package in applications/ , which is gitignored. Reports are
    written next to the application file that produced them.

Putting an application fact in an actuator file makes it silently wrong for
every other robot that uses that actuator, which is why they are separate.
"""

from . import (params, units, envelope, models, physics, db, evaluate, report,
               charts)

__all__ = ["params", "units", "envelope", "models", "physics", "db", "evaluate",
           "report", "charts"]
