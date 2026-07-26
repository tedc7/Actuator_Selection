# actuator_eval

Evaluate candidate vendor actuators against the requirements of a specific robot joint.

Built around three ideas:

1. **Heat is the real limit**, so the thermal model is the centrepiece rather than an afterthought.
2. **Missing data must never block an answer.** Every parameter has a documented default and an uncertainty band. The analysis always runs; the report tells you which conclusions actually rested on a guess.
3. **Actuators and applications are separate, reusable files**, so evaluating a new actuator against five joints is five commands, not five spreadsheets.

---

## Quick start

```bash
python3 eval_actuator.py --list
python3 eval_actuator.py -a robstride_00 -j elbow_example
python3 eval_actuator.py -a robstride_00 -j elbow_example -n 1,2,3     # compare 1/2/3 in parallel
python3 eval_actuator.py -a robstride_00 -j elbow_example --ambient 55 --bus 36
python3 eval_actuator.py -a robstride_00 -j elbow_example --surface-limit 45
python3 eval_actuator.py -a robstride_00 -j elbow_example --set R_phase=0.42
python3 eval_actuator.py -a robstride_06 -j elbow_example --save --charts
```

`--save` writes the text report and `--charts` writes a self-contained HTML file
(no CDN, no JS) with three SVG plots. Both land **in the same directory as the
application file** so results never get separated from their inputs; pass
`--outdir` to send them elsewhere, or give `--charts` an explicit filename. With
`-n 1,2,3` you get one chart file per configuration.

No dependencies beyond the Python 3.9+ standard library. Tests: `python3 tests/test_smoke.py`.

As a library:

```python
from actuator_eval import db, evaluate, report

act   = db.load_actuator("robstride_00")
joint = db.load_application("elbow_example")
ev    = evaluate.evaluate(act, joint)
print(report.render(ev))
print(ev.verdict, ev.binding.name)
```

---

## Layout

```
actuator_eval/            the engine, importable as a package
  db/actuators/           vendor reference data -- public, reusable, PR-able
    _TEMPLATE.json
applications/             one file per joint on your robot -- GITIGNORED
  examples/               ...except these, so a fresh clone can run the above
    _TEMPLATE.json
eval_actuator.py          CLI
```

Two kinds of input file, and the distinction is the whole reason for the split:

- An **actuator** file is vendor data. It is reusable across every robot, worth
  reviewing, and belongs in git where it gets diffs and pull requests.
- An **application** file is one joint on *your* machine: payload, geometry, the
  bus voltage your robot actually supplies, the ambient inside your shell. None
  of it is reusable and much of it is proprietary, so `applications/` is
  gitignored and this repo stays publishable.

Putting an application fact in an actuator file makes it silently wrong for every
other robot that uses that actuator, which is the failure this layout prevents.

To start an application: copy `applications/examples/_TEMPLATE.json` into
`applications/`. A bare name is looked up in `applications/` first, then
`applications/examples/`, so a private file shadows an example of the same name;
an explicit path always wins. Set `ACTUATOR_EVAL_APPS_DIR` to keep applications
outside the repo entirely (a private repo, a synced drive).

---

## What belongs to the actuator, what belongs to the application

The split matters, because putting an application fact in the actuator file
makes it silently wrong for every other robot that uses that actuator.

| Actuator file (vendor property) | Application file (your robot) |
|---|---|
| `V_bus_min` / `V_bus_max` — range it will accept | `bus_voltage` — what your robot actually supplies |
| `V_bus_nom` — what the vendor characterised it at | `supply_current_limit` — optional; omit if not binding |
| `T_winding_max` — insulation class | `T_ambient` — inside your shell, not room temperature |
| `I_peak_rms` — the drive's own current limit | `max_surface_temp` — touch-temperature limit for operator safety |
| `Rth_wc` — winding to housing | `mounting` — how it's bolted in, which sets `Rth_ca` |

**Bus voltage.** Set what your robot supplies; the actuator declares what it
accepts and the two get checked against each other. Running above the vendor's
*nominal* is routine — it extends the speed envelope and nothing else. Higher
voltage buys speed, not torque and not thermal margin:

```
24V: stall 14.0 N.m/act, no-load 170 rpm
48V: stall 14.0 N.m/act, no-load 339 rpm
60V: stall 14.0 N.m/act, no-load 424 rpm
```

Running above the vendor's stated *maximum* is a different matter and the tool
will not clear it on thermal grounds. That limit is set by transistor and
DC-link capacitor breakdown ratings, and regenerative braking pushes the bus
*above* its resting value, so the transient is what the silicon actually sees.
Set `"accept_overvoltage": true` to acknowledge the risk and downgrade that
check from a failure to a warning; the explanation stays in the report either
way.

**Surface temperature.** The two-node model already computes housing
temperature, so `max_surface_temp` turns it into a pass/fail criterion. This is
frequently the real constraint on a human-adjacent robot and is invisible if you
only watch the winding: on the bundled elbow example the winding settles at a
comfortable 63 degC while the housing reaches 55 degC. ISO 13732-1 is the
relevant standard for touch limits; the threshold depends on material and
contact duration, so set the number from your own safety case. The model gives a
*lumped* housing temperature — real surfaces have hot spots near the stator, so
treat it as optimistic and confirm by measurement.

---

## Units

Getting a unit wrong is the easiest way to produce a confident, wrong answer, and
it is invisible in a bare JSON number. So every physical field is checked against
the dimension it is supposed to have and converted to canonical SI before any
physics runs.

Three ways to write a value:

```jsonc
"distance_joint_axis_to_CG": {"value": 11, "units": "in"} // 1. explicit  <- preferred
"distance_joint_axis_to_CG_mm": 280                      // 2. unit in the field name
"distance_joint_axis_to_CG": 0.28                        // 3. bare -> canonical SI assumed
```

Form 3 keeps quick edits quick, but every bare number is listed in the report's
**UNIT AUDIT** with a `!` marker, so an assumption never passes silently. Set
`"strict_units": true` in a file to reject form 3 outright.

A unit from the *wrong dimension* is always a hard error, never a warning:

```
UnitError: in application file 'elbow_example.json': unit 'kg' is a mass unit, but
this field expects length. Valid units here: m, mm, cm, in, ft
```

Canonical units: mass `kg`, length `m`, inertia `kg.m^2`, angle `rad`, time `s`,
torque `N.m`, angular velocity `rad/s`, temperature `degC`, resistance `ohm`.
Imperial is accepted throughout (`in`, `ft`, `lb`, `lbf.ft`, `oz.in`, `degF`).

---

## Charts

`--charts out.html` produces three plots, each answering something the numbers
answer badly:

Every chart describes **exactly one actuator configuration** &mdash; the same one
the text report describes, so the two can never disagree. To compare candidates,
generate one chart set per candidate and put them side by side.

1. **Torque-speed envelope** with that configuration's operating points scattered
   on top, split by phase of the S-curve move, and the peak demand called out. A table says "peak margin 2.8x"; the plot
   shows whether that margin is all at low speed and nearly gone at the top of
   the stroke.
2. **Thermal warm-up from cold, for a family of four duty cycles** on the same
   configuration. The motion is fixed by the task; what varies is the rest time
   between moves, which is the knob a designer actually has. Curves that flatten
   below the insulation limit run indefinitely; ones that cross it are annotated
   with how long that takes. Rest is not free &mdash; a gravity-loaded joint
   still draws holding current at zero speed, which is why the "2x rest" curve
   settles well above ambient.
3. **Margin bars** per criterion, so the binding constraint is obvious.

`charts.duty_variants(ev)` exposes the duty-cycle family if you want to drive it
yourself; it works whether the joint was defined by a motion profile or by
explicit `duty_segments`.

---

## Why this shape (and not a spreadsheet or a web app)

A spreadsheet is the obvious first instinct and it is the wrong tool here, for three specific reasons:

- The duty-cycle thermal model needs iteration to a fixed point and a runaway check. That is painful and fragile in cells.
- Provenance tracking ("this number is a guess, that one is measured") is the core feature, and spreadsheets have nowhere to put it except a comment nobody reads.
- Actuator-vs-joint is a matrix. You want to re-run 6 actuators against 9 joints after changing one assumption, and diff the result in git.

A web UI is the right *eventual* front end, but it is a presentation layer. Build the engine first, put a UI on it once the criteria have stopped changing. The recommended progression:

| Stage | Tool | Why |
|---|---|---|
| now | Python + JSON, CLI, SVG charts | fast to change while the criteria are still in flux |
| next | Streamlit / FastAPI over the same engine | sliders for the assumed parameters; ~100 lines |
| export | CSV / Markdown out of `report.py` | for people who want the numbers in a sheet |

With `--charts` covering the visual side, a web UI mostly buys interactive
sliders. Worth it eventually, not urgent.

Keeping the physics in a library means the UI never becomes the source of truth.

---

## Should there be an actuator database?

Yes, and it should be JSON files in git rather than a real database. Sourcing and entering vendor data is the genuinely expensive part of this problem, it is done once per actuator, and it needs review. Files give you diffs, blame, and pull requests. A schema is in `actuator_eval/db/actuators/_TEMPLATE.json`, and the matching one for applications is in `applications/examples/_TEMPLATE.json`.

The important rule: **leave out what you do not know.** An omitted field gets a documented estimate and appears in the report's assumption list and sensitivity sweep. A fabricated field looks like data and silently corrupts the verdict.

---

## What it checks

| Criterion | Question |
|---|---|
| Peak torque | Can it make the highest instantaneous torque the motion needs? |
| Speed at load | Can it reach top speed *while* making the torque required there? |
| Torque-speed envelope | Does every point of the duty cycle sit inside the voltage-limited envelope? |
| Trajectory following | Can it follow the commanded S-curve &mdash; the minimum-time move under the controller's `max_velocity` / `max_accel` / `max_jerk`? |
| Thermal (winding) | Does the winding stay below its insulation limit at periodic steady state? |
| Continuous torque | Is the duty-cycle RMS current within the thermally sustainable limit? |
| Drive current limit | Does the demanded phase current exceed the drive's own rating? |
| Inertia ratio | Does reflected rotor inertia dominate the load? (low is good for QDD) |
| Mass / cost budget | Does the parallel-actuator configuration fit the budget? |
| Bus voltage | Is the application's supply within the actuator's accepted range? |
| Surface temp | Does the housing stay under the touch-temperature limit? |
| Supply current | Optional: can the robot's supply deliver the peak demanded? |
| Backdrive torque | How hard is the joint to backdrive? |

Each returns a margin (capability / demand) plus a confidence level inherited from the weakest input it used.

---

## The model

**Electrical.** Torque-speed envelope from the steady-state PMSM phasor under `id = 0` field-oriented control:

```
Vq = R*Iq + we*lambda        Vd = -we*Lq*Iq        |V| <= Vbus*k/sqrt(3)
```

solved as a quadratic in `Iq`. Conventions are fixed once in `models.py`: currents are **RMS phase**, resistance is **line-to-neutral**, and

```
Kt_rotor [N.m/A_rms] = sqrt(3) * Ke [V_rms line-line per rad/s]
```

Mixing up any of these is a factor-of-2 or factor-of-sqrt(3) error, which is why the loader converts explicitly from whatever unit the datasheet used.

**Thermal.** Two-node lumped model (winding -> case -> ambient) with:

- copper resistivity rising at 0.393 %/K, which is a *positive feedback* loop and is why thermal runaway is detected explicitly rather than returned as a large number;
- magnet strength fading at about -0.11 %/K, so a hot motor needs more current for the same torque;
- iron loss split between hysteresis (~f) and eddy (~f²) terms;
- gearbox and bearing drag.

Steady state is solved on two timescales, because a case time constant of ~12 minutes against a 1.6 s duty cycle would otherwise need tens of thousands of cycles: the case node is driven by the cycle-average loss and found by fixed-point iteration, then a few cycles are integrated in detail to recover the within-cycle winding ripple.

`time_to_limit()` answers the burst question separately — an actuator that fails the continuous check may be entirely fine if the motion only happens for 30 seconds at a time.

---

## Measurement priority

The report ends with a sensitivity section listing the assumptions that flip a verdict within their own error band. In practice, for QDD modules, it is nearly always these two:

1. **Phase resistance.** Four-wire measurement at a known temperature. If you measure line-to-line, halve it. Half an hour of work, and it is the single largest source of uncertainty in the whole model.
2. **Thermal resistance to ambient, *as mounted in your robot*.** Run a known current into a locked rotor, log the case and winding temperature to equilibrium, divide. A bench number with the module sitting in open air will flatter you badly compared to the same part buried in a forearm.

Everything else is second order by comparison.

---

## Known limitations

- Single lumped winding node: no end-turn vs slot gradient, so short high-current transients are optimistic by roughly the winding time constant.
- No mutual heating between actuators sharing a joint. Two modules bolted to the same bracket warm each other; treat the current per-actuator result as slightly optimistic for `n > 1`.
- Gearbox efficiency is a constant, not a function of load and temperature. It is worse when cold and at light load.
- No inverter loss, no duty-cycle-dependent switching loss.
- Backdrive estimate is friction-based and ignores cogging torque.
- Surface temperature is the lumped housing node, so it understates local hot
  spots near the stator and the effect of any exposed metal fastener path.
- Regenerative bus rise is warned about but not simulated; if you run near the
  actuator's voltage ceiling, measure the bus during a hard decel.
- The thermal warm-up plot uses the cycle-average loss, so it shows the envelope
  of the transient rather than the within-cycle ripple (which the steady-state
  simulation does capture).
- `L_phase` defaults to a placeholder and only matters near the top of the speed range.
