# Applications

An **application** is one joint on one robot: what it has to move, how fast, on
what bus voltage, at what internal ambient temperature. One file per joint.

This directory is **gitignored**. Application files describe real hardware and
the reports generated from them are design data, so nothing here reaches the
public repo — with one deliberate exception, `examples/`, which ships with the
repo so a fresh clone can run the quick start.

```
applications/
  examples/          <- public, tracked, safe to share
    elbow_example.json
    _TEMPLATE.json
  my_robot_elbow.json    <- yours, ignored by git
  my_robot_elbow_report_robstride_00.txt
```

## Why these are not in the actuator database

Actuator files are vendor reference data: public, reusable, worth reviewing by
pull request. Application files are the opposite — specific to your machine and
useful to nobody else. Mixing the two is how an application fact ends up in an
actuator file, where it is then silently wrong for every other robot using that
actuator.

## Getting started

```bash
cp applications/examples/_TEMPLATE.json applications/my_elbow.json
$EDITOR applications/my_elbow.json
./eval_actuator.py -a robstride_00 -j my_elbow --save --charts
```

A bare name is looked up in `applications/` first, then `applications/examples/`,
so a private file shadows an example of the same name. An explicit path always
wins:

```bash
./eval_actuator.py -a robstride_00 -j ~/design/shoulder.json --save
```

`--save` and `--charts` write results into the same directory as the application
file. Use `--outdir` to send them elsewhere.

## Keeping applications outside this repo entirely

Set `ACTUATOR_EVAL_APPS_DIR` and the tool looks there instead — useful if you
want them in a private repo with their own history:

```bash
export ACTUATOR_EVAL_APPS_DIR=~/robot-design/applications
./eval_actuator.py --list
```
