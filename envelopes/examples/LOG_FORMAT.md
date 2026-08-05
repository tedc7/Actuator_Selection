# Joint log format

What to deliver so a controller log can be turned into an actuator-sizing envelope. This is the whole specification; nothing here depends on knowing how the sizing tool works.

A reference file that satisfies every requirement below is next to this document: **`joint_log_example.csv`**.

---

## 1. Minimum deliverable

One CSV file per joint per session, containing **three columns**:

| quantity | why it is needed |
|---|---|
| time | establishes the sample interval and detects dropouts |
| torque **or** current | the load being sized against |
| speed **or** position | where on the torque-speed envelope that load sat |

Everything else in this document is either a rule about how to write those columns, or an option that improves the result.

**Preferred:** send time, position, speed *and* torque. Logged speed is better than speed derived from position afterwards, and position is worth having anyway as a record of where the joint actually worked.

---

## 2. File format

| item | requirement |
|---|---|
| encoding | UTF-8 or ASCII |
| delimiter | **comma** — not semicolon, not tab, not fixed-width |
| line endings | LF or CRLF, either is fine; mixed is fine |
| final newline | optional |
| extension | `.csv` |
| header | **required**, first non-comment line, one name per column |
| comment lines | `#` lines at the **top of the file only** are skipped |
| blank lines | leading blank lines are skipped |
| column order | any — columns are located by name |
| extra columns | allowed and ignored; send whatever else you log |
| header whitespace | `t_s, tau_Nm` is accepted; padding is stripped |
| compression | none — send the file uncompressed |

Column **names are yours to choose**. They are mapped by name when the file is
processed, so `tau_cmd_Nm`, `joint_torque`, `tau` are all equally fine. Keep
them stable between sessions so the mapping can be reused.

A leading comment block is a good place to record robot id, joint id, firmware version, sample rate and capture timestamp. It is skipped by the reader, and it means the file still identifies itself if it gets separated from its metadata.

---

## 3. Rows

- One row per controller update. **Do not decimate, average, or downsample.**
  Summarising is what the tool does; doing it twice loses the short events that decide whether a drive faults.
- **Strictly increasing timestamps.** Rows that go backwards or repeat a timestamp are discarded and counted.
- No total-row, no summary line, no trailing footer.
- Missing values: leave the field **empty** rather than writing `NaN`, `null`, `-`, or a sentinel like `-9999`. Empty and `NaN` are both dropped; a sentinel is silently read as a real measurement and corrupts the result.
- Numbers may be plain (`1.234`) or scientific (`1.234e-3`). No thousands separators, no units embedded in the value, no quoting needed.

---

## 4. Units

**Units are never assumed. They are declared once, out of band, when the file is processed** — they are not read from the header, and there is no default.

Tell us, per column, which unit it is in. Encoding it in the column name is the easiest way to make that unambiguous and self-documenting (`tau_Nm`, `q_rad`, `t_s`), but the name is a convention for humans; the declaration is what the
tool acts on.

Accepted units:

| quantity | accepted units |
|---|---|
| time | `s`, `ms`, `min`, `hr` |
| position | `rad`, `deg`, `rev`, `arcmin`, `arcsec` |
| speed | `rad/s`, `rpm`, `deg/s`, `rev/s` |
| torque | `N.m`, `Nm`, `N.cm`, `mN.m`, `kgf.cm`, `kgf.m`, `lbf.in`, `lbf.ft`, `oz.in` |
| current | `A`, `A_rms`, `mA` |

**Be consistent within a file.** One unit per column for the whole session.

---

## 5. Sampling

| item | requirement |
|---|---|
| rate | the controller's own update rate; **1 kHz is ideal**, 200 Hz is a sensible floor |
| regularity | fixed rate — the interval is taken as the median of observed gaps |
| jitter | small timestamp jitter is fine and expected |
| **max gap** | **any interval more than 5x the median counts as a dropout**, not as one long sample |
| coverage | at least **95%** of the session must be present, or the capture is refused |

The gap rule matters. A 2 s dropout recorded as a single 2 s row would be read as the joint genuinely holding that torque for two seconds, which injects phantom heating. Recorded as a gap, it is excluded and reported. So: **just stop writing rows during a dropout** — do not backfill, do not hold the last value, do not interpolate.

If the log carries no time column at all, the fixed update interval must be stated when the file is processed instead.

---

## 6. What the values must mean

**Everything is referred to the JOINT**, after any belt, linkage or extra reduction between the actuator output and the joint itself — not to the motor shaft. This is what allows one capture to be evaluated against different candidate actuators, and against different numbers of them.

If your controller reports motor-side quantities, either convert before writing, or tell us the ratio and we will.

- **torque** — the torque at the joint.
- **speed** — joint angular velocity.
- **position** — joint angle.
- **current** — only if torque is unavailable. Converting current to torque needs the actuator's torque constant, which makes the log actuator-specific; send torque if you possibly can.

Record the raw commanded or measured signal. **Do not filter, smooth, clip, or deadband it.** Spike rejection happens during processing, where it can be reported and reversed; a value already smoothed on the robot cannot be recovered.

---

## 7. Signs

**Send whatever your controller reports.** Signs are worked out on our side, so there is nothing to configure, convert, or agree in advance.

Just don't *change* convention partway through a file, and don't post-process one signal without the other. Beyond that, values may be positive or negative however your firmware happens to define them.

---

## 8. What to capture

The envelope is only as representative as the session. A log of the robot sitting still sizes an actuator optimistically.

- **Real duty, not a demo.** Normal production motion, at production speed, with the real payload — including the awkward parts, not just the easy ones.
- **Long enough to include the worst case.** Aim for **20 minutes or more**; an hour is better. Short captures cannot describe the slow thermal behaviour that a housing actually integrates over.
- **Include the idle.** Rest between cycles is part of the duty and is what makes the difference between an actuator that survives and one that cooks. Do not trim it out, and do not stitch the busy parts together.
- **One file per joint.** Do not interleave joints in one file.
- **Note anything unusual** — a collision, an e-stop, an operator intervention — in the comment block or a covering note. These show up as excursions and it helps to know whether they were real.

Multiple files from one session are fine; send them in order and say so.

---

## 9. Checklist

- [ ] One CSV per joint per session, comma-delimited, UTF-8
- [ ] Header row naming every column
- [ ] Columns for time, torque (or current), and speed and/or position
- [ ] Every value joint-referred, unfiltered, one unit per column
- [ ] Units for each column stated in the covering note
- [ ] One sign convention per file, applied to every column, start to finish
- [ ] Timestamps strictly increasing; gaps left as gaps, never backfilled
- [ ] Full session including idle; >= 20 min of real duty
- [ ] Robot id, joint id, firmware, sample rate, capture date recorded
- [ ] Anything unusual during the capture noted

---

## 10. Reference file

`joint_log_example.csv` in this directory — 12 seconds at 1 kHz of the bundled elbow example, including two motion cycles, holding periods, PI jitter and one 8 ms hard stop. It satisfies every requirement above.

```
# joint_log_example.csv -- reference format for actuator_eval envelopes
# robot=arm-004 joint=elbow fw=rs-fw-1.4.2 rate=1000Hz captured=2026-07-28T09:14:00Z
# units: t_s=s  q_rad=rad  qd_rad_s=rad/s  tau_Nm=N.m  (all joint-referred)
t_s,q_rad,qd_rad_s,tau_Nm
0.0000,-0.349066,0.00000,-1.37509
0.0010,-0.349050,0.01571,-1.15666
0.0020,-0.349019,0.03142,-0.99767
```

It is processed with:

```bash
./envelope_from_log.py --log envelopes/examples/joint_log_example.csv \
    --map   time=t_s,position=q_rad,speed=qd_rad_s,torque=tau_Nm \
    --units time=s,position=rad,speed=rad/s,torque=N.m \
    --torque-source torque_sensor \
    --name my_joint --out envelopes/my_joint.json
```

Send a short sample first — a minute or two is plenty — and it can be run through end to end before anyone commits to a long capture.
