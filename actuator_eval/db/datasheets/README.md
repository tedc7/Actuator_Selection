# Datasheets (local only, not committed)

Raw vendor manuals backing the entries in `../actuators/`. This directory is
gitignored: the PDFs are several MB each against a repo that is otherwise well
under a megabyte, they do not diff or review usefully, and they are vendor
copyright, so a public Apache-2.0 repo is not the place to redistribute them.

The reviewable artifact is the JSON. Every number in an actuator file carries a
`source`, a `tol`, and a note naming the section it came from, so a reader can
check the derivation against their own copy of the manual without needing ours.

## Convention

Name each file after the actuator entry it supports:

    db/actuators/robstride_00.json   <->   db/datasheets/robstride_00.pdf

Each actuator entry records the sha256 of its PDF in a `datasheet` block, so a
vendor reissuing a document in place is a detectable event. Verify with:

    ./eval_actuator.py --check-datasheets

`NOT LOCAL` is expected here for any manual you have not fetched. `MISMATCH`
means the file on disk is not the one the entry was written from — re-read the
affected values before trusting them, then update the hash and
`entry_revision`.

When adding a datasheet, record its hash with:

    sha256sum db/datasheets/robstride_00.pdf

## Current contents

| File               | Document                              |
|--------------------|---------------------------------------|
| `robstride_00.pdf` | RobStride RS00 User Manual, rev 260713 |
| `robstride_01.pdf` | RobStride RS01 User Manual, rev 260713 |
| `robstride_02.pdf` | RobStride RS02 User Manual, rev 260713 |
| `robstride_06.pdf` | RobStride RS06 User Manual, rev 260713 |

Sourced from RobStride's own repository, which is more current than the website
download centre: <https://github.com/RobStride/Product_Information> (under
`Product Literature/RS00/` and so on).

**These manuals lie about their own version.** The 260713 revision still declares
"Version 1.0, Initial Release, November 25, 2025" in its section 7, identical to
the 251210 revision seven months earlier, despite real spec changes between them
(winding limits moved; the RS01 and RS02 figures swapped with each other). Never
trust the stated version -- diff the hash.

## Re-deriving a value

Text and spec tables extract cleanly:

    pdftotext -layout db/datasheets/robstride_00.pdf - | less

The T-N curves and the maximum-overload endurance tables are page images, not
text. Render the relevant pages to read them:

    pdftoppm -r 150 -f 6 -l 9 -png db/datasheets/robstride_00.pdf /tmp/rs00

The overload tables (torque vs. time-to-thermal-limit at 24 rpm, 25 degC
ambient) are transcribed into the `notes` of each actuator file. They are
measured thermal transient data and are the best available check on the
estimated `Rth_wc` / `Rth_ca` / `C_w` / `C_c` values the tool falls back on.
