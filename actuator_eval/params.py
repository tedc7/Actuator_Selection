"""
Parameter model with provenance.

The central design idea of this tool: every number carries a tag saying where it
came from and how much you should trust it. Nothing is ever "missing" in a way
that blocks the calculation -- an unknown parameter gets a documented default and
a wide uncertainty band, the analysis runs anyway, and the report tells you which
conclusions actually depended on the guess.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional


# Provenance ranks, worst to best. Used to propagate confidence through results.
VENDOR_MEASURED = "measured"      # you put it on a dyno / thermal chamber
VENDOR_SPEC = "vendor"            # published on the datasheet
VENDOR_DERIVED = "derived"        # computed from other vendor numbers
ESTIMATED = "estimated"           # scaling law / class-typical default
GUESS = "guess"                   # order-of-magnitude placeholder

_RANK = {VENDOR_MEASURED: 4, VENDOR_SPEC: 3, VENDOR_DERIVED: 2, ESTIMATED: 1, GUESS: 0}

# Default relative uncertainty (+/- fraction) by provenance, when not stated.
_DEFAULT_TOL = {
    VENDOR_MEASURED: 0.05,
    VENDOR_SPEC: 0.10,
    VENDOR_DERIVED: 0.15,
    ESTIMATED: 0.50,
    GUESS: 1.00,
}


@dataclass
class P:
    """A physical parameter: value, units, provenance, uncertainty, note."""

    value: float
    units: str = ""
    source: str = ESTIMATED
    tol: Optional[float] = None          # relative, +/- fraction
    note: str = ""
    name: str = ""                       # filled in by the owning model

    def __post_init__(self):
        if self.tol is None:
            self.tol = _DEFAULT_TOL.get(self.source, 0.5)

    # -- arithmetic conveniences so P behaves like a float in formulas ------
    def __float__(self) -> float:
        return float(self.value)

    def __mul__(self, o):
        return float(self) * float(o)

    __rmul__ = __mul__

    def __truediv__(self, o):
        return float(self) / float(o)

    def __rtruediv__(self, o):
        return float(o) / float(self)

    def __add__(self, o):
        return float(self) + float(o)

    __radd__ = __add__

    def __sub__(self, o):
        return float(self) - float(o)

    def __rsub__(self, o):
        return float(o) - float(self)

    def __pow__(self, o):
        return float(self) ** float(o)

    def __lt__(self, o):
        return float(self) < float(o)

    def __gt__(self, o):
        return float(self) > float(o)

    def __le__(self, o):
        return float(self) <= float(o)

    def __ge__(self, o):
        return float(self) >= float(o)

    # -- uncertainty -------------------------------------------------------
    @property
    def lo(self) -> float:
        return self.value * (1.0 - self.tol)

    @property
    def hi(self) -> float:
        return self.value * (1.0 + self.tol)

    @property
    def rank(self) -> int:
        return _RANK.get(self.source, 0)

    @property
    def is_assumed(self) -> bool:
        """True if this number is a default rather than real data."""
        return self.rank <= _RANK[ESTIMATED]

    def scaled(self, k: float) -> "P":
        """Copy with the value multiplied by k. Used for sensitivity sweeps."""
        return P(self.value * k, self.units, self.source, self.tol,
                 self.note, self.name)

    def __repr__(self):
        flag = "" if not self.is_assumed else "  <-- ASSUMED"
        return (f"{self.value:.6g} {self.units} [{self.source} "
                f"+/-{self.tol*100:.0f}%]{flag}")

    def describe(self) -> str:
        s = f"{self.name or '?'} = {self.value:.6g} {self.units}".rstrip()
        s += f"  ({self.source}, +/-{self.tol*100:.0f}%)"
        if self.note:
            s += f"  # {self.note}"
        return s


def as_P(x, units="", source=ESTIMATED, tol=None, note="", name="") -> P:
    """Coerce a float, dict, or P into a P."""
    if isinstance(x, P):
        if name and not x.name:
            x.name = name
        return x
    if isinstance(x, dict):
        p = P(
            value=float(x["value"]),
            units=x.get("units", units),
            source=x.get("source", source),
            tol=x.get("tol"),
            note=x.get("note", ""),
            name=x.get("name", name),
        )
        return p
    return P(float(x), units, source, tol, note, name)


def worst_source(*params: P) -> str:
    """The weakest provenance among the inputs -- i.e. how much to trust a result."""
    ps = [p for p in params if isinstance(p, P)]
    if not ps:
        return GUESS
    return min(ps, key=lambda p: p.rank).source


def assumed_among(*params: P):
    """List the parameters that are defaults rather than real data."""
    return [p for p in params if isinstance(p, P) and p.is_assumed]
