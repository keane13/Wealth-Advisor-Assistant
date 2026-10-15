"""Human-in-the-loop review.

The checkpoint sits between analysis and recommendations. That is the critical step:
an unvalidated false positive (e.g. a planned house deposit flagged as a suspicious
wire) would otherwise flow straight into advice. Only HIGH/CRITICAL findings go to
review, so the human's attention is spent where it matters.

A reviewer may confirm a finding, dismiss it (and optionally have the system remember
the dismissal), or override its severity. Any finding left undecided stays *pending*,
and the system never assumes approval by default.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from ..domain.models import Finding, Severity

ACTIONS = {"confirm", "dismiss", "set_severity"}


@dataclass(frozen=True)
class ReviewDecision:
    finding_id: str
    action: str                    # confirm | dismiss | set_severity
    note: str = ""
    severity: str | None = None    # required for set_severity
    remember: bool = False         # dismiss only: suppress this finding in future runs

    def validate(self) -> str | None:
        """Return an error message, or None if the decision is well-formed."""
        if self.action not in ACTIONS:
            return f"unknown action {self.action!r}"
        if self.action == "set_severity":
            try:
                Severity(str(self.severity))
            except ValueError:
                return f"invalid severity {self.severity!r}"
        return None


class Reviewer(ABC):
    name: str = "reviewer"

    @abstractmethod
    def review(self, items: list[Finding]) -> list[ReviewDecision]:
        """Return decisions for any subset of ``items``. Undecided items remain pending."""


class DeferredReviewer(Reviewer):
    """No human is available now (batch/non-interactive run). Everything stays pending."""

    name = "deferred"

    def review(self, items: list[Finding]) -> list[ReviewDecision]:
        return []


class AutoConfirmReviewer(Reviewer):
    """Confirms everything. Only for demos and CI. It is labelled as such in the report."""

    name = "auto-confirm (non-human)"

    def review(self, items: list[Finding]) -> list[ReviewDecision]:
        return [ReviewDecision(f.id, "confirm", note="auto-confirmed (no human review)") for f in items]


class FileReviewer(Reviewer):
    """Decisions prepared in a JSON file. Reproducible, and lets review happen asynchronously.

    A decision targets one finding (``finding_id``) or every finding of a ``rule``::

        {"reviewer": "j.doe", "decisions": [
            {"finding_id": "F-1a2b3c4d5e", "action": "dismiss", "note": "...", "remember": true},
            {"rule": "spending_spike", "action": "confirm"}
        ]}
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self.name = "file"

    def review(self, items: list[Finding]) -> list[ReviewDecision]:
        try:
            doc = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ReviewUnavailable(f"cannot read review decisions from {self.path}: {exc}") from exc
        if not isinstance(doc, dict) or not isinstance(doc.get("decisions"), list):
            raise ReviewUnavailable(f"{self.path} must contain an object with a 'decisions' list")
        self.name = f"file:{doc.get('reviewer', 'unknown')}"

        decisions: list[ReviewDecision] = []
        for entry in doc["decisions"]:
            if not isinstance(entry, dict):
                continue
            targets = ([entry["finding_id"]] if entry.get("finding_id")
                       else [f.id for f in items if f.rule == entry.get("rule")])
            for fid in targets:
                decisions.append(ReviewDecision(
                    finding_id=str(fid), action=str(entry.get("action", "")), note=str(entry.get("note", "")),
                    severity=entry.get("severity"), remember=bool(entry.get("remember", False)),
                ))
        return decisions


class ConsoleReviewer(Reviewer):
    """Interactive terminal review. I/O is injectable for testing."""

    name = "console"

    def __init__(self, input_fn: Callable[[str], str] = input, output_fn: Callable[[str], None] = print) -> None:
        self.input = input_fn
        self.output = output_fn

    def review(self, items: list[Finding]) -> list[ReviewDecision]:
        self.output(f"\n=== Human review required: {len(items)} high-severity finding(s) ===")
        decisions: list[ReviewDecision] = []
        for n, f in enumerate(items, 1):
            self.output(f"\n[{n}/{len(items)}] {f.id}  {f.severity.value.upper()}  {f.title}\n    {f.detail}")
            try:
                while True:
                    choice = self.input("    [c]onfirm, [d]ismiss, [s]everity override, [k]eep pending > ").strip().lower()
                    if choice in {"c", "d", "s", "k"}:
                        break
                    self.output("    please enter c, d, s or k")
                if choice == "k":
                    continue
                note = self.input("    note (optional) > ").strip()
                if choice == "c":
                    decisions.append(ReviewDecision(f.id, "confirm", note))
                elif choice == "d":
                    remember = self.input("    remember as expected for future runs? [y/N] > ").strip().lower() == "y"
                    decisions.append(ReviewDecision(f.id, "dismiss", note, remember=remember))
                else:
                    sev = self.input("    new severity (low/medium/high/critical) > ").strip().lower()
                    decisions.append(ReviewDecision(f.id, "set_severity", note, severity=sev))
            except EOFError:  # stdin closed: stop asking; remaining items stay pending
                self.output("    (input closed; remaining findings left pending)")
                break
        return decisions


class ReviewUnavailable(Exception):
    """The reviewer could not be reached; the checkpoint degrades to 'pending'."""
