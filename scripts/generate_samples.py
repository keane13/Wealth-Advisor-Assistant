"""Regenerate the committed sample outputs in samples/output.

The scenarios run against a throwaway memory directory, so they are independent of your
local .memory state:

1. C001 with the reviewer decisions in samples/review_C001.json -> completed
2. C001 again during a CRM outage -> cached profile, remembered dismissal, run history
3. C002 with no reviewer available -> pending_review, compliance escalation
4. C003 messy data -> record-level rejection, degraded checks
5. C004 healthy and not in CRM -> no findings
6. C999 unknown client -> clean failure

    python scripts/generate_samples.py
"""

from __future__ import annotations

import json
import sys
import tempfile
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from wealth_assistant.app import build_assistant  # noqa: E402
from wealth_assistant.config import Settings  # noqa: E402
from wealth_assistant.hitl.review import DeferredReviewer, FileReviewer  # noqa: E402
from wealth_assistant.logging_setup import configure_logging  # noqa: E402
from wealth_assistant.memory.long_term import JsonFileMemoryStore  # noqa: E402

OUT = ROOT / "samples" / "output"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        log_dir = tmp_path / "logs"
        configure_logging(log_dir, console=False)
        settings = Settings(memory_dir=tmp_path / "memory", log_dir=log_dir, crm_latency_s=0.0, random_seed=1)
        memory = JsonFileMemoryStore(settings.memory_dir)

        scenarios = [
            ("01_C001_reviewed", "C001", settings, FileReviewer(ROOT / "samples" / "review_C001.json")),
            ("02_C001_crm_outage_second_run", "C001", replace(settings, crm_outage=True), DeferredReviewer()),
            ("03_C002_pending_review", "C002", settings, DeferredReviewer()),
            ("04_C003_messy_data", "C003", settings, DeferredReviewer()),
            ("05_C004_healthy_not_in_crm", "C004", settings, DeferredReviewer()),
            ("06_C999_unknown_client", "C999", settings, DeferredReviewer()),
        ]
        for name, client, cfg, reviewer in scenarios:
            report = build_assistant(cfg, reviewer, memory).review_client(client)
            (OUT / f"{name}.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
            print(f"{name}: {report['status']}")

        (OUT / "memory_C001.json").write_text(
            (settings.memory_dir / "C001.json").read_text(encoding="utf-8"), encoding="utf-8")
        lines = (log_dir / "run.jsonl").read_text(encoding="utf-8").splitlines()
        (OUT / "run_log.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
        configure_logging(None, console=False)  # release the log file before the temp dir is removed


if __name__ == "__main__":
    main()
