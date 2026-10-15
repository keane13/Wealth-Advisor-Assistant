"""Command-line interface.

    python -m wealth_assistant run C001                     # full review, interactive HITL if a TTY
    python -m wealth_assistant run C001 --review file --decisions samples/review_C001.json
    python -m wealth_assistant run C002 --crm-outage        # resilience demo
    python -m wealth_assistant run C001 --plan anomaly_scan
    python -m wealth_assistant list
    python -m wealth_assistant memory show C001

Exit codes: 0 completed, 2 completed but pending human review, 1 failed.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from .agents.orchestrator import PLANS
from .app import build_assistant
from .config import Settings
from .domain.validation import is_valid_client_id
from .hitl.review import AutoConfirmReviewer, ConsoleReviewer, DeferredReviewer, FileReviewer, Reviewer
from .logging_setup import configure_logging
from .memory.long_term import JsonFileMemoryStore
from .tools.client_data import list_client_ids

EXIT_CODES = {"completed": 0, "completed_with_warnings": 0, "pending_review": 2, "failed": 1}


def _make_reviewer(mode: str, decisions: Path | None) -> Reviewer:
    if mode == "auto":
        mode = "interactive" if sys.stdin.isatty() else "defer"
    if mode == "interactive":
        return ConsoleReviewer(output_fn=lambda s: print(s, file=sys.stderr))
    if mode == "file":
        if decisions is None:
            raise SystemExit("--review file requires --decisions PATH")
        return FileReviewer(decisions)
    if mode == "confirm-all":
        return AutoConfirmReviewer()
    return DeferredReviewer()


def _fmt(value: Any, suffix: str = "") -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:,.1f}{suffix}" if suffix else f"{value:,.2f}"
    return f"{value}{suffix}"


def _print_summary(report: dict[str, Any], path: Path) -> None:
    out = sys.stdout
    client = report.get("client", {})
    print(f"\n{'=' * 72}", file=out)
    print(f"Wealth Advisor Assistant | {report['client_id']} {client.get('name', '')} | plan={report['plan']}", file=out)
    print(f"status: {report['status'].upper()}   run_id: {report['run_id']}", file=out)
    print("=" * 72, file=out)

    dq = report.get("data_quality")
    if dq:
        print(f"Sources: {dq['sources']}  | rejected records: {dq['rejected_records']}", file=out)
    metrics = report.get("metrics", {})
    cf, risk = metrics.get("cashflow", {}), metrics.get("risk", {})
    if cf or risk:
        print(f"Net worth {_fmt(risk.get('net_worth'))} | income/mo {_fmt(cf.get('avg_monthly_income'))} | "
              f"spend/mo {_fmt(cf.get('avg_monthly_spend'))} | savings {_fmt(cf.get('savings_rate_pct'), '%')} | "
              f"equity {_fmt(risk.get('equity_pct'), '%')} | liquidity {_fmt(risk.get('liquidity_months'), ' mo')}",
              file=out)

    findings = report.get("findings", [])
    print(f"\nFindings ({len(findings)} active, {len(report.get('suppressed_findings', []))} suppressed):", file=out)
    for f in findings:
        review = f" [review: {f['review']['action']}]" if f.get("review") else ""
        print(f"  {f['severity'].upper():<8} {f['id']}  {f['title']}{review}", file=out)
    for f in report.get("suppressed_findings", []):
        print(f"  (suppressed) {f['id']}  {f['title']}: {f['suppression_reason']}", file=out)

    hr = report.get("human_review")
    if hr:
        detail = (f" (reviewer: {hr['reviewer']}, {len(hr['decisions'])} decided, "
                  f"{len(hr['pending_finding_ids'])} pending)") if hr["status"] != "not_required" else ""
        print(f"\nHuman review: {hr['status']}{detail}", file=out)

    recs = report.get("recommendations", [])
    if recs:
        print("\nRecommendations:", file=out)
        for r in recs:
            flag = " (awaiting validation)" if r["status"] == "awaiting_validation" else ""
            print(f"  P{r['priority']} {r['title']}{flag}", file=out)

    summary = report.get("summary")
    if summary:
        print(f"\nSummary [{summary['source']}]:\n  {summary['text']}", file=out)
    if report.get("errors"):
        print(f"\nErrors: {report['errors']}", file=out)
    print(f"\nFull report: {path}", file=out)


def cmd_run(args: argparse.Namespace) -> int:
    settings = Settings.from_env(
        crm_outage=True if args.crm_outage else None,
        crm_failure_rate=args.crm_failure_rate,
        use_llm=True if args.llm else None,
        random_seed=args.seed,
    )
    configure_logging(settings.log_dir, level="DEBUG" if args.verbose else "INFO", console=not args.quiet)
    assistant = build_assistant(settings, reviewer=_make_reviewer(args.review, args.decisions))
    report = assistant.review_client(args.client_id, plan=args.plan)

    output_dir = Path(args.output_dir) if args.output_dir else settings.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    safe_id = args.client_id if report["status"] != "failed" or report.get("client") else "invalid"
    path = output_dir / f"report_{safe_id}_{report['run_id']}.json"
    path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    _print_summary(report, path)
    return EXIT_CODES.get(report["status"], 1)


def cmd_list(args: argparse.Namespace) -> int:
    settings = Settings.from_env()
    for cid in list_client_ids(settings.data_dir / "clients"):
        print(cid)
    return 0


def cmd_memory(args: argparse.Namespace) -> int:
    if not is_valid_client_id(args.client_id):
        print(f"invalid client id {args.client_id!r}", file=sys.stderr)
        return 1
    store = JsonFileMemoryStore(Settings.from_env().memory_dir)
    if args.action == "clear":
        store.clear(args.client_id)
        print(f"long-term memory cleared for {args.client_id}")
    else:
        print(json.dumps({"summaries": store.summaries(args.client_id),
                          "crm_cache": store.cached_crm_profile(args.client_id),
                          "suppressions": store.suppressions(args.client_id)}, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="wealth-assistant", description="Multi-agent Wealth Advisor Assistant")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="run an advisory review for a client")
    run.add_argument("client_id")
    run.add_argument("--plan", choices=sorted(PLANS), default="full_review")
    run.add_argument("--review", choices=["auto", "interactive", "file", "defer", "confirm-all"], default="auto",
                     help="human-in-the-loop mode (auto = interactive on a TTY, otherwise defer)")
    run.add_argument("--decisions", type=Path, help="JSON file with review decisions (for --review file)")
    run.add_argument("--crm-outage", action="store_true", help="simulate a full CRM outage")
    run.add_argument("--crm-failure-rate", type=float, help="probability of a transient CRM failure per call")
    run.add_argument("--llm", action="store_true", help="write the summary with Claude (needs ANTHROPIC_API_KEY)")
    run.add_argument("--seed", type=int, help="seed for simulated CRM failures (reproducible runs)")
    run.add_argument("--output-dir", help="where to write the JSON report (default: ./output)")
    run.add_argument("-v", "--verbose", action="store_true", help="show debug logs (tool calls) on the console")
    run.add_argument("-q", "--quiet", action="store_true", help="no console logs (file logs are still written)")
    run.set_defaults(func=cmd_run)

    lst = sub.add_parser("list", help="list available client ids")
    lst.set_defaults(func=cmd_list)

    mem = sub.add_parser("memory", help="inspect or clear long-term memory")
    mem.add_argument("action", choices=["show", "clear"])
    mem.add_argument("client_id")
    mem.set_defaults(func=cmd_memory)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
