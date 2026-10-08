"""Render a Markdown summary of a built warehouse (used for the CI job summary).

python scripts/pipeline_summary.py data/warehouse/supply_chain.duckdb >> "$GITHUB_STEP_SUMMARY"
"""

from __future__ import annotations

import json
import sys

import duckdb


def render(db_path: str) -> str:
    with duckdb.connect(db_path, read_only=True) as con:
        run_id, finished, version, counts, fingerprint = con.execute(
            "SELECT run_id, finished_at, pipeline_version, row_counts, content_fingerprint FROM meta.pipeline_run"
        ).fetchone()
        rules = con.execute("SELECT rule, description, violations FROM meta.validation_results").fetchall()
        reasons = con.execute(
            "SELECT 'shipments' AS ds, quarantine_reason, count(*) FROM quarantine.shipments GROUP BY ALL "
            "UNION ALL SELECT 'port_events', quarantine_reason, count(*) FROM quarantine.port_events GROUP BY ALL "
            "ORDER BY 1, 3 DESC"
        ).fetchall()
        flags = con.execute(
            "SELECT f, count(*) FROM (SELECT unnest(dq_flags) AS f FROM curated.shipments) GROUP BY 1 ORDER BY 2 DESC"
        ).fetchall()

    counts = json.loads(counts)
    out = [
        f"### Pipeline run `{run_id[:12]}` (v{version}, finished {finished:%Y-%m-%d %H:%M:%S} UTC)",
        f"Content fingerprint: `{fingerprint[:16]}…`",
        "",
        "| Dataset | raw | curated | quarantine |",
        "|---|---:|---:|---:|",
    ]
    for ds in ("ports", "shipments", "port_events"):
        out.append(
            f"| {ds} | {counts.get(f'raw.{ds}', 0):,} | {counts.get(f'curated.{ds}', 0):,} "
            f"| {counts.get(f'quarantine.{ds}', '–') if ds != 'ports' else '–'} |"
        )
    passed = sum(v == 0 for _, _, v in rules)
    out += ["", f"**Data contract:** {passed}/{len(rules)} rules passed", "", "| Rule | Violations |", "|---|---:|"]
    out += [f"| `{r}` | {v} |" for r, _, v in rules]
    out += ["", "**Quarantine reasons**", "", "| Dataset | Reason | Rows |", "|---|---|---:|"]
    out += [f"| {ds} | `{reason}` | {n} |" for ds, reason, n in reasons]
    out += ["", "**Fixes / flags applied to curated shipments**", "", "| Flag | Rows |", "|---|---:|"]
    out += [f"| `{f}` | {n} |" for f, n in flags]
    return "\n".join(out) + "\n"


if __name__ == "__main__":
    print(render(sys.argv[1] if len(sys.argv) > 1 else "data/warehouse/supply_chain.duckdb"))
