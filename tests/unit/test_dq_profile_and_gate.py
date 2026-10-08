import json

import pandas as pd
import pytest

from dq.__main__ import main as dq_main
from dq.gate import baseline_from_report, compare, write_baseline
from dq.profile import profile_column, profile_dataset


class TestProfile:
    def test_numeric_column(self):
        p = profile_column(pd.Series(["1", "2", "", "10"]))
        assert p["inferred_type"] == "numeric"
        assert (p["blank_count"], p["min"], p["max"]) == (1, 1.0, 10.0)
        assert p["blank_pct"] == 25.0

    def test_timestamp_column(self):
        p = profile_column(pd.Series(["2025-01-01 00:00:00", "2025-02-01 00:00:00"]))
        assert p["inferred_type"] == "timestamp"
        assert p["max"].startswith("2025-02-01")

    def test_low_cardinality_string_has_top_values(self):
        p = profile_column(pd.Series(["A", "A", "b"]))
        assert p["inferred_type"] == "string"
        assert p["top_values"] == {"A": 2, "b": 1}

    def test_empty_column(self):
        assert profile_column(pd.Series(["", " "]))["inferred_type"] == "empty"

    def test_dataset_profile_is_json_serialisable(self):
        df = pd.DataFrame({"a": ["1", "1"], "b": ["x", "x"]})
        prof = profile_dataset(df)
        assert prof["exact_duplicate_rows"] == 1
        json.dumps(prof)


def report(**rows: int) -> dict:
    return {
        "checks": [{"check_name": k, "rows_affected": v, "status": "fail" if v else "pass"} for k, v in rows.items()]
    }


BASELINE = {"checks": {"a": 100, "b": 0}}


class TestGate:
    BASE = BASELINE

    def test_no_change_passes(self):
        assert compare(report(a=100, b=0), self.BASE) == []

    def test_improvement_passes(self):
        assert compare(report(a=10, b=0), self.BASE) == []

    def test_growth_within_tolerance_passes(self):
        assert compare(report(a=109, b=0), self.BASE, tolerance_pct=10) == []

    def test_growth_beyond_tolerance_fails(self):
        [r] = compare(report(a=120, b=0), self.BASE, tolerance_pct=10)
        assert r.check_name == "a"

    def test_previously_passing_check_now_failing(self):
        [r] = compare(report(a=100, b=3), self.BASE)
        assert "previously passing" in r.reason

    def test_new_failing_check_without_baseline(self):
        assert compare(report(a=100, b=0, c=5), self.BASE)[0].check_name == "c"

    def test_errored_check_fails(self):
        rep = {"checks": [{"check_name": "a", "rows_affected": 0, "status": "error", "error": "boom"}]}
        assert "errored" in compare(rep, self.BASE)[0].reason

    def test_baseline_roundtrip(self, tmp_path):
        path = tmp_path / "b.json"
        write_baseline({"report_version": "1", **report(a=1)}, path)
        assert json.loads(path.read_text())["checks"] == {"a": 1}
        assert baseline_from_report(report(a=2))["checks"] == {"a": 2}


class TestCli:
    @pytest.fixture
    def raw_dir(self, tmp_path):
        from tests.unit.test_dq_checks import data, shipment

        raw = data([shipment(), shipment(shipment_id="SHP-00002", container_count="0")])
        raw.ports.to_csv(tmp_path / "ports.csv", index=False)
        raw.shipments.to_csv(tmp_path / "shipments.csv", index=False)
        raw.port_events.to_csv(tmp_path / "port_events.csv", index=False)
        return tmp_path

    def test_gate_passes_against_own_baseline(self, raw_dir, tmp_path):
        base, out = tmp_path / "baseline.json", tmp_path / "r.json"
        assert dq_main(["--input", str(raw_dir), "--output", str(out), "--update-baseline", str(base)]) == 0
        assert dq_main(["--input", str(raw_dir), "--output", str(out), "--baseline", str(base)]) == 0

    def test_gate_fails_on_regression(self, raw_dir, tmp_path):
        base = tmp_path / "baseline.json"
        base.write_text(json.dumps({"checks": {"shipments.container_count_zero": 0}}))
        assert dq_main(["--input", str(raw_dir), "--output", str(tmp_path / "r.json"), "--baseline", str(base)]) == 1

    def test_strict_mode(self, raw_dir, tmp_path):
        args = ["--input", str(raw_dir), "--output", str(tmp_path / "r.json"), "--no-profile"]
        assert dq_main(args) == 0
        assert dq_main([*args, "--fail-on-critical"]) == 0  # container_count_zero is only a warning


def test_render_markdown_from_report(tmp_path):
    from dq.render import render
    from dq.runner import build_report
    from tests.unit.test_dq_checks import data, shipment

    raw = data([shipment(), shipment(shipment_id="SHP-00002", weight_tons="-3")])
    raw.ports.to_csv(tmp_path / "ports.csv", index=False)
    raw.shipments.to_csv(tmp_path / "shipments.csv", index=False)
    raw.port_events.to_csv(tmp_path / "port_events.csv", index=False)
    md = render(build_report(tmp_path, include_profile=False))
    assert "# Data quality report" in md
    assert "`weight_non_positive`" in md
    assert "How detected" in md
