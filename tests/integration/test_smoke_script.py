"""The post-deploy smoke script (used in the runbook) must pass against a healthy service."""

from scripts.smoke_test import main


def test_smoke_script_passes_against_running_service(base_url, capsys):
    assert main(["--base-url", base_url, "--max-latency-ms", "5000"]) == 0
    assert "FAIL" not in capsys.readouterr().out


def test_smoke_script_detects_wrong_release(base_url, capsys):
    assert main(["--base-url", base_url, "--expected-sha", "not-the-deployed-sha"]) == 1
    assert "build_sha matches release" in capsys.readouterr().out
