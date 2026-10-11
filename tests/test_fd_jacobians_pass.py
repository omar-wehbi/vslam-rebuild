"""The BA and pose-graph analytical Jacobians agree with finite differences (stage 01a).

Supersedes tests/test_fd_jacobians_expected_fail.py (stage 00b), which documented the
broken additive-rvec state.  Both harness probes are run as subprocesses exactly as the
verifier runs them (sys.executable, cwd = repo root, PYTHONPATH = repo root, --tol 1e-4)
and must exit 0 while still reporting METRIC lines.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _run_check(rel_path):
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT)
    return subprocess.run(
        [sys.executable, str(ROOT / rel_path), "--tol", "1e-4"],
        cwd=str(ROOT),
        env=env,
        capture_output=True,
        text=True,
    )


def _assert_pass(result):
    metric_lines = [line for line in result.stdout.splitlines() if line.startswith("METRIC")]
    for line in metric_lines:  # visible with pytest -s; useful when a bar is missed
        print(line)
    assert metric_lines, f"no METRIC lines\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    assert result.returncode == 0, (
        f"probe exited {result.returncode}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )


def test_fd_jacobian_ba_passes():
    _assert_pass(_run_check("system/verify/checks/fd_jacobian_ba.py"))


def test_fd_jacobian_posegraph_passes():
    _assert_pass(_run_check("system/verify/checks/fd_jacobian_posegraph.py"))
