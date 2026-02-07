"""Smoke tests for the main entrypoint and package imports."""

import subprocess
import sys


def test_entrypoint_runs():
    result = subprocess.run(
        [sys.executable, '-m', 'src.model.simulate_network'],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )  # Capture output for diagnostics on failure.
    assert result.returncode == 0, result.stderr.decode()


def test_imports():
    import src
    import src.analysis
    import src.model
    import src.io
    import src.preprocessing
    # Ensure modules import without issues
    assert True
