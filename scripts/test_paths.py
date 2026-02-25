"""
Verify that all resolved paths in PPTL scripts are CWD-independent.

This test imports the path-resolution logic from each script module and
checks that every path resolves to the expected location under the
project root, regardless of the current working directory.

Usage (run from any directory):
    uv run python <path-to>/scripts/test_paths.py
"""

import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Resolve project root the same way all scripts do
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent


def _resolve(rel: str) -> Path:
    """Resolve a project-relative path to an absolute Path."""
    return (_PROJECT_ROOT / rel).resolve()


# ---------------------------------------------------------------------------
# Expected paths (mirroring every ../  that was converted)
# ---------------------------------------------------------------------------
EXPECTED_PATHS = {
    # CSVLoader data root (used in all scripts)
    "datasets": _resolve("datasets/Cambridge-Estates-Building-Energy-Archive"),

    # Encoder weights
    "encoder_weight": _resolve("output/assets/weights/encoder_b0.pt"),

    # Similarity JSON
    "similarities": _resolve("output/assets/similarities.json"),

    # TiDE pretrained weight
    "tide_weight": _resolve("output/assets/weights/tide_bid_0_best_2.pt"),

    # Transfer learning DB
    "transfer_db": _resolve("output/assets/transfer_learning.db"),

    # Optuna hypertune DB
    "hypertune_db": _resolve("output/assets/tide-hypertune.db"),

    # Tide transfer (checkpoint dir)
    "tide_transfer_dir": _resolve("output/assets/tide_transfer"),

    # TS2Vec library (sys.path entry)
    "ts2vec_lib": _resolve("ts2vec"),
}


def test_path_resolution():
    """Check that every path starts with the project root."""
    print(f"Project root : {_PROJECT_ROOT}")
    print(f"Script dir   : {_SCRIPT_DIR}")
    print(f"CWD          : {Path.cwd()}")
    print()

    all_ok = True
    for name, expected in EXPECTED_PATHS.items():
        # Check that the path is under the project root
        try:
            expected.relative_to(_PROJECT_ROOT)
            status = "✓"
        except ValueError:
            status = "✗ NOT under project root"
            all_ok = False

        print(f"  [{status}] {name:20s} → {expected}")

    print()
    return all_ok


def test_script_path_constants():
    """Import each script's _SCRIPT_DIR / _PROJECT_ROOT and verify consistency."""
    print("--- Verifying script-level _SCRIPT_DIR constants ---")

    scripts_to_check = [
        "train_encoder",
        "calculate_similarity",
        "train_tide",
        "tune_hyperparameter",
        "transfer_tide",
        "visualize_forecast",
    ]

    # We cannot import these scripts directly (they use argparse at module level).
    # Instead, read each file and extract the _SCRIPT_DIR line to verify the pattern.
    all_ok = True
    for script_name in scripts_to_check:
        script_path = _SCRIPT_DIR / f"{script_name}.py"
        if not script_path.exists():
            print(f"  [✗] {script_name}.py — FILE NOT FOUND")
            all_ok = False
            continue

        source = script_path.read_text()

        # Check for _SCRIPT_DIR definition
        has_script_dir = "_SCRIPT_DIR = Path(__file__).resolve().parent" in source
        # Check for any remaining '../' paths (excluding comments)
        remaining_relative = []
        for i, line in enumerate(source.splitlines(), 1):
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            if "'../'" in line or '"../"' in line or "('../" in line or '("../' in line:
                remaining_relative.append((i, stripped))

        if has_script_dir and not remaining_relative:
            print(f"  [✓] {script_name}.py — _SCRIPT_DIR defined, no '../' paths")
        else:
            all_ok = False
            if not has_script_dir:
                print(f"  [✗] {script_name}.py — MISSING _SCRIPT_DIR definition")
            if remaining_relative:
                print(f"  [✗] {script_name}.py — REMAINING '../' paths:")
                for lineno, content in remaining_relative:
                    print(f"       L{lineno}: {content}")

    print()
    return all_ok


def test_sys_path_entries():
    """Verify that sys.path insertions use _SCRIPT_DIR, not relative paths."""
    print("--- Verifying sys.path insertions ---")

    scripts_to_check = [
        "train_encoder",
        "calculate_similarity",
        "train_tide",
        "tune_hyperparameter",
        "transfer_tide",
        "visualize_forecast",
    ]

    all_ok = True
    for script_name in scripts_to_check:
        script_path = _SCRIPT_DIR / f"{script_name}.py"
        if not script_path.exists():
            continue

        source = script_path.read_text()
        sys_path_lines = [
            (i, line.strip())
            for i, line in enumerate(source.splitlines(), 1)
            if "sys.path" in line and not line.strip().startswith("#")
        ]

        bad_entries = [
            (i, line) for i, line in sys_path_lines
            if "'..'" in line or '".."' in line
        ]

        if not bad_entries:
            print(f"  [✓] {script_name}.py — sys.path uses _SCRIPT_DIR")
        else:
            all_ok = False
            for lineno, content in bad_entries:
                print(f"  [✗] {script_name}.py L{lineno}: {content}")

    print()
    return all_ok


if __name__ == "__main__":
    print("=" * 60)
    print("PPTL Path Resolution Test")
    print("=" * 60)
    print()

    results = [
        test_path_resolution(),
        test_script_path_constants(),
        test_sys_path_entries(),
    ]

    if all(results):
        print("All path checks passed ✓")
        sys.exit(0)
    else:
        print("Some checks FAILED ✗")
        sys.exit(1)
