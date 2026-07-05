"""Run the whole Titanic benchmark end-to-end: XGBoost, then TabFM, then report.

This is a convenience wrapper around the individual steps. Each can also be run
on its own (see the module docstrings).

Run with:

    python -m src.run_titanic_benchmark
"""

from __future__ import annotations

from src import generate_report, run_tabfm_titanic, run_xgboost_titanic


def main() -> None:
    print("=" * 60)
    print("Step 1/3: XGBoost baseline")
    print("=" * 60)
    run_xgboost_titanic.main()

    print("\n" + "=" * 60)
    print("Step 2/3: TabFM zero-shot (first run downloads weights)")
    print("=" * 60)
    run_tabfm_titanic.main()

    print("\n" + "=" * 60)
    print("Step 3/3: Combined results + report")
    print("=" * 60)
    generate_report.main()


if __name__ == "__main__":
    main()
