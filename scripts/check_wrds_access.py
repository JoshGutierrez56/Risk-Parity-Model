"""Dry-run by default; use --live for a metadata-only WRDS entitlement check."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from riskparity.wrds_loader import CRSPRequest, DATASETS, connect_wrds


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--dataset", choices=DATASETS, default="legacy")
    args = parser.parse_args()
    request = CRSPRequest.create(["SPY"], "2024-01-02", "2024-01-31", args.dataset)
    returns_table, names_table = DATASETS[request.dataset]
    result = {
        "connection_attempted": False,
        "dataset": request.dataset,
        "required_tables": [returns_table, names_table],
        "bounded_probe": {"tickers": 1, "start": str(request.start), "end": str(request.end)},
    }
    if args.live:
        db = connect_wrds()
        try:
            library = returns_table.split(".", 1)[0]
            libraries = set(db.list_libraries())
            tables = set(db.list_tables(library=library)) if library in libraries else set()
            result.update({
                "connection_attempted": True,
                "connection_succeeded": True,
                "library_available": library in libraries,
                "returns_table_available": returns_table.split(".", 1)[1] in tables,
                "names_table_available": names_table.split(".", 1)[1] in tables,
            })
        finally:
            db.close()
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
