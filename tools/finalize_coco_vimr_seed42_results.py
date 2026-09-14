"""Add measured MAC complexity data to a completed VIMR comparison run."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


PROFILED_BASELINE_MACS_G = 36.605413888
PROFILED_VIMR_CLICK5_MACS_G = 42.335129088


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    args = parser.parse_args()

    payload = json.loads(args.results.read_text(encoding="utf-8"))
    published_baseline = float(payload["baseline_complexity"]["MACs(G)"])
    overhead = PROFILED_VIMR_CLICK5_MACS_G - PROFILED_BASELINE_MACS_G
    normalized_vimr = published_baseline + overhead
    payload["baseline_complexity"]["ProfiledMACs(G)"] = PROFILED_BASELINE_MACS_G
    payload["vimr_complexity"].update(
        {
            "MACs(G)": normalized_vimr,
            "PublishedBaselineMACs(G)": published_baseline,
            "ProfiledBaselineMACs(G)": PROFILED_BASELINE_MACS_G,
            "ProfiledVIMRClick5MACs(G)": PROFILED_VIMR_CLICK5_MACS_G,
            "ProfiledVIMROverheadMACs(G)": overhead,
            "macs_protocol": (
                "PyTorch 2.6 FlopCounterMode, MAC=FLOP/2; 512x512 image encoder plus one "
                "decoder interaction; VIMR uses the maximum-history click-5 path. The table "
                "adds measured VIMR overhead to the published 35G EfficientViT-SAM-L0 value."
            ),
        }
    )
    for row in payload["table"]:
        if row["method"] == "VIMR seed 42":
            row["MACs(G)"] = normalized_vimr

    args.results.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    comparison_path = args.results.parent / "comparison.csv"
    fields = [
        "method",
        "click",
        "mIoU",
        "mIoUS",
        "mIoUM",
        "mIoUL",
        "mAP",
        "APS",
        "APM",
        "APL",
        "Params(M)",
        "MACs(G)",
        "MeasuredPipelineThroughput(image/s)",
    ]
    with comparison_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(payload["table"])
    print(
        json.dumps(
            {
                "results": str(args.results),
                "baseline_MACs_G": published_baseline,
                "vimr_MACs_G": normalized_vimr,
                "same_counter_baseline_MACs_G": PROFILED_BASELINE_MACS_G,
                "same_counter_vimr_MACs_G": PROFILED_VIMR_CLICK5_MACS_G,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
