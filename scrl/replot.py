from __future__ import annotations

import json
import math
import os
import sys

import numpy as np

from . import plotting


def _num(x):
    if isinstance(x, str):
        return {"inf": math.inf, "-inf": -math.inf, "nan": math.nan}.get(x, x)
    return x


def _clean(obj):
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_clean(v) for v in obj]
    return _num(obj)


def replot(run_dir: str) -> None:
    with open(os.path.join(run_dir, "summary.json")) as f:
        S = _clean(json.load(f))
    kind = os.path.basename(os.path.normpath(run_dir)).split("_")[0]
    if kind == "oracle":
        plotting.plot_gap_bars({c: v["results"] for c, v in S.items()}, os.path.join(run_dir, "gaps.png"),
                               "Protocol 1: CE gap to the SC/DP optimum")
        plotting.plot_learning_curves({c: v["curves"] for c, v in S.items()},
                                      os.path.join(run_dir, "learning_curves.png"),
                                      "Protocol 1: validation CE gap during training")
    elif kind == "estimation":
        strip = {c: {k: a["gaps"] for k, a in v["aggregate"].items()} for c, v in S.items()}
        plotting.plot_estimation_strip(strip, os.path.join(run_dir, "estimation_gaps.png"),
                                       "Protocol 2: gap to the true-parameter optimum across observed histories")
    elif kind == "misspec":
        M = S["matrix"]
        plotting.plot_heatmap(np.array(M["gap_bp"], float), M["rows"], M["cols"],
                              os.path.join(run_dir, "misspec_heatmap.png"),
                              "Protocol 3: train/calibrate on row model, evaluate on column model")
    else:
        raise ValueError(f"cannot infer protocol from '{run_dir}'")
    print(f"figures regenerated in {run_dir}")


if __name__ == "__main__":
    for d in sys.argv[1:]:
        replot(d)
