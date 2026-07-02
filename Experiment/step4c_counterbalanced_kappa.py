"""
Step 4c -- Counterbalanced inter-judge agreement (Cohen's kappa + exact).

Reuses the EXISTING verdict files from step3b (original order) and step3c
(flipped order). No re-judging / no API calls.

For each (instance, expression_set, judge, dimension) we have two verdicts (one
per order). We combine them into a single order-robust ("de-biased") verdict:

    cb_sign = (sign(orig_winner) + sign(flip_winner)) / 2
        sign:  two_stage -> +1, direct -> -1, tie -> 0
    de-biased label:  cb_sign > 0  -> two_stage
                      cb_sign < 0  -> direct
                      cb_sign == 0 -> tie     (includes the case where the two
                                               orders disagree and cancel out)

Inter-judge agreement (GPT vs Claude) is then computed on these counterbalanced
labels, per dimension, mirroring Table P2 from the single-order analysis but on
the de-biased verdicts.

Run:  python step4c_counterbalanced_kappa.py
"""
import glob
import json
import os

import numpy as np
import pandas as pd

from config import RANDOM_SEED, RESULTS_DIR
from step3b_pairwise_judge import PAIRWISE_DIR
from step3c_pairwise_flip import FLIP_DIR

SETS = ["ekman", "extended", "adaptive"]
JUDGES = ["gpt", "claude"]
CATS = ["direct", "two_stage", "tie"]


def sign(winner):
    """+1 two-stage, -1 direct, 0 tie."""
    return {"two_stage": 1.0, "direct": -1.0, "tie": 0.0}.get(winner, 0.0)


def cb_label(wo, wf):
    """Combine the two ordered verdicts into one order-robust label."""
    s = (sign(wo) + sign(wf)) / 2.0
    if s > 0:
        return "two_stage"
    if s < 0:
        return "direct"
    return "tie"


def load_pass(root):
    """root/<inst>/<set>.json -> dict[(inst, set)] = record."""
    out = {}
    for pf in glob.glob(os.path.join(root, "story_*", "*.json")):
        with open(pf, encoding="utf-8") as f:
            r = json.load(f)
        out[(r["instance_id"], r["expression_set"])] = r
    return out


def build_cb_labels():
    """DataFrame with one counterbalanced label per (instance, set, judge, dim)."""
    orig, flip = load_pass(PAIRWISE_DIR), load_pass(FLIP_DIR)
    keys = sorted(set(orig) & set(flip))          # only pairs present in BOTH passes
    rows = []
    for (inst, es) in keys:
        ro, rf = orig[(inst, es)], flip[(inst, es)]
        for judge in JUDGES:
            jo, jf = ro["judges"].get(judge), rf["judges"].get(judge)
            if not jo or not jf:
                continue
            for dim in ("cc", "ef"):
                rows.append({
                    "instance_id": inst, "expression_set": es,
                    "judge": judge, "dim": dim,
                    "cb_winner": cb_label(jo[f"{dim}_winner"], jf[f"{dim}_winner"]),
                })
    return pd.DataFrame(rows), len(keys)


def cohen_kappa(a, b, cats):
    """Cohen's kappa for two raters over the same items (lists of category labels)."""
    a, b = list(a), list(b)
    n = len(a)
    if n == 0:
        return np.nan
    idx = {c: i for i, c in enumerate(cats)}
    k = len(cats)
    M = np.zeros((k, k))
    for x, y in zip(a, b):
        M[idx[x], idx[y]] += 1
    po = np.trace(M) / n
    row, col = M.sum(1) / n, M.sum(0) / n
    pe = float(np.sum(row * col))
    if abs(1 - pe) < 1e-12:
        return np.nan
    return (po - pe) / (1 - pe)


def table_cb_kappa(df):
    """Inter-judge exact agreement + Cohen's kappa on the counterbalanced labels."""
    out = []
    for dim in ("cc", "ef"):
        g = df[df.dim == dim].pivot_table(
            index=["instance_id", "expression_set"],
            columns="judge", values="cb_winner", aggfunc="first")
        g = g.dropna(subset=JUDGES)
        if len(g):
            agree = float((g["gpt"] == g["claude"]).mean())
            kappa = cohen_kappa(g["gpt"], g["claude"], CATS)
            out.append({"Dim": dim.upper(),
                        "Exact agreement": f"{agree:.2f}",
                        "Cohen's kappa": f"{kappa:.3f}",
                        "n": len(g)})
    return pd.DataFrame(out)


def main():
    df, npairs = build_cb_labels()
    if df.empty:
        print("No paired data. Run step3b (original) AND step3c (flipped) first.")
        return
    print(f"Counterbalanced over {npairs} (instance x set) pairs that have both "
          f"orders.\n")

    cbk = table_cb_kappa(df)
    cbk.to_csv(os.path.join(RESULTS_DIR, "tableCB_kappa.csv"), index=False)

    pd.set_option("display.max_columns", None, "display.width", 220)
    print("=== Table CB-kappa: inter-judge agreement on COUNTERBALANCED verdicts ===")
    print(cbk.to_string(index=False), "\n")
    print(f"CSV written to {os.path.join(RESULTS_DIR, 'tableCB_kappa.csv')}")


if __name__ == "__main__":
    main()
