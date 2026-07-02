"""
Step 4b (counterbalanced) -- merge the original pairwise pass (step3b) with the
flipped-order pass (step3c) to remove position bias by design.

For each (instance, expression_set, judge, dimension) we have two verdicts (one
per order). Each is mapped to direct / two_stage / tie and turned into a sign
(+1 two-stage, -1 direct, 0 tie); the two are averaged, so a judge that simply
follows the slot (A or B) cancels to 0. The per-instance counterbalanced score
(averaged over judges) feeds the bootstrap.

Diagnostics:
  * Flip rate: among pairs where BOTH passes gave a non-tie verdict, the fraction
    whose winner flipped when the order flipped. ~0.5 = chance; high = position bias.

Outputs Table CB1 (counterbalanced results) and CB2 (flip-rate diagnostic),
and writes data/results/tableCB1|CB2.csv.

Run:  python step4b_counterbalanced.py
"""
import glob
import json
import os

import numpy as np
import pandas as pd

from config import DATA_DIR, RANDOM_SEED, RESULTS_DIR, BOOTSTRAP_B
from step3b_pairwise_judge import PAIRWISE_DIR
from SpriteSwitchingEX.step3c_pairwise_flip import FLIP_DIR

SETS = ["ekman", "extended", "adaptive"]
JUDGES = ["gpt", "claude"]


def sign(winner):
    return {"two_stage": 1.0, "direct": -1.0, "tie": 0.0}.get(winner, 0.0)


def load_pass(root):
    """root/<inst>/<set>.json -> dict[(inst,set)] = record."""
    out = {}
    for pf in glob.glob(os.path.join(root, "story_*", "*.json")):
        with open(pf, encoding="utf-8") as f:
            r = json.load(f)
        out[(r["instance_id"], r["expression_set"])] = r
    return out


def build_rows():
    orig = load_pass(PAIRWISE_DIR)
    flip = load_pass(FLIP_DIR)
    keys = sorted(set(orig) & set(flip))   # only pairs that have BOTH passes
    rows = []
    for (inst, es) in keys:
        ro, rf = orig[(inst, es)], flip[(inst, es)]
        for judge in JUDGES:
            jo, jf = ro["judges"].get(judge), rf["judges"].get(judge)
            if not jo or not jf:
                continue
            for dim in ("cc", "ef"):
                wo, wf = jo[f"{dim}_winner"], jf[f"{dim}_winner"]
                rows.append({
                    "instance_id": inst, "expression_set": es, "judge": judge,
                    "dim": dim,
                    "orig_winner": wo, "flip_winner": wf,
                    "cb_sign": (sign(wo) + sign(wf)) / 2.0,  # counterbalanced sign
                })
    return pd.DataFrame(rows), len(keys)


def bootstrap_ci(per_instance_values, B=BOOTSTRAP_B, rng=None):
    v = np.asarray(per_instance_values, float)
    if len(v) == 0:
        return np.nan, np.nan, np.nan
    rng = rng or np.random.default_rng(RANDOM_SEED)
    idx = rng.integers(0, len(v), size=(B, len(v)))
    m = v[idx].mean(axis=1)
    return float(v.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def table_cb1(df):
    rng = np.random.default_rng(RANDOM_SEED)
    out = []
    for es in SETS:
        for dim in ("cc", "ef"):
            sub = df[(df.expression_set == es) & (df.dim == dim)]
            if sub.empty:
                continue
            # W-T-L over (instance, judge) using the counterbalanced sign.
            wins = int((sub.cb_sign > 0).sum())
            losses = int((sub.cb_sign < 0).sum())
            ties = int((sub.cb_sign == 0).sum())
            decided = wins + losses
            win_rate = wins / decided if decided else np.nan
            # per-instance score = mean over judges, then bootstrap over instances
            per_inst = [g.cb_sign.mean() for _, g in sub.groupby("instance_id")]
            net = bootstrap_ci(per_inst, rng=rng)
            out.append({
                "Expression Set": es, "Dim": dim.upper(),
                "TwoStage W-T-L": f"{wins}-{ties}-{losses}",
                "WinRate(excl.tie)": (f"{win_rate:.2f}" if decided else "n/a"),
                "Net(counterbalanced) [95% CI]":
                    f"{net[0]:+.2f} [{net[1]:+.2f}, {net[2]:+.2f}]",
            })
    return pd.DataFrame(out)


def table_cb2(df):
    """Flip rate: among both-non-tie pairs, how often the winner flipped with order."""
    out = []
    for judge in JUDGES:
        for dim in ("cc", "ef"):
            sub = df[(df.judge == judge) & (df.dim == dim)
                     & (df.orig_winner != "tie") & (df.flip_winner != "tie")]
            n = len(sub)
            flipped = int((sub.orig_winner != sub.flip_winner).sum())
            rate = flipped / n if n else np.nan
            out.append({
                "Judge": judge, "Dim": dim.upper(),
                "Flip rate (pos. bias)": (f"{rate:.2f}  (n={n})" if n else "n/a"),
            })
    # overall
    sub = df[(df.orig_winner != "tie") & (df.flip_winner != "tie")]
    n = len(sub)
    rate = int((sub.orig_winner != sub.flip_winner).sum()) / n if n else np.nan
    out.append({"Judge": "ALL", "Dim": "ALL",
                "Flip rate (pos. bias)": (f"{rate:.2f}  (n={n})" if n else "n/a")})
    return pd.DataFrame(out)


def main():
    df, npairs = build_rows()
    if df.empty:
        print("No paired data. Run step3b (original) AND step3c (flipped) first.")
        return
    print(f"Counterbalanced over {npairs} (instance x set) pairs that have both "
          f"orders.\n")

    cb1 = table_cb1(df)
    cb2 = table_cb2(df)
    cb1.to_csv(os.path.join(RESULTS_DIR, "tableCB1_counterbalanced.csv"), index=False)
    cb2.to_csv(os.path.join(RESULTS_DIR, "tableCB2_flip_rate.csv"), index=False)

    pd.set_option("display.max_columns", None, "display.width", 220)
    print("=== Table CB1: counterbalanced pairwise (Net > 0 favors two-stage) ===")
    print(cb1.to_string(index=False), "\n")
    print("=== Table CB2: flip rate (0.50 = no position bias, high = biased) ===")
    print(cb2.to_string(index=False), "\n")
    print(f"CSVs written to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
