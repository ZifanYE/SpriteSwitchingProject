"""
Step 4b -- Analysis of the PAIRWISE comparison (direct vs two-stage).

Produces:
  Table P1: per expression set x dimension -- two-stage wins / ties / losses,
            two-stage win-rate (excluding ties), and a net-preference score
            (two-stage advantage) with a 95% bootstrap CI. A confidence-weighted
            net score is also reported.
  Table P2: sanity checks -- position bias (how often the "A" slot was chosen)
            and inter-judge agreement (% agreement + Cohen's kappa).

Net-preference score per (instance, judge): +1 two-stage win, -1 direct win,
0 tie; positive mean => two-stage preferred. Bootstrap resamples instances
(the independent unit), averaging the two judges within each instance.

Run:  python step4b_pairwise_analysis.py
"""
import glob
import json
import os

import numpy as np
import pandas as pd

from config import DATA_DIR, RANDOM_SEED, RESULTS_DIR, BOOTSTRAP_B

PAIRWISE_DIR = os.path.join(DATA_DIR, "pairwise")
SETS = ["ekman", "extended", "adaptive"]
CONF_W = {"slight": 1.0, "clear": 2.0, "strong": 3.0}
JUDGES = ["gpt", "claude"]


def sign(winner):
    """+1 two-stage, -1 direct, 0 tie."""
    return {"two_stage": 1.0, "direct": -1.0, "tie": 0.0}.get(winner, 0.0)


def load_rows():
    rows = []
    for pf in glob.glob(os.path.join(PAIRWISE_DIR, "story_*", "*.json")):
        with open(pf, encoding="utf-8") as f:
            r = json.load(f)
        for jname, j in r["judges"].items():
            rows.append({
                "instance_id": r["instance_id"],
                "expression_set": r["expression_set"],
                "judge": jname,
                "A_is_direct": r["A_is_direct"],
                "cc_winner": j["cc_winner"], "cc_conf": j["cc_confidence"],
                "ef_winner": j["ef_winner"], "ef_conf": j["ef_confidence"],
                "raw_cc_ab": j.get("raw_cc_ab"), "raw_ef_ab": j.get("raw_ef_ab"),
            })
    return pd.DataFrame(rows)


def bootstrap_ci(per_instance_values, B=BOOTSTRAP_B, rng=None):
    v = np.asarray(per_instance_values, float)
    if len(v) == 0:
        return np.nan, np.nan, np.nan
    rng = rng or np.random.default_rng(RANDOM_SEED)
    idx = rng.integers(0, len(v), size=(B, len(v)))
    means = v[idx].mean(axis=1)
    return float(v.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def per_instance_score(sub, dim, weighted=False):
    """Average the two judges' (signed, optionally confidence-weighted) verdicts per instance."""
    wcol, ccol = f"{dim}_winner", f"{dim}_conf"
    vals = []
    for _, g in sub.groupby("instance_id"):
        s = []
        for _, row in g.iterrows():
            mag = CONF_W[row[ccol]] if weighted else 1.0
            s.append(sign(row[wcol]) * mag)
        vals.append(np.mean(s))
    return vals


def table_p1(df):
    rng = np.random.default_rng(RANDOM_SEED)
    out = []
    for es in SETS:
        for dim in ("cc", "ef"):
            sub = df[df.expression_set == es]
            wcol = f"{dim}_winner"
            wins = int((sub[wcol] == "two_stage").sum())
            losses = int((sub[wcol] == "direct").sum())
            ties = int((sub[wcol] == "tie").sum())
            decided = wins + losses
            win_rate = wins / decided if decided else np.nan
            net = bootstrap_ci(per_instance_score(sub, dim, weighted=False), rng=rng)
            netw = bootstrap_ci(per_instance_score(sub, dim, weighted=True), rng=rng)
            out.append({
                "Expression Set": es,
                "Dim": dim.upper(),
                "TwoStage W-T-L": f"{wins}-{ties}-{losses}",
                "TwoStage WinRate(excl.tie)": (f"{win_rate:.2f}" if decided else "n/a"),
                "Net [95% CI]": f"{net[0]:+.2f} [{net[1]:+.2f}, {net[2]:+.2f}]",
                "Net(conf-wt) [95% CI]": f"{netw[0]:+.2f} [{netw[1]:+.2f}, {netw[2]:+.2f}]",
            })
    return pd.DataFrame(out)


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


def table_p2(df):
    out = []
    # Position bias: among non-tie raw A/B verdicts, how often was "A" chosen?
    for dim, raw in (("cc", "raw_cc_ab"), ("ef", "raw_ef_ab")):
        for judge in JUDGES:
            sub = df[(df.judge == judge) & (df[raw].isin(["A", "B"]))]
            a_rate = (sub[raw] == "A").mean() if len(sub) else np.nan
            out.append({"Check": f"Position bias ({dim.upper()}, {judge}) -- P(chose A)",
                        "Value": (f"{a_rate:.2f}  (n={len(sub)})" if len(sub) else "n/a")})

    # Inter-judge agreement on the pipeline verdict (direct/two_stage/tie).
    cats = ["direct", "two_stage", "tie"]
    for dim in ("cc", "ef"):
        wcol = f"{dim}_winner"
        g = df.pivot_table(index=["instance_id", "expression_set"], columns="judge",
                           values=wcol, aggfunc="first")
        g = g.dropna(subset=JUDGES)
        if len(g):
            agree = float((g["gpt"] == g["claude"]).mean())
            kappa = cohen_kappa(g["gpt"], g["claude"], cats)
            out.append({"Check": f"Inter-judge agreement ({dim.upper()})",
                        "Value": f"{agree:.2f} exact, kappa={kappa:.3f}  (n={len(g)})"})
    return pd.DataFrame(out)


def main():
    df = load_rows()
    if df.empty:
        print("No pairwise data found. Run step3b_pairwise_judge.py first.")
        return
    print(f"Loaded {len(df)} pairwise verdicts "
          f"({df.instance_id.nunique()} instances x {df.expression_set.nunique()} sets x "
          f"{df.judge.nunique()} judges).\n")

    p1 = table_p1(df)
    p2 = table_p2(df)
    p1.to_csv(os.path.join(RESULTS_DIR, "tableP1_pairwise.csv"), index=False)
    p2.to_csv(os.path.join(RESULTS_DIR, "tableP2_pairwise_checks.csv"), index=False)

    pd.set_option("display.max_columns", None, "display.width", 220)
    print("=== Table P1: pairwise direct vs two-stage (Net > 0 favors two-stage) ===")
    print(p1.to_string(index=False), "\n")
    print("=== Table P2: sanity checks ===")
    print(p2.to_string(index=False), "\n")
    print(f"CSVs written to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
