"""
Step 4 -- Statistical analysis. Reproduces the paper's Tables 1, 2 and 3.

  Table 1: per-condition Mean +/- 95% bootstrap CI for CC/EF (GPT, Claude, Mean).
  Table 2: paired-bootstrap Direct vs Two-stage per expression set (dCC, dEF).
  Table 3: inter-judge ICC (GPT vs Claude) and human-vs-judge Spearman rho.

Requires: numpy, pandas, scipy, pingouin.
Run:  python step4_analysis.py
"""
import glob
import json
import os

import numpy as np
import pandas as pd

from config import (BOOTSTRAP_B, HUMAN_RATINGS, JUDGED_DIR, RANDOM_SEED,
                    RESULTS_DIR)

PIPELINES = ["direct", "twostage"]
SETS = ["ekman", "extended", "adaptive"]


# ----------------------------------------------------------- load scores --- #
def load_dataframe():
    rows = []
    for cond_file in glob.glob(os.path.join(JUDGED_DIR, "story_*", "*.json")):
        with open(cond_file, "r", encoding="utf-8") as f:
            r = json.load(f)
        j = r["judges"]
        rows.append({
            "instance_id": r["instance_id"],
            "pipeline": r["pipeline"],
            "expression_set": r["expression_set"],
            "personality": r.get("personality", ""),
            "cc_gpt": j["gpt"]["character_consistency"],
            "cc_claude": j["claude"]["character_consistency"],
            "ef_gpt": j["gpt"]["emotional_fidelity"],
            "ef_claude": j["claude"]["emotional_fidelity"],
        })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["cc_mean"] = (df["cc_gpt"] + df["cc_claude"]) / 2.0
    df["ef_mean"] = (df["ef_gpt"] + df["ef_claude"]) / 2.0
    return df


# ------------------------------------------------------------- bootstrap --- #
def bootstrap_ci(values, B=BOOTSTRAP_B, rng=None):
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return np.nan, np.nan, np.nan
    rng = rng or np.random.default_rng(RANDOM_SEED)
    idx = rng.integers(0, len(values), size=(B, len(values)))
    means = values[idx].mean(axis=1)
    return float(values.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def paired_bootstrap_ci(a, b, B=BOOTSTRAP_B, rng=None):
    """Mean of (b - a) with 95% CI, resampling matched pairs (b=two-stage, a=direct)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    diff = b - a
    return bootstrap_ci(diff, B=B, rng=rng)


def fmt(mean, lo, hi):
    if np.isnan(mean):
        return "n/a"
    return f"{mean:.2f} [{lo:.2f}, {hi:.2f}]"


# --------------------------------------------------------------- Table 1 --- #
def table1(df):
    rng = np.random.default_rng(RANDOM_SEED)
    metrics = ["cc_gpt", "cc_claude", "cc_mean", "ef_gpt", "ef_claude", "ef_mean"]
    out = []
    for pipe in PIPELINES:
        for es in SETS:
            sub = df[(df.pipeline == pipe) & (df.expression_set == es)]
            row = {"Pipeline": pipe, "Expression Set": es}
            for m in metrics:
                row[m] = fmt(*bootstrap_ci(sub[m].values, rng=rng))
            out.append(row)
    return pd.DataFrame(out)


# --------------------------------------------------------------- Table 2 --- #
def table2(df):
    rng = np.random.default_rng(RANDOM_SEED)
    out = []
    for es in SETS:
        d = df[(df.pipeline == "direct") & (df.expression_set == es)].set_index("instance_id")
        t = df[(df.pipeline == "twostage") & (df.expression_set == es)].set_index("instance_id")
        common = d.index.intersection(t.index)
        d, t = d.loc[common], t.loc[common]
        dcc = paired_bootstrap_ci(d["cc_mean"].values, t["cc_mean"].values, rng=rng)
        def_ = paired_bootstrap_ci(d["ef_mean"].values, t["ef_mean"].values, rng=rng)
        out.append({
            "Expression Set": es,
            "dCC (two-stage - direct)": fmt(*dcc),
            "dEF (two-stage - direct)": fmt(*def_),
        })
    return pd.DataFrame(out)


# --------------------------------------------------------------- Table 3 --- #
def compute_icc(df):
    """ICC(2,1) between GPT and Claude over all samples, per dimension and overall."""
    try:
        import pingouin as pg
    except ImportError:
        print("  [warn] pingouin not installed; skipping ICC. pip install pingouin")
        return {}

    import numpy as np

    def _icc2(ld):
        icc = pg.intraclass_corr(data=ld, targets="target", raters="rater", ratings="score")
        # ICC2 = two-way random, single rater, absolute agreement.
        # pingouin labels it "ICC2" (old) or "ICC(A,1)" (newer).
        for key in ("ICC2", "ICC(A,1)"):
            v = icc[icc.Type == key]["ICC"].values
            if len(v):
                return float(v[0])
        return np.nan

    res = {}
    for dim, (gcol, ccol) in {"CC": ("cc_gpt", "cc_claude"),
                              "EF": ("ef_gpt", "ef_claude")}.items():
        long = []
        for i, r in df.reset_index(drop=True).iterrows():
            long.append({"target": i, "rater": "gpt", "score": r[gcol]})
            long.append({"target": i, "rater": "claude", "score": r[ccol]})
        res[dim] = _icc2(pd.DataFrame(long))

    # overall: stack CC and EF
    long = []
    k = 0
    for _, r in df.iterrows():
        for gcol, ccol in [("cc_gpt", "cc_claude"), ("ef_gpt", "ef_claude")]:
            long.append({"target": k, "rater": "gpt", "score": r[gcol]})
            long.append({"target": k, "rater": "claude", "score": r[ccol]})
            k += 1
    res["overall"] = _icc2(pd.DataFrame(long))
    return res


def human_spearman(df):
    """Optional: Spearman rho of human vs each judge, if data/human_ratings.csv exists.

    Expected CSV columns: instance_id, pipeline, expression_set,
                          human_cc, human_ef
    (one row per judged sample you had humans rate).
    """
    if not os.path.exists(HUMAN_RATINGS):
        print("  [info] no human_ratings.csv -> skipping human validation.")
        return {}
    from scipy.stats import spearmanr
    hr = pd.read_csv(HUMAN_RATINGS)
    merged = hr.merge(df, on=["instance_id", "pipeline", "expression_set"], how="inner")
    if merged.empty:
        print("  [warn] human ratings did not match any judged samples.")
        return {}

    # stack CC and EF so each judge gets one rho across both dimensions
    human = np.concatenate([merged["human_cc"], merged["human_ef"]])
    out = {}
    for judge, (cc, ef) in {"gpt": ("cc_gpt", "ef_gpt"),
                            "claude": ("cc_claude", "ef_claude")}.items():
        model = np.concatenate([merged[cc], merged[ef]])
        rho, p = spearmanr(human, model)
        out[judge] = (float(rho), float(p))
    return out


# ------------------------------------------------------------------ main --- #
def main():
    df = load_dataframe()
    if df.empty:
        print("No judged data found. Run step3 first.")
        return

    print(f"Loaded {len(df)} judged samples across "
          f"{df.instance_id.nunique()} story instances.\n")

    t1 = table1(df)
    t2 = table2(df)
    icc = compute_icc(df)
    spear = human_spearman(df)

    t1.to_csv(os.path.join(RESULTS_DIR, "table1_conditions.csv"), index=False)
    t2.to_csv(os.path.join(RESULTS_DIR, "table2_pipeline_comparison.csv"), index=False)

    t3_rows = []
    for dim, v in icc.items():
        t3_rows.append({"Comparison": f"GPT vs Claude ICC2 ({dim})",
                        "Statistic": f"{v:.3f}" if not np.isnan(v) else "n/a"})
    for judge, (rho, p) in spear.items():
        t3_rows.append({"Comparison": f"Human vs {judge} (Spearman rho)",
                        "Statistic": f"{rho:.3f} (p={p:.3g})"})
    t3 = pd.DataFrame(t3_rows)
    t3.to_csv(os.path.join(RESULTS_DIR, "table3_judge_validation.csv"), index=False)

    pd.set_option("display.max_columns", None, "display.width", 200)
    print("=== Table 1: per-condition (Mean [95% bootstrap CI]) ===")
    print(t1.to_string(index=False), "\n")
    print("=== Table 2: Direct vs Two-stage (paired bootstrap) ===")
    print(t2.to_string(index=False), "\n")
    print("=== Table 3: judge validation ===")
    print(t3.to_string(index=False), "\n")
    print(f"CSVs written to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
