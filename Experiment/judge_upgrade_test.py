"""
Small-scale judge-upgrade test.

Re-judges a small subset of already-evaluated samples with a STRONGER Claude
judge (config.JUDGE_TEST_MODEL, default Claude Opus 4.8) and asks one question:
does the stronger judge agree with GPT-5.4 better than the original Haiku judge?

It is cheap: GPT-5.4 and the original Claude (Haiku) pointwise scores are reused
from data/judged/; only the NEW Claude judge is actually called, on JUDGE_TEST_N
instances x 6 conditions.

It does NOT touch data/judged/. New scores are cached under data/judge_test/.

Output: prints ICC(GPT vs new-Claude) next to ICC(GPT vs old-Claude) on the same
subset, plus mean scores (to see if the stronger judge uses the scale more
widely), and writes data/results/judge_upgrade_test.csv.

Run:  python judge_upgrade_test.py
"""
import glob
import json
import os

import numpy as np
import pandas as pd

from config import (DATA_DIR, JUDGED_DIR, JUDGE_TEST_MODEL, JUDGE_TEST_N,
                    RESULTS_DIR, STORIES_DIR)
from step3_llm_judge import build_transcript, judge_one

TEST_DIR = os.path.join(DATA_DIR, "judge_test")
os.makedirs(TEST_DIR, exist_ok=True)


def icc2(paired_a, paired_b):
    """ICC(2,1) between two raters over paired scores."""
    try:
        import pingouin as pg
    except ImportError:
        print("  [warn] pingouin not installed; ICC skipped. pip install pingouin")
        return np.nan
    long = []
    for i, (a, b) in enumerate(zip(paired_a, paired_b)):
        long.append({"target": i, "rater": "x", "score": a})
        long.append({"target": i, "rater": "y", "score": b})
    res = pg.intraclass_corr(data=pd.DataFrame(long),
                             targets="target", raters="rater", ratings="score")
    for key in ("ICC2", "ICC(A,1)"):
        v = res[res.Type == key]["ICC"].values
        if len(v):
            return float(v[0])
    return np.nan


def main():
    judged_dirs = sorted(glob.glob(os.path.join(JUDGED_DIR, "story_*")))[:JUDGE_TEST_N]
    if not judged_dirs:
        print("No judged data found. Run step3 first.")
        return
    print(f"Testing judge {JUDGE_TEST_MODEL['model']} on "
          f"{len(judged_dirs)} instance(s) x up to 6 conditions.\n")

    rows = []
    for jdir in judged_dirs:
        inst = os.path.basename(jdir)
        with open(os.path.join(STORIES_DIR, f"{inst}.json"), encoding="utf-8") as f:
            story = json.load(f)

        cache_dir = os.path.join(TEST_DIR, inst)
        os.makedirs(cache_dir, exist_ok=True)

        for jf in sorted(glob.glob(os.path.join(jdir, "*.json"))):
            cond = os.path.splitext(os.path.basename(jf))[0]
            with open(jf, encoding="utf-8") as f:
                rec = json.load(f)  # = switching record + judges{gpt, claude(old)}

            # New Claude judge (cached so re-runs don't recall the API).
            cache_path = os.path.join(cache_dir, f"{cond}.json")
            if os.path.exists(cache_path):
                with open(cache_path, encoding="utf-8") as f:
                    new = json.load(f)
            else:
                transcript = build_transcript(story, rec)
                print(f"[test] {inst}/{cond} <- {JUDGE_TEST_MODEL['model']}")
                try:
                    new = judge_one(JUDGE_TEST_MODEL, story["personality"],
                                    story["personality_profile"], transcript)
                except Exception as e:  # noqa: BLE001
                    print(f"[fail] {inst}/{cond}: {e} -- skipping; re-run to retry.")
                    continue
                with open(cache_path, "w", encoding="utf-8") as f:
                    json.dump(new, f, ensure_ascii=False, indent=2)

            rows.append({
                "instance_id": inst, "cond": cond,
                "cc_gpt": rec["judges"]["gpt"]["character_consistency"],
                "ef_gpt": rec["judges"]["gpt"]["emotional_fidelity"],
                "cc_old": rec["judges"]["claude"]["character_consistency"],
                "ef_old": rec["judges"]["claude"]["emotional_fidelity"],
                "cc_new": new["character_consistency"],
                "ef_new": new["emotional_fidelity"],
            })

    if not rows:
        print("No samples scored.")
        return
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULTS_DIR, "judge_upgrade_test.csv"), index=False)

    # Agreement with GPT: new (stronger) Claude vs old (Haiku) Claude.
    icc_cc_new = icc2(df.cc_gpt, df.cc_new)
    icc_ef_new = icc2(df.ef_gpt, df.ef_new)
    icc_cc_old = icc2(df.cc_gpt, df.cc_old)
    icc_ef_old = icc2(df.ef_gpt, df.ef_old)
    icc_all_new = icc2(pd.concat([df.cc_gpt, df.ef_gpt]), pd.concat([df.cc_new, df.ef_new]))
    icc_all_old = icc2(pd.concat([df.cc_gpt, df.ef_gpt]), pd.concat([df.cc_old, df.ef_old]))

    print(f"\nSubset: {len(df)} samples.\n")
    print(f"{'Agreement with GPT-5.4 (ICC2)':<34}{'old Haiku':>12}{'new model':>12}")
    print(f"{'  Character Consistency':<34}{icc_cc_old:>12.3f}{icc_cc_new:>12.3f}")
    print(f"{'  Emotional Fidelity':<34}{icc_ef_old:>12.3f}{icc_ef_new:>12.3f}")
    print(f"{'  Overall':<34}{icc_all_old:>12.3f}{icc_all_new:>12.3f}")

    print(f"\n{'Mean score (scale usage)':<26}{'GPT':>8}{'old':>8}{'new':>8}{'  std(new)':>10}")
    print(f"{'  CC':<26}{df.cc_gpt.mean():>8.2f}{df.cc_old.mean():>8.2f}"
          f"{df.cc_new.mean():>8.2f}{df.cc_new.std():>10.2f}")
    print(f"{'  EF':<26}{df.ef_gpt.mean():>8.2f}{df.ef_old.mean():>8.2f}"
          f"{df.ef_new.mean():>8.2f}{df.ef_new.std():>10.2f}")

    better = (np.nan_to_num(icc_all_new) > np.nan_to_num(icc_all_old))
    print("\nVerdict: the stronger Claude judge agrees with GPT-5.4 "
          + ("BETTER" if better else "NOT better")
          + " than Haiku on this subset.")
    print("(Small n -- treat as a go/no-go signal, then scale up if promising.)")
    print(f"\nCSV: {os.path.join(RESULTS_DIR, 'judge_upgrade_test.csv')}")


if __name__ == "__main__":
    main()
