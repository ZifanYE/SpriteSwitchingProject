"""
Step 3c -- Flipped-order pairwise pass (counterbalancing).

You already ran step3b once, with a randomized A/B order recorded per sample.
This script adds the OTHER order: for each (instance, expression_set) it shows
the same two sprite sequences but with A and B swapped relative to step3b, and
re-judges. Combining the two passes (step4b_counterbalanced.py) cancels position
bias by construction and yields a flip-rate diagnostic.

It reuses the original pass (does not re-run it) and only adds the flipped one.

Output: data/pairwise_flip/story_XXXX/<expression_set>.json
  ("A_is_direct" here is the NEGATION of the original pass's value.)

Run:  python step3c_pairwise_flip.py
"""
import glob
import json
import os

from config import DATA_DIR, JUDGE_MODELS, SWITCHING_DIR
from step3b_pairwise_judge import (judge_pair, map_winner, transcript_for,
                                    PAIRWISE_DIR)

FLIP_DIR = os.path.join(DATA_DIR, "pairwise_flip")
EXPRESSION_SETS = ["ekman", "extended", "adaptive"]


def main():
    orig_files = sorted(glob.glob(os.path.join(PAIRWISE_DIR, "story_*", "*.json")))
    if not orig_files:
        print("No original pairwise results found. Run step3b_pairwise_judge.py first.")
        return

    failures = []
    for of in orig_files:
        inst = os.path.basename(os.path.dirname(of))
        es = os.path.splitext(os.path.basename(of))[0]

        out_dir = os.path.join(FLIP_DIR, inst)
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"{es}.json")
        if os.path.exists(out_path):
            print(f"[skip] {inst}/{es}")
            continue

        with open(of, encoding="utf-8") as f:
            orig = json.load(f)
        # Flip the order used in the original pass.
        a_is_direct = not orig["A_is_direct"]

        story_path = os.path.normpath(
            os.path.join(SWITCHING_DIR, "..", "stories", f"{inst}.json"))
        with open(story_path, encoding="utf-8") as f:
            story = json.load(f)

        d_path = os.path.join(SWITCHING_DIR, inst, f"direct_{es}.json")
        t_path = os.path.join(SWITCHING_DIR, inst, f"twostage_{es}.json")
        if not (os.path.exists(d_path) and os.path.exists(t_path)):
            print(f"[miss] {inst}/{es}: missing switching files -- skipping.")
            continue
        with open(d_path, encoding="utf-8") as f:
            direct = json.load(f)
        with open(t_path, encoding="utf-8") as f:
            twostage = json.load(f)

        if a_is_direct:
            tA, tB = transcript_for(story, direct), transcript_for(story, twostage)
        else:
            tA, tB = transcript_for(story, twostage), transcript_for(story, direct)

        try:
            judges = {}
            for jname, jspec in JUDGE_MODELS.items():
                print(f"[flip ] {inst}/{es} <- {jname} (A={'direct' if a_is_direct else 'two_stage'})")
                raw = judge_pair(jspec, story["personality"],
                                 story["personality_profile"], tA, tB)
                judges[jname] = {
                    "cc_winner": map_winner(raw["cc"]["winner"], a_is_direct),
                    "cc_confidence": raw["cc"]["confidence"],
                    "ef_winner": map_winner(raw["ef"]["winner"], a_is_direct),
                    "ef_confidence": raw["ef"]["confidence"],
                    "raw_cc_ab": raw["cc"]["winner"],
                    "raw_ef_ab": raw["ef"]["winner"],
                    "rationale": raw["rationale"],
                }
        except Exception as e:  # noqa: BLE001
            print(f"[fail] {inst}/{es}: {e} -- skipping; re-run to retry.")
            failures.append(f"{inst}/{es}")
            continue

        record = {
            "instance_id": inst,
            "expression_set": es,
            "personality": story["personality"],
            "A_is_direct": a_is_direct,   # flipped relative to the original pass
            "judges": judges,
        }
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(record, f, ensure_ascii=False, indent=2)

    if failures:
        print(f"\nStep 3c finished with {len(failures)} unfinished pair(s): "
              f"{', '.join(failures)}\nRe-run `python step3c_pairwise_flip.py` to fill them in.")
    else:
        print("Step 3c done.")


if __name__ == "__main__":
    main()
