"""
Step 3b -- PAIRWISE LLM-as-judge (direct vs two-stage).

Pointwise Likert scoring (step3) tends to saturate at 4-5. Pairwise judging is
more sensitive: for each matched (instance x expression_set), the two sprite
sequences (direct vs two-stage) are shown SIDE BY SIDE and the judge only has to
say which one is better -- separately for Character Consistency and Emotional
Fidelity -- with a tie option and a confidence level.

Controls:
  * Blind: the two candidates are labelled A / B only; the judge never sees which
    pipeline produced which.
  * Order randomized per sample (seeded -> reproducible) and the mapping is stored,
    so any position bias can be inspected afterwards.

Expression sets are NOT compared pairwise (different label spaces => "all else
equal" fails); only the pipeline factor is compared here.

Output: data/pairwise/story_XXXX/<expression_set>.json

Run:  python step3b_pairwise_judge.py
"""
import glob
import json
import os
import random

from config import (DATA_DIR, JUDGE_MODELS, PROTAGONIST, RANDOM_SEED,
                    SWITCHING_DIR)
from llm_client import call_json
from step3_llm_judge import build_transcript  # reuse the same transcript builder

PAIRWISE_DIR = os.path.join(DATA_DIR, "pairwise")
EXPRESSION_SETS = ["ekman", "extended", "adaptive"]
CONFIDENCE = ["slight", "clear", "strong"]

JUDGE_SYSTEM = (
    "You are a rigorous, impartial judge comparing two versions of facial-sprite control for "
    "the SAME visual-novel scene. The two versions, A and B, were produced by different methods "
    "you cannot see; the dialogue, character, and personality are identical between them -- only "
    "the chosen sprite for each of the character's lines differs. Compare them head to head and "
    "decide which version is better on each dimension. Do not assume A or B is better by default; "
    "judge only the evidence. Output only JSON."
)

RUBRIC = (
    "Compare A and B on TWO dimensions independently:\n"
    "  Character Consistency (CC): which version's sprite sequence better fits THIS character's "
    "assigned personality across the whole scene?\n"
    "  Emotional Fidelity (EF): which version's sprites better match the emotion implied by each "
    "line, with a more coherent emotional progression?\n\n"
    'For each dimension, "winner" is "A", "B", or "tie" (tie only when they are genuinely '
    'indistinguishable), and "confidence" is one of "slight", "clear", "strong".\n'
)


def transcript_for(story, switching):
    """One side of the comparison: dialogue with that version's chosen sprites."""
    return build_transcript(story, switching)


def judge_pair(judge_spec, personality, profile, transcript_A, transcript_B):
    user = (
        f"# Assigned personality\n{personality} -- {profile}\n\n"
        f"# How to judge\n{RUBRIC}\n"
        "# Version A (dialogue with A's chosen sprites)\n"
        f"{transcript_A}\n\n"
        "# Version B (dialogue with B's chosen sprites)\n"
        f"{transcript_B}\n\n"
        "Return JSON only:\n"
        '{"cc": {"winner": "A|B|tie", "confidence": "slight|clear|strong"},'
        ' "ef": {"winner": "A|B|tie", "confidence": "slight|clear|strong"},'
        ' "rationale": "<one terse sentence>"}'
    )
    out = call_json(judge_spec, JUDGE_SYSTEM, user, temperature=0.0, max_tokens=400)

    def clean(dim):
        d = out.get(dim, {}) or {}
        w = str(d.get("winner", "tie")).strip().upper()
        w = w if w in ("A", "B") else "tie"
        c = str(d.get("confidence", "slight")).strip().lower()
        c = c if c in CONFIDENCE else "slight"
        return {"winner": w, "confidence": c}

    return {"cc": clean("cc"), "ef": clean("ef"),
            "rationale": str(out.get("rationale", ""))}


def map_winner(ab_winner, a_is_direct):
    """Translate an A/B/tie verdict into direct/two-stage/tie using the stored order."""
    if ab_winner == "tie":
        return "tie"
    a_pipeline = "direct" if a_is_direct else "two_stage"
    b_pipeline = "two_stage" if a_is_direct else "direct"
    return a_pipeline if ab_winner == "A" else b_pipeline


def main():
    rng = random.Random(RANDOM_SEED)
    story_dirs = sorted(glob.glob(os.path.join(SWITCHING_DIR, "story_*")))
    if not story_dirs:
        print("No switching results found. Run step2 first.")
        return

    failures = []
    for sdir in story_dirs:
        inst = os.path.basename(sdir)
        story_path = os.path.normpath(
            os.path.join(SWITCHING_DIR, "..", "stories", f"{inst}.json"))
        with open(story_path, "r", encoding="utf-8") as f:
            story = json.load(f)
        out_dir = os.path.join(PAIRWISE_DIR, inst)
        os.makedirs(out_dir, exist_ok=True)

        for es in EXPRESSION_SETS:
            out_path = os.path.join(out_dir, f"{es}.json")
            if os.path.exists(out_path):
                print(f"[skip] {inst}/{es}")
                continue

            d_path = os.path.join(sdir, f"direct_{es}.json")
            t_path = os.path.join(sdir, f"twostage_{es}.json")
            if not (os.path.exists(d_path) and os.path.exists(t_path)):
                print(f"[miss] {inst}/{es}: need both direct_ and twostage_ -- skipping.")
                continue

            with open(d_path, encoding="utf-8") as f:
                direct = json.load(f)
            with open(t_path, encoding="utf-8") as f:
                twostage = json.load(f)

            # Randomize which pipeline is shown as "A" (seeded -> reproducible).
            a_is_direct = rng.random() < 0.5
            if a_is_direct:
                tA, tB = transcript_for(story, direct), transcript_for(story, twostage)
            else:
                tA, tB = transcript_for(story, twostage), transcript_for(story, direct)

            try:
                judges = {}
                for jname, jspec in JUDGE_MODELS.items():
                    print(f"[pair ] {inst}/{es} <- {jname} (A={'direct' if a_is_direct else 'two_stage'})")
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
                "A_is_direct": a_is_direct,   # stored order, for position-bias checks
                "judges": judges,
            }
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(record, f, ensure_ascii=False, indent=2)

    if failures:
        print(f"\nStep 3b finished with {len(failures)} unfinished pair(s): "
              f"{', '.join(failures)}\nRe-run `python step3b_pairwise_judge.py` to fill them in.")
    else:
        print("Step 3b done.")


if __name__ == "__main__":
    main()
