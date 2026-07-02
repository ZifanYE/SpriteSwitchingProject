"""
Step 3 -- LLM-as-judge evaluation.

Two independent judges (GPT and Claude) score every switching result, blind to
the generation method, on a 5-point Likert scale along two dimensions:

  * Character Consistency (CC)  -- adapted from CharacterEval [9]
  * Emotional Fidelity   (EF)  -- adapted from EmoCharacter [3]

Output layout mirrors Step 2 (one subfolder per story):
  data/judged/story_0001/direct_ekman.json   (= switching result + 'judges' block)

Run:  python step3_llm_judge.py
"""
import glob
import json
import os

from config import JUDGE_MODELS, JUDGED_DIR, PROTAGONIST, SWITCHING_DIR
from llm_client import call_json

JUDGE_SYSTEM = (
    "You are a rigorous, impartial evaluator of facial-expression control in visual novels. "
    "You will be given a character's assigned personality, the full dialogue, and the facial "
    "sprite chosen for each of the character's lines (shown in [sprite: ...]). You do NOT know "
    "which method produced these sprites; judge only the evidence in front of you. "
    "Work through the scene line by line internally, then assign integer scores. Be strict and "
    "use the full 1-5 range: reserve 5 for near-flawless control and do not cluster around 3-4. "
    "Do not reward verbosity, label quantity, or anything other than correctness. "
    "Output only JSON."
)

RUBRIC = (
    "Rate TWO dimensions, each an integer from 1 to 5.\n\n"
    "Character Consistency (CC) -- does the sprite sequence, as a whole, behave the way THIS "
    "personality would express itself?\n"
    "  1 = sprites contradict the assigned personality throughout.\n"
    "  2 = frequently out of character.\n"
    "  3 = mixed; some choices fit the personality, several do not.\n"
    "  4 = mostly in character, only minor slips.\n"
    "  5 = every choice coherently reflects the personality across the whole scene.\n\n"
    "Emotional Fidelity (EF) -- does each sprite match the emotion implied by its line, and do "
    "the sprites form a coherent emotional progression across consecutive turns?\n"
    "  1 = sprites bear little relation to the lines' emotions.\n"
    "  2 = often mismatched or emotionally flat.\n"
    "  3 = roughly right but misses several emotional turns.\n"
    "  4 = accurate for most lines, with a coherent progression.\n"
    "  5 = each sprite precisely matches its line's emotion, with a smooth, coherent arc.\n"
)


def build_transcript(story, switching):
    """Pair each protagonist line with its assigned sprite label for the judge."""
    label_by_turn = {s["turn"]: s["label"] for s in switching["sprite_sequence"]}
    lines = []
    for t in story["dialogue"]:
        if t["type"] == "narration":
            lines.append(f"  ({t['text']})")
        elif t["speaker"] == PROTAGONIST:
            sprite = label_by_turn.get(t["turn"], "?")
            lines.append(f"  {t['speaker']} [sprite: {sprite}]: {t['text']}")
        else:
            lines.append(f"  {t['speaker']}: {t['text']}")
    return "\n".join(lines)


def judge_one(judge_spec, personality, profile, transcript):
    user = (
        f"# Assigned personality\n{personality} -- {profile}\n\n"
        f"# Scoring rubric\n{RUBRIC}\n"
        "# Dialogue with the character's chosen facial sprites\n"
        f"{transcript}\n\n"
        "Assess the [sprite: ...] choices against the rubric. Return JSON only:\n"
        '{"character_consistency": <1-5 integer>, '
        '"emotional_fidelity": <1-5 integer>, '
        '"rationale": "<one terse sentence citing the strongest evidence>"}'
    )
    out = call_json(judge_spec, JUDGE_SYSTEM, user, temperature=0.0, max_tokens=400)
    return {
        "character_consistency": int(out["character_consistency"]),
        "emotional_fidelity": int(out["emotional_fidelity"]),
        "rationale": str(out.get("rationale", "")),
    }


def main():
    story_dirs = sorted(glob.glob(os.path.join(SWITCHING_DIR, "story_*")))
    if not story_dirs:
        print("No switching results found. Run step2 first.")
        return

    for sdir in story_dirs:
        inst = os.path.basename(sdir)
        story_path = os.path.join(SWITCHING_DIR, "..", "stories", f"{inst}.json")
        story_path = os.path.normpath(story_path)
        with open(story_path, "r", encoding="utf-8") as f:
            story = json.load(f)

        out_dir = os.path.join(JUDGED_DIR, inst)
        os.makedirs(out_dir, exist_ok=True)

        for cond_file in sorted(glob.glob(os.path.join(sdir, "*.json"))):
            cond = os.path.splitext(os.path.basename(cond_file))[0]
            out_path = os.path.join(out_dir, f"{cond}.json")
            if os.path.exists(out_path):
                print(f"[skip] {inst}/{cond}")
                continue

            with open(cond_file, "r", encoding="utf-8") as f:
                switching = json.load(f)
            transcript = build_transcript(story, switching)

            judges = {}
            for judge_name, judge_spec in JUDGE_MODELS.items():
                print(f"[judge] {inst}/{cond} <- {judge_name}")
                judges[judge_name] = judge_one(
                    judge_spec, story["personality"],
                    story["personality_profile"], transcript)

            record = dict(switching)
            record["judges"] = judges
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(record, f, ensure_ascii=False, indent=2)

    print("Step 3 done.")


if __name__ == "__main__":
    main()