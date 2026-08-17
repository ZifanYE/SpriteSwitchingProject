"""
Step 2 -- Two-stage / Direct sprite switching across the 2 x 3 design.

For every story instance and every one of the 6 conditions, predict an
expression label for each protagonist line and emit the full sprite sequence.

Output layout (one subfolder per story, conditions one-to-one with the story):
  data/switching/story_0001/direct_ekman.json
  data/switching/story_0001/direct_extended.json
  data/switching/story_0001/direct_adaptive.json
  data/switching/story_0001/twostage_ekman.json
  data/switching/story_0001/twostage_extended.json
  data/switching/story_0001/twostage_adaptive.json

Run:  python step2_sprite_switching.py
"""
import glob
import json
import os

from config import (PROTAGONIST, STORIES_DIR, SWITCHING_DIR, SWITCHING_MODEL)
from expression_sets import (EXPRESSION_SET_NAMES, get_expression_set,
                             validate_label)
from llm_client import call_json

PIPELINES = ["direct", "twostage"]


# ---------------------------------------------------------------- helpers --- #
def history_text(dialogue, upto_turn):
    lines = []
    for t in dialogue:
        if t["turn"] >= upto_turn:
            break
        if t["type"] == "narration":
            lines.append(f"({t['text']})")
        else:
            lines.append(f"{t['speaker']}: {t['text']}")
    return "\n".join(lines) if lines else "(scene start)"


# ----------------------------------------------------------- direct path --- #
def direct_label(profile, personality, history, current, labels):
    system = (
        "You are an expression-control module for a visual-novel engine. Your job is to pick "
        "the single facial sprite that the character should show on their current line. "
        "Decide from: (a) the literal content of the current line, (b) the emotional context "
        "built up by the preceding dialogue, and (c) how THIS character's personality would "
        "outwardly display that feeling. Choose exactly one label from the allowed set and "
        "nothing else. Output only JSON."
    )
    user = (
        f"# Character profile\n{profile}\n\n"
        f"# Personality\n{personality}\n\n"
        f"# Dialogue so far (chronological)\n{history}\n\n"
        f'# Current line spoken by {PROTAGONIST}\n"{current}"\n\n'
        f"# Allowed expression labels\n{labels}\n\n"
        'Return JSON only: {"label": "<exactly one label from the allowed set>"}'
    )
    out = call_json(SWITCHING_MODEL, system, user, temperature=0.0, max_tokens=200)
    return validate_label(out.get("label"), labels), None


# -------------------------------------------------------- two-stage path --- #
def stage1_emotion(profile, personality, history, current):
    system = (
        "You are an affect-analysis module. Read the dialogue in order and infer the "
        "character's inner emotional state ON THEIR CURRENT LINE. Ground your inference in "
        "the current line plus the preceding context, filtered through the character's "
        "personality. Describe that inner state in ONE concise free-form sentence -- do NOT "
        "choose from any fixed list, and do not name a sprite. Output only JSON."
    )
    user = (
        f"# Character profile\n{profile}\n\n"
        f"# Personality\n{personality}\n\n"
        f"# Dialogue so far (chronological)\n{history}\n\n"
        f'# Current line spoken by {PROTAGONIST}\n"{current}"\n\n'
        'Return JSON only: {"emotion": "<one-sentence free-form description>"}'
    )
    out = call_json(SWITCHING_MODEL, system, user, temperature=0.0, max_tokens=200)
    return str(out.get("emotion", "")).strip()


def stage2_map(profile, personality, history, current, emotion, labels):
    system = (
        "You select the facial sprite a specific character should display, given an inferred "
        "emotional state. The mapping from emotion to outward expression is CHARACTER-DEPENDENT: "
        "the same inner feeling can surface differently depending on the character's personality "
        "(e.g. a reserved character may show a faint, restrained version of an emotion an "
        "outgoing character would show openly). Choose the single label from the allowed set "
        "that best captures how THIS character, with THIS personality, would outwardly show the "
        "described state on their current line. If several fit, pick the most specific one the "
        "evidence supports. Output only JSON."
    )
    user = (
        f"# Character profile\n{profile}\n\n"
        f"# Personality\n{personality}\n\n"
        f"# Dialogue so far (chronological)\n{history}\n\n"
        f'# Current line spoken by {PROTAGONIST}\n"{current}"\n\n'
        f'# Inferred inner emotional state\n"{emotion}"\n\n'
        f"# Allowed expression labels\n{labels}\n\n"
        'Return JSON only: {"label": "<exactly one label from the allowed set>"}'
    )
    out = call_json(SWITCHING_MODEL, system, user, temperature=0.0, max_tokens=100)
    return validate_label(out.get("label"), labels)


def twostage_label(profile, personality, history, current, labels):
    emotion = stage1_emotion(profile, personality, history, current)
    label = stage2_map(profile, personality, history, current, emotion, labels)
    return label, emotion


# ----------------------------------------------------------------- driver --- #
def run_condition(story, pipeline, set_name):
    personality = story["personality"]
    profile = f"{story['character_base_profile']} {story['personality_profile']}"
    labels = get_expression_set(set_name, personality)
    dialogue = story["dialogue"]

    sprite_sequence = []
    for t in dialogue:
        if t["type"] != "dialogue" or t["speaker"] != PROTAGONIST:
            continue
        hist = history_text(dialogue, t["turn"])
        if pipeline == "direct":
            label, emotion = direct_label(profile, personality, hist, t["text"], labels)
        else:
            label, emotion = twostage_label(profile, personality, hist, t["text"], labels)
        sprite_sequence.append({
            "turn": t["turn"],
            "text": t["text"],
            "emotion_description": emotion,   # None for direct
            "label": label,
        })

    return {
        "instance_id": story["instance_id"],
        "pipeline": pipeline,
        "expression_set": set_name,
        "personality": personality,
        "expression_labels": labels,
        "sprite_sequence": sprite_sequence,
    }


def main():
    story_files = sorted(glob.glob(os.path.join(STORIES_DIR, "story_*.json")))
    if not story_files:
        print("No stories found. Run step1 first.")
        return

    failures = []
    for sf in story_files:
        with open(sf, "r", encoding="utf-8") as f:
            story = json.load(f)
        inst = story["instance_id"]
        out_dir = os.path.join(SWITCHING_DIR, inst)
        os.makedirs(out_dir, exist_ok=True)

        for pipeline in PIPELINES:
            for set_name in EXPRESSION_SET_NAMES:
                cond = f"{pipeline}_{set_name}"
                out_path = os.path.join(out_dir, f"{cond}.json")
                if os.path.exists(out_path):
                    print(f"[skip] {inst}/{cond}")
                    continue
                print(f"[run ] {inst}/{cond}")
                try:
                    result = run_condition(story, pipeline, set_name)
                except Exception as e:  # noqa: BLE001
                    # Don't write a partial file -> this condition is retried on the next run.
                    print(f"[fail] {inst}/{cond}: {e} -- skipping; re-run to retry.")
                    failures.append(f"{inst}/{cond}")
                    continue
                with open(out_path, "w", encoding="utf-8") as f:
                    json.dump(result, f, ensure_ascii=False, indent=2)

    if failures:
        print(f"\nStep 2 finished with {len(failures)} unfinished condition(s): "
              f"{', '.join(failures)}\nRe-run `python step2_sprite_switching.py` to fill them in.")
    else:
        print("Step 2 done.")


if __name__ == "__main__":
    main()