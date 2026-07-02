"""
Step 1 -- Evaluation data preparation.

Build a benchmark of pre-generated story instances. Each instance:
  * fixed protagonist (Aoi),
  * a personality randomly assigned from {introverted, extroverted},
  * a complete VN story generated conditioned on that personality.

Every instance is saved to data/stories/story_XXXX.json and is REUSED across
all six experimental conditions in later steps.

Run:  python step1_generate_stories.py
"""
import json
import os
import random

from config import (MIN_PROTAG_LINES, N_INSTANCES, PERSONALITIES,
                    PERSONALITY_PROFILES, CHARACTER_BASE_PROFILE, FIXED_OPENING,
                    PROTAGONIST, RANDOM_SEED, STORIES_DIR, STORY_MODEL,
                    STORY_MAX_TOKENS, SCENARIO_DIRECTIONS, LLM_SEED)
from llm_client import call_json

STORY_SYSTEM = (
    "You are a professional visual-novel scenario writer. You write emotionally rich, "
    "natural scenes with narration and dialogue. Respond ONLY with JSON."
)


def opening_text():
    lines = []
    for t in FIXED_OPENING:
        if t["type"] == "narration":
            lines.append(f"({t['text']})")
        else:
            lines.append(f"{t['speaker']}: {t['text']}")
    return "\n".join(lines)


def make_prompt(personality, direction):
    return (
        f"You are continuing a visual-novel scene starring {PROTAGONIST}, a first-year "
        "high-school girl, on the first day of the new term.\n\n"
        f"Base profile: {CHARACTER_BASE_PROFILE}\n"
        f"Personality for THIS scene: {personality} -- {PERSONALITY_PROFILES[personality]}\n\n"
        "The scene has ALREADY begun with this FIXED opening (do NOT rewrite or repeat it):\n"
        "-----\n"
        f"{opening_text()}\n"
        "-----\n\n"
        f"# Required direction for THIS story ({direction['name']})\n"
        f"{direction['prompt']}\n\n"
        "Write what happens NEXT, continuing seamlessly from that last line, built around the "
        "required direction above.\n"
        "Requirements:\n"
        "- Invent WHY she is running late, consistent with the required direction.\n"
        "- Commit fully to the assigned direction -- do NOT fall back on the overused "
        "'talking cat' or 'collides with a stray animal' opening unless the direction explicitly "
        "calls for an animal. Aim for a premise you have not written before.\n"
        "- Build a real emotional arc with clear ups and downs: tension, relief, a turn, a "
        f"high or low point. Do NOT keep {PROTAGONIST} in one single mood.\n"
        "- Mix narration with spoken dialogue from "
        f"{PROTAGONIST} and 1-2 other characters who appear naturally.\n"
        f"- {PROTAGONIST} must have at least {MIN_PROTAG_LINES} spoken lines in your continuation.\n"
        f"- Keep the whole continuation to roughly {MIN_PROTAG_LINES + 2}-{MIN_PROTAG_LINES + 6} "
        f"{PROTAGONIST} lines and about 26 total dialogue/narration turns -- do not write a novella.\n"
        "- Keep each line short, like real VN text.\n\n"
        'Return ONLY the continuation as JSON (do not include the opening):\n'
        '{"title": "...", "dialogue": ['
        '{"type":"narration","text":"..."}, '
        f'{{"type":"dialogue","speaker":"{PROTAGONIST}","text":"..."}}, '
        '{"type":"dialogue","speaker":"<other>","text":"..."}'
        ']}'
    )


def normalize_dialogue(raw_dialogue):
    """Attach turn indices and clean fields."""
    dialogue = []
    for i, t in enumerate(raw_dialogue):
        ttype = t.get("type", "dialogue")
        entry = {"turn": i, "type": ttype, "text": str(t.get("text", "")).strip()}
        if ttype == "dialogue":
            entry["speaker"] = str(t.get("speaker", PROTAGONIST)).strip() or PROTAGONIST
        dialogue.append(entry)
    return dialogue


def count_protagonist_lines(dialogue):
    return sum(1 for t in dialogue if t["type"] == "dialogue" and t["speaker"] == PROTAGONIST)


def main():
    random.seed(RANDOM_SEED)
    for idx in range(1, N_INSTANCES + 1):
        instance_id = f"story_{idx:04d}"
        out_path = os.path.join(STORIES_DIR, f"{instance_id}.json")
        if os.path.exists(out_path):
            print(f"[skip] {instance_id} already exists")
            continue

        personality = random.choice(PERSONALITIES)
        # Rotate through scenario directions so genres are balanced across instances.
        direction = SCENARIO_DIRECTIONS[(idx - 1) % len(SCENARIO_DIRECTIONS)]
        # Per-instance seed: reproducible runs, but each story differs from the others.
        story_seed = LLM_SEED + idx
        print(f"[gen ] {instance_id}  personality={personality}  scenario={direction['name']}")

        # Retry until the protagonist has enough lines in the continuation.
        continuation = []
        title = ""
        for _ in range(3):
            out = call_json(STORY_MODEL, STORY_SYSTEM, make_prompt(personality, direction),
                            temperature=0.95, max_tokens=STORY_MAX_TOKENS, seed=story_seed)
            continuation = out.get("dialogue", [])
            title = out.get("title", "")
            cont_protag = sum(1 for t in continuation
                              if t.get("type") == "dialogue"
                              and str(t.get("speaker", "")).strip() == PROTAGONIST)
            if cont_protag >= MIN_PROTAG_LINES:
                break

        # Prepend the fixed opening, then reindex the whole scene.
        full = [dict(t) for t in FIXED_OPENING] + list(continuation)
        dialogue = normalize_dialogue(full)

        record = {
            "instance_id": instance_id,
            "protagonist": PROTAGONIST,
            "personality": personality,
            "personality_profile": PERSONALITY_PROFILES[personality],
            "character_base_profile": CHARACTER_BASE_PROFILE,
            "scenario": direction["name"],
            "title": title,
            "opening_turns": len(FIXED_OPENING),   # first N turns are the shared opening
            "dialogue": dialogue,
        }
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(record, f, ensure_ascii=False, indent=2)
        print(f"       saved -> {out_path}  ({count_protagonist_lines(dialogue)} protagonist lines)")

    print("Step 1 done.")


if __name__ == "__main__":
    main()