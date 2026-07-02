import json
from pathlib import Path
from collections import OrderedDict

# ==================================================
# Paths
# ==================================================
stories_dir = Path("data/stories")
switching_dir = Path("data/switching")
output_dir = Path("data/merged")

output_dir.mkdir(exist_ok=True)

# ==================================================
# Merge one story
# ==================================================
def merge_story(original_story_path, switching_json_path, output_path):

    # -----------------------------
    # Load files
    # -----------------------------
    with open(original_story_path, "r", encoding="utf-8") as f:
        story = json.load(f)

    with open(switching_json_path, "r", encoding="utf-8") as f:
        switching = json.load(f)

    # -----------------------------
    # Build sprite map
    # -----------------------------
    sprite_map = {
        label: idx
        for idx, label in enumerate(switching["expression_labels"])
    }

    # -----------------------------
    # turn -> label
    # -----------------------------
    label_map = {
        item["turn"]: item["label"]
        for item in switching["sprite_sequence"]
    }

    # -----------------------------
    # Merge label + sprite index
    # -----------------------------
    for node in story["dialogue"]:

        if node["type"] != "dialogue":
            continue

        turn = node["turn"]

        if turn in label_map:
            label = label_map[turn]
            node["label"] = label
            node["sprite"] = sprite_map[label]

    # -----------------------------
    # Build output JSON
    # -----------------------------
    merged = OrderedDict()

    merged["instance_id"] = story["instance_id"]
    merged["title"] = story["title"]
    merged["protagonist"] = story["protagonist"]

    # 可以改成其他角色，目前默认就是主角
    merged["studied_speaker"] = story["protagonist"]

    merged["pipeline"] = switching["pipeline"]
    merged["expression_set"] = switching["expression_set"]
    merged["personality"] = switching["personality"]

    merged["spriteMap"] = sprite_map

    merged["dialogue"] = story["dialogue"]

    # -----------------------------
    # Save
    # -----------------------------
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(
            merged,
            f,
            indent=2,
            ensure_ascii=False
        )


# ==================================================
# Merge all stories
# ==================================================
for story_file in sorted(stories_dir.glob("*.json")):

    story_id = story_file.stem

    switch_folder = switching_dir / story_id

    if not switch_folder.exists():
        print(f"Skip {story_id}")
        continue

    out_folder = output_dir / story_id
    out_folder.mkdir(parents=True, exist_ok=True)

    for variant in sorted(switch_folder.glob("*.json")):

        output_path = out_folder / variant.name

        merge_story(
            story_file,
            variant,
            output_path
        )

        print(f"Generated: {output_path}")

print("Done.")