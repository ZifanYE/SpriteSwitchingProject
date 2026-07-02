"""
CharacterGen_ImageOnly.py
==========================
仅负责图片生成：基于一张角色底图，批量生成不同表情的图片。

支持三种表情集模式：
  --mode ekman     : 7个Ekman基础表情
  --mode extended  : 14个表情（含VN常用）
  --mode adaptive  : 由 GPT-5.4 mini 根据角色背景故事自动决定需要哪些表情

用法示例：
  python CharacterGen_ImageOnly.py --image Eamon_base.png --mode ekman
  python CharacterGen_ImageOnly.py --image Eamon_base.png --mode extended
  python CharacterGen_ImageOnly.py --image Eamon_base.png --mode adaptive
"""

import os
import json
import base64
import argparse
from openai import OpenAI

# ── 初始化 ────────────────────────────────────────────────────────────────────

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", "sk-"))

# ── 表情集定义 ─────────────────────────────────────────────────────────────────

EKMAN_LABELS = [
    "happiness", "sadness", "anger", "fear", "disgust", "surprise", "neutral"
]

EXTENDED_LABELS = [
    "Slightly happy, faint smile", "Natural smile, cheerful face", "Serious, expressionless, unsociable", 
    "Displeased, frown", "Uneasy,confused", "Exasperated", "Interested, attentive",
    "Surprised", "sad", "angry", "Unconvinced", "Astonished", "Crying hard", "Holding back tears"
]

# ── 角色背景故事（Adaptive模式的输入依据）────────────────────────────────────────

CHARACTER_BACKSTORY = """
Character: A female high school student on her first day of high school.
A slightly impulsive, but very energetic girl.
This is a famous VN scene, There are no restrictions on the theme; it can be any type of story, such as school life, adventure, fantasy, or science fiction.
This character will be used in a visual novel. 
"""

# ── Adaptive 标签生成（GPT-5.4 mini）────────────────────────────────────────

def generate_adaptive_labels() -> list[str]:
    """
    让 GPT-5.4 mini根据角色背景故事，自主决定这个角色需要哪些表情标签。
    返回表情标签列表。
    """
    print("🤖 [Adaptive] 正在让 GPT-5.4 mini 分析角色故事，决定表情集...")

    response = client.chat.completions.create(
        model="gpt-5.4-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an expert visual novel character designer. "
                    "Given a character's background story and emotional arc, "
                    "decide what facial expression labels this character needs. "
                    "Choose labels that:\n"
                    "  1. Cover the full emotional range of the story\n"
                    "  2. Are visually distinct and clearly drawable as facial expressions\n"
                    "  3. Are specific enough to be meaningful (avoid overly generic labels)\n"
                    "  4. Number suitable for a visual novel character\n\n"
                    "Return ONLY a JSON array of lowercase English strings. "
                    "No explanation, no markdown, just the array.\n"
                    "Example: [\"happy\", \"sadness\", \"anger\"]"
                )
            },
            {
                "role": "user",
                "content": f"Character story:\n{CHARACTER_BACKSTORY.strip()}"
            }
        ],
        max_completion_tokens=300,
        temperature=0.7,
    )

    raw = response.choices[0].message.content.strip()

    try:
        labels = json.loads(raw)
        if isinstance(labels, list) and all(isinstance(l, str) for l in labels):
            return [l.lower().strip() for l in labels]
    except json.JSONDecodeError:
        pass

    # fallback: 尝试提取 [...] 部分
    import re
    match = re.search(r"\[.*?\]", raw, re.DOTALL)
    if match:
        try:
            labels = json.loads(match.group())
            return [l.lower().strip() for l in labels]
        except Exception:
            pass

    print(f"  ⚠️  解析失败，原始输出：{raw[:200]}")
    print("  ⚠️  回退至 Extended 标签集")
    return EXTENDED_LABELS

# ── 工具函数 ───────────────────────────────────────────────────────────────────

def load_image_bytes(path: str) -> bytes:
    with open(path, "rb") as f:
        return f.read()

def save_image_b64(b64_data: str, out_path: str):
    with open(out_path, "wb") as f:
        f.write(base64.b64decode(b64_data))

# ── 图片生成 ───────────────────────────────────────────────────────────────────

def generate_expression(image_bytes: bytes, expression: str, out_path: str) -> bool:
    """
    调用 gpt-image-1 的图像编辑接口，修改角色表情并保存结果。
    返回 True 表示成功。
    """
    prompt = (
        "This is an illustrated anime-style character. "
        f"Change her facial expression to: {expression}. "
        "Keep everything else exactly the same: "
        "hair color, eye color, facial features, skin tone, "
        "clothing/uniform, pose, background, line art style, and color palette. "
        "Only modify the mouth shape, eyebrow position, and eye shape "
        f"to clearly convey the emotion '{expression}'."
    )

    try:
        response = client.images.edit(
            model="gpt-image-1",
            image=("Aoi_base.png", image_bytes, "image/png"),
            prompt=prompt,
            n=1,
            size="1024x1024",
        )
        save_image_b64(response.data[0].b64_json, out_path)
        return True
    except Exception as e:
        print(f"  ❌ 生成失败 [{expression}]: {e}")
        return False

# ── 批量生成 ───────────────────────────────────────────────────────────────────

def run(image_path: str, labels: list[str], output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    image_bytes = load_image_bytes(image_path)

    print(f"\n底图: {image_path}")
    print(f"表情标签 ({len(labels)}个): {', '.join(labels)}")
    print(f"输出目录: {output_dir}\n")

    success_count = 0
    for expression in labels:
        out_path = os.path.join(output_dir, f"{expression}.png")
        print(f"⏳ 生成中: {expression}")
        ok = generate_expression(image_bytes, expression, out_path)
        if ok:
            print(f"  ✅ 保存: {out_path}")
            success_count += 1

    print(f"\n🎉 完成！成功生成 {success_count}/{len(labels)} 张图片。")

# ── 入口 ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="角色表情图片批量生成")
    parser.add_argument("--image",  required=True, help="底图路径（PNG）")
    parser.add_argument("--mode",   choices=["ekman", "extended", "adaptive"], default="ekman")
    parser.add_argument("--output", default="output", help="输出目录（默认: output）")
    args = parser.parse_args()

    if args.mode == "ekman":
        labels = EKMAN_LABELS
    elif args.mode == "extended":
        labels = EXTENDED_LABELS
    elif args.mode == "adaptive":
        labels = generate_adaptive_labels()
        print(f"  → GPT-5.5 选定的表情集: {labels}\n")

    run(args.image, labels, args.output)
