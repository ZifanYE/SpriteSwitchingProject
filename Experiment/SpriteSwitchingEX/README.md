# Sprite-Switching Experiment Pipeline

Implements the 2 x 3 experiment from *From Emotion to Expression: LLM-Driven
Automatic Character Sprite Switching in Visual Novels*.

```
config.py                 # models, scale, paths, protagonist profiles  <-- edit this first
llm_client.py             # OpenAI + Anthropic wrapper (retry + JSON parse)
expression_sets.py        # Ekman / Extended / Adaptive sets
step1_generate_stories.py # -> data/stories/story_XXXX.json
step2_sprite_switching.py # -> data/switching/story_XXXX/{pipeline}_{set}.json   (6 per story)
step3_llm_judge.py        # -> data/judged/story_XXXX/{pipeline}_{set}.json       (+ judges)
step4_analysis.py         # -> data/results/table1|2|3.csv  (+ printed tables)
run_all.py                # runs all four steps in order
```

## Data flow (matches your three-folder design)

1. **stories/** — one JSON per instance: `{instance_id, personality, dialogue, ...}`
2. **switching/** — per story, a subfolder with 6 condition files (one-to-one with the story)
3. **judged/** — mirror of switching/, each file augmented with a `judges` block (GPT + Claude)
4. **results/** — the three tables

## Setup

```bash
pip install -r requirements.txt
export OPENAI_API_KEY=sk-...
export ANTHROPIC_API_KEY=sk-ant-...
```

Open `config.py` and set the model strings (the paper uses GPT-5.4 + Claude as
judges and a smaller model for switching), `N_INSTANCES`, and the protagonist /
personality profiles.

## Run

```bash
python run_all.py          # everything
# or step by step:
python step1_generate_stories.py
python step2_sprite_switching.py
python step3_llm_judge.py
python step4_analysis.py
```

All steps are **resumable**: re-running skips files that already exist, so an
interrupted run continues where it stopped. Delete the relevant folder to redo a
stage.

## Design notes

- **Factors.** Pipeline `{direct, twostage}` x Expression set `{ekman, extended,
  adaptive}` = 6 conditions, all evaluated on the *same* story instances so
  differences come only from the method.
- **Two-stage** = Stage 1 free-form emotion description -> Stage 2 map to a label.
  **Direct** = dialogue straight to a label. This is the only difference between
  the two pipelines; inputs are identical.
- **Adaptive set** is generated once per personality and cached in
  `data/adaptive_sets.json`, then kept fixed throughout the experiment.
- **Judges** are blind to method: they only see personality + dialogue + the
  chosen sprite labels.
- **Stats**: non-parametric bootstrap (B=10,000) for per-condition CIs; paired
  bootstrap for Direct vs Two-stage; ICC(2,1) for GPT/Claude agreement.

## Judge validation (Table 3, optional)

To get the human-vs-judge Spearman rows, rate a subset of samples and save them
as `data/human_ratings.csv` (see `human_ratings.csv.template`), then re-run
step 4.

## Sprite assets

This pipeline covers the **text -> emotion -> expression-label** loop and the
evaluation. Generating the actual sprite images (DALL-E 3 base + GPT-Image-1
variants + rembg background removal, Section 3.2) is a separate preprocessing
stage; the labels predicted here index into those assets.
