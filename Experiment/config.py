"""
Central configuration for the LLM-driven sprite-switching experiment.

Edit MODELS and EXPERIMENT SCALE to match your setup, then run the steps
in order (see README.md).
"""
import os

# ------------------------------------------------------------------ #
# API keys -- paste them directly between the quotes below.           #
# Keep this file PRIVATE (don't commit to a public repo / don't share)#
# ------------------------------------------------------------------ #
OPENAI_API_KEY = ""
ANTHROPIC_API_KEY = ""

# Convenience: if you leave a placeholder above, it falls back to an
# environment variable automatically (so both styles work).
if not OPENAI_API_KEY or OPENAI_API_KEY.startswith("sk-REPLACE"):
    OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", OPENAI_API_KEY)
if not ANTHROPIC_API_KEY or ANTHROPIC_API_KEY.startswith("sk-ant-REPLACE"):
    ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", ANTHROPIC_API_KEY)

# ------------------------------------------------------------------ #
# Models                                                              #
#   provider must be "openai" or "anthropic".                         #
#   NOTE: verify the exact model id strings against your account --   #
#   the OpenAI ids below are best guesses for the GPT-5.4 family.      #
# ------------------------------------------------------------------ #
STORY_MODEL     = {"provider": "openai", "model": "gpt-5.4"}        # Step 1: story generation (strongest)
SWITCHING_MODEL = {"provider": "openai", "model": "gpt-5.4-mini"}   # Step 2: sprite switching (the "smaller" model)

# Step 3: the two judges. Paper uses a strong judge ("comparable or superior
# capability" to the system under test). GPT-5.4 is the strong judge; Claude
# Haiku 4.5 is used here as the second judge (you can upgrade it later).
JUDGE_MODELS = {
    "gpt":    {"provider": "openai",    "model": "gpt-5.4"},
    "claude": {"provider": "anthropic", "model": "claude-opus-4-8"},
}

# --- Small-scale judge-upgrade test (judge_upgrade_test.py) ---------- #
# Test whether a STRONGER Claude judge agrees better with GPT-5.4.
# Default is Claude Opus 4.8 -- the strongest currently-available Claude, on a
# tier comparable to GPT-5.4. To try Claude Fable 5 / Mythos 5 instead, swap the
# model string below; note their API access is temporarily suspended per
# Anthropic's export-control notice, so they may not be callable right now.
JUDGE_TEST_MODEL = {"provider": "anthropic", "model": "claude-opus-4-8"}
# JUDGE_TEST_MODEL = {"provider": "anthropic", "model": "claude-fable-5"}  # when access is restored
JUDGE_TEST_N = 5   # number of story instances to re-judge in the test

# ------------------------------------------------------------------ #
# Experiment scale                                                    #
# ------------------------------------------------------------------ #
N_INSTANCES      = 20                                  # number of story instances
PERSONALITIES    = ["introverted", "extroverted"]     # randomly assigned per instance
MIN_PROTAG_LINES = 8                                  # min spoken lines for the protagonist
RANDOM_SEED      = 42
LLM_SEED         = 7                                  # decoding seed (OpenAI) for reproducibility
MAX_RETRIES      = 6                                   # per-call retries on transient/network errors
STORY_MAX_TOKENS = 8000                               # story JSON is long; reasoning models also spend tokens here
BOOTSTRAP_B      = 10_000                              # bootstrap resamples (paper: 10,000)

# ------------------------------------------------------------------ #
# Protagonist                                                         #
# ------------------------------------------------------------------ #
PROTAGONIST = "Aoi"

CHARACTER_BASE_PROFILE = (
    "Aoi is a first-year high-school girl. Today is the first day of the new school term, "
    "and she is an ordinary student hoping to start the year off on the right foot."
)

# Shared, byte-identical opening prepended to EVERY story instance.
# Kept deliberately minimal: it ONLY establishes the classic galgame hook
# (first day of term, Aoi is about to be late). The REASON she is late, and
# everything that follows, is invented by the story generator -- any genre,
# any direction, as long as it has real emotional ups and downs.
FIXED_OPENING = [
    {"type": "narration",
     "text": "The first morning of the new school term -- and Aoi is already running late."},
    {"type": "dialogue", "speaker": PROTAGONIST,
     "text": "No, no, no -- not on the very first day!"},
]

# ------------------------------------------------------------------ #
# Scenario directions -- break the model out of the "cat at the       #
# corner" rut. Each story instance is assigned ONE direction (rotated #
# in order so all genres get balanced coverage), and the generator    #
# must build the scene around it. Edit / add freely.                  #
# ------------------------------------------------------------------ #
SCENARIO_DIRECTIONS = [
    {"name": "slice_of_life",
     "prompt": "A grounded, realistic first day with NO supernatural element. The trouble is "
               "ordinary -- a train delay, the wrong classroom, a forgotten gym uniform, a "
               "spilled drink. Emotion comes from everyday stakes: nerves, embarrassment, a "
               "small unexpected kindness or triumph."},
    {"name": "bump_into_someone",
     "prompt": "Rounding a corner she physically collides with another PERSON -- a transfer "
               "student, a brusque upperclassman, a stranger in a hurry. The collision reshapes "
               "her morning: a meet-cute, a budding rivalry, or a mortifying misunderstanding."},
    {"name": "mysterious_stranger",
     "prompt": "She crosses paths with someone who behaves oddly -- says something they "
               "couldn't possibly know, leaves behind a strange object, or vanishes when she "
               "looks again. A thread of mystery / suspense that need not fully resolve."},
    {"name": "helping_hand",
     "prompt": "She stops to help a person or an animal in genuine trouble -- a lost child, an "
               "injured stranger, someone being cornered. It makes her even later, but the "
               "choice changes her day. Grounded and emotionally weighty."},
    {"name": "supernatural",
     "prompt": "Something uncanny intrudes on the mundane: a ghost at the shrine gate, a "
               "morning that keeps repeating, an omen, a corridor that shouldn't exist. Eerie "
               "or wondrous rather than action-packed."},
    {"name": "sci_fi_alien",
     "prompt": "A science-fiction twist: an alien hiding in plain sight, a glitch in reality, a "
               "scrap of impossible technology, a visitor who is clearly not from here. Lead "
               "with wonder and disorientation."},
    {"name": "fantasy_portal",
     "prompt": "She slips into, or brushes against, a fantasy world -- a hidden door, a spell "
               "gone sideways, a creature out of a fairy tale walking the same street. Vivid "
               "high-fantasy color."},
    {"name": "comedy_mishap",
     "prompt": "An escalating chain of comedic disasters between her front door and the school "
               "gate. Fast, absurd, and physical -- but include one real low point before the "
               "recovery so the arc still has a dip."},
]

PERSONALITY_PROFILES = {
    "introverted": (
        "Reserved and soft-spoken. Keeps strong feelings inside, prefers to listen, "
        "and shows emotion subtly — a slight smile, a quiet pause, downcast eyes. "
        "Rarely raises her voice even when upset."
    ),
    "extroverted": (
        "Outgoing and expressive. Speaks her mind readily, laughs openly, and her face "
        "moves with her mood — bright grins, wide-eyed surprise, visible frustration. "
        "Comfortable taking the lead in a conversation."
    ),
}

# ------------------------------------------------------------------ #
# Paths                                                               #
# ------------------------------------------------------------------ #
BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
DATA_DIR      = os.path.join(BASE_DIR, "data")
STORIES_DIR   = os.path.join(DATA_DIR, "stories")    # Step 1 output
SWITCHING_DIR = os.path.join(DATA_DIR, "switching")  # Step 2 output (one subfolder per story)
JUDGED_DIR    = os.path.join(DATA_DIR, "judged")     # Step 3 output (one subfolder per story)
RESULTS_DIR   = os.path.join(DATA_DIR, "results")    # Step 4 output (tables)
HUMAN_RATINGS  = os.path.join(DATA_DIR, "human_ratings.csv")  # optional, for judge validation

for _d in (DATA_DIR, STORIES_DIR, SWITCHING_DIR, JUDGED_DIR, RESULTS_DIR):
    os.makedirs(_d, exist_ok=True)