"""
The three expression sets X used in the 2 x 3 design. ALL THREE ARE FIXED AND
PREDEFINED BEFORE THE EXPERIMENT -- nothing here is generated at runtime.

  - Ekman    (7 labels):  classical basic-emotion taxonomy.
  - Extended (14 labels): generic VN expression set (after Muraji et al. [5]).
  - Adaptive (~10 labels): the character-specific set you decided in advance for
                           the target character. Edit ADAPTIVE below to match the
                           set you set up before evaluation; it is then kept fixed
                           throughout the whole experiment.
"""

# Classical Ekman six + neutral.
EKMAN = ["anger", "disgust", "fear", "happiness", "neutral", "sadness", "surprise"]

# Generic VN set (14), after the labels reported in Muraji et al. [5] / Fig. 2.
EXTENDED = [
    "angry", "astonished", "crying_hard", "displeased_frown", "exasperated",
    "holding_back_tears", "interested_attentive", "natural_smile_cheerful",
    "sad", "serious_expressionless", "slightly_happy_faint_smile",
    "surprised", "unconvinced", "uneasy_confused",
]

# Character-specific Adaptive set -- PRESET BY YOU before the experiment.
# (Default below matches the Adaptive row in Fig. 2; replace with your own.)
ADAPTIVE = [
    "angry", "anxious", "confused", "determined", "embarrassed",
    "neutral", "sad", "shy", "smile", "surprised",
]

# Canonical names used everywhere.
EXPRESSION_SET_NAMES = ["ekman", "extended", "adaptive"]

_SETS = {"ekman": EKMAN, "extended": EXTENDED, "adaptive": ADAPTIVE}


def get_expression_set(name, personality=None):
    """Return the fixed label list for the named set.

    `personality` is accepted for call-site compatibility but ignored: every set,
    including Adaptive, is predefined and identical across all instances.
    """
    name = name.lower()
    if name not in _SETS:
        raise ValueError(f"Unknown expression set: {name}")
    return list(_SETS[name])


def validate_label(label, allowed):
    """Snap a model-returned label onto the allowed set (robust to casing/spacing)."""
    if label is None:
        label = ""
    norm = str(label).strip().lower().replace(" ", "_")
    lut = {a.lower(): a for a in allowed}
    if norm in lut:
        return lut[norm]
    # loose containment match
    for a in allowed:
        if norm and (norm in a.lower() or a.lower() in norm):
            return a
    # sensible fallback
    for fb in ("neutral", "natural_smile_cheerful", "serious_expressionless"):
        if fb in lut:
            return lut[fb]
    return allowed[0]