"""
Unified LLM client: wraps OpenAI and Anthropic behind one call, with retry,
robust JSON extraction, and reproducible decoding.

It transparently handles the GPT-5 family API differences:
  * those models use `max_completion_tokens` instead of `max_tokens`;
  * some reasoning models reject a custom `temperature` (only default allowed).
The client probes the right parameter scheme once per model and caches it, so
later calls go straight through.
"""
import json
import random
import re
import time

from config import OPENAI_API_KEY, ANTHROPIC_API_KEY, LLM_SEED, MAX_RETRIES

_openai_client = None
_anthropic_client = None
_openai_scheme = {}   # model -> (tokens_param, use_temperature)


def _get_openai():
    global _openai_client
    if _openai_client is None:
        from openai import OpenAI
        _openai_client = OpenAI(api_key=OPENAI_API_KEY)
    return _openai_client


def _get_anthropic():
    global _anthropic_client
    if _anthropic_client is None:
        from anthropic import Anthropic
        _anthropic_client = Anthropic(api_key=ANTHROPIC_API_KEY)
    return _anthropic_client


_PARAM_ERR_HINTS = (
    "temperature", "max_tokens", "max_completion_tokens",
    "unsupported", "unknown parameter", "not supported", "invalid parameter",
)


def _openai_call(model, system, user, temperature, max_tokens, seed):
    client = _get_openai()
    messages = [{"role": "system", "content": system},
                {"role": "user", "content": user}]

    def build(tokens_param, use_temp):
        kw = {"model": model, "messages": messages, tokens_param: max_tokens}
        if use_temp and temperature is not None:
            kw["temperature"] = temperature
        if seed is not None:
            kw["seed"] = seed
        return kw

    # Use a previously discovered working scheme if we have one.
    if model in _openai_scheme:
        tp, ut = _openai_scheme[model]
        resp = client.chat.completions.create(**build(tp, ut))
        return resp.choices[0].message.content

    # Probe schemes from most- to least-specific.
    schemes = [
        ("max_tokens", True),
        ("max_completion_tokens", True),
        ("max_completion_tokens", False),   # reasoning models: drop temperature
        ("max_tokens", False),
    ]
    last = None
    for tp, ut in schemes:
        try:
            resp = client.chat.completions.create(**build(tp, ut))
            _openai_scheme[model] = (tp, ut)   # remember what worked
            return resp.choices[0].message.content
        except Exception as e:  # noqa: BLE001
            last = e
            if any(h in str(e).lower() for h in _PARAM_ERR_HINTS):
                continue            # parameter problem -> try next scheme
            raise                   # genuine error (rate limit etc.) -> bubble up
    raise last


_anthropic_use_temp = {}   # model -> bool (whether temperature is accepted)


def _anthropic_call(model, system, user, temperature, max_tokens):
    client = _get_anthropic()

    def build(use_temp):
        kw = {"model": model, "max_tokens": max_tokens, "system": system,
              "messages": [{"role": "user", "content": user}]}
        if use_temp and temperature is not None:
            kw["temperature"] = temperature
        return kw

    def run(kw):
        resp = client.messages.create(**kw)
        return "".join(b.text for b in resp.content if b.type == "text")

    # Use a previously discovered scheme if we have one.
    if model in _anthropic_use_temp:
        return run(build(_anthropic_use_temp[model]))

    try:
        out = run(build(True))
        _anthropic_use_temp[model] = True
        return out
    except Exception as e:  # noqa: BLE001
        msg = str(e).lower()
        # Newer models (e.g. Opus 4.8) deprecate/forbid a custom temperature.
        if any(h in msg for h in ("temperature", "unsupported", "deprecated",
                                  "not supported", "invalid")):
            out = run(build(False))
            _anthropic_use_temp[model] = False
            return out
        raise


def call_llm(spec, system, user, temperature=0.7, max_tokens=1500,
             seed=LLM_SEED, retries=MAX_RETRIES):
    """spec = {"provider": ..., "model": ...}. Returns raw text. Retries transient errors."""
    provider, model = spec["provider"], spec["model"]
    last_err = None
    for attempt in range(retries):
        try:
            if provider == "openai":
                return _openai_call(model, system, user, temperature, max_tokens, seed)
            elif provider == "anthropic":
                return _anthropic_call(model, system, user, temperature, max_tokens)
            else:
                raise ValueError(f"Unknown provider: {provider}")
        except Exception as e:  # noqa: BLE001
            last_err = e
            wait = min(2 ** attempt, 30) + random.uniform(0, 1)  # capped backoff + jitter
            print(f"  [warn] {provider}/{model} attempt {attempt + 1}/{retries} failed: {e} "
                  f"(retry in {wait:.1f}s)")
            time.sleep(wait)
    raise RuntimeError(f"LLM call failed after {retries} retries ({provider}/{model}): {last_err}")


def extract_json(text):
    """Pull the first JSON object out of a model response, tolerating code fences."""
    text = (text or "").strip()
    text = re.sub(r"^```(?:json)?", "", text).strip()
    text = re.sub(r"```$", "", text).strip()
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        text = m.group(0)
    return json.loads(text)


def repair_truncated_json(text):
    """Best-effort salvage of JSON cut off mid-output (e.g. hit max_tokens).

    Trims to the last complete '}' and re-balances the closing brackets, which
    recovers a long {"...":[ {..}, {..}, <cut> ]} by dropping the partial tail.
    """
    text = (text or "").strip()
    text = re.sub(r"^```(?:json)?", "", text).strip()
    text = re.sub(r"```$", "", text).strip()
    start = text.find("{")
    if start == -1:
        raise ValueError("no JSON object start")
    text = text[start:]
    last = text.rfind("}")
    if last == -1:
        raise ValueError("no closing brace to salvage from")
    head = text[: last + 1]
    for suffix in ("", "]}", "}", "]}}", "}]}", "]"):
        try:
            return json.loads(head + suffix)
        except json.JSONDecodeError:
            continue
    raise ValueError("could not repair truncated JSON")


def call_json(spec, system, user, temperature=0.5, max_tokens=1500,
              seed=LLM_SEED, retries=MAX_RETRIES):
    """Call the model and parse a JSON object. Salvages truncation; retries parse failures."""
    for attempt in range(retries):
        raw = call_llm(spec, system, user, temperature=temperature,
                       max_tokens=max_tokens, seed=seed, retries=retries)
        try:
            return extract_json(raw)
        except Exception:  # noqa: BLE001
            # Most parse failures here are max_tokens truncation -> try to salvage
            # rather than burn another full (equally long) generation.
            try:
                obj = repair_truncated_json(raw)
                print(f"  [info] recovered a truncated JSON response "
                      f"(consider raising max_tokens; currently {max_tokens}).")
                return obj
            except Exception as e:  # noqa: BLE001
                print(f"  [warn] JSON parse failed (attempt {attempt + 1}): {e}\n"
                      f"  raw: {str(raw)[:200]!r}")
                if temperature is not None:
                    temperature = max(0.0, temperature - 0.1)
    raise ValueError("Could not parse JSON from model after retries.")