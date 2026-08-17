"""
Switch the Claude judge to Opus 4.8 and re-judge everything -- WITHOUT redoing
the samples already scored by Opus during judge_upgrade_test.py.

What it does:
  1. Backs up the current data/judged/ (Haiku-scored) to a timestamped folder.
  2. For each judged file, REPLACES the old Claude (Haiku) score with the Opus
     score already cached in data/judge_test/ when available -> that sample is
     done, no re-call needed.
  3. Deletes any judged file that does NOT yet have an Opus score, so Step 3
     (now configured to use Opus, see config.JUDGE_MODELS) re-judges only those.
  4. Optionally runs Step 3 and the analyses to fill the gaps and refresh tables.

Prerequisite: set config.JUDGE_MODELS["claude"] to Opus 4.8 (already done).
GPT-5.4 scores are always preserved.

Usage:
  python switch_judge_to_opus.py            # migrate + rerun steps 3,4,3b?,4b?
  python switch_judge_to_opus.py --migrate-only
  python switch_judge_to_opus.py --no-pairwise   # rerun pointwise only
"""
import argparse
import glob
import json
import os
import shutil
import time

from config import DATA_DIR, JUDGED_DIR, JUDGE_MODELS

TEST_DIR = os.path.join(DATA_DIR, "judge_test")


def opus_is_active():
    return JUDGE_MODELS.get("claude", {}).get("model", "") == "claude-opus-4-8"


def migrate():
    judged_files = sorted(glob.glob(os.path.join(JUDGED_DIR, "story_*", "*.json")))
    if not judged_files:
        print("No judged files found. Run step3 first (or nothing to migrate).")
        return 0, 0

    # Back up the current judged/ (still Haiku-scored) before touching it.
    stamp = time.strftime("%Y%m%d_%H%M%S")
    bdir = os.path.join(DATA_DIR, f"backup_judged_haiku_{stamp}")
    for f in judged_files:
        rel = os.path.relpath(f, DATA_DIR)
        dest = os.path.join(bdir, rel)
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        shutil.copy2(f, dest)
    print(f"Backed up {len(judged_files)} judged file(s) to: {bdir}")

    migrated, deleted = 0, 0
    for f in judged_files:
        inst = os.path.basename(os.path.dirname(f))
        cond = os.path.splitext(os.path.basename(f))[0]
        opus_path = os.path.join(TEST_DIR, inst, f"{cond}.json")

        if os.path.exists(opus_path):
            # Reuse the Opus score from the test phase: swap it into the claude slot.
            with open(f, encoding="utf-8") as fh:
                rec = json.load(fh)
            with open(opus_path, encoding="utf-8") as fh:
                opus = json.load(fh)
            rec["judges"]["claude"] = {
                "character_consistency": int(opus["character_consistency"]),
                "emotional_fidelity": int(opus["emotional_fidelity"]),
                "rationale": str(opus.get("rationale", "")),
            }
            with open(f, "w", encoding="utf-8") as fh:
                json.dump(rec, fh, ensure_ascii=False, indent=2)
            migrated += 1
        else:
            # No Opus score yet -> drop it so Step 3 re-judges this one with Opus.
            os.remove(f)
            deleted += 1

    print(f"Migrated {migrated} sample(s) from the Opus test cache; "
          f"deleted {deleted} stale Haiku-only sample(s) for re-judging.")
    return migrated, deleted


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--migrate-only", action="store_true",
                    help="only migrate/clean; don't rerun steps")
    ap.add_argument("--no-pairwise", action="store_true",
                    help="rerun pointwise (3,4) only, skip pairwise (3b,4b)")
    args = ap.parse_args()

    if not opus_is_active():
        print("WARNING: config.JUDGE_MODELS['claude'] is not Opus 4.8. "
              "Set it before running so Step 3 re-judges with Opus.")
    migrate()

    if args.migrate_only:
        print("migrate-only: done. Run step3/step4 (and 3b/4b) when ready.")
        return

    import step3_llm_judge
    import step4_analysis
    print("\n========== STEP 3: re-judge gaps with Opus ==========")
    step3_llm_judge.main()
    print("\n========== STEP 4: analysis (pointwise) ==========")
    step4_analysis.main()

    if not args.no_pairwise:
        # Pairwise verdicts were produced by Haiku too; clear & redo them with Opus.
        pair_dir = os.path.join(DATA_DIR, "pairwise")
        n = len(glob.glob(os.path.join(pair_dir, "story_*", "*.json")))
        if n:
            stamp = time.strftime("%Y%m%d_%H%M%S")
            shutil.move(pair_dir, os.path.join(DATA_DIR, f"backup_pairwise_haiku_{stamp}"))
            print(f"\nMoved {n} old (Haiku) pairwise file(s) to a backup; "
                  f"they will be regenerated with Opus.")
        import step3b_pairwise_judge
        import step4b_pairwise_analysis
        print("\n========== STEP 3b: re-judge pairwise with Opus ==========")
        step3b_pairwise_judge.main()
        print("\n========== STEP 4b: analysis (pairwise) ==========")
        step4b_pairwise_analysis.main()


if __name__ == "__main__":
    main()
