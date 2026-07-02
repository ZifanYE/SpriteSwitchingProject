"""
Rerun ONLY the two-stage conditions (after fixing stage-2 to be character-dependent).

All `direct_*` results are kept and reused; this script:
  1. backs up, then deletes every `twostage_*.json` under data/switching and data/judged,
  2. re-runs Step 2 (fills only the missing twostage_* conditions; direct_* show [skip]),
  3. re-runs Step 3 (judges only the new twostage_* samples),
  4. re-runs Step 4 (analysis / tables).

Usage:
  python rerun_twostage.py            # backup + clear + rerun steps 2,3,4
  python rerun_twostage.py --no-backup
  python rerun_twostage.py --clear-only   # just back up & delete, don't rerun
"""
import argparse
import glob
import os
import shutil
import time

from config import SWITCHING_DIR, JUDGED_DIR


def find_twostage_files():
    files = []
    for root in (SWITCHING_DIR, JUDGED_DIR):
        files += glob.glob(os.path.join(root, "*", "twostage_*.json"))
    return sorted(files)


def backup(files):
    if not files:
        return None
    stamp = time.strftime("%Y%m%d_%H%M%S")
    bdir = os.path.join(os.path.dirname(SWITCHING_DIR), f"backup_twostage_{stamp}")
    for f in files:
        # mirror the data/<switching|judged>/<story>/<file> structure inside the backup
        rel = os.path.relpath(f, os.path.dirname(SWITCHING_DIR))
        dest = os.path.join(bdir, rel)
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        shutil.copy2(f, dest)
    return bdir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-backup", action="store_true", help="skip backing up before deleting")
    ap.add_argument("--clear-only", action="store_true", help="back up + delete only; don't rerun")
    args = ap.parse_args()

    files = find_twostage_files()
    print(f"Found {len(files)} two-stage file(s) to clear.")

    if files and not args.no_backup:
        bdir = backup(files)
        print(f"Backed up to: {bdir}")

    for f in files:
        os.remove(f)
    print(f"Deleted {len(files)} two-stage file(s). All direct_* results are untouched.")

    if args.clear_only:
        print("clear-only: done. Re-run steps 2-4 yourself when ready.")
        return

    # Import here so the deletion above happens before any heavy imports run.
    import step2_sprite_switching
    import step3_llm_judge
    import step4_analysis

    print("\n========== STEP 2: sprite switching (two-stage only) ==========")
    step2_sprite_switching.main()
    print("\n========== STEP 3: LLM judges (two-stage only) ==========")
    step3_llm_judge.main()
    print("\n========== STEP 4: analysis ==========")
    step4_analysis.main()


if __name__ == "__main__":
    main()
