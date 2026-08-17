"""Run the full pipeline end to end: step1 -> step2 -> step3 -> step4."""
import step1_generate_stories
import step2_sprite_switching
import step3_llm_judge
import step4_analysis

if __name__ == "__main__":
    print("\n========== STEP 1: generate stories ==========")
    step1_generate_stories.main()
    print("\n========== STEP 2: sprite switching ==========")
    step2_sprite_switching.main()
    print("\n========== STEP 3: LLM judges ==========")
    step3_llm_judge.main()
    print("\n========== STEP 4: analysis ==========")
    step4_analysis.main()
