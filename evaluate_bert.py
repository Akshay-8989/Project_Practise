"""
evaluate_bertscore.py
---------------------
Semantic evaluation using BERTScore.
Better than ROUGE for concise but correct answers.

Install: pip install bert-score
"""

import os, sys
sys.path.insert(0, '.')

# Force offline mode
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

from bert_score import score

# ── Ground truths (reuse SAME TEST_CASES from ROUGE) ──
from evaluate_rouge import TEST_CASES


def run_bert_evaluation():
    print("Loading pipeline...")
    from src.rag_pipeline import ComplianceRAGPipeline
    pipeline = ComplianceRAGPipeline()

    if pipeline.vectorstore.document_count() == 0:
        print("\n❌ Vector store is empty — upload a PDF first.\n")
        return

    print("\n" + "=" * 80)
    print(f"{'Question':<42} {'Precision':>10} {'Recall':>8} {'F1':>8}  Verdict")
    print("=" * 80)

    all_f1 = []

    for tc in TEST_CASES:
        response = pipeline.query(tc["question"])
        answer = response.answer.strip()

        if not answer or "could not find" in answer.lower():
            print(f"{tc['question'][:41]:<42} {'N/A':>10} {'N/A':>8} {'N/A':>8} ⚠ No answer")
            continue

        P, R, F1 = score(
            [answer],
            [tc["ground_truth"]],
            lang="en",
            model_type="roberta-large",   # ✅ best for English
            verbose=False
        )

        p = P.mean().item()
        r = R.mean().item()
        f1 = F1.mean().item()

        if f1 >= 0.85:
            verdict = "✅ Excellent"
        elif f1 >= 0.75:
            verdict = "🟢 Good"
        elif f1 >= 0.65:
            verdict = "🟡 OK"
        else:
            verdict = "🔴 Low"

        all_f1.append(f1)

        print(
            f"{tc['question'][:41]:<42} "
            f"{p:>9.1%} {r:>7.1%} {f1:>7.1%}  {verdict}"
        )

        if f1 < 0.65:
            print(f"  ↳ Answer preview: {answer[:120]}...")

    if all_f1:
        avg_f1 = sum(all_f1) / len(all_f1)
        print("=" * 80)
        print(f"{'AVERAGE':<42} {'':>9} {'':>7} {avg_f1:>7.1%}")
        print()

        if avg_f1 >= 0.80:
            print("✅ Overall: STRONG semantic alignment")
        elif avg_f1 >= 0.70:
            print("🟡 Overall: ACCEPTABLE")
        else:
            print("🔴 Overall: WEAK semantic alignment")


if __name__ == "__main__":
    run_bert_evaluation()