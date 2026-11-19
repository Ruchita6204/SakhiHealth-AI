
"""
SAKHIHEALTH AI - TOPIC BOOST WEIGHT EXPERIMENT

Compares hybrid retrieval using topic boost weights:
0.00, 0.05, 0.10 and 0.20.

The existing application and evaluation scripts are not modified.
"""

import sys
from pathlib import Path

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# ------------------------------------------------------------
# 1. PROJECT PATH
# ------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app import main as app_main


# ------------------------------------------------------------
# 2. EVALUATION QUESTIONS
# ------------------------------------------------------------

TEST_CASES = [
    {"question": "What is menstrual health?", "relevant_ids": ["WH01"]},
    {"question": "When should I seek medical help for menstrual problems?", "relevant_ids": ["WH02"]},
    {"question": "Why do I experience menstrual pain?", "relevant_ids": ["WH03"]},
    {"question": "What is PCOS and PCOD?", "relevant_ids": ["WH04"]},
    {"question": "How is PCOS diagnosed?", "relevant_ids": ["WH05"]},
    {"question": "What is endometriosis?", "relevant_ids": ["WH06"]},
    {"question": "What are the symptoms of an underactive thyroid?", "relevant_ids": ["WH07"]},
    {"question": "What is a healthy diet for women?", "relevant_ids": ["WH08"]},
    {"question": "How can women improve their overall wellbeing?", "relevant_ids": ["WH09"]},
    {"question": "How can I maintain menstrual hygiene?", "relevant_ids": ["WH10"]},
    {"question": "How does women's health change across life stages?", "relevant_ids": ["WH11"]},
    {"question": "How should my privacy and dignity be respected during a women's health consultation?", "relevant_ids": ["WH12"]},
    {"question": "What lifestyle changes can support long-term metabolic health in women with PCOS?", "relevant_ids": ["WH13"]},
    {"question": "Can menstrual cycles vary from person to person?", "relevant_ids": ["WH01"]},
    {"question": "What symptoms of heavy or prolonged periods need medical attention?", "relevant_ids": ["WH02"]},
    {"question": "Can endometriosis cause persistent pelvic pain?", "relevant_ids": ["WH06"]},
    {"question": "Can an underactive thyroid affect menstrual periods?", "relevant_ids": ["WH07"]},
    {"question": "Does healthy eating cure PCOS?", "relevant_ids": ["WH08", "WH13"]},
]

BOOST_WEIGHTS = [0.00, 0.05, 0.10, 0.20]
TOP_K = 3


# ------------------------------------------------------------
# 3. RETRIEVAL FUNCTION
# ------------------------------------------------------------

def get_rankings(question, boost_weight):

    knowledge = app_main.KNOWLEDGE

    if not knowledge:
        return []

    expanded_question = app_main.expand_question(question)

    # TF-IDF scores
    query_vector = app_main.VECTORIZER.transform(
        [expanded_question]
    )

    tfidf_scores = cosine_similarity(
        query_vector,
        app_main.TFIDF_MATRIX
    ).ravel()

    # Semantic scores
    query_embedding = app_main.SEMANTIC_MODEL.encode(
        [expanded_question],
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False
    )[0]

    semantic_scores = np.dot(
        app_main.DOCUMENT_EMBEDDINGS,
        query_embedding
    )

    semantic_scores = np.clip(
        semantic_scores,
        0.0,
        1.0
    )

    # Hybrid score: 35% TF-IDF + 65% semantic
    final_scores = (
        0.35 * tfidf_scores
        + 0.65 * semantic_scores
    )

    # Apply adjustable topic boost
    if boost_weight > 0:

        detected_intents = app_main.detect_intents(question)

        for index, item in enumerate(knowledge):

            topic = str(
                item.get("topic", "")
            ).lower()

            for intent in detected_intents:

                matching_topics = (
                    app_main.TOPIC_MAPPING.get(intent, [])
                )

                if any(
                    target in topic
                    for target in matching_topics
                ):
                    final_scores[index] += boost_weight

    # Ranking and existing threshold logic
    order = np.argsort(final_scores)[::-1][:TOP_K]

    if len(order) == 0:
        return []

    best_score = float(final_scores[order[0]])

    threshold = 0.30

    if best_score < threshold:
        return []

    minimum_score = max(
        threshold,
        best_score * 0.70
    )

    results = []

    for index in order:

        score = float(final_scores[index])

        if score >= minimum_score:

            results.append({
                "id": str(
                    knowledge[int(index)].get("id")
                ),
                "title": knowledge[int(index)].get("title", ""),
                "score": score
            })

    return results


# ------------------------------------------------------------
# 4. METRIC CALCULATION
# ------------------------------------------------------------

def calculate_metrics(all_results):

    total_queries = len(TEST_CASES)

    total_relevant = sum(
        len(test["relevant_ids"])
        for test in TEST_CASES
    )

    relevant_retrieved = 0
    successful_queries = 0
    reciprocal_rank_sum = 0.0
    query_precisions = []

    for test, results in zip(TEST_CASES, all_results):

        relevant_ids = set(test["relevant_ids"])

        relevant_count = sum(
            1 for item in results
            if item["id"] in relevant_ids
        )

        relevant_retrieved += relevant_count

        if relevant_count > 0:
            successful_queries += 1

        first_relevant_rank = None

        for rank, item in enumerate(results, start=1):

            if item["id"] in relevant_ids:
                first_relevant_rank = rank
                break

        if first_relevant_rank is not None:
            reciprocal_rank_sum += 1 / first_relevant_rank

        if results:
            query_precisions.append(
                relevant_count / len(results)
            )

    precision_at_3 = (
        relevant_retrieved / (total_queries * TOP_K)
    )

    recall_at_3 = (
        relevant_retrieved / total_relevant
    )

    hit_rate = successful_queries / total_queries

    mrr = reciprocal_rank_sum / total_queries

    returned_precision = (
        sum(query_precisions) / len(query_precisions)
        if query_precisions else 0
    )

    return {
        "precision": precision_at_3,
        "recall": recall_at_3,
        "hit_rate": hit_rate,
        "mrr": mrr,
        "returned_precision": returned_precision,
        "successful": successful_queries
    }


# ------------------------------------------------------------
# 5. RUN EXPERIMENTS
# ------------------------------------------------------------

def run_experiment():

    if app_main.SEMANTIC_MODEL is None:
        raise RuntimeError("Semantic model is unavailable.")

    if app_main.VECTORIZER is None:
        raise RuntimeError("TF-IDF vectorizer is unavailable.")

    if app_main.DOCUMENT_EMBEDDINGS is None:
        raise RuntimeError("Document embeddings are unavailable.")

    print("\n" + "=" * 75)
    print("SAKHIHEALTH AI - TOPIC BOOST WEIGHT EXPERIMENT")
    print("=" * 75)

    print("Knowledge documents:", len(app_main.KNOWLEDGE))
    print("Evaluation questions:", len(TEST_CASES))
    print("Boost weights:", BOOST_WEIGHTS)

    experiment_results = {}

    for weight in BOOST_WEIGHTS:

        print("\n" + "-" * 75)
        print(f"RUNNING EXPERIMENT: TOPIC BOOST = {weight:.2f}")
        print("-" * 75)

        all_results = []

        for index, test in enumerate(TEST_CASES, start=1):

            results = get_rankings(
                test["question"],
                weight
            )

            all_results.append(results)

            relevant_ids = set(test["relevant_ids"])

            relevant_rank = next(
                (
                    rank
                    for rank, item in enumerate(results, start=1)
                    if item["id"] in relevant_ids
                ),
                None
            )

            print(
                f"Q{index:02d} | "
                f"First relevant rank: "
                f"{relevant_rank if relevant_rank else 'NOT FOUND'}"
            )

        metrics = calculate_metrics(all_results)

        experiment_results[weight] = {
            "metrics": metrics,
            "results": all_results
        }

        print("\nMetrics for boost", weight)

        for metric, value in metrics.items():

            if metric == "successful":
                print(f"{metric}: {value}/{len(TEST_CASES)}")
            else:
                print(f"{metric}: {value * 100:.2f}%")

    # --------------------------------------------------------
    # 6. FINAL COMPARISON
    # --------------------------------------------------------

    print("\n")
    print("=" * 95)
    print("FINAL TOPIC BOOST WEIGHT COMPARISON")
    print("=" * 95)

    print(
        f"{'BOOST':<10}"
        f"{'PRECISION':<14}"
        f"{'RECALL':<12}"
        f"{'HIT RATE':<12}"
        f"{'MRR':<12}"
        f"{'RETURNED PREC.':<16}"
    )

    print("-" * 95)

    for weight, data in experiment_results.items():

        m = data["metrics"]

        print(
            f"{weight:<10.2f}"
            f"{m['precision'] * 100:<14.2f}"
            f"{m['recall'] * 100:<12.2f}"
            f"{m['hit_rate'] * 100:<12.2f}"
            f"{m['mrr'] * 100:<12.2f}"
            f"{m['returned_precision'] * 100:<16.2f}"
        )

    # --------------------------------------------------------
    # 7. QUESTION 17 RANKING COMPARISON
    # --------------------------------------------------------

    print("\n")
    print("=" * 75)
    print("QUESTION 17 - UNDERACTIVE THYROID RANKING")
    print("=" * 75)

    for weight, data in experiment_results.items():

        results = data["results"][16]

        print(f"\nBoost weight: {weight:.2f}")

        for rank, item in enumerate(results, start=1):

            marker = (
                " <-- EXPECTED DOCUMENT"
                if item["id"] == "WH07"
                else ""
            )

            print(
                f"{rank}. {item['id']} | "
                f"{item['title']} | "
                f"{item['score']:.3f}"
                f"{marker}"
            )

    print("\n")
    print("=" * 75)
    print("TOPIC BOOST EXPERIMENT COMPLETED")
    print("=" * 75)


if __name__ == "__main__":
    run_experiment()
