
"""
SAKHIHEALTH AI - PER-QUERY RANKING ANALYSIS

Compares:
B. Semantic-only retrieval
C. TF-IDF + Semantic retrieval
D. TF-IDF + Semantic + Topic Boost

Identifies:
- Retrieved document rankings
- Expected document rank
- Ranking changes between configurations
- Questions affected by topic boosting
"""

import sys
from pathlib import Path

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# 1. PROJECT PATH AND BACKEND IMPORT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app import main as app_main


# ============================================================
# 2. EVALUATION QUESTIONS
# ============================================================

TEST_CASES = [
    {"question": "What is menstrual health?",
     "relevant_ids": ["WH01"]},

    {"question": "When should I seek medical help for menstrual problems?",
     "relevant_ids": ["WH02"]},

    {"question": "Why do I experience menstrual pain?",
     "relevant_ids": ["WH03"]},

    {"question": "What is PCOS and PCOD?",
     "relevant_ids": ["WH04"]},

    {"question": "How is PCOS diagnosed?",
     "relevant_ids": ["WH05"]},

    {"question": "What is endometriosis?",
     "relevant_ids": ["WH06"]},

    {"question": "What are the symptoms of an underactive thyroid?",
     "relevant_ids": ["WH07"]},

    {"question": "What is a healthy diet for women?",
     "relevant_ids": ["WH08"]},

    {"question": "How can women improve their overall wellbeing?",
     "relevant_ids": ["WH09"]},

    {"question": "How can I maintain menstrual hygiene?",
     "relevant_ids": ["WH10"]},

    {"question": "How does women's health change across life stages?",
     "relevant_ids": ["WH11"]},

    {"question": "How should my privacy and dignity be respected during a women's health consultation?",
     "relevant_ids": ["WH12"]},

    {"question": "What lifestyle changes can support long-term metabolic health in women with PCOS?",
     "relevant_ids": ["WH13"]},

    {"question": "Can menstrual cycles vary from person to person?",
     "relevant_ids": ["WH01"]},

    {"question": "What symptoms of heavy or prolonged periods need medical attention?",
     "relevant_ids": ["WH02"]},

    {"question": "Can endometriosis cause persistent pelvic pain?",
     "relevant_ids": ["WH06"]},

    {"question": "Can an underactive thyroid affect menstrual periods?",
     "relevant_ids": ["WH07"]},

    {"question": "Does healthy eating cure PCOS?",
     "relevant_ids": ["WH08", "WH13"]},
]


# ============================================================
# 3. RETRIEVAL CONFIGURATIONS
# ============================================================

CONFIGURATIONS = {
    "B - SEMANTIC ONLY": "semantic",
    "C - TFIDF + SEMANTIC": "hybrid",
    "D - HYBRID + TOPIC BOOST": "hybrid_topic",
}


# ============================================================
# 4. CALCULATE DOCUMENT SCORES
# ============================================================

def get_rankings(question, mode, top_k=3):

    knowledge = app_main.KNOWLEDGE

    if not knowledge:
        return []

    expanded_question = app_main.expand_question(question)

    tfidf_scores = np.zeros(len(knowledge), dtype=float)
    semantic_scores = np.zeros(len(knowledge), dtype=float)

    # --------------------------------------------------------
    # TF-IDF
    # --------------------------------------------------------

    if mode in ("hybrid", "hybrid_topic"):

        query_vector = app_main.VECTORIZER.transform(
            [expanded_question]
        )

        tfidf_scores = cosine_similarity(
            query_vector,
            app_main.TFIDF_MATRIX
        ).ravel()

    # --------------------------------------------------------
    # SEMANTIC SIMILARITY
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # COMBINE SCORES
    # --------------------------------------------------------

    if mode == "semantic":

        final_scores = semantic_scores.copy()
        threshold = 0.30

    else:

        final_scores = (
            0.35 * tfidf_scores
            + 0.65 * semantic_scores
        )

        threshold = 0.30

    # --------------------------------------------------------
    # TOPIC BOOST
    # --------------------------------------------------------

    if mode == "hybrid_topic":

        detected_intents = app_main.detect_intents(
            question
        )

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

                    final_scores[index] += 0.10

    # --------------------------------------------------------
    # RANK AND APPLY EXISTING THRESHOLD
    # --------------------------------------------------------

    order = np.argsort(final_scores)[::-1][:top_k]

    if len(order) == 0:
        return []

    best_score = float(final_scores[order[0]])

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
                "id": str(knowledge[int(index)].get("id")),
                "title": knowledge[int(index)].get("title", ""),
                "score": score,
            })

    return results


# ============================================================
# 5. FIND FIRST RELEVANT DOCUMENT RANK
# ============================================================

def find_relevant_rank(results, relevant_ids):

    for rank, result in enumerate(results, start=1):

        if result["id"] in relevant_ids:
            return rank

    return None


# ============================================================
# 6. RUN PER-QUERY ANALYSIS
# ============================================================

def run_ranking_analysis():

    if app_main.SEMANTIC_MODEL is None:
        raise RuntimeError("Semantic model is unavailable.")

    if app_main.VECTORIZER is None:
        raise RuntimeError("TF-IDF vectorizer is unavailable.")

    if app_main.DOCUMENT_EMBEDDINGS is None:
        raise RuntimeError("Document embeddings are unavailable.")

    all_results = {}

    print("\n" + "=" * 75)
    print("SAKHIHEALTH AI - PER-QUERY RANKING ANALYSIS")
    print("=" * 75)

    print("Evaluation questions:", len(TEST_CASES))
    print("Knowledge documents:", len(app_main.KNOWLEDGE))
    print("Top K: 3")

    for index, test in enumerate(TEST_CASES, start=1):

        question = test["question"]
        relevant_ids = set(test["relevant_ids"])

        print("\n" + "-" * 75)
        print(f"QUESTION {index}: {question}")
        print("Expected documents:", sorted(relevant_ids))

        query_results = {}

        for config_name, mode in CONFIGURATIONS.items():

            results = get_rankings(
                question,
                mode,
                top_k=3
            )

            relevant_rank = find_relevant_rank(
                results,
                relevant_ids
            )

            query_results[config_name] = {
                "results": results,
                "relevant_rank": relevant_rank,
            }

            print(f"\n{config_name}")

            if not results:
                print("No documents returned.")
                continue

            for rank, item in enumerate(results, start=1):

                marker = (
                    " <-- RELEVANT"
                    if item["id"] in relevant_ids
                    else ""
                )

                print(
                    f"{rank}. {item['id']} | "
                    f"{item['title']} | "
                    f"Score: {item['score']:.3f}"
                    f"{marker}"
                )

            print(
                "First relevant document rank:",
                relevant_rank if relevant_rank else "NOT FOUND"
            )

        all_results[index] = query_results

    # --------------------------------------------------------
    # SUMMARY OF RANKING CHANGES
    # --------------------------------------------------------

    print("\n")
    print("=" * 75)
    print("RANKING CHANGE SUMMARY")
    print("=" * 75)

    semantic_name = "B - SEMANTIC ONLY"
    hybrid_name = "C - TFIDF + SEMANTIC"
    topic_name = "D - HYBRID + TOPIC BOOST"

    topic_changed = []
    hybrid_changed = []

    for index, query_results in all_results.items():

        semantic_rank = query_results[
            semantic_name
        ]["relevant_rank"]

        hybrid_rank = query_results[
            hybrid_name
        ]["relevant_rank"]

        topic_rank = query_results[
            topic_name
        ]["relevant_rank"]

        if semantic_rank != hybrid_rank:
            hybrid_changed.append(index)

        if hybrid_rank != topic_rank:
            topic_changed.append(index)

    print(
        "Queries where adding TF-IDF changed "
        "the first relevant document rank:",
        hybrid_changed if hybrid_changed else "None"
    )

    print(
        "Queries where adding topic boost changed "
        "the first relevant document rank:",
        topic_changed if topic_changed else "None"
    )

    print("\n")
    print("=" * 75)
    print("PER-QUERY ANALYSIS COMPLETED")
    print("=" * 75)


# ============================================================
# 7. MAIN
# ============================================================

if __name__ == "__main__":

    run_ranking_analysis()
