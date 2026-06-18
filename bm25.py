import pandas as pd
import re
from rank_bm25 import BM25Okapi
from docx import Document


def run_bm25(
    candidate_file="sample_candidates.parquet",
    jd_file="job_description.docx",
    output_file="top5000_candidates.csv"
):

    # =====================================================
    # LOAD DATA
    # =====================================================

    flat_df = pd.read_parquet(candidate_file)

    print("Candidates:", len(flat_df))

    # =====================================================
    # LOAD JOB DESCRIPTION
    # =====================================================

    doc = Document(jd_file)

    jd_text = "\n".join(
        para.text
        for para in doc.paragraphs
    )

    # =====================================================
    # CLEAN JD
    # =====================================================

    jd_text = jd_text.lower()

    jd_text = re.sub(
        r"[^a-z0-9+#.\s]",
        " ",
        jd_text
    )

    jd_text = re.sub(
        r"\s+",
        " ",
        jd_text
    )

    query_tokens = jd_text.split()

    print("JD Tokens:", len(query_tokens))

    # =====================================================
    # BUILD BM25 INDEX
    # =====================================================

    corpus = (
        flat_df["candidate_document"]
        .fillna("")
        .astype(str)
        .tolist()
    )

    tokenized_corpus = [
        doc.split()
        for doc in corpus
    ]

    bm25 = BM25Okapi(tokenized_corpus)

    print("BM25 Index Built")

    # =====================================================
    # SCORE CANDIDATES
    # =====================================================

    scores = bm25.get_scores(query_tokens)

    flat_df["bm25_score"] = scores

    # =====================================================
    # TOP 5000
    # =====================================================

    top_candidates = (
        flat_df
        .sort_values(
            by="bm25_score",
            ascending=False
        )
        .head(5000)
        .copy()
    )

    # =====================================================
    # SAVE
    # =====================================================

    top_candidates.to_csv(
        output_file,
        index=False
    )

    print(
        f"Saved {output_file}"
    )

    return top_candidates


if __name__ == "__main__":
    run_bm25()