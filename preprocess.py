import pandas as pd
import numpy as np
import re

def preprocess_candidates(
    INPUT_FILE="flat_candidates.parquet",
    OUTPUT_FILE="clean_candidates.parquet"
):
    def clean_skills(skills):

        if not isinstance(skills, (list, np.ndarray)):
            return ""

        parts = []

        for skill in skills:

            if not isinstance(skill, dict):
                continue

            name = str(
                skill.get("name", "")
            )

            prof = str(
                skill.get(
                    "proficiency",
                    ""
                )
            )

            # BM25 weighting
            parts.extend(
                [name, name, name]
            )

            parts.append(prof)

        return " ".join(parts)


    def clean_education(education):

        if not isinstance(
            education,
            (list, np.ndarray)
        ):
            return ""

        parts = []

        for edu in education:

            if not isinstance(
                edu,
                dict
            ):
                continue

            parts.append(
                str(
                    edu.get(
                        "degree",
                        ""
                    )
                )
            )

            parts.append(
                str(
                    edu.get(
                        "field_of_study",
                        ""
                    )
                )
            )

            parts.append(
                str(
                    edu.get(
                        "institution",
                        ""
                    )
                )
            )

            parts.append(
                str(
                    edu.get(
                        "tier",
                        ""
                    )
                )
            )

        return " ".join(parts)


    def clean_languages(languages):

        if not isinstance(
            languages,
            (list, np.ndarray)
        ):
            return ""

        parts = []

        for lang in languages:

            if not isinstance(
                lang,
                dict
            ):
                continue

            parts.append(
                str(
                    lang.get(
                        "language",
                        ""
                    )
                )
            )

        return " ".join(parts)


    def clean_certifications(certs):

        if not isinstance(
            certs,
            (list, np.ndarray)
        ):
            return ""

        parts = []

        for cert in certs:

            if not isinstance(
                cert,
                dict
            ):
                continue

            parts.append(
                str(
                    cert.get(
                        "name",
                        ""
                    )
                )
            )

        return " ".join(parts)


    def clean_career_history(career_history):

        if not isinstance(
            career_history,
            (list, np.ndarray)
        ):
            return ""

        parts = []

        for job in career_history:

            if not isinstance(
                job,
                dict
            ):
                continue

            parts.append(
                str(
                    job.get(
                        "company",
                        ""
                    )
                )
            )

            parts.append(
                str(
                    job.get(
                        "title",
                        ""
                    )
                )
            )

            parts.append(
                str(
                    job.get(
                        "industry",
                        ""
                    )
                )
            )

            parts.append(
                str(
                    job.get(
                        "description",
                        ""
                    )
                )
            )

        return " ".join(parts)

    # =====================================================
    # LOAD DATA
    # =====================================================

    print("Loading candidates...")

    flat_df = pd.read_parquet(
        INPUT_FILE
    )

    print(
        "Candidates:",
        len(flat_df)
    )

    # =====================================================
    # CREATE TEXT COLUMNS
    # =====================================================

    flat_df["skills_text"] = (
        flat_df["skills"]
        .apply(clean_skills)
    )

    flat_df["education_text"] = (
        flat_df["education"]
        .apply(clean_education)
    )

    flat_df["languages_text"] = (
        flat_df["languages"]
        .apply(clean_languages)
    )

    flat_df["certifications_text"] = (
        flat_df["certifications"]
        .apply(clean_certifications)
    )

    flat_df["career_text"] = (
        flat_df["career_history"]
        .apply(clean_career_history)
    )

    # =====================================================
    # BUILD BM25 DOCUMENT
    # =====================================================

    flat_df["candidate_document"] = (

        flat_df["career_text"]
        .fillna("")

        + " "

        + flat_df["education_text"]
        .fillna("")

        + " "

        + flat_df["education_text"]
        .fillna("")

        + " "

        + flat_df["skills_text"]
        .fillna("")

        + " "

        + flat_df["skills_text"]
        .fillna("")

        + " "

        + flat_df["profile_headline"]
        .fillna("")
        .astype(str)

        + " "

        + flat_df["profile_summary"]
        .fillna("")
        .astype(str)

        + " "

        + flat_df["profile_current_title"]
        .fillna("")
        .astype(str)

        + " "

        + flat_df["profile_current_industry"]
        .fillna("")
        .astype(str)

        + " "

        + flat_df["profile_current_company"]
        .fillna("")
        .astype(str)

        + " "

        + flat_df["languages_text"]
        .fillna("")
        .astype(str)
    )

    # =====================================================
    # CLEAN DOCUMENT
    # =====================================================

    flat_df["candidate_document"] = (

        flat_df["candidate_document"]

        .str.lower()

        .str.replace(
            r"[^a-z0-9+#.\s]",
            " ",
            regex=True
        )

        .str.replace(
            r"\s+",
            " ",
            regex=True
        )

        .str.strip()

    )

    # =====================================================
    # SAVE
    # =====================================================

    flat_df.to_parquet(
        OUTPUT_FILE,
        index=False
    )

    print(
        f"Saved {OUTPUT_FILE}"
    )
if __name__ == "__main__":
    preprocess_candidates()