import pandas as pd
import numpy as np
import ast
from docx import Document


def generate_candidate_summaries(
    input_file="top100.csv",
    jd_file="job_description.docx",
    output_file="final_submission.csv"
):

    df = pd.read_csv(input_file)


    

    doc = Document(jd_file)

    jd_text = "\n".join(
        para.text
        for para in doc.paragraphs
    )

    jd_lower = jd_text.lower()
    

    all_skills = set()

    for s in df["skills_text"]:
            all_skills.update(s)

    jd_skills = {
        skill
        for skill in all_skills
        if skill in jd_lower
    }

    # ---------------------------------------------------
    # Generate summary
    # ---------------------------------------------------

    def generate_summary(row):

        strengths = []
        weaknesses = []
        sk = set()
        sk.update(row["skills_text"].lower().split())
        if len(jd_skills) == 0:
            return 0

        

        matched = list(
            sk & jd_skills
        )

        missing = list(
            jd_skills - sk
        )

        # -------------------------
        # Strengths
        # -------------------------

        if row["skill_overlap"] > 0.5:

            strengths.append(
                "strong alignment with JD skills"
            )

        if row["assessment_feature"] > 70:

            strengths.append(
                "high assessment performance"
            )

        if row["profile_years_of_experience"] >= 5:

            strengths.append(
                f"{row['profile_years_of_experience']:.1f} years of experience"
            )

        if len(matched) > 0:

            strengths.append(
                f"experience with {', '.join(matched[:3])}"
            )

        # -------------------------
        # Weaknesses
        # -------------------------

        if len(missing) > 0:

            weaknesses.append(
                f"limited evidence of {', '.join(missing[:3])}"
            )

        if row["assessment_feature"] < 40:

            weaknesses.append(
                "few relevant assessment signals"
            )

        if row["profile_years_of_experience"] < 2:

            weaknesses.append(
                "relatively limited industry experience"
            )

        # -------------------------
        # Current role
        # -------------------------

        title = str(
            row.get(
                "profile_current_title",
                ""
            )
        )

        company = str(
            row.get(
                "profile_current_company",
                ""
            )
        )

        summary = (
            f"Currently working as {title}"
        )

        if company and company != "nan":

            summary += f" at {company}"

        summary += ". "

        if strengths:

            summary += (
                "Key strengths include "
                + "; ".join(strengths)
                + ". "
            )

        if weaknesses:

            summary += (
                "Potential concerns include "
                + "; ".join(weaknesses)
                + "."
            )

        return summary

    # ---------------------------------------------------
    # Create summaries
    # ---------------------------------------------------

    df["summary"] = (
        df.apply(
            generate_summary,
            axis=1
        )
    )

    # ---------------------------------------------------
    # Export
    # ---------------------------------------------------

    submission = df[
        [
            "candidate_id",
            "rank",
            "final_score",
            "summary"
        ]
    ]

    submission.to_csv(
    output_file,
    index=False
)

    print(
        f"Saved {output_file}"
    )

    return submission

if __name__ == "__main__":
    generate_candidate_summaries()
