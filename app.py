import streamlit as st
import pandas as pd
import os
from preprocess import preprocess_candidates
from bm25 import run_bm25
from ranking import rank_candidates
from summary import generate_candidate_summaries

st.title("Resume Ranking Sandbox")

jd_file = st.file_uploader(
    "Upload Job Description",
    type=["docx"]
)

candidate_file = st.file_uploader(
    "Upload Candidate Dataset",
    type=["csv"]
)

if st.button("Run Pipeline"):

    if jd_file is None:
        st.error("Please upload a JD")
        st.stop()

    if candidate_file is None:
        st.error("Please upload a candidate parquet")
        st.stop()

    # -----------------------------------
    # Save uploaded files
    # -----------------------------------

    with open("job_description.docx", "wb") as f:
        f.write(jd_file.getbuffer())

    with open("top5000_candidates.csv", "wb") as f:
        f.write(candidate_file.getbuffer())

    
    
    st.write("Uploaded size:", candidate_file.size)
    st.write(
        "Saved size:",
        os.path.getsize("top5000_candidates.csv")
    )
    # Run pipeline
    # -----------------------------------

    
    with st.spinner("Ranking candidates..."):
        rank_candidates()

    with st.spinner("Generating summaries..."):
        final_df = generate_candidate_summaries()

    st.success("Pipeline Complete")

    st.subheader("Top Candidates")

    st.dataframe(
        final_df.head(20)
    )

    csv_data = final_df.to_csv(index=False)

    st.download_button(
        label="Download Results",
        data=csv_data,
        file_name="final_submission.csv",
        mime="text/csv"
    )