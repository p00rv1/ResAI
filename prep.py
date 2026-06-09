import json
import pandas as pd
import numpy as np

all_candidates_flattened = []
processed_count = 0

print("Starting 100% complete data flattening (No filters applied)...")

with open("candidates.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        cand = json.loads(line)
        processed_count += 1
        
        # --- 1. IDENTITY & PRIMARY METADATA ---
        cand_id = cand.get("candidate_id")
        
        profile = cand.get("profile", {})
        anonymized_name = profile.get("anonymized_name", "")
        headline = profile.get("headline", "")
        summary = profile.get("summary", "")
        years_exp = float(profile.get("years_of_experience", 0.0))
        location = profile.get("location", "")
        country = profile.get("country", "")
        current_title = profile.get("current_title", "")
        current_company = profile.get("current_company", "")
        current_company_size = profile.get("current_company_size", "")
        current_industry = profile.get("current_industry", "")
        
        # --- 2. CAREER HISTORY FLATTENING ---
        history = cand.get("career_history", [])
        num_past_companies = len(history)
        total_months_employed = sum([int(job.get("duration_months", 0)) for job in history if job.get("duration_months")])
        
        # Extract and string-join lists of historical elements
        past_titles = [job.get("title", "") for job in history if job.get("title")]
        past_companies = [job.get("company", "") for job in history if job.get("company")]
        past_descs = [job.get("description", "") for job in history if job.get("description")]
        past_industries = [job.get("industry", "") for job in history if job.get("industry")]
        
        flattened_titles = ", ".join(past_titles)
        flattened_companies = ", ".join(past_companies)
        flattened_job_descriptions = " ".join(past_descs)
        flattened_past_industries = " ".join(past_industries)
        
        # --- 3. EDUCATION FLATTENING ---
        education_entries = cand.get("education", [])
        num_edu_degrees = len(education_entries)
        
        edu_institutions = [e.get("institution", "") for e in education_entries if e.get("institution")]
        edu_degrees = [e.get("degree", "") for e in education_entries if e.get("degree")]
        edu_fields = [e.get("field_of_study", "") for e in education_entries if e.get("field_of_study")]
        edu_tiers = [e.get("tier", "unknown") for e in education_entries if e.get("tier")]
        
        flattened_institutions = ", ".join(edu_institutions)
        flattened_degrees = ", ".join(edu_degrees)
        flattened_edu_fields = ", ".join(edu_fields)
        primary_education_tier = edu_tiers[0] if edu_tiers else "unknown"
        
        # --- 4. DECLARED SKILLS FLATTENING ---
        skills = cand.get("skills", [])
        num_skills_declared = len(skills)
        
        skill_names = [s.get("name", "") for s in skills if s.get("name")]
        skill_proficiencies = [s.get("proficiency", "") for s in skills if s.get("proficiency")]
        total_skill_endorsements = sum([int(s.get("endorsements", 0)) for s in skills if s.get("endorsements")])
        total_skill_duration_months = sum([int(s.get("duration_months", 0)) for s in skills if s.get("duration_months")])
        
        flattened_declared_skills = ", ".join(skill_names)
        flattened_skill_proficiencies = " ".join(skill_proficiencies)
        
        # --- 5. THE ALL-INCLUSIVE MASTER TEXT PROFILE ---
        # Combines every raw text scrap into one cell for downstream TF-IDF / keyword scanning
        master_text_profile = (
            f"{headline} {summary} {flattened_titles} {flattened_job_descriptions} "
            f"{flattened_declared_skills} {flattened_edu_fields}"
        )
        
        # --- 6. REDROB BEHAVIORAL SIGNALS FLATTENING ---
        signals = cand.get("redrob_signals", {})
        
        profile_completeness_score = float(signals.get("profile_completeness_score", 0.0))
        signup_date = signals.get("signup_date", "")
        last_active_date = signals.get("last_active_date", "")
        open_to_work_flag = 1.0 if signals.get("open_to_work_flag") else 0.0
        profile_views_received_30d = float(signals.get("profile_views_received_30d", 0.0))
        applications_submitted_30d = float(signals.get("applications_submitted_30d", 0.0))
        recruiter_response_rate = float(signals.get("recruiter_response_rate", 0.0))
        avg_response_time_hours = float(signals.get("avg_response_time_hours", 0.0))
        connection_count = float(signals.get("connection_count", 0.0))
        endorsements_received = float(signals.get("endorsements_received", 0.0))
        notice_period_days = float(signals.get("notice_period_days", 90.0))
        github_activity_score = float(signals.get("github_activity_score", 0.0))
        search_appearance_30d = float(signals.get("search_appearance_30d", 0.0))
        saved_by_recruiters_30d = float(signals.get("saved_by_recruiters_30d", 0.0))
        interview_completion_rate = float(signals.get("interview_completion_rate", 0.0))
        offer_acceptance_rate = float(signals.get("offer_acceptance_rate", 0.0))
        
        verified_email = 1.0 if signals.get("verified_email") else 0.0
        verified_phone = 1.0 if signals.get("verified_phone") else 0.0
        linkedin_connected = 1.0 if signals.get("linkedin_connected") else 0.0
        
        preferred_work_mode = signals.get("preferred_work_mode", "any")
        willing_to_relocate = 1.0 if signals.get("willing_to_relocate") else 0.0
        
        # Flattening Nested Expected Salary Ranges
        salary_range = signals.get("expected_salary_range_inr_lpa", {})
        expected_salary_min_lpa = float(salary_range.get("min", 0.0))
        expected_salary_max_lpa = float(salary_range.get("max", 0.0))
        
        # Flattening Nested Verified Skill Assessment Scores
        assessments = signals.get("skill_assessment_scores", {})
        if assessments:
            test_scores = list(assessments.values())
            avg_verified_skill_score = float(np.mean(test_scores))
            max_verified_skill_score = float(np.max(test_scores))
            min_verified_skill_score = float(np.min(test_scores))
            num_verified_tests_taken = len(test_scores)
            verified_test_names = ", ".join(list(assessments.keys()))
        else:
            avg_verified_skill_score = 0.0
            max_verified_skill_score = 0.0
            min_verified_skill_score = 0.0
            num_verified_tests_taken = 0
            verified_test_names = ""
            
        # --- 7. APPEND THE ABSOLUTE FLAT ROW ---
        all_candidates_flattened.append({
            "candidate_id": cand_id,
            "anonymized_name": anonymized_name,
            "years_of_experience": years_exp,
            "headline": headline,
            "location": location,
            "country": country,
            "current_title": current_title,
            "current_company": current_company,
            "current_company_size": current_company_size,
            "current_industry": current_industry,
            "num_past_companies": num_past_companies,
            "total_months_employed": total_months_employed,
            "past_titles_flattened": flattened_titles,
            "past_companies_flattened": flattened_companies,
            "past_job_descriptions_flattened": flattened_job_descriptions,
            "past_industries_flattened": flattened_past_industries,
            "num_education_degrees": num_edu_degrees,
            "education_institutions_flattened": flattened_institutions,
            "education_degrees_flattened": flattened_degrees,
            "education_fields_flattened": flattened_edu_fields,
            "primary_education_tier": primary_education_tier,
            "num_skills_declared": num_skills_declared,
            "total_skill_endorsements": total_skill_endorsements,
            "total_skill_duration_months": total_skill_duration_months,
            "declared_skills_flattened": flattened_declared_skills,
            "skill_proficiencies_flattened": flattened_skill_proficiencies,
            "profile_completeness_score": profile_completeness_score,
            "signup_date": signup_date,
            "last_active_date": last_active_date,
            "open_to_work_flag": open_to_work_flag,
            "profile_views_received_30d": profile_views_received_30d,
            "applications_submitted_30d": applications_submitted_30d,
            "recruiter_response_rate": recruiter_response_rate,
            "avg_response_time_hours": avg_response_time_hours,
            "connection_count": connection_count,
            "endorsements_received": endorsements_received,
            "notice_period_days": notice_period_days,
            "github_activity_score": github_activity_score,
            "search_appearance_30d": search_appearance_30d,
            "saved_by_recruiters_30d": saved_by_recruiters_30d,
            "interview_completion_rate": interview_completion_rate,
            "offer_acceptance_rate": offer_acceptance_rate,
            "verified_email": verified_email,
            "verified_phone": verified_phone,
            "linkedin_connected": linkedin_connected,
            "preferred_work_mode": preferred_work_mode,
            "willing_to_relocate": willing_to_relocate,
            "expected_salary_min_lpa": expected_salary_min_lpa,
            "expected_salary_max_lpa": expected_salary_max_lpa,
            "avg_verified_skill_score": avg_verified_skill_score,
            "max_verified_skill_score": max_verified_skill_score,
            "min_verified_skill_score": min_verified_skill_score,
            "num_verified_tests_taken": num_verified_tests_taken,
            "verified_test_names_flattened": verified_test_names,
            "master_text_profile": master_text_profile
        })

# Save 100% of rows into a high-speed columnar Parquet format
df = pd.DataFrame(all_candidates_flattened)
df.to_parquet("clean_candidates.parquet", index=False)

print(f"\n--- DATA FLATTENING SUCCESSFUL ---")
print(f"Total Rows Saved: {len(df)}")
print(f"Total Flat Columns Generated: {df.shape[1]}")