import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import re

# Load BERT model for skill embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')

# Mock placement data (replace with actual 3-month placement data)
MOCK_PLACEMENT_STATS = {
    "Data Analyst": {"avg_days_to_placement": 60, "placed_count": 20},
    "Data Scientist": {"avg_days_to_placement": 75, "placed_count": 15},
    "Software Engineer": {"avg_days_to_placement": 50, "placed_count": 30}
}

# Skill taxonomy for roadmap recommendations
SKILL_RECOMMENDATIONS = {
    "python": [("Python Basics Course (Coursera)", "2 weeks"), ("Python Project: Data Analysis", "2 weeks")],
    "sql": [("SQL Fundamentals (Khan Academy)", "2 weeks"), ("SQL Query Project", "1 week")],
    "excel": [("Excel for Data Analysis (Udemy)", "2 weeks"), ("Excel Dashboard Project", "2 weeks")],
    "power bi": [("Power BI Essentials (Microsoft Learn)", "3 weeks"), ("Power BI Dashboard Project", "2 weeks")],
    "statistics": [("Statistics for Data Science (edX)", "4 weeks")]
}

def normalize_skills(skills_text):
    """Normalize skills text to a list of lowercase tokens."""
    if not skills_text or pd.isna(skills_text):
        return []
    skills_text = skills_text.lower().replace("|", ",").replace(";", ",")
    skills = [s.strip() for s in skills_text.split(",") if s.strip()]
    # Canonical mapping
    mapping = {"py": "python", "ml": "machine learning"}
    return [mapping.get(s, s) for s in skills]

def predict_placement(known_skills, preferred_job_role, education_level, experience_months, projects_count, matches_df, recommendations_df):
    """Predict placement date and generate skills report and roadmap."""
    # Normalize inputs
    student_skills = normalize_skills(known_skills)
    job_role = preferred_job_role.strip().lower()

    # Get job skills from matches_df
    job_skills = set()
    if not matches_df.empty:
        job_row = matches_df[matches_df["job_title"].str.lower() == job_role]
        if not job_row.empty:
            job_skills = set(normalize_skills(job_row.iloc[0]["missing_skills"] + "," + job_row.iloc[0]["matched_count"]))

    # BERT embeddings for skill matching
    all_skills = list(set(student_skills + list(job_skills)))
    if all_skills:
        skill_embeddings = model.encode(all_skills)
        student_indices = [all_skills.index(s) for s in student_skills if s in all_skills]
        job_indices = [all_skills.index(s) for s in job_skills if s in all_skills]
        
        matching_skills = []
        for s in student_skills:
            if s in job_skills:
                matching_skills.append((s, "exact", 1.0))
            else:
                s_idx = all_skills.index(s) if s in all_skills else None
                if s_idx is not None:
                    similarities = cosine_similarity([skill_embeddings[s_idx]], skill_embeddings[job_indices])[0]
                    max_sim = max(similarities) if similarities.size > 0 else 0.0
                    match_type = "very strong" if max_sim >= 0.90 else "good" if max_sim >= 0.75 else "partial"
                    matching_skills.append((s, match_type, max_sim))

    else:
        matching_skills = [(s, "no match", 0.0) for s in student_skills]

    # Skill gaps
    skill_gaps = [s for s in job_skills if s not in student_skills]

    # Top required skills (mock, replace with job posting frequency)
    top_required_skills = list(job_skills)[:5] or ["python", "sql", "excel", "power bi", "statistics"]

    # Advantages (skills not commonly required but present)
    advantages = [s for s in student_skills if s not in top_required_skills]

    # Roadmap generation
    roadmap = []
    for gap in skill_gaps:
        if gap in SKILL_RECOMMENDATIONS:
            roadmap.extend(SKILL_RECOMMENDATIONS[gap])

    # Placement prediction (mock, replace with trained model)
    base_days = MOCK_PLACEMENT_STATS.get(job_role, {"avg_days_to_placement": 60})["avg_days_to_placement"]
    confidence = 0.8 - (len(skill_gaps) * 0.1) if len(skill_gaps) < 5 else 0.5
    confidence = max(0.5, min(confidence, 0.95))
    
    # Adjust based on experience and projects
    days_adjustment = -min(experience_months / 12 * 10, 20) - min(projects_count * 5, 15)
    predicted_days = max(30, base_days + days_adjustment)
    predicted_date = (datetime.now() + timedelta(days=predicted_days)).strftime("%Y-%m-%d")

    # Mock placement stats
    placed_in_last_3_months = MOCK_PLACEMENT_STATS.get(job_role, {"placed_count": 0})["placed_count"] > 0

    return {
        "predicted_placement_date": predicted_date,
        "predicted_days_to_placement": int(predicted_days),
        "placement_confidence": confidence,
        "top_required_skills": top_required_skills,
        "matching_skills": matching_skills,
        "skill_gaps": skill_gaps,
        "roadmap": roadmap,
        "advantages": advantages,
        "placed_in_last_3_months_similar": placed_in_last_3_months,
        "notes": "Mock prediction; replace with trained model and real placement data."
    }