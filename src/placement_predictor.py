import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import os

model = SentenceTransformer('all-MiniLM-L6-v2')
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "Data")
SKILLS_CSV_PATH = os.path.join(DATA_DIR, "skills_dataset.csv")

def normalize_skills(skills_text):
    if not skills_text or pd.isna(skills_text):
        return []
    skills_text = skills_text.lower().replace("|", ",").replace(";", ",")
    skills = [s.strip() for s in skills_text.split(",") if s.strip()]
    mapping = {"py": "python", "ml": "machine learning", "nlp": "natural language processing"}
    return [mapping.get(s, s) for s in skills]

def load_job_skills(job_role):
    if not os.path.exists(SKILLS_CSV_PATH):
        return []
    df = pd.read_csv(SKILLS_CSV_PATH)
    match = df[df["title"].str.lower() == job_role.lower()]
    if not match.empty and "skills" in match.columns:
        return [x.strip() for x in match.iloc[0]["skills"].split(",")]
    return ["python", "sql", "machine learning", "aws", "digital marketing"]  # India-specific fallback

def predict_placement(known_skills, preferred_job_role, education_level, experience_months, projects_count, matches_df, recommendations_df):
    student_skills = normalize_skills(known_skills)
    job_role = preferred_job_role.strip().lower()
    job_skills = set(load_job_skills(job_role))
    all_skills = list(set(student_skills + list(job_skills)))

    matching_skills = []
    if all_skills:
        skill_embeddings = model.encode(all_skills)
        student_indices = [all_skills.index(s) for s in student_skills if s in all_skills]
        job_indices = [all_skills.index(s) for s in job_skills if s in all_skills]
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

    skill_gaps = [s for s in job_skills if s not in student_skills]
    top_required_skills = list(job_skills)[:5]
    advantages = [s for s in student_skills if s not in top_required_skills]
    roadmap = [(g, f"Learn {g} on SWAYAM", "medium (4 weeks)") for g in skill_gaps[:3]]

    base_days = 60
    confidence = 0.8 - (len(skill_gaps) * 0.1) if len(skill_gaps) < 5 else 0.5
    confidence = max(0.5, min(confidence, 0.95))

    days_adjustment = -min(experience_months / 12 * 10, 20) - min(projects_count * 5, 15)
    predicted_days = max(30, base_days + days_adjustment)

    youth_factor = 0.1 if "student" in education_level.lower() or experience_months < 12 else 0
    predicted_days += int(youth_factor * 30)
    predicted_date = (datetime(2025, 9, 25) + timedelta(days=predicted_days)).strftime("%Y-%m-%d")

    return {
        "predicted_placement_date": predicted_date,
        "predicted_days_to_placement": int(predicted_days),
        "placement_confidence": confidence,
        "top_required_skills": top_required_skills,
        "matching_skills": matching_skills,
        "skill_gaps": skill_gaps,
        "roadmap": roadmap,
        "advantages": advantages,
        "notes": "Enhanced for Indian market."
    }

if __name__ == "__main__":
    print(predict_placement("Python, SQL", "Data Scientist", "Bachelor", 12, 2, pd.DataFrame(), pd.DataFrame()))