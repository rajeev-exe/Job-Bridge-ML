import pandas as pd
from datetime import datetime, timedelta
from advanced_skill_extractor import IndustrySkillExtractor
from gnn_skill_predictor import GINXMLC, predict_missing_skills, graph_dict_to_data
from typing import Dict, List
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import json
import os

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "Data")

SKILL_DIFFICULTY = {
    "python": ("easy", 2), "sql": ("easy", 2), "java": ("medium", 4),
    "machine learning": ("hard", 8), "aws": ("hard", 8), "react": ("medium", 4),
    "html": ("easy", 2), "css": ("easy", 2), "tensorflow": ("hard", 8),
    "deep learning": ("hard", 8), "power bi": ("medium", 4), "statistics": ("medium", 4),
    "git": ("easy", 1), "javascript": ("medium", 4), "c++": ("hard", 8),
    "digital marketing": ("medium", 4), "hindi-nlp": ("medium", 4)
}

def forecast_placement(student_skills: str, job_role: str, extractor: IndustrySkillExtractor, gnn_model: GINXMLC,
                      graph_dict: Dict, ontology: List[str], jobs_df: pd.DataFrame, projects_count: int = 0,
                      project_matches: pd.DataFrame = None) -> Dict:
    graph_data = graph_dict_to_data(graph_dict, ontology)
    student_skills_list = [s.strip().lower() for s in student_skills.split(",") if s.strip()]
    job_row = jobs_df[jobs_df['title'].str.lower() == job_role.lower()]
    if not job_row.empty:
        job_skills = [s.strip().lower() for s in str(job_row.iloc[0]['skills']).split(",")]
    else:
        job_skills = ["python", "sql", "machine learning", "aws", "react", "digital marketing"]

    matching_skills = []
    text_emb = extractor.bi_encoder.encode([",".join(student_skills_list)])
    job_emb = extractor.bi_encoder.encode(job_skills)
    for i, job_skill in enumerate(job_skills):
        if job_skill in student_skills_list:
            sim = 1.0
            match_type = "exact"
        else:
            sim = cosine_similarity(text_emb, [job_emb[i]])[0][0]
            match_type = "semantic" if sim > 0.7 else "partial"
        if sim > 0.5:
            matching_skills.append((job_skill, match_type, sim * 100))

    match_percentage = (sum([sim for sim in [cosine_similarity(text_emb, [job_emb[i]])[0][0] for i in range(len(job_skills))] if sim > 0.5]) / len(job_skills)) * 100 if job_skills else 0
    gaps = [s for s in job_skills if s not in student_skills_list]
    missing = predict_missing_skills(gnn_model, graph_data, student_skills_list, ontology)
    gaps = list(set(gaps + missing))

    total_days = 0
    roadmap = []

    courses_df = pd.read_csv(os.path.join(DATA_DIR, "courses.csv"))
    skill_col = next((c for c in courses_df.columns if c.lower() in ["skill", "course", "course_name", "topic"]), "skill")

    difficulty_multiplier = 1.2 if len(gaps) > 3 else 1.0
    if "machine learning" in job_role.lower():
        difficulty_multiplier *= 1.5
    if projects_count > 2:
        difficulty_multiplier -= 0.2

    project_boost = 0
    if project_matches is not None and not project_matches.empty:
        project_boost = project_matches['project_boost'].str.extract(r'\+(\d\.\d)').astype(float).fillna(0).sum()
    difficulty_multiplier -= project_boost / 100

    for g in gaps:
        diff_level, base_weeks = SKILL_DIFFICULTY.get(g, ("medium", 4))
        estimated_weeks = base_weeks * difficulty_multiplier
        total_days += estimated_weeks * 7
        course = courses_df[courses_df[skill_col].str.lower() == g.lower()]
        resource = (f"{course['provider'].values[0]}: {course['course_name'].values[0]}"
                   if not course.empty and 'provider' in course.columns and 'course_name' in course.columns
                   else f"Self-study {g} on SWAYAM")
        roadmap.append((g, resource, f"{diff_level} ({estimated_weeks:.1f} weeks)"))

    difficulty_factor = sum(1 if level == "hard" else 0.5 if level == "medium" else 0.2
                          for _, level in [SKILL_DIFFICULTY.get(g, ("medium", 4)) for g in gaps])

    X_train = np.array([[len(gaps), match_percentage, difficulty_factor] for _ in range(20)])
    y_train = np.random.randint(30, 120, 20)
    rf = RandomForestRegressor(n_estimators=100)
    rf.fit(X_train, y_train)
    predicted_days = int(rf.predict([[len(gaps), match_percentage, difficulty_factor]])[0])

    predicted_days = max(30, predicted_days + total_days - min(projects_count * 5, 15) - int(project_boost * 10))
    youth_adjustment = 0.1 if projects_count < 3 or total_days > 90 else 0
    predicted_days += int(youth_adjustment * 30)
    predicted_date = (datetime(2025, 9, 25) + timedelta(days=predicted_days)).strftime("%Y-%m-%d")

    feature_importance = {
        "Number of Gaps": rf.feature_importances_[0],
        "Skill Match %": rf.feature_importances_[1],
        "Difficulty Factor": rf.feature_importances_[2],
        "Project Boost": project_boost / 100
    }

    return {
        "predicted_date": predicted_date,
        "match_percentage": round(match_percentage, 2),
        "gaps": gaps,
        "confidence": max(0.5, 0.95 - 0.05 * len(gaps) + (match_percentage / 1000) + min(projects_count * 0.05, 0.15) + project_boost / 100),
        "explanation": feature_importance,
        "roadmap": roadmap,
        "matching_skills": matching_skills,
        "estimated_total_weeks": round(total_days / 7, 1),
        "projects_count": projects_count,
    }