import pandas as pd
import json
from typing import List, Dict
import os
from collections import defaultdict

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "Data")
SKILLS_CSV_PATH = os.path.join(DATA_DIR, "skills_dataset.csv")

def load_skills_from_dataset(csv_path=SKILLS_CSV_PATH) -> List[str]:
    try:
        df = pd.read_csv(csv_path)
        if 'skill' in df.columns:
            return df['skill'].dropna().unique().tolist()
        elif 'skills' in df.columns:
            skills_set = set()
            for skill_str in df['skills'].dropna():
                skills = [x.strip() for x in skill_str.split(',')]
                skills_set.update(skills)
            return sorted(list(skills_set))
        else:
            raise Exception("Skill column not found in dataset")
    except Exception as e:
        print("Error loading skills dataset:", str(e))
        return []

def generate_synthetic_job_post(skill_set: List[str], num_samples: int = 10) -> List[Dict]:
    synthetic_data = []
    if os.path.exists(SKILLS_CSV_PATH):
        df = pd.read_csv(SKILLS_CSV_PATH)
        for i, row in df.head(num_samples).iterrows():
            skills = row['skills'].split(',') if 'skills' in row else skill_set[:5]
            synthetic_data.append({
                "id": f"syn_{i}",
                "title": row['title'] if 'title' in row else f"Role_{i+1}_India",
                "description": row['description'] if 'description' in row else "Tech role in India.",
                "skills": [x.strip() for x in skills]
            })
    else:
        synthetic_data = [{
            "id": "syn_fallback",
            "title": "Generic Job India",
            "description": "Tech role in India.",
            "skills": skill_set if skill_set else []
        } for _ in range(num_samples)]
    for job in synthetic_data:
        job['demand_score'] = sum(1 for skill in job['skills'] if skill in ["python", "machine learning", "aws", "digital marketing"])
    return synthetic_data[:num_samples]

def build_skill_graph(jobs_df: pd.DataFrame) -> Dict:
    graph = defaultdict(list)
    for _, row in jobs_df.iterrows():
        job_id = row["id"]
        skills = row["skills"] if isinstance(row["skills"], list) else row["skills"].split(",")
        for skill in skills:
            graph[job_id].append(skill.strip())
            graph[skill.strip()].append(job_id)
    for node in graph:
        if node in jobs_df['id'].values:
            graph[node] = list(set(graph[node]))
    return dict(graph)

def load_pre_generated_data():
    if not os.path.exists(SKILLS_CSV_PATH):
        raise FileNotFoundError(f"Skills dataset not found at {SKILLS_CSV_PATH}")
    jobs_df = pd.read_csv(SKILLS_CSV_PATH)
    jobs_df['skills'] = jobs_df['skills'].apply(lambda s: [x.strip() for x in s.split(',')] if isinstance(s, str) else [])
    job_list = jobs_df.to_dict("records")
    graph = build_skill_graph(jobs_df)
    with open(os.path.join(DATA_DIR, "skill_graph.json"), "w") as f:
        json.dump(graph, f)
    return job_list, graph

if __name__ == "__main__":
    skill_list = load_skills_from_dataset()
    print("Skills loaded:", skill_list[:10])
    jobs, graph = load_pre_generated_data()
    print("Loaded", len(jobs), "jobs and skill graph with", len(graph), "nodes.")