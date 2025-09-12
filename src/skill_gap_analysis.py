# skill_gap_analysis.py
import os
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
import ast
from io import BytesIO

# Ensure output dir exists (used if we ever save files)
OUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "uploaded_data"))
os.makedirs(OUT_DIR, exist_ok=True)


def read_csv_flexible(source):
    """
    Read CSV from a filepath or a file-like object (Streamlit uploader).
    Strips whitespace from column names.
    """
    if isinstance(source, str):
        df = pd.read_csv(source, encoding="utf-8", on_bad_lines="skip")
    else:
        # file-like (UploadedFile). Reset pointer then try reading.
        try:
            source.seek(0)
        except Exception:
            pass
        df = pd.read_csv(source)
    # strip column names of leading/trailing whitespace
    df.columns = df.columns.str.strip()
    return df


def parse_skills(cell):
    """Return a set of normalized skills from various input styles."""
    if pd.isna(cell):
        return set()
    if isinstance(cell, (list, set)):
        return set([str(x).strip().lower() for x in cell if x])
    s = str(cell).strip()
    # try python list literal
    if s.startswith("[") and s.endswith("]"):
        try:
            lst = ast.literal_eval(s)
            return set([str(x).strip().lower() for x in lst if x])
        except Exception:
            pass
    # normalize separators
    s = s.replace("|", ",").replace(";", ",")
    parts = [p.strip().lower() for p in s.split(",") if p.strip()]
    return set(parts)


def _find_column(df, candidates):
    """Return first matching column name from candidates (case-insensitive)."""
    cols = list(df.columns)
    # exact name first
    for c in candidates:
        if c in cols:
            return c
    # case-insensitive
    lowmap = {col.lower(): col for col in cols}
    for c in candidates:
        if c.lower() in lowmap:
            return lowmap[c.lower()]
    return None


def _bar_chart_bytes(counter_obj, title, top_n=10, color="#2b8cbe"):
    items = counter_obj.most_common(top_n)
    if not items:
        return None
    skills, counts = zip(*items)
    # horizontal bar, reversed to show largest on top
    skills_rev = list(skills)[::-1]
    counts_rev = list(counts)[::-1]

    fig, ax = plt.subplots(figsize=(8, max(3, len(skills)*0.35)))
    ax.barh(skills_rev, counts_rev, color=color)
    ax.set_title(title)
    ax.set_xlabel("Frequency")
    plt.tight_layout()

    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=150)
    plt.close(fig)
    buf.seek(0)
    return buf


def run_analysis(jobs_source, students_source, courses_source=None, top_n=15):
    """
    Main function.
    - jobs_source, students_source, courses_source: either file paths or file-like objects (e.g. Streamlit uploads)
    Returns: matches_df, recommendations_df, top_demand_chart_bytes, top_missing_chart_bytes
    """
    # 1) Read files flexibly and strip column names
    jobs_df = read_csv_flexible(jobs_source)
    students_df = read_csv_flexible(students_source)
    courses_df = read_csv_flexible(courses_source) if courses_source is not None else pd.DataFrame()

    # 2) Find sensible column names
    job_id_col = _find_column(jobs_df, ["job_id", "id", "jobId"])
    job_title_col = _find_column(jobs_df, ["title", "job_title", "job"])
    job_skills_col = _find_column(jobs_df, ["skills", "required_skills", "required skills", "extracted_skills", "skill_list", "skill(s)", "requirements"])

    student_id_col = _find_column(students_df, ["student_id", "id", "studentId"])
    student_name_col = _find_column(students_df, ["student_name", "name", "student"])
    student_skills_col = _find_column(students_df, ["skills", "skill_list", "student_skills", "known_skills"])

    # informative errors if missing
    if job_skills_col is None:
        raise KeyError(f"No job skills column found in jobs file. Available columns: {list(jobs_df.columns)}")
    if student_skills_col is None:
        raise KeyError(f"No student skills column found in students file. Available columns: {list(students_df.columns)}")

    # 3) Normalize & parse skills into sets
    jobs_df["_job_id"] = jobs_df[job_id_col] if job_id_col else jobs_df.index
    jobs_df["_job_title"] = jobs_df[job_title_col] if job_title_col else jobs_df["_job_id"]
    jobs_df["_job_skills_set"] = jobs_df[job_skills_col].apply(parse_skills)

    students_df["_student_id"] = students_df[student_id_col] if student_id_col else students_df.index
    students_df["_student_name"] = students_df[student_name_col] if student_name_col else students_df["_student_id"]
    students_df["_student_skills_set"] = students_df[student_skills_col].apply(parse_skills)

    # 4) Build maps and counters
    job_skill_map = dict(zip(jobs_df["_job_id"], jobs_df["_job_skills_set"]))
    job_title_map = dict(zip(jobs_df["_job_id"], jobs_df["_job_title"]))

    rows = []
    missing_counter = Counter()
    demand_counter = Counter()

    # 5) Compute per-student per-job match & missing skills
    for _, srow in students_df.iterrows():
        sid = srow["_student_id"]
        sname = srow["_student_name"]
        sskills = srow["_student_skills_set"]

        for jid, jskills in job_skill_map.items():
            if not jskills:
                # skip jobs that don't list any skills
                continue
            matched = jskills & sskills
            missing = jskills - sskills
            match_score = (len(matched) / len(jskills)) if jskills else 0.0

            rows.append({
                "student_id": sid,
                "student_name": sname,
                "job_id": jid,
                "job_title": job_title_map.get(jid, ""),
                "match_score": round(match_score, 4),
                "matched_count": len(matched),
                "missing_count": len(missing),
                "missing_skills": ";".join(sorted(missing))
            })

            for ms in missing:
                missing_counter[ms] += 1

    # demand counts
    for skills in job_skill_map.values():
        for sk in skills:
            demand_counter[sk] += 1

    matches_df = pd.DataFrame(rows)
    if not matches_df.empty:
        matches_df.sort_values(["student_id", "match_score"], ascending=[True, False], inplace=True)

    # 6) Recommendations per student (top 5 jobs by match_score)
    recommendations = matches_df.groupby("student_id").head(5).copy()

    # Attach resources if courses_df provided
    skill_to_resource = {}
    if not courses_df.empty:
        # find columns in courses_df for skill and resource
        skill_col = _find_column(courses_df, ["skill", "skills", "topic"])
        resource_col = _find_column(courses_df, ["resource", "link", "course", "url"])
        if skill_col and resource_col:
            # normalize keys to lowercase
            skill_to_resource = {str(s).strip().lower(): str(r).strip() for s, r in zip(courses_df[skill_col], courses_df[resource_col])}

    def _resources_for_missing(ms_str):
        if not ms_str:
            return ""
        skills = [s.strip().lower() for s in ms_str.split(";") if s.strip()]
        res = [skill_to_resource.get(s) for s in skills if skill_to_resource.get(s)]
        return ";".join(res)

    if not recommendations.empty:
        recommendations["recommended_resources"] = recommendations["missing_skills"].apply(_resources_for_missing)
    else:
        recommendations["recommended_resources"] = []

    # 7) Charts as in-memory bytes
    top_demand_chart = _bar_chart_bytes(demand_counter, "Top Demanded Skills", top_n=top_n, color="#2b8cbe")
    top_missing_chart = _bar_chart_bytes(missing_counter, "Top Missing Skills", top_n=top_n, color="#de2d26")

    return matches_df, recommendations, top_demand_chart, top_missing_chart
