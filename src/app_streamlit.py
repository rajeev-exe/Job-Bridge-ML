import streamlit as st
import pandas as pd
import os
from pathlib import Path
st.set_page_config(page_title="JobBridge Demo", layout="wide")

ROOT = Path(__file__).parents[1]
data_dir = ROOT / "data"
out_dir = ROOT / "outputs"

st.title("JobBridge — Skill Gap Demo")

jobs = pd.read_csv(data_dir / "sample_jobs_with_skills.csv")
students = pd.read_csv(data_dir / "student_skills.csv")

st.header("Jobs (sample)")
st.dataframe(jobs[['job_id','title','extracted_skills']].head())

st.header("Students (sample)")
st.dataframe(students.head())

if st.button("Run skill gap analysis (quick)"):
    # naive mapping reusing code from skill_gap_analysis.py but minimal
    st.info("Running quick gap analysis...")
    import subprocess, sys
    subprocess.run([sys.executable, str(ROOT / "src" / "skill_gap_analysis.py")])
    st.success("Done. Check outputs and data files.")
    st.image(str(out_dir / "top_demanded_skills.png"))
    st.image(str(out_dir / "top_missing_skills.png"))
