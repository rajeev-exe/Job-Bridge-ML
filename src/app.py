import streamlit as st
import pandas as pd
import os
from datetime import datetime
import uuid
import skill_gap_analysis as sga
import placement_predictor as pp
from io import BytesIO

# ---------- Page config ----------
st.set_page_config(page_title="JobBridge — Skill Gap & Placement Prediction", layout="wide", page_icon="🧭")

# ---------- Custom CSS for attractive UI ----------
st.markdown("""
    <style>
    .main { background-color: #f0f2f6; }
    .stButton>button { background-color: #2b8cbe; color: white; border-radius: 8px; }
    .stTextInput>div>input { border-radius: 8px; }
    .dashboard-container { background-color: white; padding: 20px; border-radius: 10px; box-shadow: 0 4px 8px rgba(0,0,0,0.1); }
    .section-header { font-size: 24px; color: #2b8cbe; font-weight: bold; }
    .metric-box { background-color: #e6f3fa; padding: 10px; border-radius: 8px; }
    </style>
""", unsafe_allow_html=True)

# ---------- Database and data paths ----------
DATABASE_DIR = os.path.join(os.path.dirname(__file__), "..", "database")
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "Data")
os.makedirs(DATABASE_DIR, exist_ok=True)
INPUT_CSV = os.path.join(DATABASE_DIR, "students_input.csv")
RESULTS_CSV = os.path.join(DATABASE_DIR, "students_results.csv")
JOBS_PATH = os.path.join(DATA_DIR, "jobs.csv")
STUDENTS_PATH = os.path.join(DATA_DIR, "student_skills.csv")
COURSES_PATH = os.path.join(DATA_DIR, "courses.csv")

# ---------- Input form ----------
st.title("🚀 JobBridge — Your Path to Placement")
st.markdown("Enter your details to predict your placement date and get a personalized skill roadmap.")
with st.form("student_input_form"):
    st.markdown("**Your Details**")
    name = st.text_input("Name", placeholder="e.g., Ravi Kumar")
    known_skills = st.text_input("Known Skills (comma-separated)", placeholder="e.g., Python, SQL, Machine Learning")
    preferred_job_role = st.text_input("Preferred Job Role", placeholder="e.g., Data Analyst")
    submit_button = st.form_submit_button("Analyze My Skills")

# ---------- Process form submission ----------
if submit_button:
    if not name or not known_skills or not preferred_job_role:
        st.error("Please fill in all fields.")
        st.stop()
    if not os.path.exists(JOBS_PATH) or not os.path.exists(STUDENTS_PATH):
        st.error("Required data files (jobs.csv or student_skills.csv) not found in Data folder.")
        st.stop()

    try:
        # Save student input to CSV
        submission_id = str(uuid.uuid4())
        submission_date = datetime.now().strftime("%Y-%m-%d")
        input_data = {
            "submission_id": submission_id,
            "submission_date": submission_date,
            "name": name,
            "known_skills": known_skills,
            "preferred_job_role": preferred_job_role,
            "notes": ""
        }
        input_df = pd.DataFrame([input_data])
        if not os.path.exists(INPUT_CSV):
            input_df.to_csv(INPUT_CSV, index=False)
        else:
            input_df.to_csv(INPUT_CSV, mode='a', header=False, index=False)

        st.info("Running analysis — please wait...")

        # Prepare temporary student CSV for skill gap analysis
        temp_student_data = {
            "student_id": submission_id,
            "name": name,
            "skills": known_skills
        }
        temp_student_df = pd.DataFrame([temp_student_data])
        temp_student_path = os.path.join(DATABASE_DIR, "temp_student.csv")
        temp_student_df.to_csv(temp_student_path, index=False)

        # Run skill gap analysis
        with st.spinner("Analyzing skills and predicting placement..."):
            matches_df, recommendations_df, top_demand_img, top_missing_img = sga.run_analysis(
                JOBS_PATH, temp_student_path, COURSES_PATH if os.path.exists(COURSES_PATH) else None, top_n=10
            )
            # Run placement prediction
            placement_results = pp.predict_placement(
                known_skills, preferred_job_role, matches_df, recommendations_df
            )

        st.success("Analysis Complete! 🎉")

        # Save results to CSV
        results_data = {
            "submission_id": submission_id,
            "predicted_placement_date": placement_results["predicted_placement_date"],
            "predicted_days_to_placement": placement_results["predicted_days_to_placement"],
            "placement_confidence": placement_results["placement_confidence"],
            "top_required_skills_for_role": ";".join(placement_results["top_required_skills"]),
            "student_matching_skills": ";".join([f"{s}|{t}|{sc:.2f}" for s, t, sc in placement_results["matching_skills"]]),
            "skill_gap_list": ";".join(placement_results["skill_gaps"]),
            "roadmap": ";".join([f"{step} ({dur})" for step, dur in placement_results["roadmap"]]),
            "advantages": ";".join(placement_results["advantages"]),
            "placed_in_last_3_months_similar": placement_results["placed_in_last_3_months_similar"],
            "notes": placement_results["notes"]
        }
        results_df = pd.DataFrame([results_data])
        if not os.path.exists(RESULTS_CSV):
            results_df.to_csv(RESULTS_CSV, index=False)
        else:
            results_df.to_csv(RESULTS_CSV, mode='a', header=False, index=False)

        # ---------- Dashboard output ----------
        with st.container():
            st.markdown('<div class="dashboard-container">', unsafe_allow_html=True)
            st.markdown('<p class="section-header">📅 Your Placement Prediction</p>', unsafe_allow_html=True)
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"**Predicted Placement Date**: {placement_results['predicted_placement_date']}")
                st.markdown(f"**Confidence**: {placement_results['placement_confidence']:.2%}")
            with col2:
                st.markdown(f"**Recent Placements (last 3 months)**: {'Yes' if placement_results['placed_in_last_3_months_similar'] else 'No'}")
            
            st.markdown('<p class="section-header">📊 Skill Analysis</p>', unsafe_allow_html=True)
            col3, col4 = st.columns(2)
            with col3:
                st.markdown("**Top Required Skills**")
                for skill in placement_results["top_required_skills"]:
                    st.markdown(f"- {skill}")
            with col4:
                st.markdown("**Your Matching Skills**")
                for skill, match_type, score in placement_results["matching_skills"]:
                    st.markdown(f"- {skill} ({match_type}, {score:.2f})")
            
            st.markdown("**Skill Gaps**")
            for gap in placement_results["skill_gaps"]:
                st.markdown(f"- {gap}")
            
            st.markdown("**Your Advantages**")
            for adv in placement_results["advantages"]:
                st.markdown(f"- {adv}")

            st.markdown('<p class="section-header">🛤️ Your Roadmap to Success</p>', unsafe_allow_html=True)
            for step, duration in placement_results["roadmap"]:
                st.markdown(f"- {step} ({duration})")

            st.markdown('<p class="section-header">📈 Market Insights</p>', unsafe_allow_html=True)
            col5, col6 = st.columns(2)
            with col5:
                st.markdown("**Top Demanded Skills**")
                if top_demand_img:
                    st.image(top_demand_img, use_container_width=True)
                else:
                    st.info("No demand chart available.")
            with col6:
                st.markdown("**Top Missing Skills**")
                if top_missing_img:
                    st.image(top_missing_img, use_container_width=True)
                else:
                    st.info("No missing chart available.")

            # Downloads
            csv_matches = matches_df.to_csv(index=False).encode("utf-8")
            st.download_button("⬇ Download Skill Gap Results", data=csv_matches, file_name="skill_gap_results.csv", mime="text/csv")
            if not recommendations_df.empty:
                csv_recs = recommendations_df.to_csv(index=False).encode("utf-8")
                st.download_button("⬇ Download Recommendations", data=csv_recs, file_name="recommendations.csv", mime="text/csv")
            results_csv = results_df.to_csv(index=False).encode("utf-8")
            st.download_button("⬇ Download Placement Results", data=results_csv, file_name="placement_results.csv", mime="text/csv")

            st.markdown('</div>', unsafe_allow_html=True)

        st.balloons()
    except Exception as e:
        st.error(f"Error: {e}")
        st.exception(e)
else:
    st.info("Enter your details above and click 'Analyze My Skills' to get started.")