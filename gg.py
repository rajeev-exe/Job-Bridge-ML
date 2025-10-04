import streamlit as st
import pandas as pd
import numpy as np
import re
import time
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Dict, List, Tuple, Optional
import io
import base64
import os
import sys
import torch

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import ML modules
try:
    from advanced_skill_extractor import IndustrySkillExtractor
    from gnn_skill_predictor import GINXMLC, predict_missing_skills, graph_dict_to_data
    from enhanced_placement_forecaster import forecast_placement
    from data_synthesizer import load_pre_generated_data, load_skills_from_dataset
    from placement_predictor import predict_placement
    ML_AVAILABLE = True
    print("ML modules loaded successfully in terminal")
except ImportError as e:
    st.warning(f"ML modules not fully loaded: {e}. Running in basic mode.")
    ML_AVAILABLE = False
    print(f"ML modules failed to load: {e}")

# ==================== DATA PATHS ====================
DATA_DIR = os.path.join(os.path.dirname(__file__), "Data")
COURSES_PATH = os.path.join(DATA_DIR, "courses.csv")
EMERGING_TECH_PATH = os.path.join(DATA_DIR, "emerging_tech.csv")
JOBS_PATH = os.path.join(DATA_DIR, "jobs.csv")
SKILLS_DATASET_PATH = os.path.join(DATA_DIR, "skills_dataset.csv")
SYNTHETIC_JOBS_PATH = os.path.join(DATA_DIR, "synthetic_jobs.csv")

# ==================== CONFIGURATION ====================
st.set_page_config(
    page_title="Job Bridge",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="collapsed",
    menu_items={
        'Get Help': None,
        'Report a bug': None,
        'About': "Job Bridge | ML Skill Gap Analyzer for Pre-final & Final Year Students"
    }
)

# ==================== PREMIUM DARK THEME CSS ====================
def inject_premium_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    /* Premium Dark Theme */
    .stApp {
        background: linear-gradient(135deg, #0f1419 0%, #1a1f2e 50%, #0f1419 100%);
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        color: #e2e8f0;
    }
    
    /* Main Container */
    .main-container {
        background: linear-gradient(145deg, #1e293b 0%, #334155 100%);
        border-radius: 34px;
        padding: 40px;
        margin: 20px;
        box-shadow: 
            0 25px 50px -12px rgba(0, 0, 0, 0.6),
            0 0 0 1px rgba(255, 255, 255, 0.05),
            inset 0 1px 0 rgba(255, 255, 255, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
    }
    
    /* Glassmorphism Header */
    .header-section {
        text-align: center;
        padding: 60px 40px;
        background: linear-gradient(135deg, rgba(59, 130, 246, 0.1) 0%, rgba(147, 51, 234, 0.1) 100%);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 20px;
        margin-bottom: 40px;
        position: relative;
        overflow: hidden;
    }
    
    .header-section::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: radial-gradient(circle at 30% 20%, rgba(59, 130, 246, 0.3) 0%, transparent 50%),
                    radial-gradient(circle at 70% 80%, rgba(147, 51, 234, 0.3) 0%, transparent 50%);
        pointer-events: none;
    }
    
    .header-title {
        font-size: 4rem;
        font-weight: 800;
        margin-bottom: 16px;
        background: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 50%, #06b6d4 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-shadow: 0 0 50px rgba(59, 130, 246, 0.3);
        position: relative;
        z-index: 1;
    }
    
    .header-subtitle {
        font-size: 1.5rem;
        color: #94a3b8;
        font-weight: 400;
        position: relative;
        z-index: 1;
    }
    
    .header-badge {
        display: inline-block;
        background: linear-gradient(135deg, #059669 0%, #10b981 100%);
        color: white;
        padding: 8px 20px;
        border-radius: 50px;
        font-size: 0.9rem;
        font-weight: 600;
        margin-top: 20px;
        box-shadow: 0 8px 25px rgba(16, 185, 129, 0.3);
        position: relative;
        z-index: 1;
    }
    
    /* Student-Focused Upload Section */
    .upload-section {
        background: linear-gradient(145deg, rgba(15, 23, 42, 0.8) 0%, rgba(30, 41, 59, 0.8) 100%);
        border: 2px dashed rgba(59, 130, 246, 0.4);
        border-radius: 20px;
        padding: 50px;
        text-align: center;
        margin: 40px 0;
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        backdrop-filter: blur(10px);
        position: relative;
        overflow: hidden;
    }
    
    .upload-section::before {
        content: '';
        position: absolute;
        top: -2px;
        left: -2px;
        right: -2px;
        bottom: -2px;
        background: linear-gradient(45deg, #3b82f6, #8b5cf6, #06b6d4, #3b82f6);
        border-radius: 20px;
        z-index: -1;
        animation: borderRotate 4s linear infinite;
        opacity: 0;
        transition: opacity 0.3s ease;
    }
    
    .upload-section:hover::before {
        opacity: 1;
    }
    
    .upload-section:hover {
        border-color: rgba(59, 130, 246, 0.8);
        background: linear-gradient(145deg, rgba(15, 23, 42, 0.9) 0%, rgba(30, 41, 59, 0.9) 100%);
        transform: translateY(-4px);
        box-shadow: 0 20px 40px rgba(0, 0, 0, 0.4);
    }
    
    /* Premium Score Card */
    .score-card {
        background: linear-gradient(145deg, #1e293b 0%, #334155 100%);
        border-radius: 24px;
        padding: 40px;
        text-align: center;
        box-shadow: 
            0 20px 40px rgba(0, 0, 0, 0.4),
            inset 0 1px 0 rgba(255, 255, 255, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.1);
        margin: 30px 0;
        position: relative;
        overflow: hidden;
    }
    
    .score-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 6px;
        background: linear-gradient(90deg, #ef4444 0%, #f59e0b 25%, #10b981 50%, #3b82f6 75%, #8b5cf6 100%);
    }
    
    .score-display {
        font-size: 5rem;
        font-weight: 900;
        background: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin: 30px 0;
        text-shadow: 0 0 30px rgba(59, 130, 246, 0.3);
        filter: drop-shadow(0 4px 8px rgba(0, 0, 0, 0.3));
    }
    
    .score-label {
        font-size: 1.3rem;
        color: #94a3b8;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 2px;
        margin-bottom: 10px;
    }
    
    .score-status {
        font-size: 1.1rem;
        font-weight: 600;
        padding: 12px 24px;
        border-radius: 50px;
        display: inline-block;
        margin-top: 20px;
    }
    
    /* Analysis Cards with Glassmorphism */
    .analysis-card {
        background: linear-gradient(145deg, rgba(30, 41, 59, 0.8) 0%, rgba(51, 65, 85, 0.8) 100%);
        backdrop-filter: blur(20px);
        border-radius: 20px;
        padding: 30px;
        margin: 20px 0;
        box-shadow: 
            0 8px 32px rgba(0, 0, 0, 0.3),
            inset 0 1px 0 rgba(255, 255, 255, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.1);
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
    }
    
    .analysis-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        width: 4px;
        height: 100%;
        background: linear-gradient(180deg, #3b82f6 0%, #8b5cf6 100%);
    }
    
    .analysis-card:hover {
        transform: translateY(-8px) scale(1.02);
        box-shadow: 0 20px 40px rgba(0, 0, 0, 0.4);
        border-color: rgba(59, 130, 246, 0.3);
    }
    
    .card-title {
        font-size: 1.5rem;
        font-weight: 700;
        color: #e2e8f0;
        margin-bottom: 20px;
        display: flex;
        align-items: center;
        gap: 12px;
    }
    
    .card-content {
        color: #cbd5e1;
        line-height: 1.8;
        font-size: 1rem;
    }
    
    /* Modern Skill Tags */
    .skill-tag {
        display: inline-block;
        padding: 10px 18px;
        border-radius: 50px;
        margin: 6px;
        font-size: 0.9rem;
        font-weight: 600;
        transition: all 0.3s ease;
        border: 1px solid transparent;
        position: relative;
        overflow: hidden;
    }
    
    .skill-tag.missing {
        background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
        color: white;
        box-shadow: 0 4px 15px rgba(239, 68, 68, 0.4);
    }
    
    .skill-tag.present {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        color: white;
        box-shadow: 0 4px 15px rgba(16, 185, 129, 0.4);
    }
    
    .skill-tag:hover {
        transform: translateY(-2px) scale(1.05);
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.3);
    }
    
    /* Premium Progress Bars */
    .progress-container {
        background: rgba(15, 23, 42, 0.8);
        border-radius: 15px;
        height: 16px;
        margin: 15px 0;
        overflow: hidden;
        box-shadow: inset 0 2px 4px rgba(0, 0, 0, 0.3);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .progress-bar {
        height: 100%;
        border-radius: 15px;
        transition: width 1s cubic-bezier(0.4, 0, 0.2, 1);
        background: linear-gradient(90deg, #3b82f6 0%, #8b5cf6 50%, #06b6d4 100%);
        box-shadow: 0 0 20px rgba(59, 130, 246, 0.5);
        position: relative;
    }
    
    .progress-bar::after {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: linear-gradient(45deg, transparent 25%, rgba(255, 255, 255, 0.1) 25%, rgba(255, 255, 255, 0.1) 50%, transparent 50%);
        background-size: 20px 20px;
        animation: progressShine 2s linear infinite;
    }
    
    @keyframes progressShine {
        0% { background-position: -20px 0; }
        100% { background-position: 40px 0; }
    }
    
    /* Enhanced Loading Animation */
    .loading-container {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        padding: 80px 40px;
        background: linear-gradient(145deg, rgba(15, 23, 42, 0.9) 0%, rgba(30, 41, 59, 0.9) 100%);
        backdrop-filter: blur(20px);
        border-radius: 24px;
        margin: 40px 0;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .loading-spinner {
        width: 100px;
        height: 100px;
        border: 8px solid rgba(59, 130, 246, 0.2);
        border-top: 8px solid #3b82f6;
        border-radius: 50%;
        animation: spin 1.5s cubic-bezier(0.68, -0.55, 0.265, 1.55) infinite;
        position: relative;
    }
    
    .loading-spinner::after {
        content: '';
        position: absolute;
        top: -8px;
        left: -8px;
        right: -8px;
        bottom: -8px;
        border: 8px solid transparent;
        border-top: 8px solid #8b5cf6;
        border-radius: 50%;
        animation: spin 2s linear infinite reverse;
    }
    
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    
    .timer-display {
        font-size: 3rem;
        font-weight: 800;
        background: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin: 30px 0;
        font-family: 'Courier New', monospace;
    }
    
    .progress-steps {
        display: flex;
        justify-content: center;
        margin: 40px 0;
        gap: 15px;
        flex-wrap: wrap;
    }
    
    .step-item {
        display: flex;
        align-items: center;
        padding: 15px 25px;
        background: rgba(30, 41, 59, 0.8);
        backdrop-filter: blur(10px);
        border-radius: 50px;
        border: 2px solid rgba(255, 255, 255, 0.1);
        transition: all 0.4s ease;
        font-weight: 600;
        color: #94a3b8;
        position: relative;
    }
    
    .step-item.active {
        background: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%);
        color: white;
        border-color: #3b82f6;
        box-shadow: 0 8px 25px rgba(59, 130, 246, 0.4);
    }
    
    .step-item.completed {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        color: white;
        border-color: #10b981;
        box-shadow: 0 8px 25px rgba(16, 185, 129, 0.4);
    }
    
    .recommendation-item {
        background: linear-gradient(145deg, rgba(15, 23, 42, 0.8) 0%, rgba(30, 41, 59, 0.8) 100%);
        backdrop-filter: blur(10px);
        border-left: 6px solid #3b82f6;
        padding: 20px 25px;
        margin: 15px 0;
        border-radius: 0 16px 16px 0;
        transition: all 0.3s ease;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .recommendation-item:hover {
        background: linear-gradient(145deg, rgba(30, 41, 59, 0.9) 0%, rgba(51, 65, 85, 0.9) 100%);
        border-left-color: #8b5cf6;
        transform: translateX(8px);
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.3);
    }
    
    .recommendation-priority {
        display: inline-block;
        padding: 6px 16px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 700;
        margin-left: 15px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .priority-high {
        background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
        color: white;
        box-shadow: 0 4px 15px rgba(239, 68, 68, 0.3);
    }
    
    .priority-medium {
        background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%);
        color: white;
        box-shadow: 0 4px 15px rgba(245, 158, 11, 0.3);
    }
    
    .priority-low {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        color: white;
        box-shadow: 0 4px 15px rgba(16, 185, 129, 0.3);
    }
    
    /* Premium Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%);
        color: white;
        border: none;
        padding: 16px 32px;
        border-radius: 50px;
        font-size: 1.1rem;
        font-weight: 700;
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: 0 8px 25px rgba(59, 130, 246, 0.4);
        text-transform: none;
        letter-spacing: 0.5px;
        border: 2px solid transparent;
        position: relative;
        overflow: hidden;
    }
    
    .stButton > button::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.2), transparent);
        transition: left 0.5s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-4px) scale(1.05);
        box-shadow: 0 20px 40px rgba(59, 130, 246, 0.6);
        background: linear-gradient(135deg, #1d4ed8 0%, #7c3aed 100%);
        border-color: rgba(255, 255, 255, 0.2);
    }
    
    .stButton > button:hover::before {
        left: 100%;
    }
    
    /* Career Journey Section */
    .career-journey {
        background: linear-gradient(145deg, rgba(15, 23, 42, 0.8) 0%, rgba(30, 41, 59, 0.8) 100%);
        backdrop-filter: blur(20px);
        border-radius: 24px;
        padding: 40px;
        margin: 40px 0;
        border: 1px solid rgba(255, 255, 255, 0.1);
        text-align: center;
    }
    
    .journey-step {
        display: inline-block;
        background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
        border-radius: 16px;
        padding: 25px;
        margin: 15px;
        min-width: 200px;
        text-align: center;
        border: 2px solid rgba(59, 130, 246, 0.3);
        transition: all 0.3s ease;
        position: relative;
    }
    
    .journey-step:hover {
        border-color: #3b82f6;
        transform: translateY(-8px);
        box-shadow: 0 15px 35px rgba(59, 130, 246, 0.3);
    }
    
    .journey-step-number {
        background: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%);
        color: white;
        width: 40px;
        height: 40px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 700;
        margin: 0 auto 15px;
    }
    
    /* Achievement Badges */
    .achievement-badge {
        display: inline-block;
        background: linear-gradient(135deg, #059669 0%, #10b981 100%);
        color: white;
        padding: 8px 20px;
        border-radius: 50px;
        font-size: 0.9rem;
        font-weight: 600;
        margin: 8px;
        box-shadow: 0 4px 15px rgba(16, 185, 129, 0.3);
        transition: all 0.3s ease;
    }
    
    .achievement-badge:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(16, 185, 129, 0.5);
    }
    
    /* Hide Streamlit Elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .stDeployButton {display: none;}
    
    /* Mobile Responsiveness */
    @media (max-width: 768px) {
        .header-title { font-size: 2.5rem; }
        .score-display { font-size: 3.5rem; }
        .main-container { margin: 10px; padding: 25px; }
        .progress-steps { flex-direction: column; align-items: center; }
        .journey-step { margin: 10px 0; }
    }
    
    /* Custom Scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: rgba(15, 23, 42, 0.5);
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%);
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(135deg, #1d4ed8 0%, #7c3aed 100%);
    }
    </style>
    """, unsafe_allow_html=True)

# ==================== DATA LOADING ====================

@st.cache_data
def load_datasets():
    """Load all CSV datasets"""
    datasets = {}
    
    try:
        if not os.path.exists(DATA_DIR):
            os.makedirs(DATA_DIR)
        
        # Create sample data if not exists
        if not os.path.exists(COURSES_PATH):
            sample_courses = pd.DataFrame({
                "skills": ["python", "sql", "java", "machine learning", "aws", "react", "html", "css", "tensorflow", "deep learning", "power bi", "statistics", "git", "javascript", "c++", "digital marketing", "hindi-nlp"],
                "course_name": ["Python for Everybody", "SQL for Data Science", "Java Programming Masterclass", "Machine Learning by Andrew Ng", "AWS Certified Cloud Practitioner", "React - The Complete Guide", "HTML and CSS for Beginners", "CSS - The Complete Guide", "TensorFlow Developer Certificate", "Deep Learning Specialization", "Power BI Data Analyst", "Statistics for Data Science", "Version Control with Git", "JavaScript: Understanding the Weird Parts", "C++ Programming", "Digital Marketing Specialization", "Natural Language Processing in Hindi"],
                "provider": ["Coursera", "Coursera", "Udemy", "Coursera", "AWS", "Udemy", "freeCodeCamp", "Udemy", "Google", "Coursera", "Microsoft", "Coursera", "Udacity", "Udemy", "Udemy", "Coursera", "Coursera"],
                "duration_weeks": [4, 3, 6, 8, 4, 5, 2, 2, 8, 8, 4, 4, 1, 4, 8, 4, 4]
            })
            sample_courses.to_csv(COURSES_PATH, index=False)
        
        if not os.path.exists(EMERGING_TECH_PATH):
            sample_emerging = pd.DataFrame({
                "technology": ["AI", "Blockchain", "VR", "AR", "Big Data", "Cybersecurity"],
                "description": ["Artificial Intelligence involves machine learning and deep learning", "Decentralized ledger technology for secure transactions", "Virtual Reality creates immersive digital environments", "Augmented Reality overlays digital information on real world", "Large and complex data sets analyzed for insights", "Protects systems from digital attacks and threats"],
                "growth_rate": ["15%", "12%", "10%", "8%", "14%", "11%"]
            })
            sample_emerging.to_csv(EMERGING_TECH_PATH, index=False)
        
        if not os.path.exists(JOBS_PATH):
            sample_jobs = pd.DataFrame({
                "title": ["Software Engineer", "Data Scientist", "Frontend Developer", "Full Stack Developer", "Mobile App Developer", "AI Engineer", "Blockchain Developer", "VR/AR Developer", "Big Data Engineer", "Data Analyst", "Cyber Security Analyst"],
                "description": ["Develop software applications", "Analyze data for insights", "Build user interfaces", "Handle frontend and backend", "Develop mobile apps", "Build AI models", "Develop blockchain solutions", "Create VR/AR experiences", "Manage big data pipelines", "Analyze business data", "Protect systems from threats"],
                "skills": ["python, java, git", "python, sql, machine learning", "html, css, react", "react, node.js, sql", "android, ios, flutter", "python, tensorflow, deep learning", "solidity, ethereum", "unity, c#", "hadoop, spark", "sql, power bi", "wireshark, kali linux"]
            })
            sample_jobs.to_csv(JOBS_PATH, index=False)
        
        if not os.path.exists(SKILLS_DATASET_PATH):
            sample_skills = pd.DataFrame({
                "title": ["Software Engineer", "Data Scientist", "Frontend Developer", "Full Stack Developer", "Mobile App Developer", "Artificial Intelligence", "Blockchain", "VR & AR", "Big Data", "Data Science", "Cyber Security"],
                "skills": ["python, java, javascript, git, data structures, algorithms", "python, r, sql, statistics, machine learning", "html, css, javascript, typescript, react", "html, css, javascript, react, node.js", "android, ios, react native, flutter", "python, r, java, c++, julia", "solidity, javascript, go, rust, python", "c#, c++, javascript, python", "python, scala, java, sql", "python, r, sql, julia", "python, c, javascript, bash"]
            })
            sample_skills.to_csv(SKILLS_DATASET_PATH, index=False)
        
        if not os.path.exists(SYNTHETIC_JOBS_PATH):
            sample_synthetic = pd.DataFrame({
                "id": ["syn_1", "syn_2", "syn_3"],
                "title": ["Synthetic Software Engineer India", "Synthetic Data Scientist India", "Synthetic Frontend Developer India"],
                "description": ["Synthetic tech role in India", "Synthetic data role in India", "Synthetic web role in India"],
                "skills": ["python, java", "python, sql, machine learning", "html, css, react"]
            })
            sample_synthetic.to_csv(SYNTHETIC_JOBS_PATH, index=False)
        
        datasets['courses'] = pd.read_csv(COURSES_PATH) if os.path.exists(COURSES_PATH) else pd.DataFrame()
        datasets['emerging_tech'] = pd.read_csv(EMERGING_TECH_PATH) if os.path.exists(EMERGING_TECH_PATH) else pd.DataFrame()
        datasets['jobs'] = pd.read_csv(JOBS_PATH) if os.path.exists(JOBS_PATH) else pd.DataFrame()
        datasets['skills_dataset'] = pd.read_csv(SKILLS_DATASET_PATH) if os.path.exists(SKILLS_DATASET_PATH) else pd.DataFrame()
        datasets['synthetic_jobs'] = pd.read_csv(SYNTHETIC_JOBS_PATH) if os.path.exists(SYNTHETIC_JOBS_PATH) else pd.DataFrame()
        
    except Exception as e:
        st.error(f"Error loading datasets: {str(e)}")
        print(f"Error loading datasets in terminal: {str(e)}")
    
    return datasets

# ==================== HELPER FUNCTIONS ====================

def extract_text_from_file(uploaded_file):
    """Extract text content from uploaded file"""
    try:
        if uploaded_file.type == "application/pdf":
            # Mock PDF extraction
            return f"""John Doe
Software Engineering Student | Final Year
Email: john.doe@university.edu | Phone: +91-9876543210
LinkedIn: linkedin.com/in/johndoe | GitHub: github.com/johndoe

EDUCATION
Bachelor of Technology in Computer Science Engineering
ABC University of Technology (2021-2025)
CGPA: 8.5/10.0

TECHNICAL SKILLS
Programming Languages: Python, Java, JavaScript, C++
Web Development: HTML, CSS, React.js, Node.js, Express.js
Database: MySQL, MongoDB
Tools & Technologies: Git, Docker, AWS, Linux
Data Structures and Algorithms: Proficient in problem-solving

PROJECTS
1. E-commerce Website (Jan 2024 - Mar 2024)
   - Developed full-stack e-commerce platform using MERN stack
   - Implemented user authentication, payment gateway integration
   - Deployed on AWS with 99.9% uptime

2. Student Management System (Sep 2023 - Nov 2023)
   - Built desktop application using Java and MySQL
   - Implemented CRUD operations and user management
   - Used by 500+ students in college

INTERNSHIPS
Software Development Intern | TechCorp Solutions (Jun 2023 - Aug 2023)
- Worked on backend API development using Python and Django
- Improved application performance by 25%
- Collaborated with team of 5 developers

ACHIEVEMENTS
- Winner, National Level Hackathon 2023
- Google Summer of Code Participant 2023
- Technical Head, Computer Science Club

CERTIFICATIONS
- AWS Certified Cloud Practitioner
- Oracle Java SE 11 Programmer
- Google Data Analytics Certificate"""
        elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            return f"Mock extracted content from {uploaded_file.name}. Resume content with professional experience and technical skills for a student profile."
        else:
            return uploaded_file.read().decode("utf-8")
    except Exception as e:
        return f"Error extracting text: {str(e)}"

def get_role_requirements():
    """Enhanced role requirements for students"""
    return {
        "Software Engineer": {
            "core_skills": ["python", "java", "javascript", "git", "data structures", "algorithms"],
            "frameworks": ["react", "node.js", "django", "spring boot", "express"],
            "databases": ["sql", "mysql", "mongodb", "postgresql"],
            "tools": ["docker", "aws", "linux", "postman", "jenkins"],
            "concepts": ["oop", "rest api", "microservices", "testing", "version control"]
        },
        "Data Scientist": {
            "core_skills": ["python", "r", "sql", "statistics", "machine learning"],
            "libraries": ["pandas", "numpy", "matplotlib", "seaborn", "scikit-learn", "tensorflow"],
            "tools": ["jupyter", "tableau", "power bi", "excel", "git"],
            "databases": ["sql", "mongodb", "hadoop"],
            "concepts": ["data mining", "deep learning", "nlp", "computer vision", "big data"]
        },
        "Frontend Developer": {
            "core_skills": ["html", "css", "javascript", "typescript", "react"],
            "frameworks": ["vue.js", "angular", "next.js", "bootstrap", "tailwind"],
            "tools": ["git", "webpack", "npm", "yarn", "figma"],
            "concepts": ["responsive design", "spa", "pwa", "accessibility", "performance optimization"],
            "testing": ["jest", "cypress", "testing library"]
        },
        "Full Stack Developer": {
            "frontend": ["html", "css", "javascript", "react", "vue.js"],
            "backend": ["node.js", "python", "java", "express", "django"],
            "databases": ["mysql", "mongodb", "postgresql"],
            "tools": ["git", "docker", "aws", "heroku"],
            "concepts": ["rest api", "authentication", "deployment", "testing"]
        },
        "Mobile App Developer": {
            "platforms": ["android", "ios", "react native", "flutter"],
            "languages": ["java", "kotlin", "swift", "dart", "javascript"],
            "tools": ["android studio", "xcode", "firebase", "git"],
            "concepts": ["ui/ux", "api integration", "local storage", "push notifications"],
            "testing": ["unit testing", "ui testing", "device testing"]
        },
        "Artificial Intelligence": {
            "platforms": ["cloud", "on-premise", "edge devices"],
            "languages": ["python", "r", "java", "c++", "julia"],
            "tools": ["tensorflow", "pytorch", "jupyter", "git", "docker"],
            "concepts": ["machine learning", "deep learning", "natural language processing", "computer vision", "reinforcement learning"],
            "testing": ["model validation", "unit testing", "integration testing", "performance testing"]
        },
        "Blockchain": {
            "platforms": ["ethereum", "hyperledger", "binance smart chain", "solana"],
            "languages": ["solidity", "javascript", "go", "rust", "python"],
            "tools": ["truffle", "remix", "metamask", "ganache", "git"],
            "concepts": ["smart contracts", "decentralized finance", "consensus algorithms", "tokenization", "cryptography"],
            "testing": ["smart contract auditing", "unit testing", "integration testing", "security testing"]
        },
        "VR & AR": {
            "platforms": ["oculus", "hololens", "unity", "unreal engine"],
            "languages": ["c#", "c++", "javascript", "python"],
            "tools": ["unity", "unreal engine", "blender", "git", "ar foundation"],
            "concepts": ["3d modeling", "spatial computing", "gesture recognition", "immersive storytelling"],
            "testing": ["usability testing", "performance testing", "device compatibility testing"]
        },
        "Big Data": {
            "platforms": ["hadoop", "spark", "cloud platforms", "kafka"],
            "languages": ["python", "scala", "java", "sql"],
            "tools": ["apache spark", "hadoop", "tableau", "jupyter", "git"],
            "concepts": ["data lakes", "etl pipelines", "real-time processing", "data warehousing"],
            "testing": ["data validation", "performance testing", "scalability testing"]
        },
        "Data Science": {
            "platforms": ["cloud", "local environments", "colab"],
            "languages": ["python", "r", "sql", "julia"],
            "tools": ["jupyter", "pandas", "scikit-learn", "tableau", "git"],
            "concepts": ["statistical modeling", "machine learning", "data visualization", "feature engineering"],
            "testing": ["model validation", "unit testing", "integration testing"]
        },
        "Cyber Security": {
            "core_skills": ["python", "c", "javascript", "bash"],
            "tools": ["wireshark", "metasploit", "nmap", "kali linux", "burp suite"],
            "concepts": ["ethical hacking", "penetration testing", "network security", "cryptography", "incident response"],
            "certifications": ["ceh", "oscp", "cissp"],
            "testing": ["vulnerability assessment", "security auditing"]
        }
    }

def initialize_ml_models():
    """Initialize ML models"""
    if not ML_AVAILABLE:
        print("ML models not initialized due to missing dependencies")
        return None, None, None
    
    try:
        extractor = IndustrySkillExtractor()
        gnn_model = GINXMLC()
        graph_data = load_pre_generated_data()
        print("ML models initialized successfully in terminal")
        return extractor, gnn_model, graph_data
    except Exception as e:
        st.warning(f"Error initializing ML models: {str(e)}")
        print(f"Error initializing ML models in terminal: {str(e)}")
        return None, None, None

def extract_resume_components(resume_content: str) -> Dict:
    """Extract key components from resume content"""
    components = {
        'has_education': False,
        'has_projects': False,
        'has_internship': False,
        'has_achievements': False,
        'has_leadership': False,
        'word_count': len(resume_content.split()),
        'skills': []
    }
    
    content_lower = resume_content.lower()
    
    # Check for sections
    components['has_education'] = any(keyword in content_lower for keyword in ['education', 'university', 'degree', 'bachelor', 'master'])
    components['has_projects'] = any(keyword in content_lower for keyword in ['projects', 'portfolio', 'developed', 'built'])
    components['has_internship'] = any(keyword in content_lower for keyword in ['internship', 'intern', 'work experience', 'professional experience'])
    components['has_achievements'] = any(keyword in content_lower for keyword in ['achievements', 'awards', 'certifications', 'honors'])
    components['has_leadership'] = any(keyword in content_lower for keyword in ['lead', 'leader', 'president', 'head', 'organizer'])
    
    # Basic skill extraction
    skills_list = ['python', 'java', 'javascript', 'sql', 'html', 'css', 'react', 'node.js', 'django', 
                   'aws', 'docker', 'git', 'mysql', 'mongodb', 'tensorflow', 'pandas', 'numpy']
    components['skills'] = [skill for skill in skills_list if skill in content_lower]
    
    return components

def analyze_resume_with_ml(resume_content: str, target_role: str, extractor, gnn_model, graph_data, datasets):
    """Analyze resume using ML models"""
    results = {
        'found_skills': [],
        'missing_skills': [],
        'skill_match_score': 0,
        'content_quality_score': 0,
        'experience_score': 0,
        'overall_score': 0,
        'readiness_level': {},
        'course_recommendations': [],
        'job_matches': [],
        'salary_info': {'entry_level': 0, 'mid_level': 0, 'senior_level': 0},
        'emerging_tech_analysis': {'trending': [], 'recommendations': []},
        'placement_forecast': {},
        'career_suggestions': [],
        'strengths': [],
        'weaknesses': [],
        'has_education': False,
        'has_projects': False,
        'has_internship': False,
        'has_achievements': False,
        'has_leadership': False,
        'word_count': 0
    }
    
    try:
        # Extract components
        components = extract_resume_components(resume_content)
        results.update(components)
        
        # ML-based skill extraction
        if ML_AVAILABLE and extractor:
            found_skills = extractor.extract_skills(resume_content)
            results['found_skills'] = found_skills if found_skills else components['skills']
        else:
            results['found_skills'] = components['skills']
        
        # Get role requirements
        role_requirements = get_role_requirements()
        role_skills = []
        for category in role_requirements.get(target_role, {}).values():
            if isinstance(category, list):
                role_skills.extend(category)
        
        # Calculate missing skills
        results['missing_skills'] = [skill for skill in role_skills if skill not in results['found_skills']]
        
        # Calculate scores
        total_role_skills = len(role_skills)
        results['skill_match_score'] = (len(results['found_skills']) / total_role_skills * 100) if total_role_skills > 0 else 0
        results['content_quality_score'] = min(100, components['word_count'] // 2)
        results['experience_score'] = (50 if components['has_projects'] else 0) + (30 if components['has_internship'] else 0) + (20 if components['has_achievements'] else 0)
        results['overall_score'] = round((results['skill_match_score'] * 0.5 + results['content_quality_score'] * 0.3 + results['experience_score'] * 0.2), 1)
        
        # Readiness level
        if results['overall_score'] >= 80:
            results['readiness_level'] = {
                'level': 'Excellent',
                'color': '#10b981',
                'description': 'Your resume is highly competitive for your target role!',
                'next_step': 'Apply to top companies and prepare for interviews'
            }
        elif results['overall_score'] >= 60:
            results['readiness_level'] = {
                'level': 'Good',
                'color': '#3b82f6',
                'description': 'Your resume is strong but could use some enhancements.',
                'next_step': 'Focus on adding missing skills and projects'
            }
        elif results['overall_score'] >= 40:
            results['readiness_level'] = {
                'level': 'Fair',
                'color': '#f59e0b',
                'description': 'Your resume needs significant improvements.',
                'next_step': 'Work on building projects and learning key skills'
            }
        else:
            results['readiness_level'] = {
                'level': 'Needs Work',
                'color': '#ef4444',
                'description': 'Your resume needs substantial updates to be competitive.',
                'next_step': 'Start with foundational skills and build a portfolio'
            }
        
        # Course recommendations
        if datasets.get('courses') is not None and not datasets['courses'].empty:
            for skill in results['missing_skills'][:3]:
                course_match = datasets['courses'][datasets['courses']['skills'].str.contains(skill, case=False, na=False)]
                if not course_match.empty:
                    course = course_match.iloc[0]
                    results['course_recommendations'].append({
                        'skill': skill,
                        'course': course['course_name'],
                        'provider': course['provider'],
                        'duration': f"{course['duration_weeks']} weeks"
                    })
        
        # Job matches
        if datasets.get('jobs') is not None and not datasets['jobs'].empty:
            for _, job in datasets['jobs'].iterrows():
                job_skills = job['skills'].lower().split(', ')
                match_count = len([s for s in job_skills if s in results['found_skills']])
                match_percentage = (match_count / len(job_skills) * 100) if job_skills else 0
                if match_percentage >= 50:
                    results['job_matches'].append({
                        'title': job['title'],
                        'description': job['description'],
                        'match_percentage': round(match_percentage, 1)
                    })
        
        # Salary info (mock)
        results['salary_info'] = {
            'entry_level': 600000 + len(results['found_skills']) * 50000,
            'mid_level': 1200000 + len(results['found_skills']) * 75000,
            'senior_level': 2000000 + len(results['found_skills']) * 100000
        }
        
        # Emerging tech analysis
        if datasets.get('emerging_tech') is not None and not datasets['emerging_tech'].empty:
            results['emerging_tech_analysis']['trending'] = datasets['emerging_tech']['technology'].tolist()[:3]
            results['emerging_tech_analysis']['recommendations'] = datasets['emerging_tech']['technology'].tolist()[3:]
        
        # Placement forecast
        if ML_AVAILABLE:
            try:
                forecast = forecast_placement(resume_content, target_role)
                results['placement_forecast'] = {
                    'probability': forecast.get('probability', 50),
                    'timeframe': forecast.get('timeframe', 6),
                    'key_factors': forecast.get('key_factors', ['Skills', 'Projects', 'Experience'])
                }
            except:
                results['placement_forecast'] = {}
        
        # Career suggestions
        for role in ['Software Engineer', 'Data Scientist', 'Frontend Developer']:
            if role != target_role:
                role_skills = []
                for category in role_requirements.get(role, {}).values():
                    if isinstance(category, list):
                        role_skills.extend(category)
                match_count = len([s for s in role_skills if s in results['found_skills']])
                match_percentage = (match_count / len(role_skills) * 100) if role_skills else 0
                if match_percentage >= 40:
                    results['career_suggestions'].append({
                        'role': role,
                        'match': f"{round(match_percentage, 1)}% match",
                        'reason': f"Your skills in {', '.join(results['found_skills'][:3])} align well with this role"
                    })
        
        # Strengths and weaknesses
        results['strengths'] = identify_student_strengths(results['found_skills'], results['content_quality_score'], results['has_projects'], results['has_internship'])
        results['weaknesses'] = identify_student_weaknesses(results['missing_skills'], results['content_quality_score'], results['has_projects'])
        results['recommendations'] = generate_student_recommendations(results['missing_skills'], results['overall_score'], results['has_projects'], results['has_internship'], target_role)
        
        print("Resume analysis completed successfully in terminal")
        return results
    
    except Exception as e:
        st.error(f"Analysis error: {str(e)}")
        print(f"Analysis error in terminal: {str(e)}")
        return results

def analyze_resume_content(resume_content: str, target_role: str):
    """Analyze resume content without ML"""
    results = {
        'found_skills': [],
        'missing_skills': [],
        'skill_match_score': 0,
        'content_quality_score': 0,
        'experience_score': 0,
        'presentation_score': 75,  # Default value
        'overall_score': 0,
        'readiness_level': {},
        'career_suggestions': [],
        'strengths': [],
        'weaknesses': [],
        'has_education': False,
        'has_projects': False,
        'has_internship': False,
        'has_achievements': False,
        'has_leadership': False,
        'word_count': 0,
        'recommendations': []
    }
    
    # Extract components
    components = extract_resume_components(resume_content)
    results.update(components)
    
    # Basic skill extraction
    results['found_skills'] = components['skills']
    
    # Get role requirements
    role_requirements = get_role_requirements()
    role_skills = []
    for category in role_requirements.get(target_role, {}).values():
        if isinstance(category, list):
            role_skills.extend(category)
    
    # Calculate missing skills
    results['missing_skills'] = [skill for skill in role_skills if skill not in results['found_skills']]
    
    # Calculate scores
    total_role_skills = len(role_skills)
    results['skill_match_score'] = (len(results['found_skills']) / total_role_skills * 100) if total_role_skills > 0 else 0
    results['content_quality_score'] = min(100, components['word_count'] // 2)
    results['experience_score'] = (50 if components['has_projects'] else 0) + (30 if components['has_internship'] else 0) + (20 if components['has_achievements'] else 0)
    results['overall_score'] = round((results['skill_match_score'] * 0.4 + results['content_quality_score'] * 0.3 + results['experience_score'] * 0.2 + results['presentation_score'] * 0.1), 1)
    
    # Readiness level
    if results['overall_score'] >= 80:
        results['readiness_level'] = {
            'level': 'Excellent',
            'color': '#10b981',
            'description': 'Your resume is highly competitive for your target role!',
            'next_step': 'Apply to top companies and prepare for interviews'
        }
    elif results['overall_score'] >= 60:
        results['readiness_level'] = {
            'level': 'Good',
            'color': '#3b82f6',
            'description': 'Your resume is strong but could use some enhancements.',
            'next_step': 'Focus on adding missing skills and projects'
        }
    elif results['overall_score'] >= 40:
        results['readiness_level'] = {
            'level': 'Fair',
            'color': '#f59e0b',
            'description': 'Your resume needs significant improvements.',
            'next_step': 'Work on building projects and learning key skills'
        }
    else:
        results['readiness_level'] = {
            'level': 'Needs Work',
            'color': '#ef4444',
            'description': 'Your resume needs substantial updates to be competitive.',
            'next_step': 'Start with foundational skills and build a portfolio'
        }
    
    # Career suggestions
    for role in ['Software Engineer', 'Data Scientist', 'Frontend Developer']:
        if role != target_role:
            role_skills = []
            for category in role_requirements.get(role, {}).values():
                if isinstance(category, list):
                    role_skills.extend(category)
            match_count = len([s for s in role_skills if s in results['found_skills']])
            match_percentage = (match_count / len(role_skills) * 100) if role_skills else 0
            if match_percentage >= 40:
                results['career_suggestions'].append({
                    'role': role,
                    'match': f"{round(match_percentage, 1)}% match",
                    'reason': f"Your skills in {', '.join(results['found_skills'][:3])} align well with this role"
                })
    
    # Strengths and weaknesses
    results['strengths'] = identify_student_strengths(results['found_skills'], results['content_quality_score'], results['has_projects'], results['has_internship'])
    results['weaknesses'] = identify_student_weaknesses(results['missing_skills'], results['content_quality_score'], results['has_projects'])
    results['recommendations'] = generate_student_recommendations(results['missing_skills'], results['overall_score'], results['has_projects'], results['has_internship'], target_role)
    
    print("Resume analysis (non-ML) completed successfully in terminal")
    return results

def identify_student_strengths(found_skills: List[str], content_score: int, has_projects: bool, has_internship: bool) -> List[str]:
    """Identify student strengths"""
    strengths = []
    
    if len(found_skills) >= 10:
        strengths.append(f"Diverse skill set with {len(found_skills)} technical skills")
    elif len(found_skills) >= 6:
        strengths.append("Good technical foundation across multiple areas")
    
    if has_projects:
        strengths.append("Practical experience through projects")
    
    if has_internship:
        strengths.append("Industry exposure via internships")
    
    if content_score >= 75:
        strengths.append("Well-structured resume with complete sections")
    
    if content_score >= 60:
        strengths.append("Strong balance of theory and practice")
    
    return strengths if strengths else ["Foundational skills present"]

def identify_student_weaknesses(missing_skills: List[str], content_score: int, has_projects: bool) -> List[str]:
    """Identify weaknesses"""
    weaknesses = []
    
    if len(missing_skills) >= 10:
        weaknesses.append(f"Need to acquire {len(missing_skills)} key industry skills")
    elif len(missing_skills) >= 5:
        weaknesses.append("Several important skills require development")
    
    if not has_projects:
        weaknesses.append("Portfolio needs more practical projects")
    
    if content_score < 60:
        weaknesses.append("Resume content could be more comprehensive")
    
    return weaknesses if weaknesses else ["Minor improvements suggested"]

def generate_student_recommendations(missing_skills: List[str], overall_score: float, has_projects: bool, has_internship: bool, target_role: str) -> List[Dict]:
    """Generate recommendations"""
    recommendations = []
    
    if overall_score < 65:
        recommendations.append({
            'text': 'Build 3-4 strong projects showcasing technical skills',
            'priority': 'high',
            'action': 'Project Development',
            'timeline': '2-3 months'
        })
    
    if not has_projects:
        recommendations.append({
            'text': 'Create portfolio with diverse projects',
            'priority': 'high',
            'action': 'Portfolio Building',
            'timeline': '1-2 months'
        })
    
    if len(missing_skills) >= 8:
        recommendations.append({
            'text': f"Priority skills: {', '.join(missing_skills[:4])}",
            'priority': 'high',
            'action': 'Skill Development',
            'timeline': '3-4 months'
        })
    
    recommendations.extend([
        {
            'text': 'Add GitHub links and live demos',
            'priority': 'medium',
            'action': 'Online Presence',
            'timeline': '1 week'
        },
        {
            'text': 'Include quantified project outcomes',
            'priority': 'medium',
            'action': 'Content Enhancement',
            'timeline': '1 week'
        },
        {
            'text': 'Get relevant certifications',
            'priority': 'low',
            'action': 'Certifications',
            'timeline': '2-4 weeks'
        },
        {
            'text': 'Participate in hackathons',
            'priority': 'low',
            'action': 'Competitions',
            'timeline': 'Ongoing'
        }
    ])
    
    return recommendations

# ==================== UI COMPONENTS ====================

def create_enhanced_loading_screen(step: str, progress: float, elapsed_time: float):
    """Create premium loading screen with student-focused messaging"""
    steps = [
        "📄 Scanning Your Resume",
        "🎯 Analyzing Technical Skills", 
        "📊 Evaluating Projects & Experience",
        "🚀 Calculating Job Readiness",
        "✨ Generating Career Roadmap"
    ]
    
    current_step_index = min(int(progress / 20), 4)
    
    tips = [
        "💡 Tip: Include GitHub links to showcase your coding projects",
        "💡 Tip: Quantify your project impact with numbers and metrics",
        "💡 Tip: List technologies used in each project clearly",
        "💡 Tip: Mention any internships or work experience prominently",
        "💡 Tip: Include relevant certifications and online courses"
    ]
    
    current_tip = tips[current_step_index] if current_step_index < len(tips) else tips[0]
    
    loading_html = f"""
    <div class="loading-container">
        <div class="loading-spinner"></div>
        <div style="margin-top: 40px;">
            <h3 style="color: #e2e8f0; margin-bottom: 20px; font-size: 1.8rem;">{step}</h3>
            <div class="timer-display">{elapsed_time:.1f}s</div>
            <div style="width: 400px; height: 12px; background: rgba(15, 23, 42, 0.8); border-radius: 15px; margin: 30px auto; overflow: hidden; border: 1px solid rgba(255, 255, 255, 0.1);">
                <div style="width: {progress}%; height: 100%; background: linear-gradient(90deg, #3b82f6, #8b5cf6, #06b6d4); transition: width 0.5s ease; border-radius: 15px; box-shadow: 0 0 20px rgba(59, 130, 246, 0.5);"></div>
            </div>
            <div class="progress-steps">
                {"".join([f'<div class="step-item {"completed" if i < current_step_index else "active" if i == current_step_index else ""}">{steps[i]}</div>' for i in range(5)])}
            </div>
            <div style="margin-top: 30px; color: #94a3b8; font-size: 1.1rem; font-style: italic;">
                {current_tip}
            </div>
        </div>
    </div>
    """
    return loading_html

def create_readiness_gauge(score: float, readiness_data: Dict):
    """Create student job readiness gauge"""
    color = readiness_data['color']
    
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = score,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {
            'text': f"<b>Job Readiness Score</b><br><span style='color:{color};font-size:18px'>{readiness_data['level']}</span>", 
            'font': {'size': 24, 'color': '#e2e8f0'}
        },
        delta = {'reference': 70, 'increasing': {'color': "#10b981"}, 'decreasing': {'color': "#ef4444"}},
        gauge = {
            'axis': {
                'range': [None, 100], 
                'tickwidth': 2, 
                'tickcolor': "#64748b",
                'tickfont': {'color': '#94a3b8', 'size': 14}
            },
            'bar': {'color': color, 'thickness': 0.4},
            'bgcolor': "rgba(15, 23, 42, 0.8)",
            'borderwidth': 2,
            'bordercolor': "rgba(255, 255, 255, 0.1)",
            'steps': [
                {'range': [0, 40], 'color': 'rgba(239, 68, 68, 0.2)'},
                {'range': [40, 55], 'color': 'rgba(245, 158, 11, 0.2)'},
                {'range': [55, 70], 'color': 'rgba(59, 130, 246, 0.2)'},
                {'range': [70, 100], 'color': 'rgba(16, 185, 129, 0.2)'}
            ],
            'threshold': {
                'line': {'color': "#ffffff", 'width': 6},
                'thickness': 0.8,
                'value': 85
            }
        }
    ))
    
    fig.update_layout(
        height=350,
        margin=dict(l=20, r=20, t=60, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={'color': "#e2e8f0", 'family': "Inter"}
    )
    
    return fig

def create_skills_radar_chart(found_skills: List[str], missing_skills: List[str], target_role: str):
    """Create radar chart showing skill coverage"""
    role_requirements = get_role_requirements()
    role_data = role_requirements.get(target_role, role_requirements["Software Engineer"])
    
    categories = []
    scores = []
    
    if isinstance(role_data, dict):
        for category, skills in role_data.items():
            if isinstance(skills, list):
                category_found = len([s for s in skills if s in found_skills])
                category_total = len(skills)
                category_score = (category_found / category_total) * 100 if category_total > 0 else 0
                categories.append(category.replace('_', ' ').title())
                scores.append(category_score)
    
    if not categories:
        categories = ['Technical Skills', 'Frameworks', 'Databases', 'Tools', 'Concepts']
        total_skills = len(found_skills) + len(missing_skills)
        base_score = (len(found_skills) / total_skills) * 100 if total_skills > 0 else 0
        scores = [base_score + np.random.randint(-15, 15) for _ in categories]
        scores = [max(0, min(100, score)) for score in scores]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=scores,
        theta=categories,
        fill='toself',
        name='Your Skills',
        line=dict(color='#3b82f6', width=3),
        fillcolor='rgba(59, 130, 246, 0.2)'
    ))
    
    fig.update_layout(
        polar=dict(
            bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                tickfont=dict(color='#94a3b8', size=12),
                gridcolor='rgba(255, 255, 255, 0.1)'
            ),
            angularaxis=dict(
                tickfont=dict(color='#e2e8f0', size=14)
            )
        ),
        showlegend=False,
        margin=dict(l=50, r=50, t=50, b=50),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=400
    )
    
    return fig

# ==================== MAIN APPLICATION ====================

def main():
    # Inject premium CSS
    inject_premium_css()
    
    # Load data and models
    datasets = load_datasets()
    extractor, gnn_model, graph_data = initialize_ml_models()
    
    # Header Section
    st.markdown("""
    <div class="header-section">
        <div class="header-title">🚀 Job Bridge</div>
        <div class="header-subtitle">ML Skill Gap Analyzer for Students</div>
        <div class="header-badge">Designed for Pre-final & Final Year Students</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Student Journey Section
    st.markdown("""
    <div class="career-journey">
        <h2 style="color: #e2e8f0; margin-bottom: 30px; font-size: 2rem;">📈 Your Career Journey</h2>
        <div style="display: flex; justify-content: center; flex-wrap: wrap; gap: 20px;">
            <div class="journey-step">
                <div class="journey-step-number">1</div>
                <h4 style="color: #e2e8f0; margin: 10px 0;">Analyze Resume</h4>
                <p style="color: #94a3b8; margin: 0;">Get detailed feedback on your current profile</p>
            </div>
            <div class="journey-step">
                <div class="journey-step-number">2</div>
                <h4 style="color: #e2e8f0; margin: 10px 0;">Skill Gap Analysis</h4>
                <p style="color: #94a3b8; margin: 0;">Identify missing skills for your target role</p>
            </div>
            <div class="journey-step">
                <div class="journey-step-number">3</div>
                <h4 style="color: #e2e8f0; margin: 10px 0;">Build Projects</h4>
                <p style="color: #94a3b8; margin: 0;">Create portfolio projects to showcase skills</p>
            </div>
            <div class="journey-step">
                <div class="journey-step-number">4</div>
                <h4 style="color: #e2e8f0; margin: 10px 0;">Land Your Job</h4>
                <p style="color: #94a3b8; margin: 0;">Apply confidently to your dream companies</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Upload Section
    st.markdown("""
    <div class="upload-section">
        <h2 style="color: #e2e8f0; margin-bottom: 25px; font-size: 2.2rem;">📤 Upload Your Resume or Enter Skills</h2>
        <p style="color: #94a3b8; font-size: 1.2rem; margin-bottom: 15px;">Get instant feedback on your resume or skills and discover your job readiness score</p>
        <div style="color: #64748b; font-size: 1rem;">
            <span style="color: #10b981;">✓</span> Technical skill assessment &nbsp;&nbsp;
            <span style="color: #10b981;">✓</span> Project portfolio review &nbsp;&nbsp;
            <span style="color: #10b981;">✓</span> Career roadmap generation
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Input Section
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="analysis-card">
            <div class="card-title">📁 Upload Resume File</div>
            <div class="card-content">
        """, unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader(
            "Choose your resume file",
            type=['pdf', 'docx', 'txt'],
            help="Supported formats: PDF, DOCX, TXT (Max size: 5MB)",
            label_visibility="collapsed"
        )
        
        st.markdown("</div></div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="analysis-card">
            <div class="card-title">✍ Or Paste Resume Text/Skills</div>
            <div class="card-content">
        """, unsafe_allow_html=True)
        
        resume_text = st.text_area(
            "Paste your resume content or list your skills (comma-separated)",
            height=200,
            placeholder="Paste resume content or enter skills (e.g., python, java, sql)...",
            label_visibility="collapsed"
        )
        
        st.markdown("</div></div>", unsafe_allow_html=True)
    
    # Target role selection
    st.markdown("""
    <div class="analysis-card">
        <div class="card-title">🎯 Select Your Target Role</div>
        <div class="card-content">
    """, unsafe_allow_html=True)
    
    target_role = st.selectbox(
        "Choose the role you're targeting for better analysis",
        ["Software Engineer", "Data Scientist", "Frontend Developer", "Full Stack Developer", 
         "Mobile App Developer", "Artificial Intelligence", "Blockchain", "VR & AR", 
         "Big Data", "Data Science", "Cyber Security"],
        help="This helps us provide role-specific skill gap analysis and recommendations",
        label_visibility="collapsed"
    )
    
    st.markdown("</div></div>", unsafe_allow_html=True)
    
    # Analysis button
    if st.button("🚀 Analyze My Resume/Skills & Get Job Readiness Score", type="primary", use_container_width=True):
        
        # Get content
        if uploaded_file is not None:
            resume_content = extract_text_from_file(uploaded_file)
            if "Error" not in resume_content:
                st.success(f"✅ Successfully processed {uploaded_file.name}")
        elif resume_text.strip():
            resume_content = resume_text
        else:
            st.error("❌ Please upload a resume file or enter skills/resume text to continue.")
            st.stop()
        
        # Enhanced loading animation
        loading_placeholder = st.empty()
        start_time = time.time()
        
        steps = [
            ("📄 Scanning Your Input...", 20),
            ("🎯 Analyzing Technical Skills...", 40), 
            ("📊 Evaluating Experience...", 60),
            ("🚀 Calculating Job Readiness...", 80),
            ("✨ Generating Career Roadmap...", 100)
        ]
        
        for step_text, progress in steps:
            elapsed = time.time() - start_time
            with loading_placeholder.container():
                st.markdown(create_enhanced_loading_screen(step_text, progress, elapsed), unsafe_allow_html=True)
            time.sleep(1.2)
        
        loading_placeholder.empty()
        
        # Perform analysis
        try:
            if ML_AVAILABLE and "," not in resume_content:
                results = analyze_resume_with_ml(resume_content, target_role, extractor, gnn_model, graph_data, datasets)
            else:
                # Handle comma-separated skills
                if "," in resume_content:
                    resume_content = f"TECHNICAL SKILLS: {resume_content}"
                results = analyze_resume_content(resume_content, target_role)
        except Exception as e:
            st.error(f"Analysis error: {str(e)}")
            print(f"Analysis error in terminal: {str(e)}")
            st.stop()
        
        # Success message
        st.success("🎉 Analysis Complete! Here's your comprehensive career assessment:")
        
        # Job Readiness Score Section
        readiness_data = results['readiness_level']
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.plotly_chart(create_readiness_gauge(results['overall_score'], readiness_data), use_container_width=True)

        with col2:
            st.markdown(f"""
            <div class="score-card">
                <div class="score-label">Overall Readiness</div>
                <div class="score-display">{results['overall_score']}/100</div>
                <div class="score-status" style="background: {readiness_data['color']};">
                    {readiness_data['level']}
                </div>
                <p style="color: #94a3b8; margin-top: 20px;">{readiness_data['description']}</p>
                <p style="color: #94a3b8; font-size: 0.9rem;">Next Step: {readiness_data['next_step']}</p>
            </div>
            """, unsafe_allow_html=True)

        # Skills Radar Chart
        st.markdown("## 🎯 Skills Coverage Analysis")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            radar_fig = create_skills_radar_chart(results['found_skills'], results['missing_skills'], target_role)
            st.plotly_chart(radar_fig, use_container_width=True)
        
        with col2:
            st.markdown(f"""
                            <div class="card-content">
                    <p><strong>Skill Match Score:</strong> {results['skill_match_score']:.1f}%</p>
                    <p><strong>Content Quality:</strong> {results['content_quality_score']:.1f}/100</p>
                    <p><strong>Experience Score:</strong> {results['experience_score']:.1f}/100</p>
                    <p><strong>Presentation Score:</strong> {results['presentation_score']:.1f}/100</p>
                    <p style="margin-top: 20px; color: #94a3b8;">
                        Your skills cover {len(results['found_skills'])} out of {len(results['found_skills']) + len(results['missing_skills'])} key areas for {target_role}.
                    </p>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        # Skills Breakdown
        st.markdown("## 🛠 Technical Skills Assessment")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div class="analysis-card">
                <div class="card-title">✅ Present Skills</div>
                <div class="card-content">
            """, unsafe_allow_html=True)
            
            if results['found_skills']:
                for skill in results['found_skills']:
                    st.markdown(f'<span class="skill-tag present">{skill.title()}</span>', unsafe_allow_html=True)
            else:
                st.markdown('<p style="color: #94a3b8;">No skills detected. Consider adding more technical skills.</p>', unsafe_allow_html=True)
            
            st.markdown("</div></div>", unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="analysis-card">
                <div class="card-title">❌ Missing Skills</div>
                <div class="card-content">
            """, unsafe_allow_html=True)
            
            if results['missing_skills']:
                for skill in results['missing_skills']:
                    st.markdown(f'<span class="skill-tag missing">{skill.title()}</span>', unsafe_allow_html=True)
            else:
                st.markdown('<p style="color: #94a3b8;">All key skills present! Great job!</p>', unsafe_allow_html=True)
            
            st.markdown("</div></div>", unsafe_allow_html=True)
        
        # Resume Structure Analysis
        st.markdown("## 📝 Resume Structure Analysis")
        st.markdown("""
        <div class="analysis-card">
            <div class="card-title">📊 Structure Breakdown</div>
            <div class="card-content">
                <div style="display: flex; flex-wrap: wrap; gap: 20px;">
        """, unsafe_allow_html=True)
        
        structure_components = [
            ('Education', results['has_education'], 'Academic background'),
            ('Projects', results['has_projects'], 'Practical experience'),
            ('Internships', results['has_internship'], 'Industry exposure'),
            ('Achievements', results['has_achievements'], 'Awards & certifications'),
            ('Leadership', results['has_leadership'], 'Leadership roles')
        ]
        
        for component, present, desc in structure_components:
            status = "✅ Present" if present else "❌ Missing"
            color = "#10b981" if present else "#ef4444"
            st.markdown(f"""
            <div style="flex: 1; min-width: 200px;">
                <div style="color: {color}; font-weight: 600;">{status}</div>
                <div style="color: #e2e8f0; font-weight: 500;">{component}</div>
                <div style="color: #94a3b8; font-size: 0.9rem;">{desc}</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("</div></div>", unsafe_allow_html=True)
        
        # Strengths and Weaknesses
        st.markdown("## 💪 Strengths & Areas for Improvement")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div class="analysis-card">
                <div class="card-title">✅ Your Strengths</div>
                <div class="card-content">
            """, unsafe_allow_html=True)
            
            for strength in results['strengths']:
                st.markdown(f'<div class="achievement-badge">{strength}</div>', unsafe_allow_html=True)
            
            st.markdown("</div></div>", unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="analysis-card">
                <div class="card-title">⚠ Areas to Improve</div>
                <div class="card-content">
            """, unsafe_allow_html=True)
            
            for weakness in results['weaknesses']:
                st.markdown(f'<div class="achievement-badge" style="background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%); box-shadow: 0 4px 15px rgba(239, 68, 68, 0.3);">{weakness}</div>', unsafe_allow_html=True)
            
            st.markdown("</div></div>", unsafe_allow_html=True)
        
        # Course Recommendations
        if results['course_recommendations']:
            st.markdown("## 📚 Recommended Courses")
            st.markdown("""
            <div class="analysis-card">
                <div class="card-title">🎓 Learn These Skills</div>
                <div class="card-content">
            """, unsafe_allow_html=True)
            
            for course in results['course_recommendations']:
                st.markdown(f"""
                <div class="recommendation-item">
                    <div style="color: #e2e8f0; font-weight: 600;">{course['course']}</div>
                    <div style="color: #94a3b8; font-size: 0.9rem;">
                        For: <span style="color: #3b82f6;">{course['skill'].title()}</span> | 
                        Provider: {course['provider']} | 
                        Duration: {course['duration']}
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("</div></div>", unsafe_allow_html=True)
        
        # Job Matches
        if results['job_matches']:
            st.markdown("## 💼 Potential Job Matches")
            st.markdown("""
            <div class="analysis-card">
                <div class="card-title">🎯 Matching Roles</div>
                <div class="card-content">
            """, unsafe_allow_html=True)
            
            for job in results['job_matches']:
                st.markdown(f"""
                <div class="recommendation-item">
                    <div style="color: #e2e8f0; font-weight: 600;">{job['title']}</div>
                    <div style="color: #94a3b8; font-size: 0.9rem;">
                        Match: <span style="color: #10b981;">{job['match_percentage']}%</span> | 
                        {job['description']}
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("</div></div>", unsafe_allow_html=True)
        
        # Career Suggestions
        if results['career_suggestions']:
            st.markdown("## 🌟 Alternative Career Paths")
            st.markdown("""
            <div class="analysis-card">
                <div class="card-title">🔄 Explore These Roles</div>
                <div class="card-content">
            """, unsafe_allow_html=True)
            
            for suggestion in results['career_suggestions']:
                st.markdown(f"""
                <div class="recommendation-item">
                    <div style="color: #e2e8f0; font-weight: 600;">{suggestion['role']}</div>
                    <div style="color: #94a3b8; font-size: 0.9rem;">
                        Match: <span style="color: #3b82f6;">{suggestion['match']}</span> | 
                        {suggestion['reason']}
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("</div></div>", unsafe_allow_html=True)
        
        # Salary Information
        st.markdown("## 💰 Expected Salary Range")
        st.markdown("""
        <div class="analysis-card">
            <div class="card-title">📈 Salary Projections</div>
            <div class="card-content">
                <div style="display: flex; flex-wrap: wrap; gap: 20px;">
                    <div style="flex: 1; min-width: 200px;">
                        <div style="color: #10b981; font-weight: 600;">Entry Level</div>
                        <div style="color: #e2e8f0; font-size: 1.2rem;">₹{results['salary_info']['entry_level']:,}/year</div>
                    </div>
                    <div style="flex: 1; min-width: 200px;">
                        <div style="color: #3b82f6; font-weight: 600;">Mid Level</div>
                        <div style="color: #e2e8f0; font-size: 1.2rem;">₹{results['salary_info']['mid_level']:,}/year</div>
                    </div>
                    <div style="flex: 1; min-width: 200px;">
                        <div style="color: #8b5cf6; font-weight: 600;">Senior Level</div>
                        <div style="color: #e2e8f0; font-size: 1.2rem;">₹{results['salary_info']['senior_level']:,}/year</div>
                    </div>
                </div>
                <p style="color: #94a3b8; font-size: 0.9rem; margin-top: 20px;">
                    *Based on industry standards and your current skill set
                </p>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Emerging Tech Analysis
        if results['emerging_tech_analysis']['trending']:
            st.markdown("## 🚀 Emerging Technologies")
            st.markdown("""
            <div class="analysis-card">
                <div class="card-title">🌟 Trending Technologies</div>
                <div class="card-content">
            """, unsafe_allow_html=True)
            
            for tech in results['emerging_tech_analysis']['trending']:
                st.markdown(f'<span class="skill-tag present">{tech}</span>', unsafe_allow_html=True)
            
            if results['emerging_tech_analysis']['recommendations']:
                st.markdown('<div style="margin-top: 20px; color: #e2e8f0;">Recommended to Learn:</div>', unsafe_allow_html=True)
                for tech in results['emerging_tech_analysis']['recommendations']:
                    st.markdown(f'<span class="skill-tag missing">{tech}</span>', unsafe_allow_html=True)
            
            st.markdown("</div></div>", unsafe_allow_html=True)
        
        # Placement Forecast
        if results['placement_forecast']:
            st.markdown("## 📅 Placement Forecast")
            st.markdown("""
            <div class="analysis-card">
                <div class="card-title">🔮 Job Placement Outlook</div>
                <div class="card-content">
                    <div class="progress-container">
                        <div class="progress-bar" style="width: {results['placement_forecast']['probability']}%;"></div>
                    </div>
                    <p style="color: #e2e8f0;">
                        <strong>Probability of Placement:</strong> {results['placement_forecast']['probability']}%
                    </p>
                    <p style="color: #e2e8f0;">
                        <strong>Estimated Timeframe:</strong> {results['placement_forecast']['timeframe']} months
                    </p>
                    <p style="color: #e2e8f0; margin-top: 20px;">Key Factors:</p>
            """, unsafe_allow_html=True)
            
            for factor in results['placement_forecast']['key_factors']:
                st.markdown(f'<div class="achievement-badge">{factor}</div>', unsafe_allow_html=True)
            
            st.markdown("</div></div>", unsafe_allow_html=True)
        
        # Actionable Recommendations
        st.markdown("## 🚀 Your Career Roadmap")
        st.markdown("""
        <div class="analysis-card">
            <div class="card-title">📋 Actionable Next Steps</div>
            <div class="card-content">
        """, unsafe_allow_html=True)
        
        for rec in results['recommendations']:
            priority_class = {
                'high': 'priority-high',
                'medium': 'priority-medium',
                'low': 'priority-low'
            }.get(rec['priority'], 'priority-low')
            
            st.markdown(f"""
            <div class="recommendation-item">
                <div style="color: #e2e8f0; font-weight: 600;">
                    {rec['text']}
                    <span class="recommendation-priority {priority_class}">{rec['priority'].title()}</span>
                </div>
                <div style="color: #94a3b8; font-size: 0.9rem;">
                    Action: {rec['action']} | Timeline: {rec['timeline']}
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("</div></div>", unsafe_allow_html=True)
        
        # Download Report Button
        report_content = f"""
        Job Bridge Analysis Report
        Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        Target Role: {target_role}
        
        Overall Readiness Score: {results['overall_score']}/100
        Readiness Level: {results['readiness_level']['level']}
        Description: {results['readiness_level']['description']}
        Next Step: {results['readiness_level']['next_step']}
        
        Skills Assessment:
        - Present Skills: {', '.join(results['found_skills'])}
        - Missing Skills: {', '.join(results['missing_skills'])}
        - Skill Match Score: {results['skill_match_score']:.1f}%
        
        Resume Structure:
        - Education: {'Present' if results['has_education'] else 'Missing'}
        - Projects: {'Present' if results['has_projects'] else 'Missing'}
        - Internships: {'Present' if results['has_internship'] else 'Missing'}
        - Achievements: {'Present' if results['has_achievements'] else 'Missing'}
        - Leadership: {'Present' if results['has_leadership'] else 'Missing'}
        
        Strengths:
        {chr(10).join(f'- {s}' for s in results['strengths'])}
        
        Areas for Improvement:
        {chr(10).join(f'- {w}' for w in results['weaknesses'])}
        
        Recommended Courses:
        {chr(10).join(f'- {c['course']} (For: {c['skill']}, Provider: {c['provider']}, Duration: {c['duration']})' for c in results['course_recommendations'])}
        
        Job Matches:
        {chr(10).join(f'- {j['title']} ({j['match_percentage']}% match)' for j in results['job_matches'])}
        
        Alternative Career Paths:
        {chr(10).join(f'- {s['role']} ({s['match']})' for s in results['career_suggestions'])}
        
        Salary Projections:
        - Entry Level: ₹{results['salary_info']['entry_level']:,}/year
        - Mid Level: ₹{results['salary_info']['mid_level']:,}/year
        - Senior Level: ₹{results['salary_info']['senior_level']:,}/year
        
        Emerging Technologies:
        - Trending: {', '.join(results['emerging_tech_analysis']['trending'])}
        - Recommended: {', '.join(results['emerging_tech_analysis']['recommendations'])}
        
        Placement Forecast:
        - Probability: {results['placement_forecast'].get('probability', 'N/A')}%
        - Timeframe: {results['placement_forecast'].get('timeframe', 'N/A')} months
        - Key Factors: {', '.join(results['placement_forecast'].get('key_factors', []))}
        
        Actionable Recommendations:
        {chr(10).join(f'- {r['text']} (Priority: {r['priority'].title()}, Action: {r['action']}, Timeline: {r['timeline']})' for r in results['recommendations'])}
        """
        
        st.markdown("""
        <div class="analysis-card">
            <div class="card-title">📥 Download Your Report</div>
            <div class="card-content">
        """, unsafe_allow_html=True)
        
        st.download_button(
            label="Download Analysis Report",
            data=report_content,
            file_name=f"Job_Bridge_Analysis_{target_role.replace(' ', '_')}.txt",
            mime="text/plain",
            type="primary",
            use_container_width=True
        )
        
        st.markdown("</div></div>", unsafe_allow_html=True)
        
        # Footer
        st.markdown("""
        <div style="text-align: center; color: #94a3b8; margin-top: 40px; padding: 20px;">
            Powered by Job Bridge | Designed for Students | © 2025
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()