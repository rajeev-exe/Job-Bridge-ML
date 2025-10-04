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
except ImportError as e:
    st.warning(f"ML modules not fully loaded: {e}. Running in basic mode.")
    ML_AVAILABLE = False

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
    
    /* Student Journey Steps */
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
    
    /* Student-Focused Recommendations */
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
            # Load datasets
        datasets['courses'] = pd.read_csv(COURSES_PATH) if os.path.exists(COURSES_PATH) else pd.DataFrame()
        datasets['emerging_tech'] = pd.read_csv(EMERGING_TECH_PATH) if os.path.exists(EMERGING_TECH_PATH) else pd.DataFrame()
        datasets['jobs'] = pd.read_csv(JOBS_PATH) if os.path.exists(JOBS_PATH) else pd.DataFrame()
        datasets['skills'] = pd.read_csv(SKILLS_DATASET_PATH) if os.path.exists(SKILLS_DATASET_PATH) else pd.DataFrame()
        datasets['synthetic_jobs'] = pd.read_csv(SYNTHETIC_JOBS_PATH) if os.path.exists(SYNTHETIC_JOBS_PATH) else pd.DataFrame()
    except Exception as e:
        st.error(f"Error loading datasets: {e}")
    
    return datasets

@st.cache_resource
def initialize_ml_models():
    """Initialize ML models"""
    if not ML_AVAILABLE:
        return None, None, None
    
    try:
        extractor = IndustrySkillExtractor()
        gnn_model = GINXMLC(num_skills=100)
        try:
            state_dict = torch.load('path_to_checkpoint.pth')
            gnn_model.load_state_dict(state_dict, strict=False)
        except:
            pass
        jobs_list, graph_dict = load_pre_generated_data()
        return extractor, gnn_model, (jobs_list, graph_dict)
    except:
        return None, None, None

# ==================== FILE EXTRACTION ====================

def extract_text_from_file(uploaded_file):
    """Extract text from uploaded file"""
    try:
        if uploaded_file.type == "application/pdf":
            try:
                import PyPDF2
                pdf_reader = PyPDF2.PdfReader(uploaded_file)
                text = ""
                for page in pdf_reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
                return text.strip()
            except:
                return "Error: Could not extract text from PDF"
        elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            try:
                from docx import Document
                doc = Document(uploaded_file)
                text = "\n".join([paragraph.text for paragraph in doc.paragraphs if paragraph.text])
                return text.strip()
            except:
                return "Error: Could not extract text from DOCX"
        else:
            return uploaded_file.read().decode("utf-8")
    except Exception as e:
        return f"Error extracting text: {str(e)}"

def is_resume(text: str) -> bool:
    """Check if text is a resume with enhanced validation"""
    if not text or not isinstance(text, str):
        return False
    
    text_lower = text.lower()
    resume_indicators = [
        'education', 'experience', 'skills', 'projects', 'work', 'university', 'college', 
        'degree', 'bachelor', 'master', 'cgpa', 'gpa', 'internship', 'job', 'certification',
        'achievements', 'contact', 'email', 'phone', 'linkedin', 'github', 'portfolio'
    ]
    
    # Check for section headers or patterns
    section_patterns = [
        r'\b(education|experience|skills|projects|work history|qualifications|certifications)\b[:\n-]',
        r'[\w\s]+university|[\w\s]+college',
        r'[\w\s]+degree\b',
        r'[\w\s]+internship\b',
        r'[\w\s]+job\b',
        r'[\w\s]+project[s]?\b[:\n-]'
    ]
    
    # Count indicators and patterns
    indicator_count = sum(1 for indicator in resume_indicators if indicator in text_lower)
    pattern_count = sum(1 for pattern in section_patterns if re.search(pattern, text_lower))
    
    # Require at least 2 indicators or 1 pattern with some text length
    return (indicator_count >= 2 or (pattern_count >= 1 and len(text_lower.split()) > 50))

# ==================== SKILL EXTRACTION ====================

def normalize_skill(skill: str) -> str:
    """Normalize skill name for better matching"""
    return skill.lower().strip().replace('-', '').replace('.', '').replace(' ', '')

def extract_skills_advanced(text: str, all_skills: List[str]) -> List[str]:
    """Advanced skill extraction with pattern matching"""
    text_normalized = normalize_skill(text)
    found_skills = []
    
    for skill in all_skills:
        skill_normalized = normalize_skill(skill)
        skill_words = skill.lower().split()
        
        # Check for exact match
        if skill_normalized in text_normalized:
            found_skills.append(skill)
            continue
        
        # Check for word-by-word match
        if all(word in text_normalized for word in skill_words):
            found_skills.append(skill)
            continue
        
        # Check for variations
        variations = [
            skill.lower().replace(' ', '-'),
            skill.lower().replace(' ', '_'),
            skill.lower().replace('.', ''),
            skill.upper(),
            skill.title()
        ]
        
        if any(var in text for var in variations):
            found_skills.append(skill)
    
    return list(set(found_skills))

def get_role_requirements():
    """Comprehensive role requirements"""
    return {
        "Software Engineer": {
            "core_skills": ["python", "java", "javascript", "c++", "c#", "git", "data structures", "algorithms", "oop", "design patterns"],
            "frameworks": ["react", "node.js", "django", "spring boot", "express", "flask", "angular", "vue.js"],
            "databases": ["sql", "mysql", "mongodb", "postgresql", "redis", "oracle"],
            "tools": ["docker", "aws", "linux", "postman", "jenkins", "kubernetes", "ci/cd"],
            "concepts": ["rest api", "microservices", "testing", "version control", "agile", "devops"]
        },
        "Data Scientist": {
            "core_skills": ["python", "r", "sql", "statistics", "machine learning", "data analysis", "mathematics"],
            "libraries": ["pandas", "numpy", "matplotlib", "seaborn", "scikit-learn", "tensorflow", "pytorch", "keras"],
            "tools": ["jupyter", "tableau", "power bi", "excel", "git", "spark", "hadoop"],
            "databases": ["sql", "mongodb", "hadoop", "hive"],
            "concepts": ["data mining", "deep learning", "nlp", "computer vision", "big data", "feature engineering", "model evaluation"]
        },
        "Frontend Developer": {
            "core_skills": ["html", "css", "javascript", "typescript", "react", "responsive design"],
            "frameworks": ["vue.js", "angular", "next.js", "bootstrap", "tailwind", "material ui", "sass"],
            "tools": ["git", "webpack", "npm", "yarn", "figma", "vscode"],
            "concepts": ["responsive design", "spa", "pwa", "accessibility", "performance optimization", "seo"],
            "testing": ["jest", "cypress", "testing library", "enzyme"]
        },
        "Full Stack Developer": {
            "frontend": ["html", "css", "javascript", "react", "vue.js", "angular"],
            "backend": ["node.js", "python", "java", "express", "django", "spring boot", "php"],
            "databases": ["mysql", "mongodb", "postgresql", "redis"],
            "tools": ["git", "docker", "aws", "heroku", "nginx"],
            "concepts": ["rest api", "authentication", "deployment", "testing", "websockets"]
        },
        "Mobile App Developer": {
            "platforms": ["android", "ios", "react native", "flutter", "kotlin", "swift"],
            "languages": ["java", "kotlin", "swift", "dart", "javascript", "objective-c"],
            "tools": ["android studio", "xcode", "firebase", "git", "gradle"],
            "concepts": ["ui/ux", "api integration", "local storage", "push notifications", "app store", "play store"],
            "testing": ["unit testing", "ui testing", "espresso", "xctest"]
        },
        "Artificial Intelligence": {
            "platforms": ["cloud", "tensorflow", "pytorch", "keras"],
            "languages": ["python", "r", "java", "c++", "julia"],
            "tools": ["tensorflow", "pytorch", "jupyter", "git", "docker", "cuda"],
            "concepts": ["machine learning", "deep learning", "nlp", "computer vision", "reinforcement learning", "neural networks"],
            "libraries": ["scikit-learn", "pandas", "numpy", "opencv", "transformers"]
        },
        "Blockchain": {
            "platforms": ["ethereum", "hyperledger", "binance smart chain", "solana", "polygon"],
            "languages": ["solidity", "javascript", "go", "rust", "python", "vyper"],
            "tools": ["truffle", "remix", "metamask", "ganache", "git", "hardhat"],
            "concepts": ["smart contracts", "defi", "consensus algorithms", "tokenization", "cryptography", "web3"],
            "testing": ["smart contract auditing", "unit testing", "integration testing"]
        },
        "VR & AR": {
            "platforms": ["oculus", "hololens", "unity", "unreal engine", "arkit", "arcore"],
            "languages": ["c#", "c++", "javascript", "python"],
            "tools": ["unity", "unreal engine", "blender", "git", "ar foundation", "vuforia"],
            "concepts": ["3d modeling", "spatial computing", "gesture recognition", "immersive design"],
            "testing": ["usability testing", "performance testing"]
        },
        "Big Data": {
            "platforms": ["hadoop", "spark", "kafka", "flink", "databricks"],
            "languages": ["python", "scala", "java", "sql", "r"],
            "tools": ["apache spark", "hadoop", "hive", "tableau", "jupyter", "git"],
            "concepts": ["data lakes", "etl pipelines", "real-time processing", "data warehousing"],
            "databases": ["mongodb", "cassandra", "hbase", "redshift"]
        },
        "Data Science": {
            "platforms": ["jupyter", "colab", "kaggle"],
            "languages": ["python", "r", "sql", "julia"],
            "tools": ["pandas", "scikit-learn", "tableau", "git", "tensorflow"],
            "concepts": ["statistical modeling", "machine learning", "data visualization", "feature engineering"],
            "libraries": ["numpy", "matplotlib", "seaborn", "plotly"]
        },
        "Cyber Security": {
            "platforms": ["kali linux", "parrot os", "cloud security"],
            "languages": ["python", "c", "javascript", "bash", "powershell"],
            "tools": ["wireshark", "metasploit", "burp suite", "nmap", "git"],
            "concepts": ["ethical hacking", "penetration testing", "cryptography", "network security", "vulnerability assessment"],
            "testing": ["vulnerability scanning", "penetration testing", "security audits"]
        }
    }

# ==================== MAIN ANALYSIS FUNCTION ====================

def analyze_resume_with_ml(text: str, target_role: str, extractor, gnn_model, graph_data, datasets) -> Dict:
    """Enhanced ML-powered resume analysis"""
    
    # Get all required skills for the role
    role_requirements = get_role_requirements()
    role_data = role_requirements.get(target_role, role_requirements["Software Engineer"])
    
    all_required_skills = []
    if isinstance(role_data, dict):
        for category_skills in role_data.values():
            if isinstance(category_skills, list):
                all_required_skills.extend(category_skills)
    
    all_required_skills = list(set(all_required_skills))
    
    # Advanced skill extraction
    found_skills = extract_skills_advanced(text, all_required_skills)
    
    # ML enhancement if available
    if ML_AVAILABLE and extractor:
        try:
            ml_results = extractor.extract_skills_advanced(text, target_role.lower())
            ml_skills = ml_results.get('extracted_skills', [])
            found_skills = list(set(found_skills + ml_skills))
        except:
            pass
    
    missing_skills = [skill for skill in all_required_skills if skill not in found_skills]
    
    # GNN predictions for missing skills
    if ML_AVAILABLE and gnn_model and graph_data:
        try:
            jobs_list, graph_dict = graph_data
            ontology = load_skills_from_dataset()
            graph_pyg = graph_dict_to_data(graph_dict, ontology)
            predicted_missing = predict_missing_skills(gnn_model, graph_pyg, found_skills, ontology)
            missing_skills = list(set(missing_skills + predicted_missing[:10]))
        except:
            pass
    
    # Calculate scores
    skill_match_score = (len(found_skills) / len(all_required_skills)) * 100 if all_required_skills else 0
    
    # Resume content analysis
    text_lower = text.lower()
    word_count = len(text.split())
    
    has_contact_info = any(keyword in text_lower for keyword in ['email', 'phone', '@', '.com', 'linkedin', 'github'])
    has_education = any(keyword in text_lower for keyword in ['education', 'university', 'college', 'degree', 'bachelor', 'master', 'cgpa', 'gpa', 'b.tech', 'm.tech', 'bca', 'mca'])
    has_projects = any(keyword in text_lower for keyword in ['project', 'github', 'repository', 'built', 'developed', 'created', 'implemented'])
    has_internship = any(keyword in text_lower for keyword in ['intern', 'internship', 'training', 'apprentice', 'work experience', 'experience'])
    has_achievements = any(keyword in text_lower for keyword in ['achievement', 'award', 'winner', 'certificate', 'certification', 'hackathon', 'competition'])
    has_leadership = any(keyword in text_lower for keyword in ['lead', 'president', 'head', 'coordinator', 'captain', 'volunteer', 'organizer'])
    
    projects_count = max(text_lower.count('project'), text_lower.count('github'))
    
    # Content quality scoring
    content_quality_score = 0
    if word_count > 200: content_quality_score += 15
    elif word_count > 150: content_quality_score += 10
    if has_contact_info: content_quality_score += 20
    if has_education: content_quality_score += 20
    if has_projects: content_quality_score += 25
    if has_internship: content_quality_score += 10
    if has_achievements: content_quality_score += 10
    
    # Experience scoring
    experience_score = 0
    if has_projects: experience_score += 40
    if projects_count >= 3: experience_score += 10
    if has_internship: experience_score += 30
    if has_achievements: experience_score += 15
    if has_leadership: experience_score += 5
    
    # Presentation score
    presentation_score = 70
    if word_count >= 200 and word_count <= 800: presentation_score += 15
    if has_contact_info and has_education: presentation_score += 10
    if text.count('\n') > 10: presentation_score += 5
    
    # Overall score calculation
    overall_score = (
        skill_match_score * 0.45 +
        content_quality_score * 0.25 +
        experience_score * 0.20 +
        presentation_score * 0.10
    )
    
    readiness_level = get_student_readiness_level(overall_score)
    career_suggestions = get_career_suggestions(found_skills, target_role)
    
    # Get recommendations
    course_recommendations = get_course_recommendations(missing_skills[:8], datasets.get('courses'))
    salary_info = get_salary_predictions(found_skills, target_role, datasets.get('skills'))
    job_matches = get_job_matches(found_skills, target_role, datasets.get('jobs', datasets.get('synthetic_jobs')))
    emerging_tech_analysis = analyze_emerging_tech(found_skills, datasets.get('emerging_tech'))
    
    # Placement forecast
    placement_forecast = None
    if ML_AVAILABLE and extractor and gnn_model and graph_data:
        try:
            jobs_list, graph_dict = graph_data
            jobs_df = pd.DataFrame(jobs_list)
            placement_forecast = forecast_placement(
                ",".join(found_skills),
                target_role,
                extractor,
                gnn_model,
                graph_dict,
                load_skills_from_dataset(),
                jobs_df,
                projects_count,
                None
            )
        except:
            pass
    
    if not placement_forecast:
        try:
            placement_forecast = predict_placement(
                ",".join(found_skills),
                target_role,
                "Bachelor",
                0,
                projects_count,
                pd.DataFrame(),
                pd.DataFrame()
            )
        except:
            pass
    
    return {
        'overall_score': round(overall_score, 1),
        'skill_match_score': round(skill_match_score, 1),
        'content_quality_score': content_quality_score,
        'experience_score': experience_score,
        'presentation_score': presentation_score,
        'found_skills': found_skills,
        'missing_skills': missing_skills[:15],
        'word_count': word_count,
        'has_contact_info': has_contact_info,
        'has_education': has_education,
        'has_projects': has_projects,
        'has_internship': has_internship,
        'has_achievements': has_achievements,
        'has_leadership': has_leadership,
        'readiness_level': readiness_level,
        'career_suggestions': career_suggestions,
        'strengths': identify_student_strengths(found_skills, content_quality_score, experience_score, has_projects, has_internship),
        'weaknesses': identify_student_weaknesses(missing_skills, content_quality_score, has_projects),
        'recommendations': generate_student_recommendations(missing_skills, overall_score, has_projects, has_internship, target_role),
        'course_recommendations': course_recommendations,
        'salary_info': salary_info,
        'job_matches': job_matches,
        'emerging_tech_analysis': emerging_tech_analysis,
        'placement_forecast': placement_forecast,
        'projects_count': projects_count
    }

# ==================== HELPER FUNCTIONS ====================

def get_course_recommendations(missing_skills: List[str], courses_df) -> List[Dict]:
    """Get course recommendations"""
    if courses_df is None or courses_df.empty:
        return []
    
    recommendations = []
    courses_df.columns = courses_df.columns.str.strip().str.lower()
    
    skill_column = 'skills' if 'skills' in courses_df.columns else 'skill' if 'skill' in courses_df.columns else None
    if not skill_column:
        return []
    
    for skill in missing_skills[:8]:
        try:
            matching_courses = courses_df[courses_df[skill_column].str.contains(skill, case=False, na=False)]
            if not matching_courses.empty:
                course = matching_courses.iloc[0]
                recommendations.append({
                    'skill': skill,
                    'course': course.get('course_name', 'N/A'),
                    'provider': course.get('provider', 'Online'),
                    'duration': f"{course.get('duration_weeks', 4)} weeks"
                })
        except:
            pass
    
    return recommendations

def get_salary_predictions(skills: List[str], role: str, skills_df) -> Dict:
    """Predict salary based on skills"""
    base_salaries = {
        "Software Engineer": {"entry": 400000, "mid": 800000, "senior": 1500000},
        "Data Scientist": {"entry": 600000, "mid": 1200000, "senior": 2000000},
        "Frontend Developer": {"entry": 350000, "mid": 700000, "senior": 1200000},
        "Full Stack Developer": {"entry": 450000, "mid": 900000, "senior": 1600000},
        "Mobile App Developer": {"entry": 400000, "mid": 800000, "senior": 1400000},
        "Artificial Intelligence": {"entry": 700000, "mid": 1500000, "senior": 2500000},
        "Blockchain": {"entry": 600000, "mid": 1300000, "senior": 2200000},
        "VR & AR": {"entry": 500000, "mid": 1000000, "senior": 1800000},
        "Big Data": {"entry": 550000, "mid": 1100000, "senior": 1900000},
        "Data Science": {"entry": 600000, "mid": 1200000, "senior": 2000000},
        "Cyber Security": {"entry": 500000, "mid": 1000000, "senior": 1700000}
    }
    
    base = base_salaries.get(role, base_salaries["Software Engineer"])
    skill_multiplier = 1 + (len(skills) * 0.015)
    
    return {
        "entry_level": int(base["entry"] * skill_multiplier),
        "mid_level": int(base["mid"] * skill_multiplier),
        "senior_level": int(base["senior"] * skill_multiplier),
        "currency": "INR"
    }

def get_job_matches(skills: List[str], role: str, jobs_df) -> List[Dict]:
    """Find matching jobs"""
    if jobs_df is None or jobs_df.empty:
        return []
    
    matches = []
    for _, job in jobs_df.iterrows():
        job_title = str(job.get('title', job.get('Title', ''))).lower()
        job_skills_str = str(job.get('skills', job.get('Skills', '')))
        job_skills = [s.strip().lower() for s in job_skills_str.split(',')]
        
        if role.lower() in job_title or any(word in job_title for word in role.lower().split()):
            match_count = len([s for s in skills if s.lower() in job_skills])
            if match_count > 0:
                match_percentage = (match_count / len(job_skills)) * 100 if job_skills else 0
                matches.append({
                    'title': job.get('title', job.get('Title', 'N/A')),
                    'description': job.get('description', job.get('Description', 'N/A'))[:100] + "...",
                    'match_percentage': round(match_percentage, 1),
                    'matching_skills': match_count
                })
    
    return sorted(matches, key=lambda x: x['match_percentage'], reverse=True)[:5]

def analyze_emerging_tech(skills: List[str], emerging_df) -> Dict:
    """Analyze emerging technologies"""
    if emerging_df is None or emerging_df.empty:
        return {"trending": [], "recommendations": []}
    
    emerging_skills = []
    for _, tech in emerging_df.iterrows():
        tech_name = str(tech.get('technology', tech.get('Technology', 'N/A')))
        if any(skill.lower() in tech_name.lower() for skill in skills):
            emerging_skills.append(tech_name)
    
    recommendations = []
    if len(emerging_skills) < 3 and len(emerging_df) > 0:
        sample_tech = emerging_df.sample(min(3, len(emerging_df)))
        recommendations = sample_tech['technology'].tolist() if 'technology' in sample_tech.columns else []
    
    return {
        "trending": emerging_skills,
        "recommendations": recommendations[:3]
    }

def get_student_readiness_level(score: float) -> Dict:
    """Determine readiness level"""
    if score >= 85:
        return {
            "level": "Job Ready",
            "description": "Excellent profile! Ready for entry-level positions",
            "color": "#10b981",
            "next_step": "Start applying and prepare for interviews"
        }
    elif score >= 70:
        return {
            "level": "Almost Ready",
            "description": "Good foundation! Few improvements needed",
            "color": "#3b82f6",
            "next_step": "Focus on missing skills and add projects"
        }
    elif score >= 55:
        return {
            "level": "Developing",
            "description": "On track! Strengthen technical skills",
            "color": "#f59e0b",
            "next_step": "Build projects and learn in-demand tech"
        }
    else:
        return {
            "level": "Beginning",
            "description": "Good start! Focus on core competencies",
            "color": "#ef4444",
            "next_step": "Learn fundamentals and start projects"
        }

def get_career_suggestions(found_skills: List[str], target_role: str) -> List[Dict]:
    """Suggest career paths"""
    suggestions = []
    
    if any(skill in found_skills for skill in ["python", "machine learning", "data"]):
        suggestions.append({
            "role": "Data Analyst",
            "match": "85%",
            "reason": "Strong Python and data skills"
        })
    
    if any(skill in found_skills for skill in ["react", "javascript", "html", "css"]):
        suggestions.append({
            "role": "Frontend Developer",
            "match": "90%",
            "reason": "Solid web development foundation"
        })
    
    if any(skill in found_skills for skill in ["java", "python", "algorithms"]):
        suggestions.append({
            "role": "Backend Developer",
            "match": "80%",
            "reason": "Strong programming fundamentals"
        })
    
    if not suggestions:
        suggestions.append({
            "role": f"Junior {target_role}",
            "match": "70%",
            "reason": "Entry-level position match"
        })
    
    return suggestions[:3]

def identify_student_strengths(found_skills: List[str], content_score: int, exp_score: int, has_projects: bool, has_internship: bool) -> List[str]:
    """Identify strengths"""
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
    
    if exp_score >= 60:
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
    
    # Create categories and scores
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
    
    if not categories:  # Fallback for simple list format
        categories = ['Technical Skills', 'Frameworks', 'Databases', 'Tools', 'Concepts']
        total_skills = len(found_skills) + len(missing_skills)
        base_score = (len(found_skills) / total_skills) * 100 if total_skills > 0 else 0
        scores = [base_score + np.random.randint(-15, 15) for _ in categories]
        scores = [max(0, min(100, score)) for score in scores]  # Clamp to 0-100
    
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
        <h2 style="color: #e2e8f0; margin-bottom: 25px; font-size: 2.2rem;">📤 Upload Your Resume</h2>
        <p style="color: #94a3b8; font-size: 1.2rem; margin-bottom: 15px;">Get instant feedback on your resume and discover your job readiness score</p>
        <div style="color: #64748b; font-size: 1rem;">
            <span style="color: #10b981;">✓</span> Technical skill assessment &nbsp;&nbsp;
            <span style="color: #10b981;">✓</span> Project portfolio review &nbsp;&nbsp;
            <span style="color: #10b981;">✓</span> Career roadmap generation
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # File upload and text input with enhanced styling
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
            <div class="card-title">✍ Or Paste Resume Text</div>
            <div class="card-content">
        """, unsafe_allow_html=True)
        
        resume_text = st.text_area(
            "Paste your resume content here",
            height=200,
            placeholder="Paste your complete resume content here for instant analysis...",
            label_visibility="collapsed"
        )
        
        st.markdown("</div></div>", unsafe_allow_html=True)
    
    # Target role selection with enhanced design
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
    
    # Analysis button with enhanced styling
    if st.button("🚀 Analyze My Resume & Get Job Readiness Score", type="primary", use_container_width=True):
                # Get text content
        text = ""
        if uploaded_file is not None:
            text = extract_text_from_file(uploaded_file)
        elif resume_text.strip():
            text = resume_text.strip()
        else:
            st.warning("⚠️ Please upload a file or paste resume text to proceed.")
            return

        # Validate resume content
        if not is_resume(text):
            st.error("""
            <div style="background: rgba(239, 68, 68, 0.2); padding: 20px; border-radius: 10px; border-left: 6px solid #ef4444; color: #e2e8f0;">
                <h3 style="color: #ef4444; margin-bottom: 10px;">⚠️ The content doesn't appear to be a resume.</h3>
                <p>Please provide a valid resume with sections like:</p>
                <ul style="color: #94a3b8; margin-left: 20px;">
                    <li>Education</li>
                    <li>Experience</li>
                    <li>Skills</li>
                    <li>Projects</li>
                </ul>
                <p>Include details such as university, job history, technical skills, or project descriptions.</p>
            </div>
            """, unsafe_allow_html=True)
            return

        # Start analysis with loading screen
        progress_bar = st.empty()
        start_time = time.time()
        with st.spinner():
            for progress in range(0, 101, 5):
                elapsed_time = time.time() - start_time
                progress_bar.markdown(create_enhanced_loading_screen(
                    step=["Scanning Resume", "Analyzing Skills", "Evaluating Experience", "Calculating Readiness", "Generating Roadmap"][min(progress // 25, 4)],
                    progress=progress,
                    elapsed_time=elapsed_time
                ), unsafe_allow_html=True)
                time.sleep(0.1)

        # Perform analysis
        results = analyze_resume_with_ml(text, target_role, extractor, gnn_model, graph_data, datasets)
        readiness_data = results['readiness_level']

        # Display results
        st.markdown('<div class="main-container">', unsafe_allow_html=True)

        # Job Readiness Score
        col1, col2 = st.columns([1, 1])
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

        # Detailed Analysis Section
        st.markdown("""
        <div class="analysis-card">
            <div class="card-title">📋 Detailed Analysis</div>
            <div class="card-content">
        """, unsafe_allow_html=True)

        # Score Breakdown
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Skill Match", f"{results['skill_match_score']}%")
        with col2:
            st.metric("Content Quality", f"{results['content_quality_score']}/100")
        with col3:
            st.metric("Experience", f"{results['experience_score']}/100")

        st.markdown("</div></div>", unsafe_allow_html=True)

        # Skills Overview
        st.markdown("""
        <div class="analysis-card">
            <div class="card-title">💡 Skills Overview</div>
            <div class="card-content">
        """, unsafe_allow_html=True)

        st.subheader("Found Skills")
        for skill in results['found_skills']:
            st.markdown(f'<span class="skill-tag present">{skill}</span>', unsafe_allow_html=True)

        st.subheader("Missing Skills")
        for skill in results['missing_skills']:
            st.markdown(f'<span class="skill-tag missing">{skill}</span>', unsafe_allow_html=True)

        st.markdown("</div></div>", unsafe_allow_html=True)

        # Radar Chart for Skill Coverage
        st.plotly_chart(create_skills_radar_chart(results['found_skills'], results['missing_skills'], target_role), use_container_width=True)

        # Recommendations Section
        st.markdown("""
        <div class="analysis-card">
            <div class="card-title">📚 Recommendations</div>
            <div class="card-content">
        """, unsafe_allow_html=True)

        for rec in results['recommendations']:
            st.markdown(f"""
            <div class="recommendation-item">
                <span>{rec['text']}</span>
                <span class="recommendation-priority {rec['priority']}-priority">{rec['priority'].title()}</span>
                <p style="color: #94a3b8; font-size: 0.9rem; margin: 5px 0;">Action: {rec['action']} | Timeline: {rec['timeline']}</p>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("</div></div>", unsafe_allow_html=True)

        # Course Recommendations
        if results['course_recommendations']:
            st.markdown("""
            <div class="analysis-card">
                <div class="card-title">🎓 Course Recommendations</div>
                <div class="card-content">
            """, unsafe_allow_html=True)

            for course in results['course_recommendations']:
                st.markdown(f"""
                <div class="recommendation-item">
                    <span>Learn {course['skill']} with {course['course']} ({course['provider']})</span>
                    <p style="color: #94a3b8; font-size: 0.9rem; margin: 5px 0;">Duration: {course['duration']}</p>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("</div></div>", unsafe_allow_html=True)

        # Salary Prediction
        st.markdown("""
        <div class="analysis-card">
            <div class="card-title">💰 Salary Prediction (INR)</div>
            <div class="card-content">
        """, unsafe_allow_html=True)

        salary = results['salary_info']
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Entry Level", f"₹{salary['entry_level']:,.0f}")
        with col2:
            st.metric("Mid Level", f"₹{salary['mid_level']:,.0f}")
        with col3:
            st.metric("Senior Level", f"₹{salary['senior_level']:,.0f}")

        st.markdown("</div></div>", unsafe_allow_html=True)

        # Job Matches
        if results['job_matches']:
            st.markdown("""
            <div class="analysis-card">
                <div class="card-title">🔍 Job Matches</div>
                <div class="card-content">
            """, unsafe_allow_html=True)

            for job in results['job_matches']:
                st.markdown(f"""
                <div class="recommendation-item">
                    <span>{job['title']} ({job['match_percentage']}% match)</span>
                    <p style="color: #94a3b8; font-size: 0.9rem; margin: 5px 0;">{job['description']}</p>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("</div></div>", unsafe_allow_html=True)

        # Emerging Tech Analysis
        st.markdown("""
        <div class="analysis-card">
            <div class="card-title">🌟 Emerging Tech Analysis</div>
            <div class="card-content">
        """, unsafe_allow_html=True)

        st.subheader("Trending Technologies")
        for tech in results['emerging_tech_analysis']['trending']:
            st.markdown(f'<span class="skill-tag present">{tech}</span>', unsafe_allow_html=True)

        st.subheader("Recommended to Explore")
        for tech in results['emerging_tech_analysis']['recommendations']:
            st.markdown(f'<span class="skill-tag missing">{tech}</span>', unsafe_allow_html=True)

        st.markdown("</div></div>", unsafe_allow_html=True)

        # Placement Forecast
        if results['placement_forecast']:
            st.markdown("""
            <div class="analysis-card">
                <div class="card-title">🎯 Placement Forecast</div>
                <div class="card-content">
                    <p style="color: #e2e8f0;">Based on your current profile, you have a {results['placement_forecast']['probability']:.0f}% chance of placement in the next {results['placement_forecast']['timeframe']} months as a {target_role}.</p>
                    <p style="color: #94a3b8;">Key Factors: {', '.join(results['placement_forecast']['key_factors'])}</p>
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="analysis-card">
                <div class="card-title">🎯 Placement Forecast</div>
                <div class="card-content">
                    <p style="color: #e2e8f0;">Placement forecast unavailable. Improve your profile with recommended skills and projects for better prediction.</p>
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()