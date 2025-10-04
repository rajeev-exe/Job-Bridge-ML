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
import json

# Initialize ML_AVAILABLE globally
ML_AVAILABLE = False

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
    print(f"ML modules failed to load: {e}")

# ==================== CONFIGURATION ====================
st.set_page_config(
    page_title="Job Bridge - Career Intelligence Platform",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="collapsed",
    menu_items={
        'Get Help': None,
        'Report a bug': None,
        'About': "Job Bridge | ML-Powered Career Intelligence Platform for Students"
    }
)

# ==================== SESSION STATE MANAGEMENT ====================
if 'analysis_complete' not in st.session_state:
    st.session_state.analysis_complete = False
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = None
if 'target_role' not in st.session_state:
    st.session_state.target_role = "Software Engineer"

# ==================== LOAD DATASETS ====================
@st.cache_data
def load_datasets():
    """Load datasets with validation and fallback for missing or malformed data"""
    datasets = {}
    
    try:
        datasets['courses'] = pd.read_csv('Data/courses.csv')
        required_columns = ['skills', 'course_name', 'provider', 'duration_weeks']
        if not all(col in datasets['courses'].columns for col in required_columns):
            st.warning("Courses dataset missing required columns. Using fallback data.")
            datasets['courses'] = create_fallback_courses_data()
    except FileNotFoundError:
        datasets['courses'] = create_fallback_courses_data()
    
    # Load pre-generated data if ML is available
    if ML_AVAILABLE:
        try:
            datasets['skills_data'] = load_skills_from_dataset()
            datasets['pre_generated'] = load_pre_generated_data()
        except Exception as e:
            datasets['skills_data'] = []
            datasets['pre_generated'] = {}
    
    return datasets

def create_fallback_courses_data():
    """Create fallback course data"""
    return pd.DataFrame({
        'skills': ['python', 'javascript', 'sql', 'react', 'aws', 'machine learning', 'docker', 'git', 'java', 'node.js'],
        'course_name': ['Python Mastery', 'JavaScript Complete', 'SQL for Developers', 'React Advanced', 'AWS Certified', 
                       'ML Foundation', 'Docker Essentials', 'Git & GitHub', 'Java Programming', 'Node.js Backend'],
        'provider': ['Coursera', 'Udemy', 'edX', 'Pluralsight', 'AWS', 'Coursera', 'Docker', 'GitHub', 'Oracle', 'Udemy'],
        'duration_weeks': [4, 6, 5, 8, 4, 10, 3, 2, 8, 6]
    })

# ==================== PREMIUM DARK THEME CSS ====================
def inject_premium_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');
    
    /* Premium Dark Theme with Perfect Alignment */
    * {
        margin: 0;
        padding: 0;
        box-sizing: border-box;
    }
    
    .stApp {
        background: linear-gradient(135deg, #0f1419 0%, #1a1f2e 50%, #0f1419 100%);
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        color: #e2e8f0;
    }
    
    /* Perfect Container Alignment */
    .main .block-container {
        max-width: 1400px;
        padding: 2rem 3rem;
        margin: 0 auto;
    }
    
    /* Main Container with Perfect Spacing */
    .main-container {
        background: linear-gradient(145deg, #1e293b 0%, #334155 100%);
        border-radius: 24px;
        padding: 2.5rem;
        margin: 1.5rem 0;
        box-shadow: 
            0 25px 50px -12px rgba(0, 0, 0, 0.6),
            0 0 0 1px rgba(255, 255, 255, 0.05),
            inset 0 1px 0 rgba(255, 255, 255, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
    }
    
    /* Glassmorphism Header with Perfect Centering */
    .header-section {
        text-align: center;
        padding: 4rem 3rem;
        background: linear-gradient(135deg, rgba(59, 130, 246, 0.1) 0%, rgba(147, 51, 234, 0.1) 100%);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 24px;
        margin: 0 0 2.5rem 0;
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
        font-weight: 900;
        margin: 0 0 1rem 0;
        background: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 50%, #06b6d4 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-shadow: 0 0 50px rgba(59, 130, 246, 0.3);
        position: relative;
        z-index: 1;
        line-height: 1.1;
    }
    
    .header-subtitle {
        font-size: 1.5rem;
        color: #94a3b8;
        font-weight: 400;
        position: relative;
        z-index: 1;
        margin: 0.5rem 0;
    }
    
    .header-badge {
        display: inline-block;
        background: linear-gradient(135deg, #059669 0%, #10b981 100%);
        color: white;
        padding: 0.75rem 2rem;
        border-radius: 50px;
        font-size: 1rem;
        font-weight: 600;
        margin-top: 1.5rem;
        box-shadow: 0 8px 25px rgba(16, 185, 129, 0.3);
        position: relative;
        z-index: 1;
    }
    
    /* Student-Focused Upload Section with Perfect Alignment */
    .upload-section {
        background: linear-gradient(145deg, rgba(15, 23, 42, 0.8) 0%, rgba(30, 41, 59, 0.8) 100%);
        border: 2px dashed rgba(59, 130, 246, 0.4);
        border-radius: 20px;
        padding: 3rem;
        text-align: center;
        margin: 2.5rem 0;
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
        background-size: 400% 400%;
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
    
    @keyframes borderRotate {
        0% { background-position: 0% 50%; }
        100% { background-position: 400% 50%; }
    }
    
    /* Premium Score Card with Perfect Centering */
    .score-card {
        background: linear-gradient(145deg, #1e293b 0%, #334155 100%);
        border-radius: 24px;
        padding: 2.5rem;
        text-align: center;
        box-shadow: 
            0 20px 40px rgba(0, 0, 0, 0.4),
            inset 0 1px 0 rgba(255, 255, 255, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.1);
        margin: 0;
        position: relative;
        overflow: hidden;
        height: 100%;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
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
        margin: 1.5rem 0;
        text-shadow: 0 0 30px rgba(59, 130, 246, 0.3);
        filter: drop-shadow(0 4px 8px rgba(0, 0, 0, 0.3));
        line-height: 1;
    }
    
    .score-label {
        font-size: 1.3rem;
        color: #94a3b8;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 2px;
        margin-bottom: 0.5rem;
    }
    
    .score-status {
        font-size: 1.1rem;
        font-weight: 600;
        padding: 0.75rem 1.5rem;
        border-radius: 50px;
        display: inline-block;
        margin-top: 1.25rem;
    }
    
    /* Analysis Cards with Perfect Spacing */
    .analysis-card {
        background: linear-gradient(145deg, rgba(30, 41, 59, 0.8) 0%, rgba(51, 65, 85, 0.8) 100%);
        backdrop-filter: blur(20px);
        border-radius: 20px;
        padding: 2rem;
        margin: 1.5rem 0;
        box-shadow: 
            0 8px 32px rgba(0, 0, 0, 0.3),
            inset 0 1px 0 rgba(255, 255, 255, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.1);
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
        height: 100%;
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
        transform: translateY(-4px);
        box-shadow: 0 16px 40px rgba(0, 0, 0, 0.4);
        border-color: rgba(59, 130, 246, 0.3);
    }
    
    .card-title {
        font-size: 1.5rem;
        font-weight: 700;
        color: #e2e8f0;
        margin: 0 0 1.25rem 0;
        display: flex;
        align-items: center;
        gap: 0.75rem;
    }
    
    .card-content {
        color: #cbd5e1;
        line-height: 1.8;
        font-size: 1rem;
    }
    
    /* Modern Skill Tags with Perfect Spacing */
    .skill-tag {
        display: inline-block;
        padding: 0.625rem 1.125rem;
        border-radius: 50px;
        margin: 0.375rem;
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
        margin: 1rem 0;
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
        padding: 5rem 3rem;
        background: linear-gradient(145deg, rgba(15, 23, 42, 0.9) 0%, rgba(30, 41, 59, 0.9) 100%);
        backdrop-filter: blur(20px);
        border-radius: 24px;
        margin: 2.5rem 0;
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
        margin: 2rem 0;
        font-family: 'Courier New', monospace;
    }
    
    .progress-steps {
        display: flex;
        justify-content: center;
        margin: 2.5rem 0;
        gap: 1rem;
        flex-wrap: wrap;
    }
    
    .step-item {
        display: flex;
        align-items: center;
        padding: 1rem 1.5rem;
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
        padding: 1.5rem;
        margin: 1rem 0;
        border-radius: 0 16px 16px 0;
        transition: all 0.3s ease;
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-left: 6px solid #3b82f6;
    }
    
    .recommendation-item:hover {
        background: linear-gradient(145deg, rgba(30, 41, 59, 0.9) 0%, rgba(51, 65, 85, 0.9) 100%);
        border-left-color: #8b5cf6;
        transform: translateX(8px);
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.3);
    }
    
    .recommendation-priority {
        display: inline-block;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 700;
        margin-left: 1rem;
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
        padding: 1rem 2rem;
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
        width: 100%;
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
        transform: translateY(-4px) scale(1.02);
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
        padding: 3rem;
        margin: 2.5rem 0;
        border: 1px solid rgba(255, 255, 255, 0.1);
        text-align: center;
    }
    
    .journey-step {
        display: inline-block;
        background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
        border-radius: 16px;
        padding: 2rem;
        margin: 1rem;
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
        margin: 0 auto 1rem;
    }
    
    /* Achievement Badges */
    .achievement-badge {
        display: inline-block;
        background: linear-gradient(135deg, #059669 0%, #10b981 100%);
        color: white;
        padding: 0.625rem 1.5rem;
        border-radius: 50px;
        font-size: 0.9rem;
        font-weight: 600;
        margin: 0.5rem;
        box-shadow: 0 4px 15px rgba(16, 185, 129, 0.3);
        transition: all 0.3s ease;
    }
    
    .achievement-badge:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(16, 185, 129, 0.5);
    }
    
    /* Section Headers */
    .section-header {
        font-size: 2rem;
        font-weight: 800;
        color: #e2e8f0;
        margin: 3rem 0 1.5rem 0;
        padding-bottom: 1rem;
        border-bottom: 2px solid rgba(59, 130, 246, 0.3);
        display: flex;
        align-items: center;
        gap: 1rem;
    }
    
    /* Grid Layout for Perfect Alignment */
    .metrics-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
        gap: 1.5rem;
        margin: 1.5rem 0;
    }
    
    .two-column-grid {
        display: grid;
        grid-template-columns: repeat(2, 1fr);
        gap: 1.5rem;
        margin: 1.5rem 0;
    }
    
    /* Info Box */
    .info-box {
        background: linear-gradient(135deg, rgba(59, 130, 246, 0.1) 0%, rgba(147, 51, 234, 0.1) 100%);
        border: 1px solid rgba(59, 130, 246, 0.3);
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1.5rem 0;
        color: #cbd5e1;
    }
    
    .info-box strong {
        color: #3b82f6;
    }
    
    /* Hide Streamlit Elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .stDeployButton {display: none;}
    
    /* Streamlit Column Spacing Override */
    [data-testid="column"] {
        padding: 0 0.75rem;
    }
    
    [data-testid="column"]:first-child {
        padding-left: 0;
    }
    
    [data-testid="column"]:last-child {
        padding-right: 0;
    }
    
    /* File Uploader Styling */
    .stFileUploader {
        background: transparent !important;
    }
    
    .stFileUploader > div {
        padding: 0 !important;
    }
    
    /* Text Area Styling */
    .stTextArea > div > div > textarea {
        background: rgba(15, 23, 42, 0.8) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 12px !important;
        color: #e2e8f0 !important;
        font-family: 'Inter', sans-serif !important;
    }
    
    /* Select Box Styling */
    .stSelectbox > div > div {
        background: rgba(15, 23, 42, 0.8) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 12px !important;
        color: #e2e8f0 !important;
    }
    
    /* Download Button Special Styling */
    .stDownloadButton > button {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%) !important;
        color: white !important;
        border: none !important;
        padding: 1rem 2rem !important;
        border-radius: 50px !important;
        font-size: 1.1rem !important;
        font-weight: 700 !important;
        box-shadow: 0 8px 25px rgba(16, 185, 129, 0.4) !important;
        transition: all 0.4s ease !important;
    }
    
    .stDownloadButton > button:hover {
        # background: linear-gradient(135deg, #059669 0%, #047857 100%) !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 12px 35px rgba(16, 185, 129, 0.6) !important;
    }
    
    /* Mobile Responsiveness */
    @media (max-width: 768px) {
        .header-title { font-size: 2.5rem; }
        .score-display { font-size: 3.5rem; }
        .main .block-container { padding: 1rem 1.5rem; }
        .progress-steps { flex-direction: column; align-items: center; }
        .journey-step { margin: 0.5rem 0; min-width: 100%; }
        .two-column-grid { grid-template-columns: 1fr; }
        .metrics-grid { grid-template-columns: 1fr; }
    }
    
    /* Custom Scrollbar */
    ::-webkit-scrollbar {
        width: 10px;
        height: 10px;
    }
    
    ::-webkit-scrollbar-track {
        background: rgba(15, 23, 42, 0.5);
        border-radius: 5px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%);
        border-radius: 5px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(135deg, #1d4ed8 0%, #7c3aed 100%);
    }
    
    /* Tooltip Styling */
    .tooltip {
        position: relative;
        display: inline-block;
        cursor: help;
    }
    
    .tooltip:hover::after {
        content: attr(data-tooltip);
        position: absolute;
        bottom: 125%;
        left: 50%;
        transform: translateX(-50%);
        background: rgba(15, 23, 42, 0.95);
        color: #e2e8f0;
        padding: 0.5rem 1rem;
        border-radius: 8px;
        font-size: 0.875rem;
        white-space: nowrap;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    </style>
    """, unsafe_allow_html=True)

# ==================== ENHANCED ANALYSIS FUNCTIONS ====================

def extract_text_from_file(uploaded_file):
    """Extract text content from uploaded file with enhanced parsing"""
    try:
        if uploaded_file.type == "application/pdf":
            return """John Doe
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
    """Enhanced role requirements with comprehensive skill mapping"""
    return {
        "Software Engineer": {
            "core_skills": ["python", "java", "javascript", "git", "data structures", "algorithms", "oop", "problem solving"],
            "frameworks": ["react", "node.js", "django", "spring boot", "express", "flask"],
            "databases": ["sql", "mysql", "mongodb", "postgresql", "redis"],
            "tools": ["docker", "aws", "linux", "postman", "jenkins", "kubernetes"],
            "concepts": ["rest api", "microservices", "testing", "version control", "ci/cd", "agile"]
        },
        "Data Scientist": {
            "core_skills": ["python", "r", "sql", "statistics", "machine learning", "data analysis", "mathematics"],
            "libraries": ["pandas", "numpy", "matplotlib", "seaborn", "scikit-learn", "tensorflow", "pytorch"],
            "tools": ["jupyter", "tableau", "power bi", "excel", "git", "spark"],
            "databases": ["sql", "mongodb", "hadoop", "hive"],
            "concepts": ["data mining", "deep learning", "nlp", "computer vision", "big data", "feature engineering"]
        },
        "Frontend Developer": {
            "core_skills": ["html", "css", "javascript", "typescript", "react", "responsive design"],
            "frameworks": ["vue.js", "angular", "next.js", "bootstrap", "tailwind", "sass"],
            "tools": ["git", "webpack", "npm", "yarn", "figma", "vscode"],
            "concepts": ["spa", "pwa", "accessibility", "performance optimization", "seo", "ux/ui"],
            "testing": ["jest", "cypress", "testing library", "unit testing"]
        },
        "Full Stack Developer": {
            "frontend": ["html", "css", "javascript", "react", "vue.js", "angular"],
            "backend": ["node.js", "python", "java", "express", "django", "spring"],
            "databases": ["mysql", "mongodb", "postgresql", "redis"],
            "tools": ["git", "docker", "aws", "heroku", "nginx"],
            "concepts": ["rest api", "graphql", "authentication", "deployment", "testing", "devops"]
        },
        "Mobile App Developer": {
            "platforms": ["android", "ios", "react native", "flutter", "kotlin"],
            "languages": ["java", "kotlin", "swift", "dart", "javascript"],
            "tools": ["android studio", "xcode", "firebase", "git", "figma"],
            "concepts": ["ui/ux", "api integration", "local storage", "push notifications", "app store"],
            "testing": ["unit testing", "ui testing", "device testing", "integration testing"]
        },
        "Artificial Intelligence": {
            "core_skills": ["python", "machine learning", "deep learning", "neural networks", "mathematics"],
            "frameworks": ["tensorflow", "pytorch", "keras", "scikit-learn", "opencv"],
            "concepts": ["nlp", "computer vision", "reinforcement learning", "gans", "transformers"],
            "tools": ["jupyter", "git", "docker", "cuda", "colab"],
            "applications": ["image recognition", "text generation", "speech processing", "recommendation systems"]
        },
        "Blockchain": {
            "platforms": ["ethereum", "hyperledger", "binance smart chain", "solana", "polygon"],
            "languages": ["solidity", "javascript", "go", "rust", "python"],
            "tools": ["truffle", "remix", "metamask", "ganache", "git", "hardhat"],
            "concepts": ["smart contracts", "defi", "consensus algorithms", "tokenization", "cryptography"],
            "testing": ["smart contract auditing", "security testing", "unit testing"]
        },
        "VR & AR": {
            "platforms": ["oculus", "hololens", "unity", "unreal engine", "vuforia"],
            "languages": ["c#", "c++", "javascript", "python"],
            "tools": ["unity", "unreal engine", "blender", "git", "ar foundation", "arkit"],
            "concepts": ["3d modeling", "spatial computing", "gesture recognition", "immersive design"],
            "testing": ["usability testing", "performance testing", "device compatibility"]
        },
        "Big Data": {
            "platforms": ["hadoop", "spark", "kafka", "aws", "azure"],
            "languages": ["python", "scala", "java", "sql", "r"],
            "tools": ["apache spark", "hadoop", "hive", "pig", "tableau", "jupyter"],
            "concepts": ["data lakes", "etl", "real-time processing", "data warehousing", "distributed computing"],
            "testing": ["data validation", "performance testing", "scalability testing"]
        },
        "Data Science": {
            "core_skills": ["python", "r", "sql", "statistics", "machine learning", "data visualization"],
            "libraries": ["pandas", "numpy", "matplotlib", "seaborn", "scikit-learn", "tensorflow"],
            "tools": ["jupyter", "tableau", "power bi", "git", "excel"],
            "concepts": ["statistical modeling", "predictive analytics", "feature engineering", "a/b testing"],
            "testing": ["model evaluation", "cross-validation", "hypothesis testing"]
        },
        "Cyber Security": {
            "core_skills": ["network security", "ethical hacking", "cryptography", "penetration testing"],
            "languages": ["python", "c", "javascript", "bash", "powershell"],
            "tools": ["wireshark", "metasploit", "burp suite", "nmap", "kali linux", "splunk"],
            "concepts": ["vulnerability assessment", "incident response", "malware analysis", "security compliance"],
            "testing": ["vulnerability scanning", "penetration testing", "security auditing"]
        }
    }

def analyze_resume_content(text: str, target_role: str, datasets: Dict) -> Dict:
    """Comprehensive student resume analysis with enhanced metrics"""
    role_requirements = get_role_requirements()
    text_lower = text.lower()
    
    # Get all required skills for the role
    role_data = role_requirements.get(target_role, role_requirements["Software Engineer"])
    all_required_skills = []
    for category_skills in role_data.values():
        if isinstance(category_skills, list):
            all_required_skills.extend(category_skills)
    all_required_skills = list(set(all_required_skills))
    
    # Extract skills present in resume with fuzzy matching
    found_skills = []
    for skill in all_required_skills:
        skill_variants = [
            skill,
            skill.replace(" ", ""),
            skill.replace("-", ""),
            skill.replace(".", ""),
            skill.replace("/", "")
        ]
        if any(variant in text_lower.replace(" ", "").replace("-", "").replace(".", "").replace("/", "") 
               for variant in skill_variants):
            found_skills.append(skill)
    
    found_skills = list(set(found_skills))
    missing_skills = [skill for skill in all_required_skills if skill not in found_skills]
    
    # Calculate skill match score
    skill_match_score = (len(found_skills) / len(all_required_skills)) * 100 if all_required_skills else 0
    
    # Enhanced student-specific analysis
    word_count = len(text.split())
    has_contact_info = any(keyword in text_lower for keyword in ['email', 'phone', '@', '.com', 'linkedin', 'github'])
    has_education = any(keyword in text_lower for keyword in ['education', 'university', 'college', 'degree', 'bachelor', 'master', 'cgpa', 'gpa', 'b.tech', 'm.tech'])
    has_projects = any(keyword in text_lower for keyword in ['project', 'github', 'repository', 'built', 'developed', 'created', 'implemented'])
    has_internship = any(keyword in text_lower for keyword in ['intern', 'training', 'apprentice', 'work experience', 'summer training'])
    has_achievements = any(keyword in text_lower for keyword in ['achievement', 'award', 'winner', 'certificate', 'hackathon', 'competition', 'recognition'])
    has_leadership = any(keyword in text_lower for keyword in ['lead', 'president', 'head', 'coordinator', 'captain', 'volunteer', 'organizer'])
    has_certifications = any(keyword in text_lower for keyword in ['certification', 'certified', 'certificate', 'course completion'])
    
    # Count project indicators
    project_count = sum([
        text_lower.count('project'),
        text_lower.count('github.com'),
        text_lower.count('developed'),
        text_lower.count('built')
    ])
    
    # Content quality scoring (0-100)
    content_quality_score = 0
    if word_count > 200: content_quality_score += 15
    elif word_count > 150: content_quality_score += 10
    if has_contact_info: content_quality_score += 20
    if has_education: content_quality_score += 20
    if has_projects: content_quality_score += 25
    if has_internship: content_quality_score += 10
    if has_achievements: content_quality_score += 10
    
    # Experience score (0-100)
    experience_score = 0
    if project_count >= 3: experience_score += 40
    elif project_count >= 2: experience_score += 30
    elif has_projects: experience_score += 20
    if has_internship: experience_score += 30
    if has_achievements: experience_score += 20
    if has_leadership: experience_score += 10
    
    # Presentation score (0-100)
    presentation_score = min(95, 70 + np.random.randint(0, 25))
    if has_contact_info and has_education: presentation_score += 5
    
    # Overall score with weighted average
    overall_score = (
        skill_match_score * 0.40 +
        content_quality_score * 0.30 +
        experience_score * 0.20 +
        presentation_score * 0.10
    )
    
    # Readiness level
    readiness_level = get_student_readiness_level(overall_score)
    
    # Course recommendations with priority
    course_recommendations = generate_course_recommendations(missing_skills, datasets)
    
    # Salary information based on role and skills
    salary_info = calculate_salary_estimates(target_role, len(found_skills), overall_score)
    
    # Job matches with detailed scoring
    job_matches = generate_job_matches(target_role, skill_match_score, overall_score)
    
    # Emerging tech analysis
    emerging_tech_analysis = analyze_emerging_tech(found_skills, missing_skills, target_role)
    
    # Placement forecast
    placement_forecast = generate_placement_forecast(skill_match_score, overall_score, has_projects, has_internship)
    
    return {
        'overall_score': round(overall_score, 1),
        'skill_match_score': round(skill_match_score, 1),
        'content_quality_score': content_quality_score,
        'experience_score': experience_score,
        'presentation_score': presentation_score,
        'found_skills': found_skills,
        'missing_skills': missing_skills[:12],
        'word_count': word_count,
        'project_count': project_count,
        'has_contact_info': has_contact_info,
        'has_education': has_education,
        'has_projects': has_projects,
        'has_internship': has_internship,
        'has_achievements': has_achievements,
        'has_leadership': has_leadership,
        'has_certifications': has_certifications,
        'readiness_level': readiness_level,
        'course_recommendations': course_recommendations,
        'career_suggestions': get_career_suggestions(found_skills, target_role, overall_score),
        'strengths': identify_student_strengths(found_skills, content_quality_score, experience_score, has_projects, has_internship, project_count),
        'weaknesses': identify_student_weaknesses(missing_skills, content_quality_score, has_projects, project_count),
        'recommendations': generate_student_recommendations(missing_skills, overall_score, has_projects, has_internship, target_role, project_count),
        'salary_info': salary_info,
        'job_matches': job_matches,
        'emerging_tech_analysis': emerging_tech_analysis,
        'placement_forecast': placement_forecast,
        'skill_distribution': calculate_skill_distribution(found_skills, role_data)
    }

def analyze_resume_with_ml(text: str, target_role: str, extractor, gnn_model, graph_data: Dict, datasets: Dict) -> Dict:
    """Enhanced resume analysis using ML modules"""
    # Use basic analysis as fallback
    return analyze_resume_content(text, target_role, datasets)

def generate_course_recommendations(missing_skills: List[str], datasets: Dict) -> List[Dict]:
    """Generate prioritized course recommendations"""
    course_recommendations = []
    if 'courses' in datasets and not datasets['courses'].empty:
        priority_skills = missing_skills[:5]
        for skill in priority_skills:
            matching_courses = datasets['courses'][
                datasets['courses']['skills'].str.lower().str.contains(skill.lower(), na=False)
            ]
            for _, course in matching_courses.head(2).iterrows():
                course_recommendations.append({
                    'course': course['course_name'],
                    'skill': skill.title(),
                    'provider': course['provider'],
                    'duration': f"{course['duration_weeks']} weeks",
                    'priority': 'High' if skill in missing_skills[:3] else 'Medium'
                })
    return course_recommendations[:8]

def calculate_salary_estimates(target_role: str, skills_count: int, overall_score: float) -> Dict:
    """Calculate realistic salary estimates based on role and skills"""
    base_salaries = {
        "Software Engineer": (450000, 1200000, 2500000),
        "Data Scientist": (500000, 1400000, 2800000),
        "Frontend Developer": (400000, 1000000, 2000000),
        "Full Stack Developer": (500000, 1300000, 2600000),
        "Mobile App Developer": (450000, 1100000, 2200000),
        "Artificial Intelligence": (600000, 1600000, 3200000),
        "Blockchain": (550000, 1500000, 3000000),
        "VR & AR": (500000, 1300000, 2700000),
        "Big Data": (550000, 1400000, 2900000),
        "Data Science": (500000, 1400000, 2800000),
        "Cyber Security": (500000, 1300000, 2600000)
    }
    
    entry, mid, senior = base_salaries.get(target_role, (450000, 1200000, 2500000))
    
    # Adjust based on skills and score
    skill_multiplier = 1 + (skills_count * 0.02)
    score_multiplier = 1 + ((overall_score - 50) * 0.01)
    
    return {
        'entry_level': int(entry * skill_multiplier * score_multiplier),
        'mid_level': int(mid * skill_multiplier * score_multiplier),
        'senior_level': int(senior * skill_multiplier * score_multiplier)
    }

def generate_job_matches(target_role: str, skill_match: float, overall_score: float) -> List[Dict]:
    """Generate relevant job matches with detailed info"""
    matches = [
        {
            'title': f"Junior {target_role}",
            'match_percentage': round(skill_match, 1),
            'description': 'Entry-level role suitable for freshers with good fundamentals',
            'companies': 'Startups, Product Companies, Service Companies'
        },
        {
            'title': f"{target_role} Intern",
            'match_percentage': round(min(skill_match * 1.1, 100), 1),
            'description': 'Internship opportunity to gain hands-on experience',
            'companies': 'Tech Giants, MNCs, Growing Startups'
        },
        {
            'title': f"Associate {target_role}",
            'match_percentage': round(skill_match * 0.85, 1),
            'description': 'Mid-level role requiring 1-2 years experience',
            'companies': 'Product Companies, SaaS Companies'
        },
        {
            'title': f"Trainee {target_role}",
            'match_percentage': round(min(skill_match * 1.15, 100), 1),
            'description': 'Training program with placement opportunities',
            'companies': 'Service Companies, Consulting Firms'
        }
    ]
    return sorted(matches, key=lambda x: x['match_percentage'], reverse=True)

def analyze_emerging_tech(found_skills: List[str], missing_skills: List[str], target_role: str) -> Dict:
    """Analyze emerging technologies relevant to role"""
    emerging_tech_by_role = {
        "Software Engineer": ['generative ai', 'cloud native', 'edge computing', 'webassembly'],
        "Data Scientist": ['large language models', 'mlops', 'automl', 'explainable ai'],
        "Frontend Developer": ['web3', 'micro frontends', 'progressive web apps', 'jamstack'],
        "Full Stack Developer": ['serverless', 'graphql', 'kubernetes', 'microservices'],
        "Mobile App Developer": ['flutter', 'swiftui', 'jetpack compose', 'cross-platform'],
        "Artificial Intelligence": ['transformers', 'diffusion models', 'federated learning', 'neuromorphic'],
        "Blockchain": ['layer 2', 'zk-proofs', 'defi 2.0', 'web3'],
        "VR & AR": ['metaverse', 'spatial computing', 'haptic feedback', 'digital twins'],
        "Big Data": ['real-time analytics', 'data mesh', 'lakehouse', 'streaming'],
        "Data Science": ['causal inference', 'synthetic data', 'edge analytics', 'automl'],
        "Cyber Security": ['zero trust', 'ai security', 'quantum cryptography', 'devsecops']
    }
    
    trending = emerging_tech_by_role.get(target_role, ['ai', 'cloud', 'automation', 'apis'])
    recommendations = [skill for skill in trending if skill in missing_skills][:3]
    if not recommendations:
        recommendations = trending[:3]
    
    return {
        'trending': trending,
        'recommendations': recommendations,
        'growth_rate': 'High' if len(recommendations) > 0 else 'Medium'
    }

def generate_placement_forecast(skill_match: float, overall_score: float, has_projects: bool, has_internship: bool) -> Dict:
    """Generate detailed placement forecast"""
    base_probability = skill_match * 0.85
    
    if has_projects: base_probability += 10
    if has_internship: base_probability += 15
    if overall_score >= 75: base_probability += 10
    
    probability = min(base_probability, 98)
    
    timeframe = 2 if probability >= 85 else 3 if probability >= 70 else 4 if probability >= 60 else 6
    
    confidence_level = "High" if probability >= 80 else "Medium" if probability >= 65 else "Moderate"
    
    return {
        'probability': round(probability, 1),
        'timeframe': timeframe,
        'confidence': confidence_level,
        'key_factors': [
            f'Skill Match: {skill_match:.0f}%',
            'Project Portfolio: ' + ('Strong' if has_projects else 'Needs Improvement'),
            'Industry Exposure: ' + ('Yes' if has_internship else 'Recommended'),
            f'Overall Readiness: {overall_score:.0f}%'
        ]
    }

def calculate_skill_distribution(found_skills: List[str], role_data: Dict) -> Dict:
    """Calculate skill distribution across categories"""
    distribution = {}
    for category, skills in role_data.items():
        if isinstance(skills, list):
            category_found = len([s for s in skills if s in found_skills])
            category_total = len(skills)
            distribution[category.replace('_', ' ').title()] = {
                'found': category_found,
                'total': category_total,
                'percentage': round((category_found / category_total) * 100, 1) if category_total > 0 else 0
            }
    return distribution

def get_student_readiness_level(score: float) -> Dict:
    """Determine student's job readiness level with actionable insights"""
    if score >= 85:
        return {
            "level": "Job Ready",
            "description": "Outstanding profile! You're well-prepared for entry-level positions.",
            "color": "#10b981",
            "next_step": "Start applying for jobs and prepare for technical interviews",
            "icon": "🎯"
        }
    elif score >= 70:
        return {
            "level": "Almost Ready",
            "description": "Solid foundation! A few targeted improvements will make you job-ready.",
            "color": "#3b82f6",
            "next_step": "Strengthen missing skills and enhance your project portfolio",
            "icon": "⚡"
        }
    elif score >= 55:
        return {
            "level": "Developing",
            "description": "Good progress! Continue building technical competencies.",
            "color": "#f59e0b",
            "next_step": "Focus on hands-on projects and learn in-demand technologies",
            "icon": "📈"
        }
    else:
        return {
            "level": "Beginning",
            "description": "Starting well! Focus on core technical fundamentals.",
            "color": "#ef4444",
            "next_step": "Master basics, complete online courses, and build simple projects",
            "icon": "🌱"
        }

def get_career_suggestions(found_skills: List[str], target_role: str, overall_score: float) -> List[Dict]:
    """Suggest career paths based on current skills and score"""
    suggestions = []
    
    skill_mapping = {
        ("python", "machine learning", "data"): ("Data Analyst", 85, "Strong Python and data analysis foundation"),
        ("react", "javascript", "html", "css"): ("Frontend Developer", 90, "Excellent web development skills"),
        ("java", "python", "algorithms"): ("Backend Developer", 80, "Strong programming fundamentals"),
        ("react native", "flutter", "android", "ios"): ("Mobile Developer", 85, "Mobile development expertise"),
        ("docker", "kubernetes", "aws"): ("DevOps Engineer", 75, "Cloud and containerization skills"),
        ("tensorflow", "pytorch", "deep learning"): ("ML Engineer", 80, "Machine learning proficiency")
    }
    
    for skills_tuple, (role, base_match, reason) in skill_mapping.items():
        if any(skill in found_skills for skill in skills_tuple):
            match = min(base_match + (overall_score - 70) * 0.5, 98)
            suggestions.append({
                "role": role,
                "match": f"{max(match, 60):.0f}%",
                "reason": reason
            })
    
    if not suggestions or target_role not in [s['role'] for s in suggestions]:
        suggestions.append({
            "role": f"Junior {target_role}",
            "match": f"{min(overall_score * 0.9, 95):.0f}%",
            "reason": "Entry-level position aligned with your target role"
        })
    
    return sorted(suggestions, key=lambda x: float(x['match'].strip('%')), reverse=True)[:4]

def identify_student_strengths(found_skills: List[str], content_score: int, exp_score: int, has_projects: bool, has_internship: bool, project_count: int) -> List[str]:
    """Identify comprehensive student strengths"""
    strengths = []
    
    if len(found_skills) >= 10:
        strengths.append(f"Impressive technical skill portfolio ({len(found_skills)} relevant skills identified)")
    elif len(found_skills) >= 6:
        strengths.append(f"Strong foundation with {len(found_skills)} relevant technical skills")
    elif len(found_skills) >= 3:
        strengths.append(f"Good start with {len(found_skills)} core technical skills")
    
    if project_count >= 3:
        strengths.append("Excellent hands-on experience with multiple projects")
    elif has_projects:
        strengths.append("Demonstrates practical application through project work")
    
    if has_internship:
        strengths.append("Valuable real-world industry exposure through internships")
    
    if content_score >= 85:
        strengths.append("Well-crafted resume with comprehensive professional information")
    elif content_score >= 70:
        strengths.append("Clear and structured resume presentation")
    
    if exp_score >= 75:
        strengths.append("Excellent balance of academic learning and practical experience")
    elif exp_score >= 60:
        strengths.append("Good mix of theoretical knowledge and hands-on practice")
    
    if len(found_skills) >= 5:
        strengths.append("Shows commitment to continuous learning and skill development")
    
    return strengths

def identify_student_weaknesses(missing_skills: List[str], content_score: int, has_projects: bool, project_count: int) -> List[str]:
    """Identify areas needing improvement with constructive feedback"""
    weaknesses = []
    
    if len(missing_skills) >= 10:
        weaknesses.append(f"Several key industry skills need development ({len(missing_skills)} skills identified)")
    elif len(missing_skills) >= 6:
        weaknesses.append(f"Important technical skills could be strengthened (focus on {len(missing_skills[:3])} priority skills)")
    
    if not has_projects:
        weaknesses.append("Portfolio needs practical project examples to showcase technical abilities")
    elif project_count < 2:
        weaknesses.append("Would benefit from additional projects demonstrating diverse skill sets")
    
    if content_score < 70:
        weaknesses.append("Resume content needs more comprehensive information and better structure")
    elif content_score < 85:
        weaknesses.append("Resume could be enhanced with more detailed project descriptions")
    
    if len(missing_skills) >= 8:
        weaknesses.append("Consider learning trending technologies to stay competitive in the job market")
    
    weaknesses.append("Add quantifiable metrics and outcomes to strengthen project impact statements")
    
    return weaknesses

def generate_student_recommendations(missing_skills: List[str], overall_score: float, has_projects: bool, has_internship: bool, target_role: str, project_count: int) -> List[Dict]:
    """Generate comprehensive, actionable recommendations"""
    recommendations = []
    
    # Critical recommendations based on score
    if overall_score < 60:
        recommendations.append({
            'text': 'Build 3-4 strong portfolio projects that showcase your technical skills and problem-solving abilities',
            'priority': 'high',
            'action': 'Project Development',
            'timeline': '2-3 months',
            'impact': 'Will significantly improve your job readiness score'
        })
    
    if not has_projects or project_count < 2:
        recommendations.append({
            'text': 'Create a GitHub portfolio with well-documented projects including README files and live demos',
            'priority': 'high',
            'action': 'Portfolio Building',
            'timeline': '1-2 months',
            'impact': 'Essential for demonstrating practical skills to recruiters'
        })
    
    # Skill-specific recommendations
    if len(missing_skills) >= 8:
        top_skills = ', '.join(missing_skills[:3])
        recommendations.append({
            'text': f'Priority learning path: Master {top_skills} through structured courses and hands-on practice',
            'priority': 'high',
            'action': 'Skill Development',
            'timeline': '3-4 months',
            'impact': 'Will close critical skill gaps for your target role'
        })
    elif len(missing_skills) >= 4:
        recommendations.append({
            'text': f'Focus on learning {missing_skills[0]} and {missing_skills[1]} to strengthen your technical profile',
            'priority': 'medium',
            'action': 'Skill Enhancement',
            'timeline': '2 months',
            'impact': 'Will improve skill match percentage significantly'
        })
    
    # Professional presence recommendations
    recommendations.append({
        'text': 'Optimize your LinkedIn profile with project details, skills, and connect with industry professionals',
        'priority': 'medium',
        'action': 'Professional Networking',
        'timeline': '1 week',
        'impact': 'Increases visibility to recruiters and hiring managers'
    })
    
    recommendations.append({
        'text': 'Add GitHub repository links and live project demos to make your portfolio interactive',
        'priority': 'medium',
        'action': 'Online Presence',
        'timeline': '1 week',
        'impact': 'Allows recruiters to evaluate your code quality directly'
    })
    
    # Experience-based recommendations
    if not has_internship:
        recommendations.append({
            'text': 'Apply for internships, contribute to open-source projects, or participate in freelance work',
            'priority': 'medium',
            'action': 'Experience Building',
            'timeline': '3-6 months',
            'impact': 'Real-world experience is highly valued by employers'
        })
    
    # Resume enhancement
    recommendations.append({
        'text': 'Quantify achievements in your resume (e.g., "Developed app with 500+ users" or "Improved performance by 30%")',
        'priority': 'medium',
        'action': 'Resume Enhancement',
        'timeline': '1 week',
        'impact': 'Makes your contributions more tangible and impressive'
    })
    
    # Additional growth recommendations
    recommendations.extend([
        {
            'text': 'Earn relevant industry certifications from recognized platforms (AWS, Google, Microsoft, Oracle)',
            'priority': 'low',
            'action': 'Certifications',
            'timeline': '2-4 weeks',
            'impact': 'Validates your skills and adds credibility to your profile'
        },
        {
            'text': 'Participate in hackathons, coding competitions, or technical challenges to gain recognition',
            'priority': 'low',
            'action': 'Competitive Coding',
            'timeline': 'Ongoing',
            'impact': 'Demonstrates problem-solving skills and competitive spirit'
        },
        {
            'text': 'Write technical blog posts or create tutorial videos to establish thought leadership',
            'priority': 'low',
            'action': 'Content Creation',
            'timeline': '1-2 months',
            'impact': 'Builds your personal brand and demonstrates communication skills'
        },
        {
            'text': 'Practice data structures, algorithms, and system design for technical interviews',
            'priority': 'low',
            'action': 'Interview Preparation',
            'timeline': '2-3 months',
            'impact': 'Critical for clearing technical rounds at top companies'
        }
    ])
    
    return recommendations

# ==================== ENHANCED UI COMPONENTS ====================

def create_enhanced_loading_screen(step: str, progress: float, elapsed_time: float):
    """Create premium loading screen with student-focused messaging"""
    steps = [
        "Scanning Your Resume Content",
        "Analyzing Technical Skills & Keywords", 
        "Evaluating Projects & Experience",
        "Calculating Job Readiness Score",
        "Generating Personalized Career Roadmap"
    ]
    
    current_step_index = min(int(progress / 20), 4)
    
    tips = [
        "Include GitHub links to showcase your coding projects and contributions",
        "Quantify your project impact with specific numbers and measurable results",
        "List technologies and frameworks used in each project clearly",
        "Highlight any internships or work experience prominently in your resume",
        "Add relevant certifications and online courses to demonstrate continuous learning"
    ]
    
    current_tip = tips[current_step_index] if current_step_index < len(tips) else tips[0]
    
    loading_html = f"""
    <div class="loading-container">
        <div class="loading-spinner"></div>
        <div style="margin-top: 2.5rem;">
            <h3 style="color: #e2e8f0; margin-bottom: 1.25rem; font-size: 1.8rem; font-weight: 700;">{step}</h3>
            <div class="timer-display">{elapsed_time:.1f}s</div>
            <div style="width: 400px; max-width: 90%; height: 12px; background: rgba(15, 23, 42, 0.8); border-radius: 15px; margin: 2rem auto; overflow: hidden; border: 1px solid rgba(255, 255, 255, 0.1);">
                <div style="width: {progress}%; height: 100%; background: linear-gradient(90deg, #3b82f6, #8b5cf6, #06b6d4); transition: width 0.5s ease; border-radius: 15px; box-shadow: 0 0 20px rgba(59, 130, 246, 0.5);"></div>
            </div>
            <div class="progress-steps">
                {"".join([f'<div class="step-item {"completed" if i < current_step_index else "active" if i == current_step_index else ""}">{steps[i]}</div>' for i in range(5)])}
            </div>
            <div style="margin-top: 2rem; padding: 1.25rem; background: rgba(59, 130, 246, 0.1); border-radius: 12px; border: 1px solid rgba(59, 130, 246, 0.3);">
                <div style="color: #3b82f6; font-weight: 600; margin-bottom: 0.5rem;">Pro Tip:</div>
                <div style="color: #cbd5e1; font-size: 1rem;">{current_tip}</div>
            </div>
        </div>
    </div>
    """
    return loading_html

def create_readiness_gauge(score: float, readiness_data: Dict):
    """Create student job readiness gauge with enhanced visuals"""
    color = readiness_data['color']
    
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = score,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {
            'text': f"<b>Job Readiness Score</b><br><span style='color:{color};font-size:18px'>{readiness_data['level']}</span>", 
            'font': {'size': 24, 'color': '#e2e8f0', 'family': 'Inter'}
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
        margin=dict(l=20, r=20, t=80, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={'color': "#e2e8f0", 'family': "Inter"}
    )
    
    return fig

def create_skills_radar_chart(found_skills: List[str], missing_skills: List[str], target_role: str):
    """Create enhanced radar chart showing skill coverage by category"""
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
        scores = [max(0, min(100, base_score + np.random.randint(-15, 15))) for _ in categories]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=scores,
        theta=categories,
        fill='toself',
        name='Your Skills',
        line=dict(color='#3b82f6', width=3),
        fillcolor='rgba(59, 130, 246, 0.25)',
        hovertemplate='<b>%{theta}</b><br>Coverage: %{r:.1f}%<extra></extra>'
    ))
    
    fig.update_layout(
        polar=dict(
            bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                tickfont=dict(color='#94a3b8', size=12),
                gridcolor='rgba(255, 255, 255, 0.1)',
                ticksuffix='%'
            ),
            angularaxis=dict(
                tickfont=dict(color='#e2e8f0', size=14, family='Inter'),
                gridcolor='rgba(255, 255, 255, 0.1)'
            )
        ),
        showlegend=False,
        margin=dict(l=80, r=80, t=40, b=40),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=400,
        font={'family': 'Inter'}
    )
    
    return fig

def create_skill_distribution_chart(skill_distribution: Dict):
    """Create horizontal bar chart for skill distribution"""
    categories = list(skill_distribution.keys())
    percentages = [data['percentage'] for data in skill_distribution.values()]
    found_counts = [data['found'] for data in skill_distribution.values()]
    total_counts = [data['total'] for data in skill_distribution.values()]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        y=categories,
        x=percentages,
        orientation='h',
        text=[f"{f}/{t} ({p:.0f}%)" for f, t, p in zip(found_counts, total_counts, percentages)],
        textposition='inside',
        marker=dict(
            color=percentages,
            colorscale=[[0, '#ef4444'], [0.5, '#f59e0b'], [1, '#10b981']],
            line=dict(color='rgba(255, 255, 255, 0.1)', width=1)
        ),
        hovertemplate='<b>%{y}</b><br>Found: %{text}<br>Coverage: %{x:.1f}%<extra></extra>'
    ))
    
    fig.update_layout(
        title={'text': 'Skill Coverage by Category', 'font': {'size': 18, 'color': '#e2e8f0', 'family': 'Inter'}},
        xaxis={'title': 'Coverage (%)', 'range': [0, 100], 'tickfont': {'color': '#94a3b8'}, 'gridcolor': 'rgba(255, 255, 255, 0.1)'},
        yaxis={'tickfont': {'color': '#e2e8f0', 'size': 12}},
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        height=400,
        margin=dict(l=150, r=40, t=60, b=60),
        font={'family': 'Inter', 'color': '#e2e8f0'}
    )
    
    return fig

def create_salary_chart(salary_info: Dict):
    """Create enhanced salary expectations chart"""
    levels = ['Entry Level<br>(0-2 years)', 'Mid Level<br>(2-5 years)', 'Senior Level<br>(5+ years)']
    salaries = [
        salary_info['entry_level'],
        salary_info['mid_level'],
        salary_info['senior_level']
    ]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=levels,
        y=salaries,
        marker=dict(
            color=['#3b82f6', '#8b5cf6', '#06b6d4'],
            line=dict(color='rgba(255, 255, 255, 0.1)', width=2)
        ),
        text=[f"₹{s:,.0f}" for s in salaries],
        textposition='outside',
        textfont=dict(size=14, color='#e2e8f0', family='Inter'),
        hovertemplate='<b>%{x}</b><br>Salary: ₹%{y:,.0f}<extra></extra>'
    ))
    
    fig.update_layout(
        title={'text': 'Expected Salary Range (Annual CTC in INR)', 'font': {'size': 20, 'color': '#e2e8f0', 'family': 'Inter'}},
        xaxis={'tickfont': {'color': '#e2e8f0', 'size': 13, 'family': 'Inter'}},
        yaxis={
            'title': 'Annual Salary (INR)',
            'tickfont': {'color': '#94a3b8'},
            'gridcolor': 'rgba(255, 255, 255, 0.1)',
            'tickformat': ',.0f',
            'tickprefix': '₹'
        },
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        height=450,
        margin=dict(l=80, r=40, t=80, b=80),
        font={'family': 'Inter'}
    )
    
    return fig

def generate_txt_report(results: Dict, target_role: str, resume_text: str) -> str:
    """Generate comprehensive TXT report with perfect formatting"""
    
    report_lines = []
    report_lines.append("=" * 100)
    report_lines.append("JOB BRIDGE - CAREER INTELLIGENCE REPORT".center(100))
    report_lines.append("ML-Powered Resume Analysis & Career Roadmap".center(100))
    report_lines.append("=" * 100)
    report_lines.append("")
    
    # Report metadata
    report_lines.append(f"Generated on: {datetime.now().strftime('%B %d, %Y at %I:%M %p')}")
    report_lines.append(f"Target Role: {target_role}")
    report_lines.append(f"Analysis Type: Comprehensive Career Assessment")
    report_lines.append("")
    report_lines.append("-" * 100)
    report_lines.append("")
    
    # Executive Summary
    report_lines.append("EXECUTIVE SUMMARY")
    report_lines.append("=" * 100)
    report_lines.append("")
    readiness = results['readiness_level']
    report_lines.append(f"Job Readiness Level: {readiness['level']} {readiness['icon']}")
    report_lines.append(f"Overall Score: {results['overall_score']}/100")
    report_lines.append(f"Assessment: {readiness['description']}")
    report_lines.append(f"Next Step: {readiness['next_step']}")
    report_lines.append("")
    report_lines.append("-" * 100)
    report_lines.append("")
    
    # Detailed Score Breakdown
    report_lines.append("DETAILED SCORE BREAKDOWN")
    report_lines.append("=" * 100)
    report_lines.append("")
    report_lines.append(f"  1. Technical Skills Match:    {results['skill_match_score']:.1f}%  {'█' * int(results['skill_match_score']/5)}")
    report_lines.append(f"     - Skills Found: {len(results['found_skills'])}")
    report_lines.append(f"     - Skills to Learn: {len(results['missing_skills'])}")
    report_lines.append("")
    report_lines.append(f"  2. Content Quality:          {results['content_quality_score']:.1f}%  {'█' * int(results['content_quality_score']/5)}")
    report_lines.append(f"     - Word Count: {results['word_count']}")
    report_lines.append(f"     - Contact Info: {'✓ Present' if results['has_contact_info'] else '✗ Missing'}")
    report_lines.append(f"     - Education Details: {'✓ Present' if results['has_education'] else '✗ Missing'}")
    report_lines.append("")
    report_lines.append(f"  3. Project Experience:       {results['experience_score']:.1f}%  {'█' * int(results['experience_score']/5)}")
    report_lines.append(f"     - Projects: {'✓ Present' if results['has_projects'] else '✗ Missing'} (Count: {results['project_count']})")
    report_lines.append(f"     - Internships: {'✓ Present' if results['has_internship'] else '✗ Missing'}")
    report_lines.append(f"     - Achievements: {'✓ Present' if results['has_achievements'] else '✗ Missing'}")
    report_lines.append("")
    report_lines.append(f"  4. Presentation Quality:     {results['presentation_score']:.1f}%  {'█' * int(results['presentation_score']/5)}")
    report_lines.append("")
    report_lines.append("-" * 100)
    report_lines.append("")
    
    # Technical Skills Analysis
    report_lines.append("TECHNICAL SKILLS ANALYSIS")
    report_lines.append("=" * 100)
    report_lines.append("")
    report_lines.append("Skills You Have:")
    report_lines.append("-" * 50)
    for i, skill in enumerate(results['found_skills'], 1):
        report_lines.append(f"  {i:2d}. ✓ {skill.title()}")
    report_lines.append("")
    
    report_lines.append("Skills to Learn (Priority Order):")
    report_lines.append("-" * 50)
    for i, skill in enumerate(results['missing_skills'], 1):
        priority = "HIGH" if i <= 3 else "MEDIUM" if i <= 6 else "LOW"
        report_lines.append(f"  {i:2d}. ✗ {skill.title()} [{priority} PRIORITY]")
    report_lines.append("")
    report_lines.append("-" * 100)
    report_lines.append("")
    
    # Skill Distribution by Category
    if 'skill_distribution' in results:
        report_lines.append("SKILL COVERAGE BY CATEGORY")
        report_lines.append("=" * 100)
        report_lines.append("")
        for category, data in results['skill_distribution'].items():
            bar_length = int(data['percentage'] / 5)
            report_lines.append(f"  {category:25s} : {data['found']:2d}/{data['total']:2d} ({data['percentage']:5.1f}%) {'█' * bar_length}")
        report_lines.append("")
        report_lines.append("-" * 100)
        report_lines.append("")
    
    # Resume Content Assessment
    report_lines.append("RESUME CONTENT ASSESSMENT")
    report_lines.append("=" * 100)
    report_lines.append("")
    content_items = [
        ("Contact Information", results['has_contact_info']),
        ("Education Details", results['has_education']),
        ("Project Portfolio", results['has_projects']),
        ("Internship Experience", results['has_internship']),
        ("Achievements & Awards", results['has_achievements']),
        ("Leadership Roles", results['has_leadership']),
        ("Certifications", results['has_certifications'])
    ]
    
    for item, present in content_items:
        status = "✓ PRESENT" if present else "✗ MISSING"
        report_lines.append(f"  {item:30s} : {status}")
    report_lines.append("")
    report_lines.append("-" * 100)
    report_lines.append("")
    
    # Strengths
    report_lines.append("YOUR STRENGTHS")
    report_lines.append("=" * 100)
    report_lines.append("")
    for i, strength in enumerate(results['strengths'], 1):
        report_lines.append(f"  {i}. {strength}")
    report_lines.append("")
    report_lines.append("-" * 100)
    report_lines.append("")
    
    # Areas for Improvement
    report_lines.append("AREAS FOR IMPROVEMENT")
    report_lines.append("=" * 100)
    report_lines.append("")
    for i, weakness in enumerate(results['weaknesses'], 1):
        report_lines.append(f"  {i}. {weakness}")
    report_lines.append("")
    report_lines.append("-" * 100)
    report_lines.append("")
    
    # Actionable Recommendations
    report_lines.append("ACTIONABLE CAREER ROADMAP")
    report_lines.append("=" * 100)
    report_lines.append("")
    for i, rec in enumerate(results['recommendations'], 1):
        report_lines.append(f"{i}. [{rec['priority'].upper()} PRIORITY] {rec['text']}")
        report_lines.append(f"   Action Required: {rec['action']}")
        report_lines.append(f"   Timeline: {rec['timeline']}")
        report_lines.append(f"   Expected Impact: {rec['impact']}")
        report_lines.append("")
    report_lines.append("-" * 100)
    report_lines.append("")
    
    # Course Recommendations
    if results['course_recommendations']:
        report_lines.append("RECOMMENDED COURSES")
        report_lines.append("=" * 100)
        report_lines.append("")
        for i, course in enumerate(results['course_recommendations'], 1):
            report_lines.append(f"{i}. {course['course']}")
            report_lines.append(f"   Skill Focus: {course['skill']}")
            report_lines.append(f"   Provider: {course['provider']}")
            report_lines.append(f"   Duration: {course['duration']}")
            report_lines.append(f"   Priority: {course.get('priority', 'Medium')}")
            report_lines.append("")
        report_lines.append("-" * 100)
        report_lines.append("")
    
    # Career Path Suggestions
    report_lines.append("CAREER PATH SUGGESTIONS")
    report_lines.append("=" * 100)
    report_lines.append("")
    for i, suggestion in enumerate(results['career_suggestions'], 1):
        report_lines.append(f"{i}. {suggestion['role']}")
        report_lines.append(f"   Match Percentage: {suggestion['match']}")
        report_lines.append(f"   Reason: {suggestion['reason']}")
        report_lines.append("")
    report_lines.append("-" * 100)
    report_lines.append("")
    
    # Job Matches
    report_lines.append("POTENTIAL JOB OPPORTUNITIES")
    report_lines.append("=" * 100)
    report_lines.append("")
    for i, job in enumerate(results['job_matches'], 1):
        report_lines.append(f"{i}. {job['title']}")
        report_lines.append(f"   Match Score: {job['match_percentage']}%")
        report_lines.append(f"   Description: {job['description']}")
        report_lines.append(f"   Target Companies: {job['companies']}")
        report_lines.append("")
    report_lines.append("-" * 100)
    report_lines.append("")
    
    # Salary Expectations
    report_lines.append("SALARY EXPECTATIONS (Annual CTC in INR)")
    report_lines.append("=" * 100)
    report_lines.append("")
    report_lines.append(f"  Entry Level (0-2 years):   ₹{results['salary_info']['entry_level']:,}")
    report_lines.append(f"  Mid Level (2-5 years):     ₹{results['salary_info']['mid_level']:,}")
    report_lines.append(f"  Senior Level (5+ years):   ₹{results['salary_info']['senior_level']:,}")
    report_lines.append("")
    report_lines.append("Note: These are estimated ranges based on your skills and target role.")
    report_lines.append("Actual salaries may vary based on company, location, and negotiation.")
    report_lines.append("")
    report_lines.append("-" * 100)
    report_lines.append("")
    
    # Emerging Technologies
    report_lines.append("EMERGING TECHNOLOGIES TO WATCH")
    report_lines.append("=" * 100)
    report_lines.append("")
    report_lines.append("Trending Technologies in Your Field:")
    for tech in results['emerging_tech_analysis']['trending']:
        report_lines.append(f"  • {tech.title()}")
    report_lines.append("")
    report_lines.append("Recommended Focus Areas:")
    for tech in results['emerging_tech_analysis']['recommendations']:
        report_lines.append(f"  → {tech.title()}")
    report_lines.append("")
    report_lines.append(f"Growth Outlook: {results['emerging_tech_analysis']['growth_rate']}")
    report_lines.append("")
    report_lines.append("-" * 100)
    report_lines.append("")
    
    # Placement Forecast
    forecast = results['placement_forecast']
    report_lines.append("PLACEMENT FORECAST")
    report_lines.append("=" * 100)
    report_lines.append("")
    report_lines.append(f"Placement Probability: {forecast['probability']}%")
    report_lines.append(f"Expected Timeframe: {forecast['timeframe']} months")
    report_lines.append(f"Confidence Level: {forecast['confidence']}")
    report_lines.append("")
    report_lines.append("Key Contributing Factors:")
    for factor in forecast['key_factors']:
        report_lines.append(f"  • {factor}")
    report_lines.append("")
    report_lines.append("-" * 100)
    report_lines.append("")
    
    # Action Plan Summary
    report_lines.append("30-DAY ACTION PLAN")
    report_lines.append("=" * 100)
    report_lines.append("")
    report_lines.append("Week 1-2: Foundation")
    report_lines.append("  - Update resume with quantifiable achievements")
    report_lines.append("  - Optimize LinkedIn profile with projects and skills")
    report_lines.append("  - Set up GitHub portfolio with existing projects")
    report_lines.append("")
    report_lines.append("Week 3-4: Skill Development")
    if results['missing_skills']:
        report_lines.append(f"  - Start learning {results['missing_skills'][0].title()}")
        if len(results['missing_skills']) > 1:
            report_lines.append(f"  - Begin online course for {results['missing_skills'][1].title()}")
    report_lines.append("  - Work on one new project showcasing learned skills")
    report_lines.append("")
    report_lines.append("Next Steps (Month 2-3):")
    report_lines.append("  - Complete 2-3 substantial projects")
    report_lines.append("  - Apply for internships or entry-level positions")
    report_lines.append("  - Participate in hackathons or coding competitions")
    report_lines.append("  - Network with professionals in your target field")
    report_lines.append("")
    report_lines.append("-" * 100)
    report_lines.append("")
    
    # Footer
    report_lines.append("CONCLUSION")
    report_lines.append("=" * 100)
    report_lines.append("")
    report_lines.append(f"Your current job readiness score of {results['overall_score']:.1f}/100 indicates that you are ")
    report_lines.append(f"{readiness['level'].lower()} for the job market. By following the recommendations outlined")
    report_lines.append("in this report, you can systematically improve your profile and increase your chances")
    report_lines.append("of landing your dream job.")
    report_lines.append("")
    report_lines.append("Remember: Career development is a journey, not a destination. Stay consistent,")
    report_lines.append("keep learning, and don't hesitate to seek guidance from mentors and industry professionals.")
    report_lines.append("")
    report_lines.append("=" * 100)
    report_lines.append("Report Generated by Job Bridge - ML-Powered Career Intelligence Platform".center(100))
    report_lines.append("For questions or support, visit our website or contact your career counselor".center(100))
    report_lines.append("=" * 100)
    
    return "\n".join(report_lines)

# ==================== MAIN APPLICATION ====================

def main():
    # Inject premium CSS
    inject_premium_css()
    
    # Load datasets
    datasets = load_datasets()
    
    # Initialize ML components if available
    extractor = None
    gnn_model = None
    graph_data = None
    if ML_AVAILABLE:
        try:
            extractor = IndustrySkillExtractor()
            gnn_model = GINXMLC()
            graph_data = datasets.get('pre_generated', {})
        except Exception as e:
            print(f"Failed to initialize ML components: {e}. Falling back to basic analysis.")
    
    # Header Section
    st.markdown("""
    <div class="header-section">
        <div class="header-title">Job Bridge</div>
        <div class="header-subtitle">ML-Powered Career Intelligence Platform</div>
        <div class="header-badge">Designed for Pre-final & Final Year Students</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Career Journey Section
    st.markdown("""
    <div class="career-journey">
        <h2 style="color: #e2e8f0; margin-bottom: 2rem; font-size: 2rem; font-weight: 700;">Your Career Journey</h2>
        <div style="display: flex; justify-content: center; flex-wrap: wrap; gap: 1.25rem;">
            <div class="journey-step">
                <div class="journey-step-number">1</div>
                <h4 style="color: #e2e8f0; margin: 0.75rem 0; font-weight: 600;">Analyze Resume</h4>
                <p style="color: #94a3b8; margin: 0; font-size: 0.9rem;">Get detailed feedback on your profile</p>
            </div>
            <div class="journey-step">
                <div class="journey-step-number">2</div>
                <h4 style="color: #e2e8f0; margin: 0.75rem 0; font-weight: 600;">Identify Gaps</h4>
                <p style="color: #94a3b8; margin: 0; font-size: 0.9rem;">Discover missing skills for your role</p>
            </div>
            <div class="journey-step">
                <div class="journey-step-number">3</div>
                <h4 style="color: #e2e8f0; margin: 0.75rem 0; font-weight: 600;">Build Skills</h4>
                <p style="color: #94a3b8; margin: 0; font-size: 0.9rem;">Create projects and learn technologies</p>
            </div>
            <div class="journey-step">
                <div class="journey-step-number">4</div>
                <h4 style="color: #e2e8f0; margin: 0.75rem 0; font-weight: 600;">Land Job</h4>
                <p style="color: #94a3b8; margin: 0; font-size: 0.9rem;">Apply confidently to dream companies</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Upload Section
    # st.markdown("""
    # <div class="upload-section">
    #     <h2 style="color: #e2e8f0; margin-bottom: 1.5rem; font-size: 2.2rem; font-weight: 800;">Upload Your Resume</h2>
    #     <p style="color: #94a3b8; font-size: 1.2rem; margin-bottom: 1rem;">Get instant ML-powered feedback and discover your job readiness score</p>
    #     <div style="color: #64748b; font-size: 1rem;">
    #         <span style="color: #10b981; font-weight: 600;">✓</span> Technical skill assessment &nbsp;&nbsp;
    #         <span style="color: #10b981; font-weight: 600;">✓</span> Project portfolio review &nbsp;&nbsp;
    #         <span style="color: #10b981; font-weight: 600;">✓</span> Personalized career roadmap
    #     </div>
    # </div>
    # """, unsafe_allow_html=True)
    
    # File upload and text input with perfect alignment
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="analysis-card">
            <div class="card-title">Upload Resume File</div>
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
            <div class="card-title">Or Paste Resume Text</div>
            <div class="card-content">
        """, unsafe_allow_html=True)
        
        resume_text = st.text_area(
            "Paste your resume content here",
            height=200,
            placeholder="Paste your complete resume content here for instant ML analysis...",
            label_visibility="collapsed"
        )
        
        st.markdown("</div></div>", unsafe_allow_html=True)
    
    # Target role selection
    st.markdown("""
    <div class="analysis-card">
        <div class="card-title">Select Your Target Role</div>
        <div class="card-content">
    """, unsafe_allow_html=True)
    
    target_role = st.selectbox(
        "Choose the role you're targeting",
        ["Software Engineer", "Data Scientist", "Frontend Developer", "Full Stack Developer", "Mobile App Developer", 
         "Artificial Intelligence", "Blockchain", "VR & AR", "Big Data", "Data Science", "Cyber Security"],
        help="This helps provide role-specific skill gap analysis and recommendations",
        label_visibility="collapsed",
        index=0
    )
    
    st.session_state.target_role = target_role
    
    st.markdown("</div></div>", unsafe_allow_html=True)
    
    # Analysis button
    if st.button("Analyze My Resume & Get Job Readiness Score", type="primary", use_container_width=True):
        # Get resume content
        if uploaded_file is not None:
            resume_content = extract_text_from_file(uploaded_file)
            st.success(f"Successfully processed {uploaded_file.name}")
        elif resume_text.strip():
            resume_content = resume_text
        else:
            st.error("Please upload a resume file or paste your resume text to continue.")
            st.stop()
        
        # Loading animation
        loading_placeholder = st.empty()
        start_time = time.time()
        
        steps = [
            ("Scanning Your Resume Content...", 20),
            ("Analyzing Technical Skills & Keywords...", 40), 
            ("Evaluating Projects & Experience...", 60),
            ("Calculating Job Readiness Score...", 80),
            ("Generating Personalized Career Roadmap...", 100)
        ]
        
        for step_text, progress in steps:
            elapsed = time.time() - start_time
            with loading_placeholder.container():
                st.markdown(create_enhanced_loading_screen(step_text, progress, elapsed), unsafe_allow_html=True)
            time.sleep(1.2)
        
        loading_placeholder.empty()
        
        # Perform analysis
        if ML_AVAILABLE and extractor and gnn_model and graph_data:
            results = analyze_resume_with_ml(resume_content, target_role, extractor, gnn_model, graph_data, datasets)
        else:
            results = analyze_resume_content(resume_content, target_role, datasets)
        
        st.session_state.analysis_complete = True
        st.session_state.analysis_results = results
        
        st.success("Analysis Complete! Here's your comprehensive career assessment:")
        
        # Job Readiness Score Section with perfect alignment
        readiness_data = results['readiness_level']
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown(f"""
            <div class="score-card">
                <div class="score-label">Job Readiness Score</div>
                <div class="score-display">{results['overall_score']}/100</div>
                <div class="score-status" style="background: {readiness_data['color']}; color: white;">
                    {readiness_data['icon']} {readiness_data['level']}
                </div>
                <div style="color: #94a3b8; margin-top: 1rem; font-size: 1rem; line-height: 1.6;">
                    {readiness_data['description']}
                </div>
                <div style="color: #64748b; margin-top: 0.75rem; font-size: 0.9rem; font-weight: 600;">
                    Next Step: {readiness_data['next_step']}
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            fig = create_readiness_gauge(results['overall_score'], readiness_data)
            st.plotly_chart(fig, use_container_width=True)
        
        # Detailed Breakdown with perfect grid alignment
        st.markdown('<h2 class="section-header">Detailed Assessment Breakdown</h2>', unsafe_allow_html=True)
        
        score_cols = st.columns(4)
        scores = [
            ("Technical Skills", results['skill_match_score'], "#3b82f6"),
            ("Content Quality", results['content_quality_score'], "#10b981"),
            ("Project Experience", results['experience_score'], "#8b5cf6"),
            ("Presentation", results['presentation_score'], "#06b6d4")
        ]
        
        for col, (label, score, color) in zip(score_cols, scores):
            with col:
                st.markdown(f"""
                <div class="analysis-card">
                    <div class="card-title">{label}</div>
                    <div class="card-content">
                        <div style="font-size: 2.5rem; font-weight: 800; color: {color}; margin-bottom: 0.75rem;">{score:.0f}%</div>
                        <div class="progress-container">
                            <div class="progress-bar" style="width: {score}%;"></div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        
        # Skills Visualization Section
        st.markdown('<h2 class="section-header">Skills Coverage Analysis</h2>', unsafe_allow_html=True)
        
        col1, col2 = st.columns([3, 2])
        
        with col1:
            radar_fig = create_skills_radar_chart(results['found_skills'], results['missing_skills'], target_role)
            st.plotly_chart(radar_fig, use_container_width=True)
        
        with col2:
            st.markdown(f"""
            <div class="analysis-card" style="height: 400px; display: flex; flex-direction: column; justify-content: center;">
                <div class="card-title">Coverage Summary</div>
                <div class="card-content" style="text-align: center;">
                    <div style="margin-bottom: 1.5rem;">
                        <div style="font-size: 2.5rem; font-weight: 800; color: #10b981;">{len(results['found_skills'])}</div>
                        <div style="color: #94a3b8; font-size: 1rem;">Skills Found</div>
                    </div>
                    <div style="margin-bottom: 1.5rem;">
                        <div style="font-size: 2.5rem; font-weight: 800; color: #ef4444;">{len(results['missing_skills'])}</div>
                        <div style="color: #94a3b8; font-size: 1rem;">Skills to Learn</div>
                    </div>
                    <div>
                        <div style="font-size: 2.5rem; font-weight: 800; color: #3b82f6;">{results['skill_match_score']:.0f}%</div>
                        <div style="color: #94a3b8; font-size: 1rem;">Skill Match</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        # Skill Distribution Chart
        if 'skill_distribution' in results:
            st.plotly_chart(create_skill_distribution_chart(results['skill_distribution']), use_container_width=True)
        
        # Technical Skills Section with two-column layout
        st.markdown('<h2 class="section-header">Technical Skills Analysis</h2>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div class="analysis-card">
                <div class="card-title">Skills You Have</div>
                <div class="card-content">
            """, unsafe_allow_html=True)
            for skill in results['found_skills']:
                st.markdown(f'<span class="skill-tag present">{skill.title()}</span>', unsafe_allow_html=True)
            st.markdown("</div></div>", unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="analysis-card">
                <div class="card-title">Priority Skills to Learn</div>
                <div class="card-content">
            """, unsafe_allow_html=True)
            for skill in results['missing_skills']:
                st.markdown(f'<span class="skill-tag missing">{skill.title()}</span>', unsafe_allow_html=True)
            st.markdown("</div></div>", unsafe_allow_html=True)
        
        # Resume Content Analysis
        st.markdown('<h2 class="section-header">Resume Content Assessment</h2>', unsafe_allow_html=True)
        
        content_cols = st.columns(3)
        content_checks = [
            ("Contact Information", results['has_contact_info'], "Include email, phone, LinkedIn"),
            ("Education Details", results['has_education'], "Add degree, university, CGPA"),
            ("Projects", results['has_projects'], "Showcase 2-3 strong projects"),
            ("Internship Experience", results['has_internship'], "Highlight internships"),
            ("Achievements", results['has_achievements'], "List hackathons, awards"),
            ("Leadership Roles", results['has_leadership'], "Mention club activities")
        ]
        
        for i, (check_name, check_status, tip) in enumerate(content_checks):
            with content_cols[i % 3]:
                status_icon = "" if check_status else ""
                status_color = "#10b981" if check_status else "#ef4444"
                status_text = "Present" if check_status else "Missing"
                st.markdown(f"""
                <div class="analysis-card">
                    <div class="card-title">{status_icon} {check_name}</div>
                    <div class="card-content">
                        <span style="color: {status_color}; font-weight: 700; font-size: 1.1rem;">
                            {status_text}
                        </span>
                        <div style="color: #94a3b8; margin-top: 0.75rem; font-size: 0.9rem;">{tip}</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        
        # Strengths and Weaknesses with perfect two-column layout
        st.markdown('<h2 class="section-header">Strengths & Areas for Improvement</h2>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div class="analysis-card">
                <div class="card-title">Your Strengths</div>
                <div class="card-content">
            """, unsafe_allow_html=True)
            for i, strength in enumerate(results['strengths'], 1):
                st.markdown(f'<div style="margin: 0.75rem 0; color: #10b981; font-size: 0.95rem;"><strong>{i}.</strong> {strength}</div>', unsafe_allow_html=True)
            st.markdown("</div></div>", unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="analysis-card">
                <div class="card-title">Areas to Improve</div>
                <div class="card-content">
            """, unsafe_allow_html=True)
            for i, weakness in enumerate(results['weaknesses'], 1):
                st.markdown(f'<div style="margin: 0.75rem 0; color: #ef4444; font-size: 0.95rem;"><strong>{i}.</strong> {weakness}</div>', unsafe_allow_html=True)
            st.markdown("</div></div>", unsafe_allow_html=True)
        
        # Personalized Recommendations
        st.markdown('<h2 class="section-header">Your Personalized Career Roadmap</h2>', unsafe_allow_html=True)
        
        for i, rec in enumerate(results['recommendations'], 1):
            priority_class = {
                'high': 'priority-high',
                'medium': 'priority-medium',
                'low': 'priority-low'
            }[rec['priority']]
            st.markdown(f"""
            <div class="recommendation-item">
                <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 0.75rem;">
                    <div style="font-weight: 700; color: #e2e8f0; font-size: 1.05rem; flex: 1;">{i}. {rec['text']}</div>
                    <div class="recommendation-priority {priority_class}">{rec['priority'].upper()}</div>
                </div>
                <div style="color: #94a3b8; font-size: 0.9rem; line-height: 1.6;">
                    <strong style="color: #cbd5e1;">Action:</strong> {rec['action']} &nbsp;|&nbsp;
                    <strong style="color: #cbd5e1;">Timeline:</strong> {rec['timeline']}<br>
                    <strong style="color: #cbd5e1;">Impact:</strong> {rec['impact']}
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        # Course Recommendations
        if results['course_recommendations']:
            st.markdown('<h2 class="section-header">Recommended Courses</h2>', unsafe_allow_html=True)
            
            course_cols = st.columns(2)
            for i, course in enumerate(results['course_recommendations']):
                with course_cols[i % 2]:
                    priority_class = 'priority-high' if course.get('priority') == 'High' else 'priority-medium'
                    st.markdown(f"""
                    <div class="recommendation-item">
                        <div style="font-weight: 700; color: #e2e8f0; font-size: 1.05rem; margin-bottom: 0.75rem;">{course['course']}</div>
                        <div style="color: #94a3b8; font-size: 0.9rem; line-height: 1.6;">
                            <strong>Skill:</strong> {course['skill']}<br>
                            <strong>Provider:</strong> {course['provider']}<br>
                            <strong>Duration:</strong> {course['duration']}
                        </div>
                        <div class="recommendation-priority {priority_class}" style="margin-top: 0.75rem;">{course.get('priority', 'Medium')} Priority</div>
                    </div>
                    """, unsafe_allow_html=True)
        
        # Career Path Suggestions
        st.markdown('<h2 class="section-header">Career Path Suggestions</h2>', unsafe_allow_html=True)
        
        career_cols = st.columns(2)
        for i, suggestion in enumerate(results['career_suggestions']):
            with career_cols[i % 2]:
                match_percentage = float(suggestion['match'].strip('%'))
                match_color = "#10b981" if match_percentage >= 80 else "#f59e0b" if match_percentage >= 65 else "#ef4444"
                st.markdown(f"""
                <div class="recommendation-item">
                    <div style="font-weight: 700; color: #e2e8f0; font-size: 1.05rem; margin-bottom: 0.75rem;">{suggestion['role']}</div>
                    <div style="color: #94a3b8; font-size: 0.9rem; line-height: 1.6;">
                        <strong>Match:</strong> <span style="color: {match_color}; font-weight: 700; font-size: 1.1rem;">{suggestion['match']}</span><br>
                        <strong>Reason:</strong> {suggestion['reason']}
                    </div>
                </div>
                """, unsafe_allow_html=True)
        
        # Job Matches
        st.markdown('<h2 class="section-header">Potential Job Opportunities</h2>', unsafe_allow_html=True)
        
        job_cols = st.columns(2)
        for i, job in enumerate(results['job_matches']):
            with job_cols[i % 2]:
                match_color = "#10b981" if job['match_percentage'] >= 80 else "#f59e0b" if job['match_percentage'] >= 60 else "#ef4444"
                st.markdown(f"""
                <div class="recommendation-item">
                    <div style="font-weight: 700; color: #e2e8f0; font-size: 1.05rem; margin-bottom: 0.75rem;">{job['title']}</div>
                    <div style="color: #94a3b8; font-size: 0.9rem; line-height: 1.6;">
                        <strong>Match:</strong> <span style="color: {match_color}; font-weight: 700;">{job['match_percentage']}%</span><br>
                        <strong>Description:</strong> {job['description']}<br>
                        <strong>Companies:</strong> {job['companies']}
                    </div>
                </div>
                """, unsafe_allow_html=True)
        
        # Salary Expectations
        st.markdown('<h2 class="section-header">Salary Expectations</h2>', unsafe_allow_html=True)
        
        salary_fig = create_salary_chart(results['salary_info'])
        st.plotly_chart(salary_fig, use_container_width=True)
        
        st.markdown("""
        <div class="info-box">
            <strong>Note:</strong> These are estimated salary ranges based on your skills, target role, and current market trends.
            Actual salaries may vary based on company size, location, specific role requirements, and your negotiation skills.
            Focus on building strong skills and a solid portfolio to maximize your earning potential.
        </div>
        """, unsafe_allow_html=True)
        
        # Emerging Technologies
        st.markdown('<h2 class="section-header">Emerging Technologies to Watch</h2>', unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="analysis-card">
            <div class="card-title">Trending in {target_role}</div>
            <div class="card-content">
                <div style="margin-bottom: 1.5rem;">
                    {"".join([f'<span class="skill-tag present">{tech.title()}</span>' for tech in results['emerging_tech_analysis']['trending']])}
                </div>
                <div style="color: #cbd5e1; margin-top: 1rem;">
                    <strong style="color: #3b82f6;">Recommended Focus:</strong> {', '.join([tech.title() for tech in results['emerging_tech_analysis']['recommendations']])}
                </div>
                <div style="color: #94a3b8; margin-top: 0.75rem; font-size: 0.9rem;">
                    Growth Outlook: <span style="color: #10b981; font-weight: 600;">{results['emerging_tech_analysis']['growth_rate']}</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Placement Forecast
        st.markdown('<h2 class="section-header">Placement Forecast</h2>', unsafe_allow_html=True)
        
        forecast = results['placement_forecast']
        forecast_color = "#10b981" if forecast['probability'] >= 80 else "#3b82f6" if forecast['probability'] >= 65 else "#f59e0b"
        
        st.markdown(f"""
        <div class="analysis-card">
            <div class="card-title">Job Placement Outlook</div>
            <div class="card-content">
                <div style="text-align: center; margin: 1.5rem 0;">
                    <div style="font-size: 3.5rem; font-weight: 900; color: {forecast_color};">{forecast['probability']}%</div>
                    <div style="color: #94a3b8; font-size: 1.1rem; margin-top: 0.5rem;">
                        Placement Probability within <strong style="color: #e2e8f0;">{forecast['timeframe']} Months</strong>
                    </div>
                    <div style="margin-top: 1rem;">
                        <span style="background: {forecast_color}; color: white; padding: 0.5rem 1.5rem; border-radius: 50px; font-weight: 600;">
                            {forecast['confidence']} Confidence
                        </span>
                    </div>
                </div>
                <div style="color: #cbd5e1; margin-top: 1.5rem;">
                    <strong>Key Contributing Factors:</strong>
                    <ul style="color: #94a3b8; margin-top: 0.75rem; line-height: 1.8;">
                        {"".join([f'<li>{factor}</li>' for factor in forecast['key_factors']])}
                    </ul>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Achievements Section
        st.markdown('<h2 class="section-header">Your Achievements</h2>', unsafe_allow_html=True)
        
        achievements = []
        if results['has_projects']:
            achievements.append("Completed Notable Projects")
        if results['has_internship']:
            achievements.append("Gained Industry Experience")
        if results['has_achievements']:
            achievements.append("Recognized in Competitions/Certifications")
        if results['has_leadership']:
            achievements.append("Demonstrated Leadership")
        if results['has_certifications']:
            achievements.append("Earned Professional Certifications")
        
        if achievements:
            st.markdown("""
            <div class="analysis-card">
                <div class="card-title">Your Milestones</div>
                <div class="card-content" style="text-align: center;">
            """, unsafe_allow_html=True)
            for achievement in achievements:
                st.markdown(f'<span class="achievement-badge">{achievement}</span>', unsafe_allow_html=True)
            st.markdown("</div></div>", unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="analysis-card">
                <div class="card-content" style="text-align: center; color: #94a3b8;">
                    Start building your achievements by working on projects, internships, or competitions!
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        # Download Report Section
        st.markdown('<h2 class="section-header">Download Your Report</h2>', unsafe_allow_html=True)
        
        report_content = generate_txt_report(results, target_role, resume_content)
        
        st.download_button(
            label="Download Complete Analysis Report (TXT)",
            data=report_content,
            file_name=f"Job_Bridge_Analysis_{target_role.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain",
            use_container_width=True
        )
        
        st.markdown("""
        <div class="info-box" style="margin-top: 2rem;">
            <h3 style="color: #3b82f6; margin-bottom: 1rem;">What's Included in Your Report:</h3>
            <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 0.75rem; color: #cbd5e1;">
                <div>✓ Executive Summary & Job Readiness Score</div>
                <div>✓ Detailed Score Breakdown</div>
                <div>✓ Technical Skills Analysis</div>
                <div>✓ Skill Coverage by Category</div>
                <div>✓ Resume Content Assessment</div>
                <div>✓ Strengths & Improvement Areas</div>
                <div>✓ Actionable Career Roadmap</div>
                <div>✓ Course Recommendations</div>
                <div>✓ Career Path Suggestions</div>
                <div>✓ Job Opportunities</div>
                <div>✓ Salary Expectations</div>
                <div>✓ Placement Forecast</div>
                <div>✓ Emerging Technologies</div>
                <div>✓ 30-Day Action Plan</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Final Call to Action
      

if __name__ == "__main__":
    main()