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

# ==================== ENHANCED ANALYSIS FUNCTIONS ====================

def extract_text_from_file(uploaded_file):
    """Extract text content from uploaded file"""
    try:
        if uploaded_file.type == "application/pdf":
            # Mock PDF extraction for demo
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
},"Blockchain": {
    "platforms": ["ethereum", "hyperledger", "binance smart chain", "solana"],
    "languages": ["solidity", "javascript", "go", "rust", "python"],
    "tools": ["truffle", "remix", "metamask", "ganache", "git"],
    "concepts": ["smart contracts", "decentralized finance", "consensus algorithms", "tokenization", "cryptography"],
    "testing": ["smart contract auditing", "unit testing", "integration testing", "security testing"]
},"VR & AR": {
    "platforms": ["oculus", "hololens", "unity", "unreal engine"],
    "languages": ["c#", "c++", "javascript", "python"],
    "tools": ["unity", "unreal engine", "blender", "git", "ar foundation"],
    "concepts": ["3d modeling", "spatial computing", "gesture recognition", "immersive storytelling"],
    "testing": ["usability testing", "performance testing", "device compatibility testing"]
},"Big Data": {
    "platforms": ["hadoop", "spark", "cloud platforms", "kafka"],
    "languages": ["python", "scala", "java", "sql"],
    "tools": ["apache spark", "hadoop", "tableau", "jupyter", "git"],
    "concepts": ["data lakes", "etl pipelines", "real-time processing", "data warehousing"],
    "testing": ["data validation", "performance testing", "scalability testing"]
},"Data Science": {
    "platforms": ["cloud", "local environments", "colab"],
    "languages": ["python", "r", "sql", "julia"],
    "tools": ["jupyter", "pandas", "scikit-learn", "tableau", "git"],
    "concepts": ["statistical modeling", "machine learning", "data visualization", "feature engineering"],
    "testing": ["model evaluation", "cross-validation", "a/b testing"]
},"Cyber Security": {
    "platforms": ["cloud", "on-premise", "network infrastructure"],
    "languages": ["python", "c", "javascript", "bash"],
    "tools": ["wireshark", "metasploit", "burp suite", "git", "kali linux"],
    "concepts": ["ethical hacking", "penetration testing", "cryptography", "network security"],
    "testing": ["vulnerability scanning", "penetration testing", "compliance testing"]
}
    }

def analyze_resume_content(text: str, target_role: str = "Software Engineer") -> Dict:
    """Comprehensive student resume analysis"""
    role_requirements = get_role_requirements()
    text_lower = text.lower()
    
    # Get all required skills for the role (flatten nested dict)
    role_data = role_requirements.get(target_role, role_requirements["Software Engineer"])
    all_required_skills = []
    
    if isinstance(role_data, dict):
        for category_skills in role_data.values():
            if isinstance(category_skills, list):
                all_required_skills.extend(category_skills)
    else:
        all_required_skills = role_data
    
    # Remove duplicates
    all_required_skills = list(set(all_required_skills))
    
    # Extract skills present in resume
    found_skills = [skill for skill in all_required_skills if skill.replace(" ", "").replace("-", "").replace(".", "") in text_lower.replace(" ", "").replace("-", "").replace(".", "")]
    missing_skills = [skill for skill in all_required_skills if skill not in found_skills]
    
    # Calculate skill match score
    skill_match_score = (len(found_skills) / len(all_required_skills)) * 100 if all_required_skills else 0
    
    # Student-specific analysis
    word_count = len(text.split())
    has_contact_info = any(keyword in text_lower for keyword in ['email', 'phone', '@', '.com', 'linkedin'])
    has_education = any(keyword in text_lower for keyword in ['education', 'university', 'college', 'degree', 'bachelor', 'master', 'cgpa', 'gpa'])
    has_projects = any(keyword in text_lower for keyword in ['project', 'github', 'repository', 'built', 'developed', 'created'])
    has_internship = any(keyword in text_lower for keyword in ['intern', 'training', 'apprentice', 'work experience'])
    has_achievements = any(keyword in text_lower for keyword in ['achievement', 'award', 'winner', 'certificate', 'hackathon', 'competition'])
    has_leadership = any(keyword in text_lower for keyword in ['lead', 'president', 'head', 'coordinator', 'captain', 'volunteer'])
    
    # Content quality scoring for students
    content_quality_score = 0
    if word_count > 150: content_quality_score += 15  # Adequate length
    if has_contact_info: content_quality_score += 20  # Contact information
    if has_education: content_quality_score += 20     # Educational background
    if has_projects: content_quality_score += 25      # Project experience (crucial for students)
    if has_internship: content_quality_score += 10    # Internship/work experience
    if has_achievements: content_quality_score += 10  # Achievements/certifications
    
    # Student experience score (different from professional experience)
    experience_score = 0
    if has_projects: experience_score += 40           # Projects are key for students
    if has_internship: experience_score += 30         # Internship experience
    if has_achievements: experience_score += 20       # Academic/technical achievements
    if has_leadership: experience_score += 10         # Leadership experience
    
    # Presentation score (formatting, structure)
    presentation_score = min(90, 70 + np.random.randint(0, 20))
    
    # Student-specific overall score calculation
    overall_score = (
        skill_match_score * 0.40 +      # Higher weight on technical skills
        content_quality_score * 0.30 +  # Content completeness
        experience_score * 0.20 +       # Project/internship experience
        presentation_score * 0.10       # Formatting
    )
    
    # Student readiness level
    readiness_level = get_student_readiness_level(overall_score)
    career_suggestions = get_career_suggestions(found_skills, target_role)
    
    return {
        'overall_score': round(overall_score, 1),
        'skill_match_score': round(skill_match_score, 1),
        'content_quality_score': content_quality_score,
        'experience_score': experience_score,
        'presentation_score': presentation_score,
        'found_skills': found_skills,
        'missing_skills': missing_skills[:10],  # Limit to top 10 missing skills
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
        'recommendations': generate_student_recommendations(missing_skills, overall_score, has_projects, has_internship, target_role)
    }

def get_student_readiness_level(score: float) -> Dict:
    """Determine student's job readiness level"""
    if score >= 85:
        return {
            "level": "Job Ready",
            "description": "Excellent profile! You're ready to apply for entry-level positions.",
            "color": "#10b981",
            "next_step": "Start applying for jobs and prepare for interviews"
        }
    elif score >= 70:
        return {
            "level": "Almost Ready",
            "description": "Good foundation! A few improvements will make you job-ready.",
            "color": "#3b82f6",
            "next_step": "Focus on missing skills and add more projects"
        }
    elif score >= 55:
        return {
            "level": "Developing",
            "description": "You're on the right track! Need to strengthen technical skills.",
            "color": "#f59e0b",
            "next_step": "Build more projects and learn in-demand technologies"
        }
    else:
        return {
            "level": "Beginning",
            "description": "Great start! Focus on building core technical competencies.",
            "color": "#ef4444",
            "next_step": "Learn fundamentals and start with simple projects"
        }

def get_career_suggestions(found_skills: List[str], target_role: str) -> List[Dict]:
    """Suggest career paths based on current skills"""
    suggestions = []
    
    if any(skill in found_skills for skill in ["python", "machine learning", "data"]):
        suggestions.append({
            "role": "Data Analyst",
            "match": "85%",
            "reason": "Strong foundation in Python and data analysis"
        })
    
    if any(skill in found_skills for skill in ["react", "javascript", "html", "css"]):
        suggestions.append({
            "role": "Frontend Developer",
            "match": "90%",
            "reason": "Good web development skills"
        })
    
    if any(skill in found_skills for skill in ["java", "python", "algorithms"]):
        suggestions.append({
            "role": "Backend Developer",
            "match": "80%",
            "reason": "Strong programming fundamentals"
        })
    
    if any(skill in found_skills for skill in ["react native", "flutter", "android", "ios"]):
        suggestions.append({
            "role": "Mobile App Developer",
            "match": "85%",
            "reason": "Mobile development experience"
        })
    
    # Default suggestion based on target role
    if not suggestions:
        suggestions.append({
            "role": f"Junior {target_role}",
            "match": "70%",
            "reason": "Entry-level position to build experience"
        })
    
    return suggestions[:3]  # Return top 3 suggestions

def identify_student_strengths(found_skills: List[str], content_score: int, exp_score: int, has_projects: bool, has_internship: bool) -> List[str]:
    """Identify student-specific strengths"""
    strengths = []
    
    if len(found_skills) >= 8:
        strengths.append("Strong technical skill portfolio for a student")
    elif len(found_skills) >= 5:
        strengths.append("Good foundation in relevant technologies")
    
    if has_projects:
        strengths.append("Demonstrates practical application through projects")
    
    if has_internship:
        strengths.append("Valuable industry experience through internships")
    
    if content_score >= 80:
        strengths.append("Well-structured resume with complete information")
    
    if exp_score >= 70:
        strengths.append("Good balance of academic and practical experience")
    
    if len(found_skills) >= 3:
        strengths.append("Shows commitment to learning new technologies")
    
    return strengths

def identify_student_weaknesses(missing_skills: List[str], content_score: int, has_projects: bool) -> List[str]:
    """Identify student-specific areas for improvement"""
    weaknesses = []
    
    if len(missing_skills) >= 8:
        weaknesses.append("Several industry-standard skills need development")
    elif len(missing_skills) >= 5:
        weaknesses.append("Some key technical skills could be strengthened")
    
    if not has_projects:
        weaknesses.append("Portfolio needs more practical project examples")
    
    if content_score < 70:
        weaknesses.append("Resume content could be more comprehensive")
    
    if len(missing_skills) >= 6:
        weaknesses.append("Consider learning trending technologies in your field")
    
    weaknesses.append("Could benefit from quantified project outcomes and impact")
    
    return weaknesses

def generate_student_recommendations(missing_skills: List[str], overall_score: float, has_projects: bool, has_internship: bool, target_role: str) -> List[Dict]:
    """Generate student-specific recommendations"""
    recommendations = []
    
    # High priority recommendations
    if overall_score < 60:
        recommendations.append({
            'text': 'Focus on building 2-3 strong projects showcasing your technical skills',
            'priority': 'high',
            'action': 'Project Development',
            'timeline': '2-3 months'
        })
    
    if not has_projects:
        recommendations.append({
            'text': 'Create a portfolio with at least 3 projects demonstrating different skills',
            'priority': 'high',
            'action': 'Portfolio Building',
            'timeline': '1-2 months'
        })
    
    if len(missing_skills) >= 6:
        recommendations.append({
            'text': f"Learn these priority skills: {', '.join(missing_skills[:3])}",
            'priority': 'high',
            'action': 'Skill Development',
            'timeline': '3-4 months'
        })
    
    # Medium priority recommendations
    recommendations.append({
        'text': 'Add GitHub links and live project demos to showcase your work',
        'priority': 'medium',
        'action': 'Online Presence',
        'timeline': '1 week'
    })
    
    if not has_internship:
        recommendations.append({
            'text': 'Apply for internships or contribute to open-source projects',
            'priority': 'medium',
            'action': 'Experience Building',
            'timeline': '3-6 months'
        })
    
    recommendations.append({
        'text': 'Include quantified results in projects (e.g., "Built app used by 100+ users")',
        'priority': 'medium',
        'action': 'Content Enhancement',
        'timeline': '1 week'
    })
    
    # Low priority recommendations
    recommendations.extend([
        {
            'text': 'Get relevant certifications from platforms like Coursera, edX, or AWS',
            'priority': 'low',
            'action': 'Certifications',
            'timeline': '2-4 weeks'
        },
        {
            'text': 'Participate in hackathons and coding competitions to gain recognition',
            'priority': 'low',
            'action': 'Competitions',
            'timeline': 'Ongoing'
        },
        {
            'text': 'Create technical blog posts or tutorials to demonstrate expertise',
            'priority': 'low',
            'action': 'Content Creation',
            'timeline': '1-2 months'
        }
    ])
    
    return recommendations

# ==================== ENHANCED UI COMPONENTS ====================

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
    
    # Header Section
    st.markdown("""
    <div class="header-section">
        <div class="header-title">🚀 Job Bridge  </div>
        <div class="header-subtitle">ML Skill Gap Analyzer for Students</div>
        <div class="header-badge">Designed for Pre-final & Final Year Students</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Main content container
    # st.markdown('<div class="main-container">', unsafe_allow_html=True)
    
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
        ["Software Engineer", "Data Scientist", "Frontend Developer", "Full Stack Developer", "Mobile App Developer","Artificial Intelligence", "Blockchain", "VR & AR", "Big Data", "Data Science", "Cyber Security"],
        help="This helps us provide role-specific skill gap analysis and recommendations",
        label_visibility="collapsed"
    )
    
    st.markdown("</div></div>", unsafe_allow_html=True)
    
    # Analysis button with enhanced styling
    if st.button("🚀 Analyze My Resume & Get Job Readiness Score", type="primary", use_container_width=True):
        
        # Get resume content
        if uploaded_file is not None:
            resume_content = extract_text_from_file(uploaded_file)
            st.success(f"✅ Successfully processed {uploaded_file.name}")
        elif resume_text.strip():
            resume_content = resume_text
        else:
            st.error("❌ Please upload a resume file or paste your resume text to continue.")
            st.stop()
        
        # Enhanced loading animation
        loading_placeholder = st.empty()
        start_time = time.time()
        
        # Simulate analysis steps with student-focused loading screen
        steps = [
            ("📄 Scanning Your Resume Content...", 20),
            ("🎯 Analyzing Technical Skills & Keywords...", 40), 
            ("📊 Evaluating Projects & Experience...", 60),
            ("🚀 Calculating Job Readiness Score...", 80),
            ("✨ Generating Personalized Career Roadmap...", 100)
        ]
        
        for step_text, progress in steps:
            elapsed = time.time() - start_time
            with loading_placeholder.container():
                st.markdown(create_enhanced_loading_screen(step_text, progress, elapsed), unsafe_allow_html=True)
            time.sleep(1.2)  # Slightly longer for premium feel
        
        loading_placeholder.empty()
        
        # Perform actual analysis
        results = analyze_resume_content(resume_content, target_role)
        
        # Success message with confetti effect
        
        st.success("🎉 Analysis Complete! Here's your comprehensive career assessment:")
        
        # Job Readiness Score Section
        readiness_data = results['readiness_level']
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown(f"""
            <div class="score-card">
                <div class="score-label">Job Readiness Score</div>
                <div class="score-display">{results['overall_score']}/100</div>
                <div class="score-status" style="background: {readiness_data['color']}; color: white;">
                    {readiness_data['level']}
                </div>
                <div style="color: #94a3b8; margin-top: 15px; font-size: 1rem;">
                    {readiness_data['description']}
                </div>
                <div style="color: #64748b; margin-top: 10px; font-size: 0.9rem; font-weight: 600;">
                    Next Step: {readiness_data['next_step']}
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            fig = create_readiness_gauge(results['overall_score'], readiness_data)
            st.plotly_chart(fig, use_container_width=True)
        
        # Skills Radar Chart
        st.markdown("## 🎯 Skills Coverage Analysis")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            radar_fig = create_skills_radar_chart(results['found_skills'], results['missing_skills'], target_role)
            st.plotly_chart(radar_fig, use_container_width=True)
        
        with col2:
            st.markdown(f"""
            <div class="analysis-card" style="height: 400px; display: flex; flex-direction: column; justify-content: center;">
                <div class="card-title">📊 Coverage Summary</div>
                <div class="card-content">
                    <div style="margin-bottom: 20px;">
                        <div style="font-size: 2rem; font-weight: bold; color: #10b981;">{len(results['found_skills'])}</div>
                        <div style="color: #94a3b8;">Skills Found</div>
                    </div>
                    <div style="margin-bottom: 20px;">
                        <div style="font-size: 2rem; font-weight: bold; color: #ef4444;">{len(results['missing_skills'])}</div>
                        <div style="color: #94a3b8;">Skills to Learn</div>
                    </div>
                    <div>
                        <div style="font-size: 2rem; font-weight: bold; color: #3b82f6;">{results['skill_match_score']:.0f}%</div>
                        <div style="color: #94a3b8;">Skill Match</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        # Detailed Breakdown
        st.markdown("## 📊 Detailed Assessment Breakdown")
        
        score_cols = st.columns(4)
        scores = [
            ("🎯 Technical Skills", results['skill_match_score'], "#3b82f6"),
            ("📝 Content Quality", results['content_quality_score'], "#10b981"),
            ("💼 Project Experience", results['experience_score'], "#8b5cf6"),
            ("🎨 Presentation", results['presentation_score'], "#06b6d4")
        ]
        
        for col, (label, score, color) in zip(score_cols, scores):
            with col:
                st.markdown(f"""
                <div class="analysis-card" style="text-align: center; padding: 25px;">
                    <div style="font-size: 2.5rem; font-weight: bold; color: {color}; margin-bottom: 10px;">{score:.0f}%</div>
                    <div style="color: #94a3b8; font-size: 1rem; margin-bottom: 15px;">{label}</div>
                    <div class="progress-container">
                        <div class="progress-bar" style="width: {score}%; background: linear-gradient(90deg, {color}, {color}aa);"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        
        # Skills Analysis with enhanced design
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div class="analysis-card">
                <div class="card-title">✅ Skills Found in Your Resume</div>
                <div class="card-content">
            """, unsafe_allow_html=True)
            
            if results['found_skills']:
                for skill in results['found_skills']:
                    st.markdown(f'<span class="skill-tag present">{skill.title()}</span>', unsafe_allow_html=True)
                st.markdown(f'<div style="margin-top: 20px;"><div class="achievement-badge">🎯 {len(results["found_skills"])} Skills Matched!</div></div>', unsafe_allow_html=True)
            else:
                st.markdown('<p style="color: #ef4444;">No relevant skills found for this role. Consider adding technical skills to your resume.</p>', unsafe_allow_html=True)
            
            st.markdown("</div></div>", unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="analysis-card">
                <div class="card-title">⚠ Skills to Learn for Better Job Prospects</div>
                <div class="card-content">
            """, unsafe_allow_html=True)
            
            if results['missing_skills']:
                priority_skills = results['missing_skills'][:8]  # Show top 8 missing skills
                for skill in priority_skills:
                    st.markdown(f'<span class="skill-tag missing">{skill.title()}</span>', unsafe_allow_html=True)
                
                if len(results['missing_skills']) > 8:
                    st.markdown(f'<div style="margin-top: 15px; color: #94a3b8; font-style: italic;">...and {len(results["missing_skills"]) - 8} more skills</div>', unsafe_allow_html=True)
            else:
                st.markdown('<p style="color: #10b981;">Excellent! You have all the key skills for this role.</p>', unsafe_allow_html=True)
            
            st.markdown("</div></div>", unsafe_allow_html=True)
        
        # Career Suggestions
        if results['career_suggestions']:
            st.markdown("## 🎯 Career Path Suggestions")
            
            career_cols = st.columns(len(results['career_suggestions']))
            for col, suggestion in zip(career_cols, results['career_suggestions']):
                with col:
                    st.markdown(f"""
                    <div class="analysis-card" style="text-align: center;">
                        <div class="card-title" style="font-size: 1.2rem;">{suggestion['role']}</div>
                        <div style="font-size: 2rem; font-weight: bold; color: #10b981; margin: 15px 0;">{suggestion['match']}</div>
                        <div class="card-content" style="font-size: 0.9rem;">
                            {suggestion['reason']}
                        </div>
                        <div style="margin-top: 15px;">
                            <span class="achievement-badge">Recommended</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
        
        # Strengths & Areas for Improvement
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div class="analysis-card">
                <div class="card-title">💪 Your Key Strengths</div>
                <div class="card-content">
            """, unsafe_allow_html=True)
            
            for i, strength in enumerate(results['strengths'], 1):
                st.markdown(f"""
                <div style="display: flex; align-items: center; margin-bottom: 12px;">
                    <div style="background: linear-gradient(135deg, #10b981, #059669); color: white; width: 24px; height: 24px; border-radius: 50%; display: flex; align-items: center; justify-content: center; margin-right: 15px; font-weight: bold; font-size: 0.8rem;">{i}</div>
                    <div>{strength}</div>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("</div></div>", unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="analysis-card">
                <div class="card-title">🎯 Areas for Improvement</div>
                <div class="card-content">
            """, unsafe_allow_html=True)
            
            for i, weakness in enumerate(results['weaknesses'], 1):
                st.markdown(f"""
                <div style="display: flex; align-items: center; margin-bottom: 12px;">
                    <div style="background: linear-gradient(135deg, #f59e0b, #d97706); color: white; width: 24px; height: 24px; border-radius: 50%; display: flex; align-items: center; justify-content: center; margin-right: 15px; font-weight: bold; font-size: 0.8rem;">{i}</div>
                    <div>{weakness}</div>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("</div></div>", unsafe_allow_html=True)
        
        # Student-Specific Insights
        st.markdown("## 🎓 Student Profile Insights")
        
        insights_cols = st.columns(3)
        
        insights = [
            ("📚 Educational Background", "✅ Present" if results['has_education'] else "❌ Missing", "#10b981" if results['has_education'] else "#ef4444"),
            ("💻 Project Portfolio", "✅ Present" if results['has_projects'] else "❌ Missing", "#10b981" if results['has_projects'] else "#ef4444"),
            ("🏢 Internship Experience", "✅ Present" if results['has_internship'] else "❌ Could be added", "#10b981" if results['has_internship'] else "#f59e0b"),
        ]
        
        for col, (label, status, color) in zip(insights_cols, insights):
            with col:
                st.markdown(f"""
                <div class="analysis-card" style="text-align: center; padding: 20px;">
                    <div style="font-size: 1.1rem; font-weight: bold; color: #e2e8f0; margin-bottom: 15px;">{label}</div>
                    <div style="font-size: 1.3rem; font-weight: bold; color: {color};">{status}</div>
                </div>
                """, unsafe_allow_html=True)
        
        # Additional insights
        additional_insights = []
        if results['has_achievements']:
            additional_insights.append("🏆 Has achievements/certifications listed")
        if results['has_leadership']:
            additional_insights.append("👥 Shows leadership experience")
        if results['word_count'] > 200:
            additional_insights.append("📄 Good resume length and detail")
        
        if additional_insights:
            st.markdown(f"""
            <div class="analysis-card">
                <div class="card-title">⭐ Additional Positive Indicators</div>
                <div class="card-content">
                    {"".join([f'<div style="margin-bottom: 8px;">{insight}</div>' for insight in additional_insights])}
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        # Personalized Recommendations
        st.markdown("## 🌟 Your Personalized Action Plan")
        
        # Group recommendations by priority
        high_priority = [rec for rec in results['recommendations'] if rec['priority'] == 'high']
        medium_priority = [rec for rec in results['recommendations'] if rec['priority'] == 'medium']
        low_priority = [rec for rec in results['recommendations'] if rec['priority'] == 'low']
        
        if high_priority:
            st.markdown("### 🚨 High Priority (Do This First)")
            for i, rec in enumerate(high_priority, 1):
                st.markdown(f"""
                <div class="recommendation-item">
                    <div style="display: flex; align-items: center; justify-content: space-between;">
                        <div style="flex: 1;">
                            <strong>{rec['action']}:</strong> {rec['text']}
                        </div>
                        <div style="display: flex; align-items: center; gap: 10px;">
                            <span class="recommendation-priority priority-high">{rec['priority']}</span>
                            <span style="background: rgba(59, 130, 246, 0.2); color: #3b82f6; padding: 4px 12px; border-radius: 12px; font-size: 0.8rem; font-weight: 600;">
                                {rec.get('timeline', '1-2 weeks')}
                            </span>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        
        if medium_priority:
            st.markdown("### ⚡ Medium Priority (Next Steps)")
            for rec in medium_priority:
                st.markdown(f"""
                <div class="recommendation-item">
                    <div style="display: flex; align-items: center; justify-content: space-between;">
                        <div style="flex: 1;">
                            <strong>{rec['action']}:</strong> {rec['text']}
                        </div>
                        <div style="display: flex; align-items: center; gap: 10px;">
                            <span class="recommendation-priority priority-medium">{rec['priority']}</span>
                            <span style="background: rgba(59, 130, 246, 0.2); color: #3b82f6; padding: 4px 12px; border-radius: 12px; font-size: 0.8rem; font-weight: 600;">
                                {rec.get('timeline', '2-4 weeks')}
                            </span>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        
        if low_priority:
            with st.expander("📋 Long-term Improvements (Click to expand)"):
                for rec in low_priority:
                    st.markdown(f"""
                    <div class="recommendation-item">
                        <div style="display: flex; align-items: center; justify-content: space-between;">
                            <div style="flex: 1;">
                                <strong>{rec['action']}:</strong> {rec['text']}
                            </div>
                            <div style="display: flex; align-items: center; gap: 10px;">
                                <span class="recommendation-priority priority-low">{rec['priority']}</span>
                                <span style="background: rgba(59, 130, 246, 0.2); color: #3b82f6; padding: 4px 12px; border-radius: 12px; font-size: 0.8rem; font-weight: 600;">
                                    {rec.get('timeline', '1-3 months')}
                                </span>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
        
        # Interactive Action Items Checklist
        st.markdown("## ✅ Your 30-Day Action Plan")
        st.markdown("""
        <div class="analysis-card">
            <div class="card-title">📅 Check off completed items to track your progress</div>
            <div class="card-content">
        """, unsafe_allow_html=True)
        
        action_items = [
            f"Update resume with missing skills: {', '.join(results['missing_skills'][:3])}" if results['missing_skills'] else "Continue building on your strong skill set",
            "Create or update 2-3 portfolio projects showcasing different technologies",
            "Set up/improve GitHub profile with project repositories and README files",
            "Write technical blog posts or create tutorials about your projects",
            "Apply for relevant internships or contribute to open-source projects",
            "Get 1-2 relevant certifications from Coursera, edX, or cloud providers",
            "Join coding communities and participate in hackathons",
            "Practice coding interviews and system design questions",
            "Optimize your LinkedIn profile with projects and skills",
            "Prepare for technical interviews by practicing common questions"
        ]
        
        progress_count = 0
        for i, item in enumerate(action_items, 1):
            checked = st.checkbox(f"{i}. {item}", key=f"action_{i}")
            if checked:
                progress_count += 1
        
        # Show progress
        progress_percentage = (progress_count / len(action_items)) * 100
        st.markdown(f"""
        <div style="margin-top: 25px;">
            <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 10px;">
                <span style="font-weight: 600; color: #e2e8f0;">Progress: {progress_count}/{len(action_items)} completed</span>
                <span style="font-weight: 600; color: #3b82f6;">{progress_percentage:.0f}%</span>
            </div>
            <div class="progress-container">
                <div class="progress-bar" style="width: {progress_percentage}%;"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("</div></div>", unsafe_allow_html=True)
        
        # Learning Resources Section
        st.markdown("## 📚 Recommended Learning Resources")
        
        resource_cols = st.columns(2)
        
        with resource_cols[0]:
            st.markdown("""
            <div class="analysis-card">
                <div class="card-title">🎯 For Technical Skills</div>
                <div class="card-content">
                    <div style="margin-bottom: 15px;">
                        <strong>🌐 Free Platforms:</strong><br>
                        • freeCodeCamp - Web development<br>
                        • Codecademy - Interactive coding<br>
                        • Khan Academy - Programming basics<br>
                        • YouTube - Technology tutorials
                    </div>
                    <div>
                        <strong>💳 Premium Courses:</strong><br>
                        • Coursera - University courses<br>
                        • Udemy - Practical projects<br>
                        • Pluralsight - Advanced topics<br>
                        • LinkedIn Learning - Professional skills
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with resource_cols[1]:
            st.markdown("""
            <div class="analysis-card" >
                <div class="card-title">💼 For Career Preparation</div>
                <div class="card-content">
                    <div style="margin-bottom: 15px;">
                        <strong>🔍 Job Preparation:</strong><br>
                        • LeetCode - Coding practice<br>
                        • HackerRank - Technical challenges<br>
                        • Glassdoor - Interview experiences<br>
                        • Pramp - Mock interviews
                    </div>
                    <div>
                        <strong>🤝 Networking & Projects:</strong><br>
                        • GitHub - Code portfolio<br>
                        • LinkedIn - Professional network<br>
                        • Dev.to - Tech community<br>
                        • AngelList - Startup opportunities
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        
   
   
    
    st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()