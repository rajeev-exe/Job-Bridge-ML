# 🚀 Job Bridge ML — Career Intelligence & Skill Gap Analysis Platform

**Job Bridge ML** is an AI-powered platform designed to empower students by analyzing resumes, identifying skill gaps, and recommending personalized learning paths aligned with industry demands. It leverages **Machine Learning**, **Natural Language Processing (NLP)**, and **Data Visualization** to bridge the gap between academia and the job market.

---

## 🧩 Project Overview

**Job Bridge ML** integrates advanced NLP, machine learning, and graph-based analytics to provide actionable career insights. Key components include:

| Component                        | Description                                                                                       |
|-----------------------------------|---------------------------------------------------------------------------------------------------|
| **app_enhanced.py**               | 🎯 Main Streamlit app with a modern UI/UX for resume analysis and recommendations.                 |
| **advanced_skill_extractor.py**   | 🧠 NLP-based skill extraction using BERT and Sentence Transformers.                                |
| **data_synthesizer.py**           | 🔄 Generates synthetic job data and builds skill-job relationship graphs.                          |
| **jobs.csv**                      | 📊 Real or sample job postings with role titles and skill requirements.                            |
| **courses.csv**                   | 📚 Curated dataset of courses with providers, durations, and associated skills.                    |
| **skills_dataset.csv**            | 🗂️ Master dataset of technical and soft skills for analysis.                                      |
| **synthetic_jobs.csv**            | 🤖 AI-generated job postings for model testing and validation.                                     |
| **skill_graph.json**              | 🌐 Graph-based mapping of skills to job roles.                                                     |
| **emerging_tech.csv**             | 📈 Dataset of trending technologies and their market growth rates.                                 |

---

## 🧠 Key Features

### 🧾 Resume Intelligence
- **Parses Resumes**: Supports PDF, DOCX, and TXT formats.
- **Extracts Key Data**: Skills, Projects, Education, Internships, Certifications, and Achievements.
- **Scoring Metrics**:
  - **Skill Match %**: Alignment with job requirements.
  - **Experience Score**: Based on projects and internships.
  - **Content Quality**: Evaluates resume structure and keywords.
  - **Readiness Score**: Assesses job market preparedness.

### 🔍 Skill Gap Analysis
- Uses **BERT + SpaCy** for advanced skill extraction.
- Compares resume skills against job market needs from `jobs.csv` and `skills_dataset.csv`.
- Identifies **missing**, **trending**, and **emerging** skills by domain.

### 📈 Career Forecasting
- Predicts **placement readiness probability** using ML models.
- Estimates **salary ranges** (Entry-level to Senior).
- Suggests **job roles** with similarity scores based on skills.

### 🎓 Personalized Learning Paths
- Recommends courses from `courses.csv` to address skill gaps.
- Prioritizes courses (High/Medium/Low) based on relevance.
- Integrates with platforms like **Coursera**, **Udemy**, **edX**, and more.

### 🌐 Emerging Technology Insights
- Leverages `emerging_tech.csv` to highlight high-demand fields (e.g., Generative AI, MLOps, Cloud Native).
- Aligns learning recommendations with future market trends.

---

## 🏗️ Project Structure

```
Job-Bridge-ML/
│
├── src/                          
│   ├── app_enhanced.py              # Main Streamlit app (UI + analysis engine)
│   ├── advanced_skill_extractor.py  # BERT-based skill extraction module
│   ├── data_synthesizer.py          # Synthetic data and skill graph generator
│   ├── gnn_skill_predictor.py       # (Optional) Graph Neural Network for skill prediction
│   ├── placement_predictor.py       # Placement readiness forecasting
│   ├── enhanced_placement_forecaster.py # Advanced probability/confidence analysis
│
├── Data/
│   ├── jobs.csv                     # Job postings with roles and skill requirements
│   ├── courses.csv                  # Courses with providers and skill mappings
│   ├── skills_dataset.csv           # Master dataset of technical and soft skills
│   ├── synthetic_jobs.csv           # AI-generated job postings for testing
│   ├── skill_graph.json             # Graph-based skill-job relationship mapping
│   ├── emerging_tech.csv            # Trending technologies and growth data
│
├── requirements.txt                 # Python dependencies
├── README.md                        # Project documentation (this file)
├── .gitignore                       # Ignored files (e.g., __pycache__, .env)
└── LICENSE                          # MIT License file
```

---

## ⚙️ Installation & Setup

### Prerequisites
- **Python**: Version 3.10 or higher
- **Conda**: Recommended for virtual environment management
- **Git**: For cloning the repository

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/rajeev-exe/Job-Bridge-ML.git
cd Job-Bridge-ML
```

### 2️⃣ Create a Virtual Environment
```bash
conda create -n jobbridge python=3.10
conda activate jobbridge
```

### 3️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

If `requirements.txt` is unavailable, install dependencies manually:
```bash
pip install streamlit pandas numpy torch transformers sentence-transformers scikit-learn plotly reportlab spacy PyPDF2 docx2txt networkx
python -m spacy download en_core_web_sm
```

### 4️⃣ Run the Application
```bash
streamlit run src/app_enhanced.py
```
- Open the provided URL (e.g., `http://localhost:8501`) in your browser.
- Upload a resume and select a target job role to begin analysis.

---

## 🧮 How It Works

1. **Upload Resume**:
   - Supports **PDF**, **DOCX**, or **TXT** formats.
   - Uses **PyMuPDF** or **docx2txt** for text extraction.

2. **Skill Extraction**:
   - `advanced_skill_extractor.py` applies **BERT** and **Sentence Transformers**.
   - Matches skills against `skills_dataset.csv` and job requirements from `jobs.csv`.

3. **Skill Graph Generation**:
   - `data_synthesizer.py` creates a bipartite graph (`skill_graph.json`) linking skills to jobs.
   - Used for predictive modeling and recommendations.

4. **Analysis Dashboard**:
   - Displays:
     - **Skill Match %**
     - **Missing Skills**
     - **Recommended Courses**
     - **Placement Probability**
     - **Career Readiness Level**

5. **Emerging Tech Insights**:
   - Analyzes `emerging_tech.csv` to recommend future-ready skills.
   - Suggests learning paths for trending fields (e.g., Generative AI, Blockchain).

---

## 📊 Output Metrics

| Metric                | Description                                                      |
|-----------------------|------------------------------------------------------------------|
| **Skill Match %**     | Percentage of job-required skills present in the resume.         |
| **Content Quality**   | Evaluates resume structure, keywords, and clarity.               |
| **Experience Score**  | Quantifies projects, internships, and practical exposure.        |
| **Placement Probability** | Likelihood of employability based on ML models.              |
| **Readiness Level**   | Classifies as Beginner, Intermediate, or Industry-Ready.         |
| **Recommended Courses** | Curated learning paths to address skill gaps.                  |

---

## 💼 Supported Job Roles

- Software Engineer
- Data Scientist
- Frontend / Full Stack Developer
- Mobile App Developer
- AI / ML Engineer
- Cybersecurity Specialist
- Blockchain / Web3 Developer
- Big Data Analyst
- AR/VR Engineer

---

## 🧰 Tech Stack

| Category        | Technology                                   |
|-----------------|---------------------------------------------|
| **Frontend**    | Streamlit, Plotly                            |
| **ML/NLP**      | Transformers (BERT), Sentence Transformers, SpaCy |
| **Backend**     | Python 3.10                                  |
| **Data Handling** | Pandas, NumPy, Scikit-learn                |
| **Visualization** | Plotly Express, Graph Objects              |
| **File Parsing** | PyPDF2, docx2txt                            |
| **Graph Processing** | NetworkX, JSON                          |
| **Deployment**  | Streamlit Cloud / Localhost                  |

---

## 📈 Example Workflow

1. Launch `app_enhanced.py` via Streamlit.
2. Upload a resume (PDF/DOCX/TXT).
3. Select a target job role (e.g., "Data Scientist").
4. Click **Analyze Resume**.
5. View results:
   - Skill Match Score
   - Missing Skills
   - Recommended Courses
   - Salary Estimates
   - Placement Probability
   - Emerging Tech Insights

---

## 📊 Sample Datasets

| File                  | Purpose                                                                |
|-----------------------|-----------------------------------------------------------------------|
| **jobs.csv**          | Job postings with titles, descriptions, and skill requirements.        |
| **courses.csv**       | Courses with providers, durations, and skill mappings.                 |
| **skills_dataset.csv**| Master list of technical and soft skills.                              |
| **synthetic_jobs.csv**| AI-generated job postings for testing.                                 |
| **skill_graph.json**  | Graph linking skills to job roles.                                     |
| **emerging_tech.csv** | Trending technologies with market growth data.                         |

---

## 🛠️ Troubleshooting

- **Streamlit not running**: Ensure port `8501` is free and dependencies are installed.
- **Missing SpaCy model**: Run `python -m spacy download en_core_web_sm`.
- **File parsing errors**: Verify resume files are valid PDF/DOCX/TXT formats.
- **Dependency issues**: Use Python 3.10 and check `requirements.txt` for conflicts.

---

## 🤝 Contributing

Contributions are welcome! To contribute:

1. Fork the repository.
2. Create a feature branch (`git checkout -b feature-name`).
3. Commit changes (`git commit -m "Add feature"`).
4. Push to the branch (`git push origin feature-name`).
5. Open a Pull Request.

Please follow the [Contributor Covenant Code of Conduct](https://www.contributor-covenant.org/).

---

## 🧾 License

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for details.

---

## 👤 Author

**Rajeev Gandhi K**  
📍 Chennai, Tamil Nadu, India  
💻 Computer Science Student | Data Science Enthusiast  
🔗 [GitHub: @rajeev-exe](https://github.com/rajeev-exe)  
✉️ Email: rajeevgandhi.exe@gmail.com

---

## ⭐ Support the Project

If you find **Job Bridge ML** helpful, please ⭐ **star** this repository on GitHub!  
Your support helps us improve and reach more students.

```bash
git clone https://github.com/rajeev-exe/Job-Bridge-ML.git
```

**Empowering students with AI-driven career intelligence and skill recommendations 🚀**
