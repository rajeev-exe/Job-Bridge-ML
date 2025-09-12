import os
import re
import pandas as pd
import docx
import PyPDF2
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')
STOPWORDS = set(stopwords.words('english'))

# Example list of skills — can be expanded
SKILL_KEYWORDS = [
    "python", "java", "c++", "sql", "machine learning", "data analysis",
    "statistics", "html", "css", "javascript", "react", "nlp", "deep learning",
    "cloud computing", "data structures", "algorithms"
]

def extract_text_from_pdf(file_path):
    text = ""
    with open(file_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            text += page.extract_text() or ""
    return text

def extract_text_from_docx(file_path):
    doc = docx.Document(file_path)
    return "\n".join([para.text for para in doc.paragraphs])

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', ' ', text)
    return " ".join([word for word in text.split() if word not in STOPWORDS])

def extract_skills_from_text(text):
    found_skills = []
    for skill in SKILL_KEYWORDS:
        if skill.lower() in text:
            found_skills.append(skill)
    return list(set(found_skills))

def process_curriculum_file(file_path):
    if file_path.endswith(".pdf"):
        raw_text = extract_text_from_pdf(file_path)
    elif file_path.endswith(".docx"):
        raw_text = extract_text_from_docx(file_path)
    elif file_path.endswith(".txt"):
        with open(file_path, "r", encoding="utf-8") as file:
            raw_text = file.read()
    else:
        raise ValueError("Unsupported file format!")

    processed_text = preprocess_text(raw_text)
    skills = extract_skills_from_text(processed_text)
    return skills

if __name__ == "__main__":
    curriculum_folder = "data/curriculum_files"
    output_file = "data/curriculum_skills.csv"

    os.makedirs(curriculum_folder, exist_ok=True)

    all_data = []

    for filename in os.listdir(curriculum_folder):
        file_path = os.path.join(curriculum_folder, filename)
        try:
            skills = process_curriculum_file(file_path)
            all_data.append({"Curriculum File": filename, "Extracted Skills": ", ".join(skills)})
            print(f"✅ Extracted skills from {filename}: {skills}")
        except Exception as e:
            print(f"❌ Error processing {filename}: {e}")

    df = pd.DataFrame(all_data)
    df.to_csv(output_file, index=False)
    print(f"📄 Saved curriculum skills to: {output_file}")
