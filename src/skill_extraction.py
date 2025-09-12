import pandas as pd
import re
import os

# Skills list for matching
SKILLS = [
    "python", "sql", "machine learning", "data visualization", "power bi",
    "javascript", "react", "node.js", "rest apis", "git",
    "tensorflow", "pytorch", "deep learning", "aws", "azure"
]

# Paths
csv_in = os.path.join(os.path.dirname(__file__), "..", "data", "sample_jobs_cleaned.csv")
csv_out = os.path.join(os.path.dirname(__file__), "..", "data", "sample_jobs_with_skills.csv")

# Load cleaned data
df = pd.read_csv(csv_in)

# Function to extract skills
def extract_skills(text):
    found = []
    if pd.isna(text):
        return found
    text_lower = text.lower()
    for skill in SKILLS:
        if re.search(r"\b" + re.escape(skill) + r"\b", text_lower):
            found.append(skill)
    return list(set(found))

# Apply skill extraction
df['extracted_skills'] = df['description'].apply(extract_skills)

# Save output
df.to_csv(csv_out, index=False)
print(f"✅ Skills extracted and saved to {csv_out}")
