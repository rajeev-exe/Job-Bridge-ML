import pandas as pd
import os
import random

# Path to save the CSV
COURSES_CSV_FILE = r"D:\GitHub\Job-Bridge-ML\Data\courses.csv"

# Example skill categories and technologies
skill_categories = [
    "Python", "Java", "C++", "C#", "JavaScript", "TypeScript", "R", "SQL", "NoSQL",
    "Machine Learning", "Deep Learning", "Data Science", "AI", "Blockchain",
    "Robotics", "IoT", "Cloud Computing", "AWS", "Azure", "GCP", "5G", "Metaverse",
    "AR", "VR", "Quantum Computing", "React", "Angular", "Vue.js", "Node.js",
    "Django", "Flask", "Spring Boot", "Microservices", "Docker", "Kubernetes",
    "Git", "CI/CD", "DevOps", "Tableau", "Power BI", "Excel", "Hadoop", "Spark",
    "Kafka", "Unity", "Unreal Engine", "TensorFlow", "PyTorch", "OpenCV", "NLP"
]

# Generate 1000 courses by combining skill names with course types
courses_data = []
for i in range(1000):
    skill = random.choice(skill_categories)
    course_name = f"{skill} Fundamentals {i+1}"
    provider = random.choice(["Coursera", "edX", "Udemy", "Pluralsight", "Skillshare", "LinkedIn Learning"])
    duration_weeks = random.randint(2, 12)
    
    courses_data.append({
        "course_name": course_name,
        "skills": skill,
        "provider": provider,
        "duration_weeks": duration_weeks
    })

# Create DataFrame
df_courses = pd.DataFrame(courses_data)

# Ensure directory exists
os.makedirs(os.path.dirname(COURSES_CSV_FILE), exist_ok=True)

# Save CSV
df_courses.to_csv(COURSES_CSV_FILE, index=False)

print(f"Generated {len(df_courses)} courses in '{COURSES_CSV_FILE}' with proper 'skills' column.")
