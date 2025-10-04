import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
from sentence_transformers import SentenceTransformer
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import List, Tuple, Dict, Optional, Set
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import docx2txt
import PyPDF2
import re
import pandas as pd
import json
import logging
from datetime import datetime
from collections import defaultdict, Counter
import networkx as nx
import spacy

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class IndustrySkillDatabase:
    """Comprehensive industry skill taxonomy with India-specific enhancements"""
    def __init__(self, db_path: str = "industry_skills.json"):
        self.db_path = db_path
        self.skill_taxonomy = self._load_skill_taxonomy()
        self.industry_mappings = self._load_industry_mappings()
        self.skill_evolution_trends = self._load_skill_trends()
        self.job_market_data = self._load_job_market_data()

    def _load_skill_taxonomy(self) -> Dict:
        """Enhanced taxonomy with India-specific skills"""
        taxonomy = {
            "technical_skills": {
                "programming": {
                    "languages": ["python", "java", "c++", "javascript", "kotlin", "hindi-nlp", "tamil-nlp"],
                    "frameworks": ["react", "django", "flask", "spring", "angular"],
                    "cloud": ["aws", "azure", "google cloud", "tcs cloud"],
                    "databases": ["mysql", "postgresql", "mongodb", "oracle"]
                },
                "data_science": {
                    "languages": ["python", "r", "sql", "sas"],
                    "libraries": ["numpy", "scikit-learn", "tensorflow", "pytorch", "h2o"],
                    "tools": ["tableau", "power bi", "qlikview"]
                }
            },
            "domain_skills": {
                "finance": ["risk management", "financial modeling", "rbi guidelines"],
                "healthcare": ["telemedicine", "clinical research", "ayush"],
                "marketing": ["seo", "digital marketing", "indian market trends"]
            },
            "soft_skills": {
                "communication": ["public speaking", "technical writing", "bilingual communication"],
                "analytical": ["problem solving", "critical thinking", "data interpretation"]
            }
        }
        return taxonomy

    def _load_industry_mappings(self) -> Dict:
        """India-specific industry mappings with weights"""
        return {
            "technology": {
                "hot_skills": ["machine learning", "cloud computing", "cybersecurity", "devops", "aiops"],
                "emerging_skills": ["edge computing", "blockchain", "ar/vr", "5g"],
                "skill_weights": {"technical_skills": 0.7, "soft_skills": 0.2, "domain_skills": 0.1},
            },
            "finance": {
                "hot_skills": ["fintech", "blockchain", "regtech"],
                "emerging_skills": ["defi", "esg investing", "digital payments"],
                "skill_weights": {"technical_skills": 0.4, "soft_skills": 0.3, "domain_skills": 0.3}
            }
        }

    def _load_skill_trends(self) -> Dict:
        """Updated trends with Indian market focus"""
        return {
            "trending_up": {
                "artificial intelligence": {"growth_rate": 0.40, "market_demand": "very_high"},
                "cloud computing": {"growth_rate": 0.35, "market_demand": "high"},
                "digital payments": {"growth_rate": 0.30, "market_demand": "high"}
            },
            "stable": {"sql": {"growth_rate": 0.03, "market_demand": "medium"}}
        }

    def _load_job_market_data(self) -> Dict:
        """India-specific salary and demand data"""
        return {
            "salary_ranges": {
                "artificial intelligence": {"entry": 600000, "mid": 1200000, "senior": 2000000},
                "cloud computing": {"entry": 500000, "mid": 1000000, "senior": 1800000}
            },
            "job_availability": {
                "high_demand": ["python", "aws", "machine learning", "react", "digital marketing"],
                "medium_demand": ["java", "sql", "project management"],
                "low_demand": ["perl", "fortran"]
            }
        }

class AdvancedSkillSpanDataset(Dataset):
    """Enhanced dataset with contextual embeddings"""
    def __init__(self, texts: List[str], labels: List[List[str]], tokenizer, max_len=512, industry_context: Optional[str] = None):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.industry_context = industry_context

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        if self.industry_context:
            text = f"[{self.industry_context}] {text}"
        encoding = self.tokenizer(text, max_length=self.max_len, padding='max_length', truncation=True, return_tensors="pt")
        label_ids = [1 if any(s in token.lower() for s in self.labels[idx]) else 0 for token in text.split()]
        label_ids = [0] * (self.max_len - len(label_ids)) + label_ids[:self.max_len]
        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": torch.tensor(label_ids)
        }

class IndustrySkillExtractor:
    """Advanced skill extraction with industry-level analysis"""
    def __init__(self, model_name="dslim/bert-base-NER", sentence_model='all-MiniLM-L6-v2', spacy_model="en_core_web_sm"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForTokenClassification.from_pretrained(model_name, num_labels=3)
        self.bi_encoder = SentenceTransformer(sentence_model)
        self.skill_db = IndustrySkillDatabase()
        self.skill_embeddings_cache = {}
        self._build_skill_embeddings()
        try:
            self.nlp = spacy.load(spacy_model)
        except OSError:
            logger.warning(f"Spacy model {spacy_model} not found. Install with: python -m spacy download {spacy_model}")
            self.nlp = None
        self.skill_graph = self._build_skill_graph()

    def _build_skill_embeddings(self):
        all_skills = []
        for category in self.skill_db.skill_taxonomy.values():
            if isinstance(category, dict):
                for subcategory in category.values():
                    if isinstance(subcategory, dict):
                        for skill_list in subcategory.values():
                            all_skills.extend(skill_list)
                    elif isinstance(subcategory, list):
                        all_skills.extend(subcategory)
        embeddings = self.bi_encoder.encode(all_skills)
        self.skill_embeddings_cache = dict(zip(all_skills, embeddings))

    def _build_skill_graph(self) -> nx.Graph:
        G = nx.Graph()
        for category_name, category in self.skill_db.skill_taxonomy.items():
            if isinstance(category, dict):
                for subcategory_name, subcategory in category.items():
                    if isinstance(subcategory, dict):
                        for skill_type, skills in subcategory.items():
                            for skill in skills:
                                G.add_node(skill, category=category_name, subcategory=subcategory_name, type=skill_type)
                    elif isinstance(subcategory, list):
                        for skill in subcategory:
                            G.add_node(skill, category=category_name, subcategory=subcategory_name)
        skills = list(G.nodes())
        for i, skill1 in enumerate(skills):
            for skill2 in skills[i+1:]:
                if skill1 in self.skill_embeddings_cache and skill2 in self.skill_embeddings_cache:
                    similarity = cosine_similarity(
                        [self.skill_embeddings_cache[skill1]], [self.skill_embeddings_cache[skill2]]
                    )[0][0]
                    if similarity > 0.7:
                        G.add_edge(skill1, skill2, weight=similarity)
        return G

    def _extract_ner_skills(self, text: str) -> List[str]:
        """Extract skills using NER"""
        nlp = self.nlp if self.nlp else spacy.load("en_core_web_sm")
        doc = nlp(text)
        return [ent.text for ent in doc.ents if ent.label_ in ["SKILL", "TECH"]]

    def _extract_semantic_skills(self, text: str, industry: Optional[str] = None) -> List[str]:
        """Extract semantically similar skills"""
        text_emb = self.bi_encoder.encode([text])
        skills = list(self.skill_embeddings_cache.keys())
        skill_embs = np.array(list(self.skill_embeddings_cache.values()))
        similarities = cosine_similarity(text_emb, skill_embs)[0]
        return [skills[i] for i in np.where(similarities > 0.7)[0]]

    def _extract_contextual_skills(self, text: str, industry: Optional[str] = None) -> List[str]:
        """Extract context-aware skills"""
        if industry and industry in self.skill_db.industry_mappings:
            return [s for s in self.skill_db.industry_mappings[industry]["hot_skills"] if s in text.lower()]
        return []

    def _extract_pattern_skills(self, text: str) -> List[str]:
        """Extract skills using regex patterns"""
        patterns = [r"\b\w+(?:\s+\w+)*(?:programming|framework|tool)\b"]
        return [match.group() for pattern in patterns for match in re.finditer(pattern, text.lower())]

    def _score_skills(self, skills: List[str], text: str, industry: Optional[str] = None) -> Dict[str, float]:
        """Score skills based on context and demand"""
        scores = {}
        text_lower = text.lower()
        for skill in skills:
            base_score = 50
            if skill in text_lower:
                base_score += 30
            if industry and skill in self.skill_db.industry_mappings.get(industry, {}).get("hot_skills", []):
                base_score += 20
            scores[skill] = min(base_score, 100)
        return scores

    def _analyze_skill_clusters(self, skills: List[str]) -> Dict:
        """Cluster skills for advanced analysis"""
        if not skills:
            return {}
        embeddings = self.bi_encoder.encode(skills)
        scaler = StandardScaler()
        scaled_embs = scaler.fit_transform(embeddings)
        kmeans = KMeans(n_clusters=min(3, len(skills)), random_state=42)
        clusters = kmeans.fit_predict(scaled_embs)
        return {i: [skills[j] for j in range(len(skills)) if clusters[j] == i] for i in range(kmeans.n_clusters)}

    def _analyze_industry_fit(self, skills: List[str], industry: Optional[str] = None) -> Dict:
        """Analyze industry fit with weights"""
        if not industry or industry not in self.skill_db.industry_mappings:
            return {}
        weights = self.skill_db.industry_mappings[industry]["skill_weights"]
        fit_score = sum(weights.get(cat, 0) * len([s for s in skills if any(s in sub for sub in cat.values())])
                       for cat in self.skill_db.skill_taxonomy)
        return {"fit_score": min(fit_score * 100, 100), "industry": industry}

    def _analyze_market_value(self, skills: List[str]) -> Dict:
        """Analyze market value based on trends"""
        market_data = self.skill_db.job_market_data
        value = {"high_demand": [], "medium_demand": [], "low_demand": []}
        for skill in skills:
            if skill in market_data["job_availability"]["high_demand"]:
                value["high_demand"].append(skill)
            elif skill in market_data["job_availability"]["medium_demand"]:
                value["medium_demand"].append(skill)
            elif skill in market_data["job_availability"]["low_demand"]:
                value["low_demand"].append(skill)
        return value

    def _identify_skill_gaps(self, skills: List[str], industry: Optional[str] = None) -> List[str]:
        """Identify skill gaps based on industry requirements"""
        if not industry or industry not in self.skill_db.industry_mappings:
            return []
        required = self.skill_db.industry_mappings[industry]["hot_skills"] + self.skill_db.industry_mappings[industry]["emerging_skills"]
        return [s for s in required if s not in skills]

    def _generate_skill_recommendations(self, skills: List[str], industry: Optional[str] = None) -> List[Dict]:
        """Generate personalized skill recommendations"""
        if not industry or industry not in self.skill_db.industry_mappings:
            return []
        gaps = self._identify_skill_gaps(skills, industry)
        return [{"skill": gap, "priority": 80, "resource": f"Learn {gap} on Coursera"} for gap in gaps[:3]]

    def extract_skills_advanced(self, text: str, industry: Optional[str] = None) -> Dict:
        """Advanced skill extraction pipeline"""
        basic_skills = self._extract_ner_skills(text)
        semantic_skills = self._extract_semantic_skills(text, industry)
        contextual_skills = self._extract_contextual_skills(text, industry)
        pattern_skills = self._extract_pattern_skills(text)
        all_skills = list(set(basic_skills + semantic_skills + contextual_skills + pattern_skills))
        return {
            "extracted_skills": all_skills,
            "skill_scores": self._score_skills(all_skills, text, industry),
            "skill_clusters": self._analyze_skill_clusters(all_skills),
            "industry_analysis": self._analyze_industry_fit(all_skills, industry),
            "market_analysis": self._analyze_market_value(all_skills),
            "skill_gaps": self._identify_skill_gaps(all_skills, industry),
            "recommendations": self._generate_skill_recommendations(all_skills, industry)
        }

if __name__ == "__main__":
    extractor = IndustrySkillExtractor()
    sample_text = "Final-year student with Python, machine learning, and cloud computing skills."
    print(extractor.extract_skills_advanced(sample_text, "technology"))