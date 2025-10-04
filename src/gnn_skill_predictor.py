import torch
import torch.nn.functional as F
from torch_geometric.nn import GINConv, global_add_pool
from torch_geometric.data import Data
import torch.nn as nn
from typing import List, Dict
from sentence_transformers import SentenceTransformer
import numpy as np
from advanced_skill_extractor import IndustrySkillExtractor
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Sentence-transformer model for embedding skills semantically
bi_encoder = SentenceTransformer('all-MiniLM-L6-v2')

class GINXMLC(nn.Module):
    """Advanced GNN for skill prediction with dynamic ontology"""
    def __init__(self, input_dim=384, hidden_dim=128, num_skills=100):
        super().__init__()
        nn1 = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))
        self.conv1 = GINConv(nn1)
        self.conv2 = GINConv(nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim)))
        self.classifier = nn.Linear(hidden_dim, num_skills)

    def forward(self, x, edge_index, batch):
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = global_add_pool(x, batch)
        return torch.sigmoid(self.classifier(x))

def graph_dict_to_data(graph: Dict, ontology: List[str]) -> Data:
    """Convert dictionary to PyG Data with enhanced node features"""
    try:
        nodes = [str(node) for node in graph.keys()]
        if not nodes:
            raise ValueError("Graph dictionary is empty")
        node_to_idx = {node: idx for idx, node in enumerate(nodes)}
        extractor = IndustrySkillExtractor()
        embeddings = extractor.bi_encoder.encode(nodes)
        x = torch.tensor(embeddings, dtype=torch.float)
        edge_index = []
        for node, neighbors in graph.items():
            node_str = str(node)
            for neigh in neighbors:
                neigh_str = str(neigh)
                if node_str in node_to_idx and neigh_str in node_to_idx:
                    edge_index.append([node_to_idx[node_str], node_to_idx[neigh_str]])
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous() if edge_index else torch.empty((2, 0), dtype=torch.long)
        batch = torch.zeros(len(nodes), dtype=torch.long)
        return Data(x=x, edge_index=edge_index, batch=batch)
    except Exception as e:
        logger.error(f"Error converting graph to data: {str(e)}")
        raise

def predict_missing_skills(model: GINXMLC, graph_data: Data, known_skills: List[str], ontology: List[str], confidence_threshold: float = 0.65) -> List[str]:
    """Predict missing skills with configurable confidence filtering"""
    try:
        if not known_skills or not ontology:
            logger.warning("Empty known_skills or ontology provided")
            return []
        model.eval()
        with torch.no_grad():
            out = model(graph_data.x, graph_data.edge_index, graph_data.batch)
            scores = out[0]
            top_indices = torch.topk(scores, k=min(10, len(ontology))).indices.tolist()
            predicted = [ontology[i % len(ontology)] for i in top_indices]
            skill_confidence = {s: score for s, score in zip(predicted, scores[top_indices].tolist())}
            missing_skills = [p for p in predicted if p.lower() not in [s.lower() for s in known_skills] and skill_confidence[p] > confidence_threshold]
            logger.info(f"Predicted {len(missing_skills)} missing skills with confidence > {confidence_threshold}")
            return missing_skills
    except Exception as e:
        logger.error(f"Error predicting missing skills: {str(e)}")
        return []

if __name__ == "__main__":
    try:
        model = GINXMLC(num_skills=100)
        print("GNN ready with enhanced ontology.")
    except Exception as e:
        logger.error(f"Error initializing GNN model: {str(e)}")