import json
import os

def load_label_mapping(model_path: str):
    """
    Charge le mapping des labels depuis le fichier de configuration du modèle
    ou retourne un mapping par défaut
    """
    config_path = os.path.join(model_path, "config.json")
    
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            config = json.load(f)
            
        # Si id2label existe dans la config
        if "id2label" in config:
            return {int(k): v for k, v in config["id2label"].items()}
    
    # Mapping par défaut basé sur le dataset Kaggle IT Service Ticket
    return {
        0: "Hardware",
        1: "HR Support", 
        2: "Access",
        3: "Miscellaneous",
        4: "Storage",
        5: "Purchase",
        6: "Network",
        7: "Software"
    }

def get_model_info(model_path: str):
    """Récupère les informations du modèle"""
    config_path = os.path.join(model_path, "config.json")
    
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            config = json.load(f)
        
        return {
            "model_type": config.get("model_type", "unknown"),
            "num_labels": config.get("num_labels", 0),
            "max_position_embeddings": config.get("max_position_embeddings", 512),
            "hidden_size": config.get("hidden_size", 768)
        }
    
    return {}