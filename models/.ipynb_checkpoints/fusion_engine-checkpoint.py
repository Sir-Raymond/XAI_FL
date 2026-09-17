# fusion_engine.py
import numpy as np

def compute_multimodal_crop_stress(leaf_disease_prob, env_stress_prob, w_leaf=0.6, w_env=0.4):
    """
    Combines leaf image disease confidence with tabular environmental risk 
    to generate an integrated crop stress index.
    """
    multimodal_score = (w_leaf * leaf_disease_prob) + (w_env * env_stress_prob)
    
    if multimodal_score >= 0.70:
        risk_level = "CRITICAL STRESS"
        action = "Immediate chemical/organic intervention and automated irrigation required."
    elif multimodal_score >= 0.40:
        risk_level = "MODERATE STRESS"
        action = "Increase field monitoring and balance soil nutrient levels."
    else:
        risk_level = "LOW RISK"
        action = "Crop health is optimal. Maintain current irrigation schedule."
        
    return {
        "multimodal_score": float(multimodal_score),
        "risk_level": risk_level,
        "recommended_action": action
    }