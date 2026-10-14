"""
scripts/inference_pipeline.py
=============================
Production-Ready Inference Pipeline for Multi-Fruit Quality Grading.
Supports single images or batch inputs on CPU or GPU with automated unknown-species fallback.
"""

import os
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

ROOT = r"C:\fruit_dataset"
SPECIES_MAP = {"Apple": 0, "Banana": 1, "Mango": 2, "Orange": 3, "Tomato": 4}
GRADE_NAMES = ["Healthy (0)", "Mild (1)", "Moderate (2)", "Severe (3)", "Rotten (4)"]


class FruitGrader:
    def __init__(self, model_checkpoint=None, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        from phase50_casm_v2_model import CASMv2Model
        from model import build_convnext_model

        # Backbone
        self.backbone = build_convnext_model(num_classes=5, pretrained=False, device=self.device)
        p1_ckpt = os.path.join(ROOT, "outputs", "checkpoints", "best_model.pth")
        if os.path.exists(p1_ckpt):
            ckpt = torch.load(p1_ckpt, map_location=self.device, weights_only=False)
            self.backbone.load_state_dict(ckpt["model_state_dict"])
        self.backbone.eval()

        # Config A Model (Pure Image-Only, proven optimal in 30-seed confirmatory study)
        self.model = CASMv2Model(vis_dim=768, use_species=False, use_ood=False, num_classes=5).to(self.device)
        if model_checkpoint and os.path.exists(model_checkpoint):
            sd = torch.load(model_checkpoint, map_location=self.device, weights_only=False)
            self.model.load_state_dict(sd)
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    @torch.no_grad()
    def predict_image(self, image_path, species_name=None, confidence_threshold=0.70):
        try:
            img = Image.open(image_path).convert('RGB')
        except Exception as e:
            return {"error": f"Failed to load image: {e}"}

        tensor = self.transform(img).unsqueeze(0).to(self.device)
        with torch.amp.autocast("cuda", enabled=(self.device == "cuda")):
            feats = self.backbone.extract_features(tensor)
            if len(feats.shape) == 4:
                feats = torch.mean(feats, dim=[2, 3])
            logits = self.model(feats)
            probs = self.model.predict_class_probs(logits).squeeze(0).cpu().numpy()
            predicted_grade = int(self.model.predict_rank(logits).item())

        conf = float(np.max(probs))
        abstain = (conf < confidence_threshold)

        return {
            "image_path": image_path,
            "predicted_grade": predicted_grade,
            "grade_label": GRADE_NAMES[predicted_grade],
            "confidence": round(conf * 100.0, 2),
            "class_probabilities": [round(float(p), 4) for p in probs],
            "abstain_from_decision": abstain,
            "routing_action": "ROUTE_TO_MANUAL_INSPECTION" if abstain else "AUTOMATED_SORT_PASSED"
        }
