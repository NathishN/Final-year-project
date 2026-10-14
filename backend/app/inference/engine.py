import os
import sys
import time
import io
import uuid
import base64
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List, Union
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

# Ensure repository root is importable for research model modules
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from model import build_convnext_model
from phase50_casm_v2_model import CASMv2Model

try:
    from app.core.config import (
        BACKBONE_CHECKPOINT,
        CORAL_CHECKPOINT,
        SUPPORTED_SPECIES,
        GRADE_NAMES,
        GRADE_DESCRIPTIONS,
        IMAGE_SIZE,
        IMAGE_MEAN,
        IMAGE_STD,
        DEFAULT_OOD_THRESHOLD,
        DEFAULT_ABSTENTION_THRESHOLD,
    )
except ImportError:
    from backend.app.core.config import (
        BACKBONE_CHECKPOINT,
        CORAL_CHECKPOINT,
        SUPPORTED_SPECIES,
        GRADE_NAMES,
        GRADE_DESCRIPTIONS,
        IMAGE_SIZE,
        IMAGE_MEAN,
        IMAGE_STD,
        DEFAULT_OOD_THRESHOLD,
        DEFAULT_ABSTENTION_THRESHOLD,
    )

class FruitInferenceEngine:
    _instance: Optional["FruitInferenceEngine"] = None

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.backbone = None
        self.coral_head = None
        self.is_loaded = False
        self.load_error: Optional[str] = None

        self.transform = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=list(IMAGE_MEAN), std=list(IMAGE_STD))
        ])

    @classmethod
    def get_instance(cls) -> "FruitInferenceEngine":
        if cls._instance is None:
            cls._instance = FruitInferenceEngine()
        return cls._instance

    def is_ready(self) -> bool:
        return self.is_loaded and (self.backbone is not None) and (self.coral_head is not None)

    def load_models(self):
        if self.is_ready():
            return
            
        try:
            # 1. Load ConvNeXt-Tiny backbone for feature extraction & species prediction
            self.backbone = build_convnext_model(num_classes=len(SUPPORTED_SPECIES), pretrained=False, device=self.device)
            if os.path.exists(BACKBONE_CHECKPOINT):
                ckpt = torch.load(BACKBONE_CHECKPOINT, map_location=self.device, weights_only=False)
                state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
                self.backbone.load_state_dict(state_dict)
            else:
                raise FileNotFoundError(f"Backbone checkpoint not found at: {BACKBONE_CHECKPOINT}")
            self.backbone.eval()

            # 2. Load Config A Monotonic CORAL head (Seed 83 Best Exemplar, 626,180 params)
            self.coral_head = CASMv2Model(vis_dim=768, use_species=False, use_ood=False, num_classes=5).to(self.device)
            if os.path.exists(CORAL_CHECKPOINT):
                sd = torch.load(CORAL_CHECKPOINT, map_location=self.device, weights_only=False)
                self.coral_head.load_state_dict(sd)
            else:
                raise FileNotFoundError(f"Monotonic CORAL head checkpoint not found at: {CORAL_CHECKPOINT}")
            self.coral_head.eval()

            self.is_loaded = True
            self.load_error = None
        except Exception as e:
            self.is_loaded = False
            self.load_error = str(e)
            print(f"[FruitInferenceEngine] Model loading warning/error: {e}")

    @torch.inference_mode()
    def predict(
        self,
        image_input: Union[Image.Image, bytes],
        original_filename: Optional[str] = None,
        confidence_threshold: Optional[float] = None
    ) -> Dict[str, Any]:
        if not self.is_ready():
            self.load_models()
            if not self.is_ready():
                raise RuntimeError(f"Models are not initialized: {self.load_error}")

        conf_thresh = confidence_threshold or DEFAULT_ABSTENTION_THRESHOLD
        start_time = time.perf_counter()

        # Handle PIL Image or bytes input
        if isinstance(image_input, Image.Image):
            pil_image = image_input.convert("RGB")
        else:
            try:
                pil_image = Image.open(io.BytesIO(image_input)).convert("RGB")
            except Exception as e:
                raise ValueError(f"Corrupt or unreadable image stream: {e}")

        # Create thumbnail for local history preview (bounded size, no disk clutter)
        thumb = pil_image.copy()
        thumb.thumbnail((160, 160))
        thumb_buf = io.BytesIO()
        thumb.save(thumb_buf, format="JPEG", quality=80)
        thumb_b64 = base64.b64encode(thumb_buf.getvalue()).decode("utf-8")

        # Preprocess tensor
        tensor = self.transform(pil_image).unsqueeze(0).to(self.device)

        with torch.amp.autocast("cuda", enabled=(self.device == "cuda")):
            # 1. Forward species classification
            species_logits = self.backbone(tensor)
            species_probs = torch.softmax(species_logits, dim=-1).squeeze(0).cpu().numpy()

            # 2. Extract 768-D visual feature embedding
            features = self.backbone.extract_features(tensor)
            if len(features.shape) == 4:
                features = torch.mean(features, dim=[2, 3])

            # 3. Forward Monotonic CORAL ordinal grading head
            ordinal_logits = self.coral_head(features)
            discrete_probs = self.coral_head.predict_class_probs(ordinal_logits).squeeze(0).cpu().numpy()
            predicted_grade = int(self.coral_head.predict_rank(ordinal_logits).item())
            
            # Cumulative probabilities: sigmoid(logits)
            cum_probs = torch.sigmoid(ordinal_logits).squeeze(0).cpu().numpy().tolist()

            # 4. Out-of-Distribution (OOD) Scoring via Free Energy: E(x) = -logsumexp(z(x))
            energy_score = float(-torch.logsumexp(species_logits, dim=-1).item())

        end_time = time.perf_counter()
        latency_ms = round((end_time - start_time) * 1000.0, 2)

        # Interpret species prediction
        species_idx = int(np.argmax(species_probs))
        species_conf = float(species_probs[species_idx])
        species_pred = SUPPORTED_SPECIES[species_idx]
        species_prob_dict = {
            s: round(float(p), 4) for s, p in zip(SUPPORTED_SPECIES, species_probs)
        }

        # Interpret ordinal severity grade
        grade_conf = float(discrete_probs[predicted_grade])
        grade_name = GRADE_NAMES[predicted_grade]
        grade_desc = GRADE_DESCRIPTIONS[predicted_grade]

        # OOD assessment: Energy score > threshold OR very low species confidence (<35%)
        is_ood = (energy_score > DEFAULT_OOD_THRESHOLD) or (species_conf < 0.35)
        if is_ood:
            ood_interp = (
                f"Flagged as Out-of-Distribution or visual anomaly (Free Energy: {energy_score:.2f}, Threshold: {DEFAULT_OOD_THRESHOLD:.2f}). "
                "The image deviates significantly from training distributions. Automated grading should not be trusted."
            )
        else:
            ood_interp = (
                f"Within known distribution boundaries (Free Energy: {energy_score:.2f}, Threshold: {DEFAULT_OOD_THRESHOLD:.2f}). "
                "Confidence metrics represent nominal in-domain expectations."
            )

        # Selective routing & abstention
        abstain = bool((grade_conf < conf_thresh) or is_ood)
        if is_ood:
            routing_action = "REJECT_UNRELIABLE_OOD_ANOMALY"
            routing_explanation = "Sample flagged as out-of-distribution anomaly. Automated grading withheld."
        elif abstain:
            routing_action = "ROUTE_TO_MANUAL_INSPECTION"
            routing_explanation = f"Model grade confidence ({grade_conf*100:.1f}%) is below operational threshold ({conf_thresh*100:.1f}%). Route to human inspector."
        else:
            routing_action = "AUTOMATED_ACCEPT"
            routing_explanation = f"High confidence ({grade_conf*100:.1f}%) in-distribution assessment passed all safety checks."

        analysis_id = str(uuid.uuid4())
        timestamp = datetime.now(timezone.utc).isoformat()

        limitations = [
            "Research benchmark: Monotonic CORAL Config A achieved QWK = 0.8143 ± 0.0116 on held-out test splits.",
            "Open-set real-image OOD evaluation demonstrated severe domain inversion challenges (FPR95 = 1.0000 on challenging transfer datasets).",
            "This application is an academic research demonstration, not a commercial food safety or USDA/EU certified inspection system.",
            "Visual artifacts, severe lighting shifts, and unknown exotic fruit varieties degrade grading reliability."
        ]

        return {
            "analysis_id": analysis_id,
            "timestamp": timestamp,
            "species": {
                "species": species_pred,
                "confidence": round(species_conf, 4),
                "probabilities": species_prob_dict,
            },
            "ordinal_grade": {
                "grade": predicted_grade,
                "grade_name": grade_name,
                "grade_description": grade_desc,
                "confidence": round(grade_conf, 4),
                "cumulative_probabilities": [round(float(c), 4) for c in cum_probs],
                "discrete_probabilities": [round(float(p), 4) for p in discrete_probs],
            },
            "ood": {
                "detector": "Free Energy Inverted Scoring",
                "energy_score": round(energy_score, 4),
                "threshold": DEFAULT_OOD_THRESHOLD,
                "is_ood": is_ood,
                "interpretation": ood_interp,
            },
            "selective_routing": {
                "abstain_from_decision": abstain,
                "routing_action": routing_action,
                "explanation": routing_explanation,
            },
            "inference_latency_ms": latency_ms,
            "device_used": self.device,
            "thumbnail_base64": thumb_b64,
            "limitations": limitations,
        }

def get_engine() -> FruitInferenceEngine:
    engine = FruitInferenceEngine.get_instance()
    if not engine.is_ready():
        engine.load_models()
    return engine
