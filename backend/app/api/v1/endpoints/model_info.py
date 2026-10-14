from fastapi import APIRouter
from app.schemas.model_info import (
    ModelInfoResponse,
    GradeDefinition,
    CheckpointInfo,
    ResearchMetrics,
)
from app.inference.engine import get_engine
from app.core.config import (
    BACKBONE_NAME,
    CORAL_HEAD_TYPE,
    GRADE_NAMES,
    GRADE_DESCRIPTIONS,
    SUPPORTED_SPECIES,
    IMAGE_SIZE,
    IMAGE_MEAN,
    IMAGE_STD,
    DEFAULT_OOD_THRESHOLD,
    DEFAULT_ABSTENTION_THRESHOLD,
    BACKBONE_CHECKPOINT,
    CORAL_CHECKPOINT,
)

router = APIRouter()

@router.get("/model-info", response_model=ModelInfoResponse)
async def get_model_info():
    engine = get_engine()
    ready = engine.is_ready()
    
    # Calculate parameter count if model is loaded
    param_count = 0
    if ready:
        if engine.backbone is not None:
            param_count += sum(p.numel() for p in engine.backbone.parameters())
        if engine.coral_head is not None:
            param_count += sum(p.numel() for p in engine.coral_head.parameters())
            
    grades = [
        GradeDefinition(
            grade=i,
            name=GRADE_NAMES[i],
            description=GRADE_DESCRIPTIONS[i]
        )
        for i in range(len(GRADE_NAMES))
    ]
    
    # Checkpoint information with sanitized relative names
    checkpoints = {
        "backbone": CheckpointInfo(
            name="ConvNeXt-Tiny Species Classifier Backbone",
            path="outputs/checkpoints/best_model.pth",
            loaded=engine.backbone is not None,
            parameter_count=param_count
        ),
        "coral_head": CheckpointInfo(
            name="Monotonic CORAL Ordinal Head (Config A, Seed 83)",
            path="outputs/casm_v2/phase50_checkpoints/model_A_seed83_best.pth",
            loaded=engine.coral_head is not None,
            parameter_count=626180
        )
    }
    
    # Authoritative Research Evaluation Metrics from Phase 5.7 / Phase 5.8 / Phase 6.0 lock
    # Confirmatory 30-seed Image-Only baseline (Config A): QWK = 0.8143 ± 0.0116
    research_metrics = ResearchMetrics(
        benchmark_name="30-Seed Confirmatory Image-Only Baseline (Config A)",
        image_only_qwk_mean=0.8143,
        image_only_qwk_std=0.0116,
        n_seeds=30,
        spearman_rho=0.8520,
        accuracy=0.7410,
        ood_fpr95_inverted_real=1.0000,
        ood_auroc_inverted_real=0.2882
    )
    
    return ModelInfoResponse(
        backbone_architecture=BACKBONE_NAME,
        coral_head_type=CORAL_HEAD_TYPE,
        num_grades=len(GRADE_NAMES),
        grades=grades,
        supported_species=SUPPORTED_SPECIES,
        input_resolution=[IMAGE_SIZE, IMAGE_SIZE],
        input_normalization={"mean": list(IMAGE_MEAN), "std": list(IMAGE_STD)},
        ood_detector="Free Energy Inverted Scoring",
        ood_threshold=DEFAULT_OOD_THRESHOLD,
        abstention_threshold=DEFAULT_ABSTENTION_THRESHOLD,
        loaded_checkpoints=checkpoints,
        device=engine.device,
        model_ready=ready,
        research_evaluation=research_metrics,
        documentation_url="https://github.com/NathishN/Final-year-project"
    )
