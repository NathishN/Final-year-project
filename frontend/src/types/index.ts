export interface SpeciesPrediction {
  species: string;
  confidence: number;
  probabilities: Record<string, number>;
}

export interface OrdinalGradePrediction {
  grade: number;
  grade_name: string;
  grade_description: string;
  confidence: number;
  cumulative_probabilities: number[];
  discrete_probabilities: number[];
}

export interface OODAssessment {
  detector: string;
  energy_score: number;
  threshold: number;
  is_ood: boolean;
  interpretation: string;
}

export interface SelectiveRouting {
  abstain_from_decision: boolean;
  routing_action: 'AUTOMATED_ACCEPT' | 'ROUTE_TO_MANUAL_INSPECTION' | 'REJECT_UNRELIABLE_OOD_ANOMALY' | string;
  explanation: string;
}

export interface PredictionResponse {
  analysis_id: string;
  timestamp: string;
  species: SpeciesPrediction;
  ordinal_grade: OrdinalGradePrediction;
  ood: OODAssessment;
  selective_routing: SelectiveRouting;
  inference_latency_ms: number;
  device_used: string;
  thumbnail_base64?: string;
  limitations: string[];
}

export interface GradeDefinition {
  grade: number;
  name: string;
  description: string;
}

export interface CheckpointInfo {
  name: string;
  path: string;
  loaded: boolean;
  parameter_count: number;
}

export interface ResearchMetrics {
  benchmark_name: string;
  image_only_qwk_mean: number;
  image_only_qwk_std: number;
  n_seeds: number;
  spearman_rho: number;
  accuracy: number;
  ood_fpr95_inverted_real: number;
  ood_auroc_inverted_real: number;
}

export interface ModelInfoResponse {
  backbone_architecture: string;
  coral_head_type: string;
  num_grades: number;
  grades: GradeDefinition[];
  supported_species: string[];
  input_resolution: number[];
  input_normalization: {
    mean: number[];
    std: number[];
  };
  ood_detector: string;
  ood_threshold: number;
  abstention_threshold: number;
  loaded_checkpoints: Record<string, CheckpointInfo>;
  device: string;
  model_ready: boolean;
  research_evaluation: ResearchMetrics;
  documentation_url: string;
}

export interface HistoryItem {
  id: string;
  timestamp: string;
  filename?: string;
  species: string;
  species_confidence: number;
  grade: number;
  grade_name: string;
  grade_confidence: number;
  is_ood: boolean;
  energy_score: number;
  routing_action: string;
  latency_ms: number;
  thumbnail_base64?: string;
}

export interface HistoryDetail extends HistoryItem {
  discrete_probabilities?: number[];
  device?: string;
}

export interface HistoryListResponse {
  total: number;
  items: HistoryItem[];
}

export interface SystemInfoResponse {
  application_name: string;
  version: string;
  status: string;
  python_version: string;
  pytorch_version: string;
  device_available: string;
  gpu_name?: string;
  gpu_memory_allocated_mb?: number;
  gpu_memory_reserved_mb?: number;
  max_upload_size_mb: number;
  allowed_extensions: string[];
  model_loaded: boolean;
}

export interface HealthResponse {
  status: string;
  service: string;
  version: string;
  model_ready: boolean;
  device: string;
}
