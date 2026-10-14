import React from 'react';
import { 
  BrainCircuit, 
  Layers, 
  Scale, 
  ShieldAlert, 
  CheckCircle2, 
  FileCode, 
  Sparkles, 
  Cpu, 
  ExternalLink,
  BookOpen,
  Sliders,
  AlertCircle
} from 'lucide-react';
import { ModelInfoResponse } from '../types';

interface ModelInfoProps {
  modelInfo: ModelInfoResponse | null;
}

export const ModelInfo: React.FC<ModelInfoProps> = ({ modelInfo }) => {
  return (
    <div className="space-y-10 max-w-5xl mx-auto">
      
      {/* Header */}
      <div>
        <h1 className="text-3xl font-extrabold text-white tracking-tight">Model Architecture & Empirical Research</h1>
        <p className="text-sm text-slate-400 mt-1">
          Detailed transparency regarding model definitions, weights lineage, mathematical formulation, and verified evaluation benchmarks.
        </p>
      </div>

      {/* Model Overview & Architecture Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        
        {/* Backbone Card */}
        <div className="glass-panel rounded-2xl p-6 space-y-4">
          <div className="flex items-center justify-between">
            <span className="text-xs font-semibold text-slate-400 uppercase tracking-wider">Visual Feature Extractor</span>
            <span className="px-2 py-0.5 rounded bg-emerald-950 text-emerald-300 border border-emerald-800 text-[11px] font-mono">
              27.8M Parameters
            </span>
          </div>
          <div>
            <h3 className="text-xl font-bold text-white">ConvNeXt-Tiny</h3>
            <p className="text-xs text-slate-400 mt-1 leading-relaxed">
              Modernized convolutional hierarchy operating on 7x7 depthwise convolutions and inverted bottleneck blocks. Extracts 768-dimensional penultimate semantic embeddings.
            </p>
          </div>
          <div className="pt-2 border-t border-slate-800 space-y-1.5 text-xs text-slate-300">
            <div className="flex justify-between">
              <span className="text-slate-400">Input Resolution:</span>
              <span className="font-mono">224 × 224 × 3 (RGB)</span>
            </div>
            <div className="flex justify-between">
              <span className="text-slate-400">Normalization:</span>
              <span className="font-mono">ImageNet Mean & Std</span>
            </div>
            <div className="flex justify-between">
              <span className="text-slate-400">Checkpoint:</span>
              <span className="font-mono text-emerald-400">outputs/checkpoints/best_model.pth</span>
            </div>
          </div>
        </div>

        {/* Head Card */}
        <div className="glass-panel rounded-2xl p-6 space-y-4">
          <div className="flex items-center justify-between">
            <span className="text-xs font-semibold text-emerald-400 uppercase tracking-wider">Ordinal Regression Head</span>
            <span className="px-2 py-0.5 rounded bg-emerald-950 text-emerald-300 border border-emerald-800 text-[11px] font-mono">
              626,180 Parameters
            </span>
          </div>
          <div>
            <h3 className="text-xl font-bold text-white">Monotonic CORAL (CASM-v2)</h3>
            <p className="text-xs text-slate-400 mt-1 leading-relaxed">
              Monotonically constrained softplus threshold decrements <code className="text-emerald-300">bₖ = bₖ₋₁ - softplus(δₖ)</code> guarantee ordered cumulative probabilities, eliminating rank violation anomalies.
            </p>
          </div>
          <div className="pt-2 border-t border-slate-800 space-y-1.5 text-xs text-slate-300">
            <div className="flex justify-between">
              <span className="text-slate-400">Architecture Configuration:</span>
              <span className="font-semibold text-slate-200">Config A (Image-Only)</span>
            </div>
            <div className="flex justify-between">
              <span className="text-slate-400">Exemplar Seed:</span>
              <span className="font-mono">Seed 83 (Verified Best Checkpoint)</span>
            </div>
            <div className="flex justify-between">
              <span className="text-slate-400">Checkpoint:</span>
              <span className="font-mono text-emerald-400">outputs/casm_v2/phase50_checkpoints/model_A_seed83_best.pth</span>
            </div>
          </div>
        </div>

      </div>

      {/* Verified Empirical Research Benchmarks */}
      <div className="glass-panel rounded-2xl p-8 space-y-6">
        <div className="flex items-center justify-between">
          <div>
            <h3 className="text-xl font-bold text-white">Confirmatory 30-Seed Empirical Benchmark</h3>
            <p className="text-xs text-slate-400 mt-0.5">
              Verified outcomes locked across Phase 5.0, Phase 5.7, Phase 5.8, and Phase 6.0 audit reports.
            </p>
          </div>
          <span className="px-3 py-1 rounded-full bg-cyan-950 text-cyan-300 border border-cyan-800 text-xs font-semibold">
            N = 30 Seeds
          </span>
        </div>

        <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
          <div className="p-4 rounded-xl bg-slate-900/80 border border-slate-800 space-y-1">
            <span className="text-slate-400 text-xs">Image-Only Mean QWK</span>
            <div className="text-2xl font-black text-emerald-400 font-mono">0.8143</div>
            <div className="text-[10px] text-slate-500 font-mono">Std Dev: ± 0.0116</div>
          </div>

          <div className="p-4 rounded-xl bg-slate-900/80 border border-slate-800 space-y-1">
            <span className="text-slate-400 text-xs">Spearman Rank (ρ)</span>
            <div className="text-2xl font-black text-teal-400 font-mono">0.8520</div>
            <div className="text-[10px] text-slate-500">Monotonic rank preservation</div>
          </div>

          <div className="p-4 rounded-xl bg-slate-900/80 border border-slate-800 space-y-1">
            <span className="text-slate-400 text-xs">Classification Accuracy</span>
            <div className="text-2xl font-black text-cyan-400 font-mono">74.10%</div>
            <div className="text-[10px] text-slate-500">Exact 5-tier matching</div>
          </div>

          <div className="p-4 rounded-xl bg-slate-900/80 border border-slate-800 space-y-1">
            <span className="text-slate-400 text-xs">Rank Consistency</span>
            <div className="text-2xl font-black text-white font-mono">100.0%</div>
            <div className="text-[10px] text-slate-500">Zero threshold inversion</div>
          </div>
        </div>

        {/* Real-Image OOD Inversion Alert */}
        <div className="p-4 rounded-xl bg-amber-950/40 border border-amber-800/80 space-y-2 text-xs">
          <div className="flex items-center space-x-2 text-amber-400 font-bold">
            <AlertCircle className="w-4 h-4" />
            <span>Open-Set Real-Image OOD Findings & Transfer Limitations</span>
          </div>
          <p className="text-slate-300 leading-relaxed">
            While synthetic noise benchmarks (Gaussian blur, adversarial perturbations) showed apparent detection capabilities, 
            evaluation on <strong>real-image foreign datasets</strong> revealed severe domain inversion challenges 
            (<span className="font-mono text-amber-300">FPR95 = 1.0000</span>, <span className="font-mono text-amber-300">AUROC = 0.2882</span>). 
            Consequently, Free Energy scoring is deployed here as an anomaly warning heuristic rather than a guaranteed defense against unencountered varieties.
          </p>
        </div>
      </div>

      {/* 5-Grade Taxonomic Standard */}
      <div className="glass-panel rounded-2xl p-8 space-y-6">
        <div>
          <h3 className="text-xl font-bold text-white">Ordinal Severity Grading Standards</h3>
          <p className="text-xs text-slate-400 mt-0.5">
            Standardized severity spectrum mapping physical post-harvest degradation.
          </p>
        </div>

        <div className="space-y-3">
          {(modelInfo?.grades || [
            { grade: 0, name: 'Pristine (0)', description: 'Export quality: zero blemishes, uniform color.' },
            { grade: 1, name: 'Mild (1)', description: 'Minor scuffing or faint spots, fully edible.' },
            { grade: 2, name: 'Moderate (2)', description: 'Localized bruising; discount retail or processing recommended.' },
            { grade: 3, name: 'Severe (3)', description: 'Extensive skin breakdown or rot; reject from shipping.' },
            { grade: 4, name: 'Critical (4)', description: 'Advanced fungal sporulation or liquefaction; immediate discard.' },
          ]).map((g) => (
            <div key={g.grade} className="p-4 rounded-xl bg-slate-900/60 border border-slate-800/80 flex items-start space-x-4">
              <span className="px-2.5 py-1 rounded bg-slate-800 border border-slate-700 font-mono text-xs font-bold text-emerald-400 shrink-0">
                Grade {g.grade}
              </span>
              <div>
                <h4 className="text-sm font-bold text-white">{g.name}</h4>
                <p className="text-xs text-slate-400 mt-0.5 leading-relaxed">{g.description}</p>
              </div>
            </div>
          ))}
        </div>
      </div>

    </div>
  );
};
