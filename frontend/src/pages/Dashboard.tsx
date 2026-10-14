import React from 'react';
import { 
  ArrowRight, 
  Sparkles, 
  ShieldCheck, 
  AlertTriangle, 
  Cpu, 
  CheckCircle2, 
  Scale, 
  Zap, 
  Layers, 
  BookOpen,
  History,
  Info
} from 'lucide-react';
import { HealthResponse, ModelInfoResponse } from '../types';

interface DashboardProps {
  onNavigate: (tab: string) => void;
  health: HealthResponse | null;
  modelInfo: ModelInfoResponse | null;
}

export const Dashboard: React.FC<DashboardProps> = ({
  onNavigate,
  health,
  modelInfo,
}) => {
  const isHealthy = health?.status === 'healthy';
  const isReady = health?.model_ready === true;

  return (
    <div className="space-y-12">
      {/* Hero Section */}
      <section className="relative overflow-hidden rounded-3xl border border-slate-800 bg-gradient-to-b from-slate-900/90 via-slate-900/40 to-slate-950 p-8 sm:p-12">
        <div className="absolute top-0 right-0 -mt-10 -mr-10 w-96 h-96 rounded-full bg-emerald-500/10 blur-3xl pointer-events-none" />
        <div className="absolute bottom-0 left-1/3 -mb-10 w-80 h-80 rounded-full bg-cyan-500/10 blur-3xl pointer-events-none" />

        <div className="relative z-10 max-w-3xl space-y-6">
          <div className="inline-flex items-center space-x-2 px-3 py-1.5 rounded-full bg-emerald-950/80 border border-emerald-800/60 text-xs font-semibold text-emerald-300">
            <Sparkles className="w-3.5 h-3.5 text-emerald-400" />
            <span>Monotonic Ordinal Quality Grading & OOD Detection</span>
          </div>

          <h1 className="text-4xl sm:text-5xl lg:text-6xl font-extrabold tracking-tight text-white leading-tight">
            Trustworthy <span className="bg-gradient-to-r from-emerald-400 via-teal-300 to-cyan-400 bg-clip-text text-transparent">Multi-Fruit</span> Quality Assessment
          </h1>

          <p className="text-base sm:text-lg text-slate-300 leading-relaxed">
            FruitGrade AI pairs deep visual representations from a frozen ConvNeXt-Tiny backbone with rank-consistent Monotonic CORAL ordinal classification and Free Energy anomaly detection. Inspect defect severity across 5 standardized grades with automated abstention routing.
          </p>

          {/* CTA Buttons */}
          <div className="flex flex-wrap gap-4 pt-2">
            <button
              onClick={() => onNavigate('analyze')}
              className="inline-flex items-center space-x-2.5 px-6 py-3.5 rounded-xl bg-gradient-to-r from-emerald-500 to-teal-600 hover:from-emerald-400 hover:to-teal-500 text-slate-950 font-bold text-sm shadow-lg shadow-emerald-500/25 transition-all duration-200 hover:scale-[1.02]"
            >
              <span>Analyze a Fruit Image</span>
              <ArrowRight className="w-4 h-4" />
            </button>

            <button
              onClick={() => onNavigate('model-info')}
              className="inline-flex items-center space-x-2 px-5 py-3.5 rounded-xl bg-slate-800/80 hover:bg-slate-700/80 text-slate-200 font-semibold text-sm border border-slate-700 transition-colors"
            >
              <BookOpen className="w-4 h-4 text-slate-400" />
              <span>Model Architecture & Metrics</span>
            </button>

            <button
              onClick={() => onNavigate('history')}
              className="inline-flex items-center space-x-2 px-5 py-3.5 rounded-xl bg-slate-900/80 hover:bg-slate-800/80 text-slate-300 font-semibold text-sm border border-slate-800 transition-colors"
            >
              <History className="w-4 h-4 text-slate-400" />
              <span>Analysis Logs</span>
            </button>
          </div>
        </div>
      </section>

      {/* Operational Status & Capabilities Grid */}
      <section className="grid grid-cols-1 md:grid-cols-3 gap-6">
        
        {/* Card 1: Engine Status */}
        <div className="glass-panel glass-panel-hover rounded-2xl p-6 space-y-4">
          <div className="flex items-center justify-between">
            <span className="text-xs font-semibold text-slate-400 uppercase tracking-wider">Engine Status</span>
            <div className={`w-3 h-3 rounded-full ${isHealthy && isReady ? 'bg-emerald-400' : 'bg-rose-400'}`} />
          </div>
          <div>
            <h3 className="text-xl font-bold text-white">
              {isHealthy && isReady ? 'Ready for Inference' : 'Service Degraded'}
            </h3>
            <p className="text-xs text-slate-400 mt-1">
              Backend is active on <code className="text-slate-300">127.0.0.1:8000</code>. Device: <span className="text-emerald-400 font-mono font-semibold">{health?.device || 'cpu'}</span>.
            </p>
          </div>
          <div className="pt-2 border-t border-slate-800/80 space-y-2 text-xs">
            <div className="flex justify-between text-slate-400">
              <span>Backbone Checkpoint:</span>
              <span className="text-slate-200 font-mono">best_model.pth</span>
            </div>
            <div className="flex justify-between text-slate-400">
              <span>Ordinal Head:</span>
              <span className="text-slate-200 font-mono">model_A_seed83_best.pth</span>
            </div>
          </div>
        </div>

        {/* Card 2: Monotonic CORAL */}
        <div className="glass-panel glass-panel-hover rounded-2xl p-6 space-y-4">
          <div className="flex items-center justify-between">
            <span className="text-xs font-semibold text-emerald-400 uppercase tracking-wider">Ordinal Architecture</span>
            <Scale className="w-4 h-4 text-emerald-400" />
          </div>
          <div>
            <h3 className="text-xl font-bold text-white">Rank Monotonicity</h3>
            <p className="text-xs text-slate-400 mt-1">
              Guaranteed ordered thresholds <code className="text-slate-300">b₁ &lt; b₂ &lt; b₃ &lt; b₄</code> preventing ranking contradictions across severity levels.
            </p>
          </div>
          <div className="pt-2 border-t border-slate-800/80 flex items-center justify-between text-xs">
            <span className="text-slate-400">Standardized Grades:</span>
            <span className="px-2 py-0.5 rounded bg-emerald-950 text-emerald-300 border border-emerald-800 font-mono">5 Ordinal Classes</span>
          </div>
        </div>

        {/* Card 3: Free Energy OOD */}
        <div className="glass-panel glass-panel-hover rounded-2xl p-6 space-y-4">
          <div className="flex items-center justify-between">
            <span className="text-xs font-semibold text-cyan-400 uppercase tracking-wider">Safety & Abstention</span>
            <ShieldCheck className="w-4 h-4 text-cyan-400" />
          </div>
          <div>
            <h3 className="text-xl font-bold text-white">Selective Routing</h3>
            <p className="text-xs text-slate-400 mt-1">
              Free Energy Scoring flag anomalous botanical inputs or extreme corruptions, automatically redirecting ambiguous samples to human sorters.
            </p>
          </div>
          <div className="pt-2 border-t border-slate-800/80 flex items-center justify-between text-xs">
            <span className="text-slate-400">Operational Policy:</span>
            <span className="text-cyan-300 font-semibold">Abstain if Conf &lt; 70% or OOD</span>
          </div>
        </div>

      </section>

      {/* Supported Species & Grade Hierarchy Section */}
      <section className="glass-panel rounded-2xl p-8 space-y-6">
        <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
          <div>
            <h2 className="text-xl font-bold text-white">Supported Botanical Species & Severity Spectrum</h2>
            <p className="text-xs text-slate-400 mt-1">
              Multi-class species recognition and 5-tier defect taxonomy calibrated on real agricultural imaging.
            </p>
          </div>
          <div className="flex items-center space-x-2">
            <span className="text-xs text-slate-400">Supported Species:</span>
            <div className="flex space-x-1.5">
              {(modelInfo?.supported_species || ["Apple", "Banana", "Mango", "Orange", "Tomato"]).map((sp) => (
                <span key={sp} className="px-2.5 py-1 rounded-md bg-slate-800 text-slate-200 text-xs font-medium border border-slate-700">
                  {sp}
                </span>
              ))}
            </div>
          </div>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-5 gap-3 pt-2">
          {[
            { grade: 0, name: 'Pristine (0)', color: 'border-emerald-500/50 bg-emerald-950/20 text-emerald-300', desc: 'Flawless export quality' },
            { grade: 1, name: 'Mild (1)', color: 'border-teal-500/50 bg-teal-950/20 text-teal-300', desc: 'Minor superficial marks' },
            { grade: 2, name: 'Moderate (2)', color: 'border-amber-500/50 bg-amber-950/20 text-amber-300', desc: 'Noticeable defect / discount' },
            { grade: 3, name: 'Severe (3)', color: 'border-orange-500/50 bg-orange-950/20 text-orange-300', desc: 'Structural breakdown' },
            { grade: 4, name: 'Critical (4)', color: 'border-rose-500/50 bg-rose-950/20 text-rose-300', desc: 'Advanced rot / cull' },
          ].map((item) => (
            <div key={item.grade} className={`rounded-xl border p-4 space-y-1.5 ${item.color}`}>
              <div className="text-xs font-mono font-semibold">Grade {item.grade}</div>
              <div className="font-bold text-sm text-white">{item.name}</div>
              <div className="text-[11px] text-slate-400">{item.desc}</div>
            </div>
          ))}
        </div>
      </section>

      {/* Verified Empirical Research Card */}
      <section className="rounded-2xl border border-slate-800/80 bg-slate-900/40 p-6 flex flex-col sm:flex-row items-start sm:items-center justify-between gap-6">
        <div className="flex items-start space-x-4">
          <div className="p-3 rounded-xl bg-emerald-500/10 text-emerald-400 border border-emerald-500/20 shrink-0">
            <Info className="w-6 h-6" />
          </div>
          <div className="space-y-1">
            <h4 className="font-bold text-white text-base">Verified Research Benchmark: Config A Image-Only Baseline</h4>
            <p className="text-xs text-slate-400 max-w-2xl leading-relaxed">
              In rigorous 30-seed confirmatory evaluation, Pure Visual Configuration A achieved superior ordinal agreement with <strong className="text-slate-200">QWK = 0.8143 ± 0.0116</strong> and <strong className="text-slate-200">Spearman ρ = 0.8520</strong>. Note that open-set real-image OOD evaluation revealed severe domain inversion challenges (FPR95 = 1.0000 on foreign cultivars).
            </p>
          </div>
        </div>
        <button
          onClick={() => onNavigate('model-info')}
          className="whitespace-nowrap px-4 py-2 rounded-lg bg-slate-800 hover:bg-slate-700 text-xs font-semibold text-slate-200 border border-slate-700 transition-colors"
        >
          View Full Findings
        </button>
      </section>
    </div>
  );
};
