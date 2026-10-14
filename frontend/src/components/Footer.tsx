import React from 'react';
import { ShieldAlert, BookOpen, Lock, Terminal } from 'lucide-react';

export const Footer: React.FC = () => {
  return (
    <footer className="mt-20 border-t border-slate-800/80 bg-slate-950/70 py-10 text-xs text-slate-400">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="grid grid-cols-1 md:grid-cols-3 gap-8 mb-8">
          
          {/* Research Caveat */}
          <div className="space-y-2">
            <div className="flex items-center space-x-2 text-amber-400 font-semibold">
              <ShieldAlert className="w-4 h-4" />
              <span>Academic Research Disclaimer</span>
            </div>
            <p className="text-slate-400 leading-relaxed text-xs">
              FruitGrade AI is an empirical computer vision prototype investigating Monotonic CORAL ordinal regression and open-set OOD reliability. 
              Outputs do not constitute certified agricultural safety certifications or USDA/EU food inspection approvals.
            </p>
          </div>

          {/* Reproducibility & Local Privacy */}
          <div className="space-y-2">
            <div className="flex items-center space-x-2 text-emerald-400 font-semibold">
              <Lock className="w-4 h-4" />
              <span>Local Air-Gapped Execution</span>
            </div>
            <p className="text-slate-400 leading-relaxed text-xs">
              All inference operations, weights loading, and analysis history logs execute 100% locally on localhost. 
              Zero telemetry, analytics tracking, or external image uploads are transmitted over the network.
            </p>
          </div>

          {/* Model Architecture Info */}
          <div className="space-y-2">
            <div className="flex items-center space-x-2 text-cyan-400 font-semibold">
              <BookOpen className="w-4 h-4" />
              <span>Verified Checkpoint Lineage</span>
            </div>
            <p className="text-slate-400 leading-relaxed text-xs">
              Backbone: ConvNeXt-Tiny (27.8M params) • Head: Monotonic CORAL (Config A, Seed 83, QWK = 0.8143 ± 0.0116) • OOD: Free Energy Inverted Scoring.
            </p>
          </div>

        </div>

        <div className="pt-6 border-t border-slate-900 flex flex-col sm:flex-row items-center justify-between text-[11px] text-slate-500 gap-4">
          <div className="flex items-center space-x-2">
            <Terminal className="w-3.5 h-3.5 text-slate-400" />
            <span>FruitGrade AI • Localhost Web Application</span>
          </div>
          <div className="flex items-center space-x-4">
            <span>FastAPI v1 REST API</span>
            <span>•</span>
            <span>PyTorch 2.5 Inference Mode</span>
            <span>•</span>
            <span>React 18 + Vite</span>
          </div>
        </div>
      </div>
    </footer>
  );
};
