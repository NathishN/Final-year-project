import React, { useState, useRef } from 'react';
import { 
  Upload, 
  Image as ImageIcon, 
  Trash2, 
  RefreshCw, 
  CheckCircle, 
  AlertTriangle, 
  ShieldAlert, 
  ShieldCheck, 
  ArrowRight, 
  Clock, 
  Cpu, 
  Sparkles,
  BarChart2,
  HelpCircle,
  XCircle,
  FileText
} from 'lucide-react';
import { api } from '../services/api';
import { PredictionResponse } from '../types';

export const Analyze: React.FC = () => {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<PredictionResponse | null>(null);

  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFileSelect = (file: File) => {
    setError(null);
    setResult(null);

    // Validate extension
    const validTypes = ['image/jpeg', 'image/png', 'image/webp', 'image/jpg'];
    if (!validTypes.includes(file.type) && !file.name.match(/\.(jpg|jpeg|png|webp)$/i)) {
      setError('Unsupported file type. Please upload a valid PNG, JPG, or JPEG fruit image.');
      return;
    }

    // Validate size (15 MB max)
    if (file.size > 15 * 1024 * 1024) {
      setError('File size exceeds the 15 MB limit. Please select a smaller image.');
      return;
    }

    setSelectedFile(file);
    const url = URL.createObjectURL(file);
    setPreviewUrl(url);
  };

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = () => {
    setIsDragging(false);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFileSelect(e.dataTransfer.files[0]);
    }
  };

  const handleRemove = () => {
    setSelectedFile(null);
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    setPreviewUrl(null);
    setResult(null);
    setError(null);
    if (fileInputRef.current) fileInputRef.current.value = '';
  };

  const handleAnalyze = async () => {
    if (!selectedFile) return;

    setLoading(true);
    setError(null);

    try {
      const data = await api.predict(selectedFile);
      setResult(data);
    } catch (err: any) {
      setError(err.message || 'Failed to analyze fruit image. Please ensure the backend is running.');
    } finally {
      setLoading(false);
    }
  };

  // Grade badge color helper
  const getGradeColor = (grade: number) => {
    switch (grade) {
      case 0: return { bg: 'bg-emerald-950/80', border: 'border-emerald-500/50', text: 'text-emerald-400', badge: 'bg-emerald-500/20 text-emerald-300' };
      case 1: return { bg: 'bg-teal-950/80', border: 'border-teal-500/50', text: 'text-teal-400', badge: 'bg-teal-500/20 text-teal-300' };
      case 2: return { bg: 'bg-amber-950/80', border: 'border-amber-500/50', text: 'text-amber-400', badge: 'bg-amber-500/20 text-amber-300' };
      case 3: return { bg: 'bg-orange-950/80', border: 'border-orange-500/50', text: 'text-orange-400', badge: 'bg-orange-500/20 text-orange-300' };
      case 4: return { bg: 'bg-rose-950/80', border: 'border-rose-500/50', text: 'text-rose-400', badge: 'bg-rose-500/20 text-rose-300' };
      default: return { bg: 'bg-slate-900', border: 'border-slate-800', text: 'text-slate-300', badge: 'bg-slate-800 text-slate-300' };
    }
  };

  return (
    <div className="space-y-8 max-w-5xl mx-auto">
      
      {/* Title Header */}
      <div>
        <h1 className="text-3xl font-extrabold text-white tracking-tight">Fruit Quality Analysis</h1>
        <p className="text-sm text-slate-400 mt-1">
          Upload an optical image of a single fruit specimen (Apple, Banana, Mango, Orange, or Tomato) to evaluate defect severity, species identification, and open-set anomaly scores.
        </p>
      </div>

      {/* Upload & Preview Card */}
      <div className="glass-panel rounded-2xl p-6 sm:p-8 space-y-6">
        
        {/* Dropzone */}
        {!previewUrl ? (
          <div
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
            onDrop={handleDrop}
            onClick={() => fileInputRef.current?.click()}
            className={`border-2 border-dashed rounded-2xl p-10 sm:p-14 text-center cursor-pointer transition-all duration-200 ${
              isDragging
                ? 'border-emerald-400 bg-emerald-950/20 scale-[0.99]'
                : 'border-slate-700/80 hover:border-emerald-500/50 bg-slate-900/40 hover:bg-slate-900/70'
            }`}
          >
            <input
              ref={fileInputRef}
              type="file"
              accept=".jpg,.jpeg,.png,.webp"
              className="hidden"
              onChange={(e) => e.target.files?.[0] && handleFileSelect(e.target.files[0])}
            />
            
            <div className="flex flex-col items-center space-y-3">
              <div className="w-16 h-16 rounded-2xl bg-emerald-500/10 border border-emerald-500/20 flex items-center justify-center text-emerald-400 shadow-inner">
                <Upload className="w-8 h-8" />
              </div>
              <div className="space-y-1">
                <p className="text-base font-semibold text-white">
                  Drag and drop a fruit photo here, or <span className="text-emerald-400 underline decoration-emerald-500/40">browse files</span>
                </p>
                <p className="text-xs text-slate-400">
                  Supported formats: JPG, JPEG, PNG, WEBP (Maximum size: 15 MB)
                </p>
              </div>
            </div>
          </div>
        ) : (
          /* Preview Mode */
          <div className="space-y-6">
            <div className="relative rounded-2xl overflow-hidden border border-slate-700/80 bg-slate-950 max-w-md mx-auto aspect-square flex items-center justify-center group">
              <img
                src={previewUrl}
                alt="Selected fruit preview"
                className="w-full h-full object-contain"
              />
              <div className="absolute inset-0 bg-slate-950/60 opacity-0 group-hover:opacity-100 transition-opacity duration-200 flex items-center justify-center space-x-3">
                <button
                  onClick={() => fileInputRef.current?.click()}
                  className="px-3.5 py-2 rounded-lg bg-slate-800/90 text-white text-xs font-semibold hover:bg-slate-700 flex items-center space-x-1.5 shadow-md"
                >
                  <RefreshCw className="w-3.5 h-3.5" />
                  <span>Replace</span>
                </button>
                <button
                  onClick={handleRemove}
                  className="px-3.5 py-2 rounded-lg bg-rose-900/90 text-rose-200 text-xs font-semibold hover:bg-rose-800 flex items-center space-x-1.5 shadow-md"
                >
                  <Trash2 className="w-3.5 h-3.5" />
                  <span>Remove</span>
                </button>
              </div>
            </div>

            {/* Hidden Input for Replace */}
            <input
              ref={fileInputRef}
              type="file"
              accept=".jpg,.jpeg,.png,.webp"
              className="hidden"
              onChange={(e) => e.target.files?.[0] && handleFileSelect(e.target.files[0])}
            />

            {/* Action Bar */}
            <div className="flex flex-col sm:flex-row items-center justify-between gap-4 max-w-md mx-auto">
              <div className="text-xs text-slate-400 truncate max-w-[200px]">
                {selectedFile?.name} ({(selectedFile ? selectedFile.size / (1024 * 1024) : 0).toFixed(2)} MB)
              </div>
              <button
                onClick={handleAnalyze}
                disabled={loading}
                className="w-full sm:w-auto px-7 py-3 rounded-xl bg-gradient-to-r from-emerald-500 to-teal-600 hover:from-emerald-400 hover:to-teal-500 disabled:opacity-50 text-slate-950 font-bold text-sm shadow-lg shadow-emerald-500/25 flex items-center justify-center space-x-2 transition-all duration-200"
              >
                {loading ? (
                  <>
                    <RefreshCw className="w-4 h-4 animate-spin" />
                    <span>Executing Inference...</span>
                  </>
                ) : (
                  <>
                    <Sparkles className="w-4 h-4" />
                    <span>Run Quality Grading</span>
                  </>
                )}
              </button>
            </div>
          </div>
        )}

        {/* Error message */}
        {error && (
          <div className="p-4 rounded-xl bg-rose-950/60 border border-rose-800 text-rose-200 text-xs flex items-start space-x-3">
            <XCircle className="w-5 h-5 text-rose-400 shrink-0 mt-0.5" />
            <div>
              <p className="font-semibold text-rose-300">Analysis Error</p>
              <p className="mt-0.5">{error}</p>
            </div>
          </div>
        )}

      </div>

      {/* Structured Result Panel */}
      {result && (
        <div className="space-y-6 animate-fadeIn">
          
          {/* Selective Routing & Decision Banner */}
          <div className={`rounded-2xl border p-6 flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4 ${
            result.selective_routing.routing_action === 'AUTOMATED_ACCEPT'
              ? 'border-emerald-500/50 bg-emerald-950/30 text-emerald-300'
              : result.selective_routing.routing_action === 'ROUTE_TO_MANUAL_INSPECTION'
              ? 'border-amber-500/50 bg-amber-950/30 text-amber-300'
              : 'border-rose-500/50 bg-rose-950/30 text-rose-300'
          }`}>
            <div className="flex items-start space-x-3.5">
              {result.selective_routing.routing_action === 'AUTOMATED_ACCEPT' ? (
                <ShieldCheck className="w-7 h-7 text-emerald-400 shrink-0" />
              ) : (
                <ShieldAlert className="w-7 h-7 text-amber-400 shrink-0" />
              )}
              <div>
                <div className="flex items-center space-x-2">
                  <span className="text-xs uppercase font-mono font-bold tracking-wider">
                    Operational Routing Decision:
                  </span>
                  <span className="px-2 py-0.5 rounded text-xs font-extrabold bg-slate-900 border border-slate-700">
                    {result.selective_routing.routing_action}
                  </span>
                </div>
                <p className="text-xs text-slate-300 mt-1 leading-relaxed">
                  {result.selective_routing.explanation}
                </p>
              </div>
            </div>

            {/* Inference Latency pill */}
            <div className="flex items-center space-x-2 text-xs text-slate-400 bg-slate-900/80 px-3 py-1.5 rounded-lg border border-slate-800 shrink-0 font-mono">
              <Clock className="w-3.5 h-3.5 text-slate-400" />
              <span>{result.inference_latency_ms} ms</span>
              <span className="text-slate-600">•</span>
              <Cpu className="w-3.5 h-3.5 text-cyan-400" />
              <span>{result.device_used}</span>
            </div>
          </div>

          {/* Core Results: Species & Ordinal Severity Grid */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            
            {/* Left: Species Classification */}
            <div className="glass-panel rounded-2xl p-6 space-y-4">
              <div className="flex items-center justify-between">
                <span className="text-xs font-semibold text-slate-400 uppercase tracking-wider">Botanical Identification</span>
                <span className="text-xs font-mono text-emerald-400 font-bold">
                  {(result.species.confidence * 100).toFixed(1)}% Confidence
                </span>
              </div>

              <div>
                <h3 className="text-2xl font-black text-white">{result.species.species}</h3>
                <p className="text-xs text-slate-400 mt-0.5">Identified via ConvNeXt-Tiny penultimate feature space</p>
              </div>

              {/* Species Probability Distribution */}
              <div className="pt-2 space-y-2 border-t border-slate-800/80">
                <span className="text-xs font-semibold text-slate-400">Class Probabilities:</span>
                {Object.entries(result.species.probabilities).map(([sp, prob]) => (
                  <div key={sp} className="space-y-1">
                    <div className="flex justify-between text-xs">
                      <span className={sp === result.species.species ? 'text-white font-semibold' : 'text-slate-400'}>
                        {sp}
                      </span>
                      <span className="font-mono text-slate-400">{(prob * 100).toFixed(1)}%</span>
                    </div>
                    <div className="w-full h-1.5 rounded-full bg-slate-800 overflow-hidden">
                      <div 
                        className={`h-full rounded-full transition-all duration-500 ${
                          sp === result.species.species ? 'bg-emerald-400' : 'bg-slate-600'
                        }`}
                        style={{ width: `${Math.max(prob * 100, 1)}%` }}
                      />
                    </div>
                  </div>
                ))}
              </div>
            </div>

            {/* Right: Monotonic CORAL Ordinal Grade */}
            {(() => {
              const c = getGradeColor(result.ordinal_grade.grade);
              return (
                <div className={`glass-panel rounded-2xl p-6 space-y-4 border ${c.border}`}>
                  <div className="flex items-center justify-between">
                    <span className="text-xs font-semibold text-slate-400 uppercase tracking-wider">Monotonic CORAL Grading</span>
                    <span className={`px-2.5 py-0.5 rounded-full text-xs font-mono font-bold ${c.badge}`}>
                      Grade {result.ordinal_grade.grade}
                    </span>
                  </div>

                  <div>
                    <h3 className={`text-2xl font-black ${c.text}`}>
                      {result.ordinal_grade.grade_name}
                    </h3>
                    <p className="text-xs text-slate-300 mt-1 leading-relaxed">
                      {result.ordinal_grade.grade_description}
                    </p>
                  </div>

                  {/* Discrete Probability PMF across 5 grades */}
                  <div className="pt-2 space-y-2 border-t border-slate-800/80">
                    <div className="flex justify-between text-xs">
                      <span className="text-slate-400 font-semibold">Severity PMF Distribution P(y=k):</span>
                      <span className="font-mono text-emerald-400 font-semibold">
                        {(result.ordinal_grade.confidence * 100).toFixed(1)}% Pred Weight
                      </span>
                    </div>

                    <div className="grid grid-cols-5 gap-1.5 pt-1">
                      {result.ordinal_grade.discrete_probabilities.map((p, idx) => {
                        const isPred = idx === result.ordinal_grade.grade;
                        return (
                          <div key={idx} className="space-y-1 text-center">
                            <div className="h-16 bg-slate-900 rounded-lg p-1 flex items-end justify-center border border-slate-800/80">
                              <div
                                className={`w-full rounded transition-all duration-500 ${
                                  isPred ? 'bg-gradient-to-t from-emerald-600 to-emerald-400' : 'bg-slate-700'
                                }`}
                                style={{ height: `${Math.max(p * 100, 6)}%` }}
                              />
                            </div>
                            <div className="font-mono text-[10px] text-slate-400">G{idx}</div>
                            <div className={`font-mono text-[10px] ${isPred ? 'text-white font-bold' : 'text-slate-500'}`}>
                              {(p * 100).toFixed(0)}%
                            </div>
                          </div>
                        );
                      })}
                    </div>
                  </div>
                </div>
              );
            })()}

          </div>

          {/* OOD Anomaly Assessment Card */}
          <div className="glass-panel rounded-2xl p-6 space-y-4">
            <div className="flex items-center justify-between">
              <div className="flex items-center space-x-2">
                <span className="text-xs font-semibold text-slate-400 uppercase tracking-wider">
                  Out-of-Distribution Detector: {result.ood.detector}
                </span>
              </div>
              <div className={`flex items-center space-x-1.5 px-2.5 py-0.5 rounded-full text-xs font-semibold ${
                result.ood.is_ood
                  ? 'bg-rose-950 text-rose-300 border border-rose-800'
                  : 'bg-emerald-950 text-emerald-300 border border-emerald-800'
              }`}>
                {result.ood.is_ood ? <AlertTriangle className="w-3.5 h-3.5" /> : <CheckCircle className="w-3.5 h-3.5" />}
                <span>{result.ood.is_ood ? 'Anomalous / OOD' : 'In-Distribution'}</span>
              </div>
            </div>

            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 text-xs">
              <div className="p-3 rounded-xl bg-slate-900/80 border border-slate-800 space-y-1">
                <span className="text-slate-400">Computed Free Energy Score E(x):</span>
                <div className="text-lg font-mono font-bold text-white">
                  {result.ood.energy_score.toFixed(4)}
                </div>
                <div className="text-[11px] text-slate-500">
                  Cutoff Threshold: <span className="font-mono text-slate-400">{result.ood.threshold.toFixed(2)}</span> (Scores above threshold suggest OOD)
                </div>
              </div>

              <div className="p-3 rounded-xl bg-slate-900/80 border border-slate-800 flex items-center">
                <p className="text-slate-300 leading-relaxed text-xs">
                  {result.ood.interpretation}
                </p>
              </div>
            </div>
          </div>

          {/* Research Limitations & Guardrails Accordion */}
          <div className="rounded-2xl border border-slate-800 bg-slate-900/30 p-5 space-y-3">
            <div className="flex items-center space-x-2 text-slate-400 text-xs font-bold uppercase tracking-wider">
              <FileText className="w-4 h-4 text-amber-400" />
              <span>Scientific Caveats & Model Limitations</span>
            </div>
            <ul className="grid grid-cols-1 sm:grid-cols-2 gap-2 text-xs text-slate-400 list-disc list-inside">
              {result.limitations.map((lim, idx) => (
                <li key={idx} className="leading-relaxed">
                  {lim}
                </li>
              ))}
            </ul>
          </div>

        </div>
      )}

    </div>
  );
};
