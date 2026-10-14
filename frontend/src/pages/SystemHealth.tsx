import React, { useState, useEffect } from 'react';
import { 
  Activity, 
  Cpu, 
  HardDrive, 
  ShieldCheck, 
  RefreshCw, 
  CheckCircle, 
  AlertCircle, 
  Terminal,
  Zap,
  Server
} from 'lucide-react';
import { api } from '../services/api';
import { SystemInfoResponse, HealthResponse } from '../types';

export const SystemHealth: React.FC = () => {
  const [systemInfo, setSystemInfo] = useState<SystemInfoResponse | null>(null);
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetchHealth = async () => {
    setLoading(true);
    setError(null);
    try {
      const [h, sys] = await Promise.all([
        api.getHealth(),
        api.getSystemInfo(),
      ]);
      setHealth(h);
      setSystemInfo(sys);
    } catch (err: any) {
      setError(err.message || 'Failed to connect to backend service.');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchHealth();
  }, []);

  return (
    <div className="space-y-8 max-w-4xl mx-auto">
      
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-extrabold text-white tracking-tight">System Health & Runtime</h1>
          <p className="text-sm text-slate-400 mt-1">
            Real-time diagnostics for hardware acceleration, PyTorch runtime, and inference service readiness.
          </p>
        </div>

        <button
          onClick={fetchHealth}
          className="px-4 py-2 rounded-xl bg-slate-800 hover:bg-slate-700 text-slate-300 border border-slate-700 text-xs font-semibold flex items-center space-x-2 transition-colors"
        >
          <RefreshCw className={`w-3.5 h-3.5 ${loading ? 'animate-spin' : ''}`} />
          <span>Refresh</span>
        </button>
      </div>

      {error && (
        <div className="p-4 rounded-xl bg-rose-950/80 border border-rose-800 text-rose-200 text-xs flex items-center space-x-3">
          <AlertCircle className="w-5 h-5 text-rose-400 shrink-0" />
          <span>{error}</span>
        </div>
      )}

      {/* Main Status Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-6">
        
        {/* Service Card */}
        <div className="glass-panel rounded-2xl p-6 space-y-3">
          <div className="flex items-center justify-between text-slate-400">
            <span className="text-xs font-semibold uppercase">API Service</span>
            <Server className="w-4 h-4 text-emerald-400" />
          </div>
          <div className="text-2xl font-black text-white">
            {health?.status === 'healthy' ? 'Operational' : 'Degraded'}
          </div>
          <div className="text-xs text-slate-400">
            FastAPI 0.115 • REST v1
          </div>
        </div>

        {/* Device Acceleration */}
        <div className="glass-panel rounded-2xl p-6 space-y-3">
          <div className="flex items-center justify-between text-slate-400">
            <span className="text-xs font-semibold uppercase">Compute Device</span>
            <Cpu className="w-4 h-4 text-cyan-400" />
          </div>
          <div className="text-2xl font-black text-white font-mono uppercase">
            {systemInfo?.device_available || 'CPU'}
          </div>
          <div className="text-xs text-slate-400 truncate">
            {systemInfo?.gpu_name || 'Host CPU Fallback'}
          </div>
        </div>

        {/* Model Readiness */}
        <div className="glass-panel rounded-2xl p-6 space-y-3">
          <div className="flex items-center justify-between text-slate-400">
            <span className="text-xs font-semibold uppercase">Model Status</span>
            <ShieldCheck className="w-4 h-4 text-emerald-400" />
          </div>
          <div className="text-2xl font-black text-emerald-400">
            {systemInfo?.model_loaded ? 'Ready (Loaded)' : 'Not Loaded'}
          </div>
          <div className="text-xs text-slate-400">
            ConvNeXt + CORAL Head
          </div>
        </div>

      </div>

      {/* Hardware & Runtime Specification */}
      <div className="glass-panel rounded-2xl p-8 space-y-6">
        <h3 className="text-lg font-bold text-white">Runtime Specifications & Bounds</h3>

        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 text-xs">
          
          <div className="p-4 rounded-xl bg-slate-900/60 border border-slate-800 space-y-1">
            <span className="text-slate-400">Python Runtime:</span>
            <div className="font-mono text-white text-sm">v{systemInfo?.python_version || '3.12.4'}</div>
          </div>

          <div className="p-4 rounded-xl bg-slate-900/60 border border-slate-800 space-y-1">
            <span className="text-slate-400">PyTorch Version:</span>
            <div className="font-mono text-white text-sm">{systemInfo?.pytorch_version || '2.5.1'}</div>
          </div>

          <div className="p-4 rounded-xl bg-slate-900/60 border border-slate-800 space-y-1">
            <span className="text-slate-400">Max Upload File Size:</span>
            <div className="font-mono text-white text-sm">{systemInfo?.max_upload_size_mb || 15} MB</div>
          </div>

          <div className="p-4 rounded-xl bg-slate-900/60 border border-slate-800 space-y-1">
            <span className="text-slate-400">Allowed Image Types:</span>
            <div className="font-mono text-white text-sm">
              {(systemInfo?.allowed_extensions || ['.jpg', '.jpeg', '.png', '.webp']).join(', ')}
            </div>
          </div>

          {systemInfo?.gpu_memory_allocated_mb !== null && systemInfo?.gpu_memory_allocated_mb !== undefined && (
            <>
              <div className="p-4 rounded-xl bg-slate-900/60 border border-slate-800 space-y-1">
                <span className="text-slate-400">GPU VRAM Allocated:</span>
                <div className="font-mono text-emerald-400 text-sm">{systemInfo.gpu_memory_allocated_mb} MB</div>
              </div>

              <div className="p-4 rounded-xl bg-slate-900/60 border border-slate-800 space-y-1">
                <span className="text-slate-400">GPU VRAM Reserved:</span>
                <div className="font-mono text-cyan-400 text-sm">{systemInfo.gpu_memory_reserved_mb} MB</div>
              </div>
            </>
          )}

        </div>
      </div>

    </div>
  );
};
