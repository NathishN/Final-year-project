import React from 'react';
import { 
  Sparkles, 
  BarChart3, 
  History, 
  BrainCircuit, 
  Activity, 
  ShieldCheck, 
  AlertTriangle,
  Cpu
} from 'lucide-react';
import { HealthResponse } from '../types';

interface NavbarProps {
  currentTab: string;
  setCurrentTab: (tab: string) => void;
  health: HealthResponse | null;
  loadingHealth: boolean;
}

export const Navbar: React.FC<NavbarProps> = ({
  currentTab,
  setCurrentTab,
  health,
  loadingHealth,
}) => {
  const isHealthy = health?.status === 'healthy';
  const isReady = health?.model_ready === true;

  const navItems = [
    { id: 'dashboard', label: 'Dashboard', icon: Sparkles },
    { id: 'analyze', label: 'Analyze Fruit', icon: BarChart3 },
    { id: 'history', label: 'History', icon: History },
    { id: 'model-info', label: 'Model Architecture', icon: BrainCircuit },
    { id: 'system-health', label: 'System Health', icon: Activity },
  ];

  return (
    <header className="sticky top-0 z-50 glass-panel border-b border-slate-800/80 bg-slate-950/80 backdrop-blur-md">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16">
          
          {/* Logo & Brand */}
          <div 
            className="flex items-center space-x-3 cursor-pointer group"
            onClick={() => setCurrentTab('dashboard')}
          >
            <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-emerald-400 via-teal-500 to-emerald-700 flex items-center justify-center shadow-lg shadow-emerald-500/20 group-hover:scale-105 transition-transform duration-200">
              <span className="text-xl font-black text-slate-950">F</span>
            </div>
            <div>
              <div className="flex items-center space-x-2">
                <span className="font-extrabold text-lg tracking-tight bg-gradient-to-r from-white via-slate-100 to-slate-400 bg-clip-text text-transparent">
                  FruitGrade AI
                </span>
                <span className="text-[10px] font-semibold px-2 py-0.5 rounded-full bg-emerald-950/80 text-emerald-400 border border-emerald-800/60 uppercase tracking-wider">
                  v1.0 Local
                </span>
              </div>
              <p className="text-xs text-slate-400 tracking-tight hidden sm:block">
                Trustworthy Open-Set Ordinal Quality Grading
              </p>
            </div>
          </div>

          {/* Navigation Items */}
          <nav className="hidden md:flex items-center space-x-1">
            {navItems.map((item) => {
              const Icon = item.icon;
              const isActive = currentTab === item.id;
              return (
                <button
                  key={item.id}
                  onClick={() => setCurrentTab(item.id)}
                  className={`flex items-center space-x-2 px-3.5 py-2 rounded-lg text-sm font-medium transition-all duration-200 ${
                    isActive
                      ? 'bg-emerald-500/10 text-emerald-400 border border-emerald-500/30 shadow-sm'
                      : 'text-slate-400 hover:text-slate-200 hover:bg-slate-900/60'
                  }`}
                >
                  <Icon className={`w-4 h-4 ${isActive ? 'text-emerald-400' : 'text-slate-400'}`} />
                  <span>{item.label}</span>
                </button>
              );
            })}
          </nav>

          {/* Status Indicators */}
          <div className="flex items-center space-x-3">
            {/* Device Badge */}
            <div className="hidden sm:flex items-center space-x-1.5 px-2.5 py-1 rounded-md bg-slate-900 border border-slate-800 text-xs text-slate-300">
              <Cpu className="w-3.5 h-3.5 text-cyan-400" />
              <span className="font-mono text-[11px] uppercase">{health?.device || 'CPU'}</span>
            </div>

            {/* Health Status Pill */}
            <div
              className={`flex items-center space-x-2 px-3 py-1 rounded-full text-xs font-medium border transition-colors ${
                loadingHealth
                  ? 'bg-slate-900 text-slate-400 border-slate-800'
                  : isHealthy && isReady
                  ? 'bg-emerald-950/60 text-emerald-300 border-emerald-800/80'
                  : 'bg-rose-950/60 text-rose-300 border-rose-800/80'
              }`}
            >
              <span className={`w-2 h-2 rounded-full ${
                loadingHealth 
                  ? 'bg-slate-500 animate-pulse'
                  : isHealthy && isReady 
                  ? 'bg-emerald-400 animate-pulse' 
                  : 'bg-rose-400'
              }`} />
              <span className="hidden sm:inline">
                {loadingHealth
                  ? 'Connecting...'
                  : isHealthy && isReady
                  ? 'Engine Online'
                  : 'Degraded / Offline'}
              </span>
            </div>
          </div>

        </div>

        {/* Mobile Navigation bar */}
        <div className="flex md:hidden overflow-x-auto py-2 space-x-1 border-t border-slate-800/60">
          {navItems.map((item) => {
            const Icon = item.icon;
            const isActive = currentTab === item.id;
            return (
              <button
                key={item.id}
                onClick={() => setCurrentTab(item.id)}
                className={`flex items-center space-x-1.5 px-2.5 py-1.5 rounded-md text-xs font-medium whitespace-nowrap ${
                  isActive
                    ? 'bg-emerald-500/10 text-emerald-400 border border-emerald-500/30'
                    : 'text-slate-400 hover:text-slate-200'
                }`}
              >
                <Icon className="w-3.5 h-3.5" />
                <span>{item.label}</span>
              </button>
            );
          })}
        </div>
      </div>
    </header>
  );
};
