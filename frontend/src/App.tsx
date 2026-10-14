import React, { useState, useEffect } from 'react';
import { Navbar } from './components/Navbar';
import { Footer } from './components/Footer';
import { Dashboard } from './pages/Dashboard';
import { Analyze } from './pages/Analyze';
import { History } from './pages/History';
import { ModelInfo } from './pages/ModelInfo';
import { SystemHealth } from './pages/SystemHealth';
import { api } from './services/api';
import { HealthResponse, ModelInfoResponse } from './types';

export const App: React.FC = () => {
  const [currentTab, setCurrentTab] = useState<string>('dashboard');
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [modelInfo, setModelInfo] = useState<ModelInfoResponse | null>(null);
  const [loadingHealth, setLoadingHealth] = useState<boolean>(true);

  const fetchGlobalState = async () => {
    try {
      const [h, m] = await Promise.allSettled([
        api.getHealth(),
        api.getModelInfo(),
      ]);

      if (h.status === 'fulfilled') setHealth(h.value);
      if (m.status === 'fulfilled') setModelInfo(m.value);
    } catch {
      // Degraded/offline fallback
    } finally {
      setLoadingHealth(false);
    }
  };

  useEffect(() => {
    fetchGlobalState();
    const interval = setInterval(fetchGlobalState, 30000); // 30s background poll
    return () => clearInterval(interval);
  }, []);

  return (
    <div className="min-h-screen bg-[#030712] text-slate-100 flex flex-col selection:bg-emerald-500 selection:text-slate-950">
      
      {/* Navigation Header */}
      <Navbar
        currentTab={currentTab}
        setCurrentTab={setCurrentTab}
        health={health}
        loadingHealth={loadingHealth}
      />

      {/* Main Content Area */}
      <main className="flex-1 max-w-7xl mx-auto w-full px-4 sm:px-6 lg:px-8 py-8 sm:py-10">
        {currentTab === 'dashboard' && (
          <Dashboard
            onNavigate={(tab) => setCurrentTab(tab)}
            health={health}
            modelInfo={modelInfo}
          />
        )}

        {currentTab === 'analyze' && <Analyze />}

        {currentTab === 'history' && <History />}

        {currentTab === 'model-info' && <ModelInfo modelInfo={modelInfo} />}

        {currentTab === 'system-health' && <SystemHealth />}
      </main>

      {/* Global Footer */}
      <Footer />

    </div>
  );
};

export default App;
