import React, { useState, useEffect } from 'react';
import { 
  History as HistoryIcon, 
  Trash2, 
  Search, 
  Filter, 
  Clock, 
  AlertTriangle, 
  CheckCircle, 
  Eye, 
  RefreshCw,
  X,
  FileX
} from 'lucide-react';
import { api } from '../services/api';
import { HistoryItem, HistoryDetail } from '../types';

export const History: React.FC = () => {
  const [items, setItems] = useState<HistoryItem[]>([]);
  const [total, setTotal] = useState(0);
  const [loading, setLoading] = useState(true);
  const [searchTerm, setSearchTerm] = useState('');
  const [selectedSpecies, setSelectedSpecies] = useState<string>('');
  const [selectedGrade, setSelectedGrade] = useState<string>('');
  const [selectedDetail, setSelectedDetail] = useState<HistoryDetail | null>(null);
  const [showClearConfirm, setShowClearConfirm] = useState(false);
  const [actionError, setActionError] = useState<string | null>(null);

  const fetchHistory = async () => {
    setLoading(true);
    setActionError(null);
    try {
      const res = await api.getHistory({
        limit: 100,
        search: searchTerm || undefined,
        species: selectedSpecies || undefined,
        grade: selectedGrade !== '' ? Number(selectedGrade) : undefined,
      });
      setItems(res.items);
      setTotal(res.total);
    } catch (err: any) {
      setActionError(err.message || 'Failed to load analysis history');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchHistory();
  }, [selectedSpecies, selectedGrade]);

  const handleSearchSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    fetchHistory();
  };

  const handleDeleteItem = async (id: string) => {
    try {
      await api.deleteHistoryItem(id);
      setItems(items.filter((item) => item.id !== id));
      setTotal((prev) => Math.max(0, prev - 1));
      if (selectedDetail?.id === id) setSelectedDetail(null);
    } catch (err: any) {
      setActionError(err.message || 'Failed to delete record');
    }
  };

  const handleClearAll = async () => {
    try {
      await api.clearHistory();
      setItems([]);
      setTotal(0);
      setShowClearConfirm(false);
      setSelectedDetail(null);
    } catch (err: any) {
      setActionError(err.message || 'Failed to clear history');
    }
  };

  const handleViewDetail = async (id: string) => {
    try {
      const detail = await api.getHistoryDetail(id);
      setSelectedDetail(detail);
    } catch (err: any) {
      setActionError(err.message || 'Failed to load details');
    }
  };

  return (
    <div className="space-y-6 max-w-6xl mx-auto">
      
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
        <div>
          <h1 className="text-3xl font-extrabold text-white tracking-tight">Analysis History</h1>
          <p className="text-sm text-slate-400 mt-1">
            Local SQLite audit records of past fruit quality evaluations ({total} total analyses recorded).
          </p>
        </div>

        {total > 0 && (
          <button
            onClick={() => setShowClearConfirm(true)}
            className="self-start sm:self-auto px-4 py-2 rounded-xl bg-rose-950/60 hover:bg-rose-900/60 text-rose-300 border border-rose-800 text-xs font-semibold flex items-center space-x-2 transition-colors"
          >
            <Trash2 className="w-3.5 h-3.5" />
            <span>Clear History</span>
          </button>
        )}
      </div>

      {/* Filter and Search Bar */}
      <div className="glass-panel rounded-2xl p-4 flex flex-col md:flex-row items-center gap-3">
        <form onSubmit={handleSearchSubmit} className="flex-1 w-full relative">
          <Search className="w-4 h-4 text-slate-400 absolute left-3.5 top-3" />
          <input
            type="text"
            placeholder="Search by fruit name, grade, or filename..."
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            className="w-full pl-10 pr-4 py-2 rounded-xl bg-slate-900 border border-slate-700/80 text-xs text-white placeholder-slate-500 focus:outline-none focus:border-emerald-500"
          />
        </form>

        <div className="flex items-center space-x-2 w-full md:w-auto">
          {/* Species Filter */}
          <select
            value={selectedSpecies}
            onChange={(e) => setSelectedSpecies(e.target.value)}
            className="px-3 py-2 rounded-xl bg-slate-900 border border-slate-700/80 text-xs text-slate-300 focus:outline-none focus:border-emerald-500"
          >
            <option value="">All Species</option>
            <option value="Apple">Apple</option>
            <option value="Banana">Banana</option>
            <option value="Mango">Mango</option>
            <option value="Orange">Orange</option>
            <option value="Tomato">Tomato</option>
          </select>

          {/* Grade Filter */}
          <select
            value={selectedGrade}
            onChange={(e) => setSelectedGrade(e.target.value)}
            className="px-3 py-2 rounded-xl bg-slate-900 border border-slate-700/80 text-xs text-slate-300 focus:outline-none focus:border-emerald-500"
          >
            <option value="">All Grades</option>
            <option value="0">Grade 0 (Pristine)</option>
            <option value="1">Grade 1 (Mild)</option>
            <option value="2">Grade 2 (Moderate)</option>
            <option value="3">Grade 3 (Severe)</option>
            <option value="4">Grade 4 (Critical)</option>
          </select>

          <button
            onClick={fetchHistory}
            className="p-2 rounded-xl bg-slate-800 hover:bg-slate-700 text-slate-300 border border-slate-700 transition-colors"
            title="Refresh"
          >
            <RefreshCw className={`w-4 h-4 ${loading ? 'animate-spin' : ''}`} />
          </button>
        </div>
      </div>

      {/* Error notification */}
      {actionError && (
        <div className="p-3 rounded-xl bg-rose-950/80 border border-rose-800 text-rose-300 text-xs flex justify-between items-center">
          <span>{actionError}</span>
          <button onClick={() => setActionError(null)}><X className="w-4 h-4" /></button>
        </div>
      )}

      {/* History Table / Empty State */}
      {loading ? (
        <div className="glass-panel rounded-2xl p-12 text-center text-slate-400 text-sm flex flex-col items-center justify-center space-y-3">
          <RefreshCw className="w-6 h-6 animate-spin text-emerald-400" />
          <span>Querying SQLite audit database...</span>
        </div>
      ) : items.length === 0 ? (
        <div className="glass-panel rounded-2xl p-12 text-center space-y-3">
          <FileX className="w-10 h-10 text-slate-500 mx-auto" />
          <h3 className="text-base font-bold text-white">No Analysis Records Found</h3>
          <p className="text-xs text-slate-400 max-w-sm mx-auto">
            Uploaded images that undergo quality grading are recorded here. Run a quality grading analysis to create a record.
          </p>
        </div>
      ) : (
        <div className="glass-panel rounded-2xl overflow-hidden border border-slate-800">
          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs">
              <thead className="bg-slate-900/90 text-slate-400 font-semibold border-b border-slate-800">
                <tr>
                  <th className="py-3 px-4">Sample</th>
                  <th className="py-3 px-4">Species</th>
                  <th className="py-3 px-4">Ordinal Severity</th>
                  <th className="py-3 px-4">Confidence</th>
                  <th className="py-3 px-4">OOD Status</th>
                  <th className="py-3 px-4">Routing</th>
                  <th className="py-3 px-4">Latency</th>
                  <th className="py-3 px-4 text-right">Actions</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-800/60">
                {items.map((item) => (
                  <tr key={item.id} className="hover:bg-slate-900/40 transition-colors">
                    
                    {/* Thumbnail & Date */}
                    <td className="py-3 px-4">
                      <div className="flex items-center space-x-3">
                        {item.thumbnail_base64 ? (
                          <img
                            src={`data:image/jpeg;base64,${item.thumbnail_base64}`}
                            alt="Sample thumbnail"
                            className="w-10 h-10 rounded-lg object-cover border border-slate-700 shrink-0"
                          />
                        ) : (
                          <div className="w-10 h-10 rounded-lg bg-slate-800 border border-slate-700 flex items-center justify-center text-slate-500 shrink-0">
                            <HistoryIcon className="w-4 h-4" />
                          </div>
                        )}
                        <div>
                          <div className="font-mono text-[11px] text-slate-300 truncate max-w-[120px]">
                            {item.filename || 'sample.jpg'}
                          </div>
                          <div className="text-[10px] text-slate-500">
                            {new Date(item.timestamp).toLocaleDateString()} {new Date(item.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                          </div>
                        </div>
                      </div>
                    </td>

                    {/* Species */}
                    <td className="py-3 px-4 font-semibold text-white">
                      {item.species}
                    </td>

                    {/* Grade */}
                    <td className="py-3 px-4">
                      <span className="font-semibold text-slate-200">
                        {item.grade_name}
                      </span>
                      <span className="text-[10px] font-mono text-slate-400 ml-1.5">
                        (G{item.grade})
                      </span>
                    </td>

                    {/* Confidence */}
                    <td className="py-3 px-4 font-mono text-emerald-400">
                      {(item.grade_confidence * 100).toFixed(1)}%
                    </td>

                    {/* OOD */}
                    <td className="py-3 px-4">
                      <span className={`inline-flex items-center space-x-1 px-2 py-0.5 rounded-full text-[10px] font-semibold ${
                        item.is_ood
                          ? 'bg-rose-950 text-rose-300 border border-rose-800'
                          : 'bg-emerald-950 text-emerald-300 border border-emerald-800'
                      }`}>
                        {item.is_ood ? 'OOD' : 'In-Dist'}
                      </span>
                    </td>

                    {/* Routing */}
                    <td className="py-3 px-4 font-mono text-[11px] text-slate-300">
                      {item.routing_action}
                    </td>

                    {/* Latency */}
                    <td className="py-3 px-4 font-mono text-slate-400 text-[11px]">
                      {item.latency_ms} ms
                    </td>

                    {/* Action buttons */}
                    <td className="py-3 px-4 text-right space-x-2 whitespace-nowrap">
                      <button
                        onClick={() => handleViewDetail(item.id)}
                        className="p-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-300 transition-colors"
                        title="View detail"
                      >
                        <Eye className="w-3.5 h-3.5" />
                      </button>
                      <button
                        onClick={() => handleDeleteItem(item.id)}
                        className="p-1.5 rounded-lg bg-rose-950/60 hover:bg-rose-900/80 text-rose-300 transition-colors"
                        title="Delete record"
                      >
                        <Trash2 className="w-3.5 h-3.5" />
                      </button>
                    </td>

                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Modal Detail View */}
      {selectedDetail && (
        <div className="fixed inset-0 z-50 bg-slate-950/80 backdrop-blur-sm flex items-center justify-center p-4">
          <div className="glass-panel border border-slate-700 rounded-3xl max-w-lg w-full p-6 space-y-6 relative max-h-[90vh] overflow-y-auto">
            <button
              onClick={() => setSelectedDetail(null)}
              className="absolute top-5 right-5 p-2 rounded-full bg-slate-800 text-slate-400 hover:text-white"
            >
              <X className="w-4 h-4" />
            </button>

            <div>
              <span className="text-xs font-mono uppercase text-emerald-400 font-bold">Analysis Audit Record</span>
              <h3 className="text-xl font-extrabold text-white mt-1">{selectedDetail.species} — {selectedDetail.grade_name}</h3>
              <p className="text-xs text-slate-400 font-mono mt-0.5">ID: {selectedDetail.id}</p>
            </div>

            {selectedDetail.thumbnail_base64 && (
              <div className="rounded-xl overflow-hidden border border-slate-800 max-w-[200px] mx-auto">
                <img
                  src={`data:image/jpeg;base64,${selectedDetail.thumbnail_base64}`}
                  alt="Saved preview"
                  className="w-full h-auto object-cover"
                />
              </div>
            )}

            <div className="space-y-3 text-xs">
              <div className="grid grid-cols-2 gap-3">
                <div className="p-3 rounded-xl bg-slate-900 border border-slate-800">
                  <span className="text-slate-400">Timestamp:</span>
                  <div className="text-white font-mono mt-0.5">{new Date(selectedDetail.timestamp).toLocaleString()}</div>
                </div>
                <div className="p-3 rounded-xl bg-slate-900 border border-slate-800">
                  <span className="text-slate-400">Inference Latency:</span>
                  <div className="text-white font-mono mt-0.5">{selectedDetail.latency_ms} ms</div>
                </div>
                <div className="p-3 rounded-xl bg-slate-900 border border-slate-800">
                  <span className="text-slate-400">Free Energy Score:</span>
                  <div className="text-white font-mono mt-0.5">{selectedDetail.energy_score.toFixed(4)}</div>
                </div>
                <div className="p-3 rounded-xl bg-slate-900 border border-slate-800">
                  <span className="text-slate-400">Device Executed:</span>
                  <div className="text-white font-mono mt-0.5 uppercase">{selectedDetail.device || 'CPU'}</div>
                </div>
              </div>

              {selectedDetail.discrete_probabilities && (
                <div className="p-3 rounded-xl bg-slate-900 border border-slate-800 space-y-1.5">
                  <span className="text-slate-400 font-semibold">Discrete Probability Vector P(y=k):</span>
                  <div className="flex justify-between font-mono text-[11px] text-slate-300">
                    {selectedDetail.discrete_probabilities.map((p, i) => (
                      <span key={i} className={i === selectedDetail.grade ? 'text-emerald-400 font-bold' : ''}>
                        G{i}: {(p * 100).toFixed(1)}%
                      </span>
                    ))}
                  </div>
                </div>
              )}
            </div>

            <div className="pt-2 flex justify-end">
              <button
                onClick={() => setSelectedDetail(null)}
                className="px-4 py-2 rounded-xl bg-slate-800 text-white text-xs font-semibold hover:bg-slate-700"
              >
                Close Audit View
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Clear Confirmation Modal */}
      {showClearConfirm && (
        <div className="fixed inset-0 z-50 bg-slate-950/80 backdrop-blur-sm flex items-center justify-center p-4">
          <div className="glass-panel border border-rose-900 rounded-3xl max-w-md w-full p-6 space-y-4">
            <div className="flex items-center space-x-3 text-rose-400">
              <AlertTriangle className="w-6 h-6" />
              <h3 className="text-lg font-bold text-white">Clear Analysis History?</h3>
            </div>
            <p className="text-xs text-slate-300 leading-relaxed">
              This action permanently purges all local SQLite records and thumbnails. It cannot be undone. Are you sure you wish to proceed?
            </p>
            <div className="flex justify-end space-x-3 pt-2">
              <button
                onClick={() => setShowClearConfirm(false)}
                className="px-4 py-2 rounded-xl bg-slate-800 hover:bg-slate-700 text-xs font-semibold text-slate-300"
              >
                Cancel
              </button>
              <button
                onClick={handleClearAll}
                className="px-4 py-2 rounded-xl bg-rose-600 hover:bg-rose-500 text-xs font-bold text-white shadow-lg shadow-rose-600/30"
              >
                Confirm & Purge All Records
              </button>
            </div>
          </div>
        </div>
      )}

    </div>
  );
};
