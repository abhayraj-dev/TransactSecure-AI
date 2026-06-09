import { useState, useEffect, useRef } from 'react';
import { ShieldAlert, Play, Square, Info, Activity } from 'lucide-react';
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer } from 'recharts';


const CustomTooltip = ({ active, payload }) => {
  if (active && payload && payload.length) {
    const data = payload[0].payload;
    const isComponent = data.feature.startsWith('V');
    const title = isComponent ? `Principal Component ${data.feature}` : `Transaction ${data.feature}`;
    const description = isComponent 
      ? "Anonymized latent feature protecting customer PII." 
      : "The scaled monetary value.";

    return (
      <div className="bg-[#0f111a] border border-slate-700 p-4 rounded shadow-2xl max-w-xs z-50 font-mono">
        <p className="text-white font-bold text-sm mb-1">{title}</p>
        <p className="text-slate-400 text-xs mb-3 leading-relaxed">{description}</p>
        <div className="bg-black/50 rounded p-2 border border-slate-800">
          <p className="text-slate-500 text-[10px] uppercase tracking-wider">Error Variance</p>
          <p className="text-red-500 text-sm font-bold">
            {data.error.toFixed(4)}
          </p>
        </div>
      </div>
    );
  }
  return null;
};

export default function App() {
  const [logs, setLogs] = useState([]);
  const [isStreaming, setIsStreaming] = useState(false);
  const [activeThreat, setActiveThreat] = useState(null);
  const [feedInterval, setFeedInterval] = useState(1500); // Default 1.5s
  const streamRef = useRef(null);


  const processNextTransaction = async () => {
    try {
      const streamRes = await fetch('http://127.0.0.1:8000/stream');
      const streamData = await streamRes.json();

      const analyzeRes = await fetch('http://127.0.0.1:8000/analyze', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ features: streamData.features })
      });
      const analysis = await analyzeRes.json();

      const newLog = {
        id: Math.random().toString(36).substring(7),
        time: new Date().toLocaleTimeString('en-US', { hour12: false }),
        amount: analysis.transaction_amount,
        score: analysis.risk_score,
        isFraud: analysis.is_fraud,
        anomalies: analysis.top_anomalies
      };

      setLogs(prev => [newLog, ...prev].slice(0, 50));
      if (analysis.is_fraud) setActiveThreat(newLog);

    } catch (error) {
      console.error("Connection Error:", error);
      setIsStreaming(false);
    }
  };

  //Attack Injector 
  const injectFraudAttack = async () => {
    try {
      const streamRes = await fetch('http://127.0.0.1:8000/stream/fraud');
      const streamData = await streamRes.json();

      const analyzeRes = await fetch('http://127.0.0.1:8000/analyze', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ features: streamData.features })
      });
      const analysis = await analyzeRes.json();

      const newLog = {
        id: "ATTACK-" + Math.random().toString(36).substring(7),
        time: new Date().toLocaleTimeString('en-US', { hour12: false }),
        amount: analysis.transaction_amount,
        score: analysis.risk_score,
        isFraud: analysis.is_fraud,
        anomalies: analysis.top_anomalies
      };

      setLogs(prev => [newLog, ...prev].slice(0, 50));
      setActiveThreat(newLog);
    } catch (error) {
      console.error("Attack Injection Failed:", error);
    }
  };


  useEffect(() => {
    if (isStreaming) {
      streamRef.current = setInterval(processNextTransaction, feedInterval);
    } else {
      clearInterval(streamRef.current);
    }
    return () => clearInterval(streamRef.current);
  }, [isStreaming, feedInterval]);

  return (
    <div className="min-h-screen bg-[#050508] text-slate-300 p-8 font-sans selection:bg-emerald-500/30">
      
      {/* HEADER */}
      <header className="flex justify-between items-center mb-10 pb-6 border-b border-white/5">
        <div className="flex items-center gap-8">
          <h1 className="text-2xl font-bold text-white tracking-[0.2em] font-mono uppercase">
            TRANSACTSECURE<span className="text-emerald-500">_AI</span>
          </h1>
        </div>
        
        <div className="flex items-center gap-6">
          {/* Feed Rate Slider */}
          <div className="hidden lg:flex items-center gap-3 bg-[#0a0b10] px-4 py-2 border border-white/10">
            <Activity className="w-4 h-4 text-slate-500" />
            <span className="text-[10px] text-slate-500 font-mono tracking-widest uppercase">Feed Rate</span>
            <input 
              type="range" 
              min="200" 
              max="3000" 
              step="100" 
              dir="rtl" 
              value={feedInterval} 
              onChange={(e) => setFeedInterval(Number(e.target.value))}
              className="w-24 accent-emerald-500 h-1 bg-slate-800 rounded-none appearance-none cursor-pointer"
            />
          </div>

          <button 
            onClick={injectFraudAttack}
            className="text-[11px] font-mono tracking-widest uppercase border border-red-500/50 text-red-400 px-6 py-2.5 hover:bg-red-500/10 hover:border-red-500 transition-all"
          >
            Inject Attack
          </button>

          <button 
            onClick={() => setIsStreaming(!isStreaming)}
            className={`flex items-center gap-2 px-6 py-2.5 text-[11px] font-mono tracking-widest uppercase border transition-all ${
              isStreaming 
                ? 'bg-red-500/10 border-red-500/50 text-red-500 hover:bg-red-500/20' 
                : 'bg-emerald-500 text-black border-emerald-500 hover:bg-emerald-400'
            }`}
          >
            {isStreaming ? <Square className="w-3 h-3" /> : <Play className="w-3 h-3 fill-current" />}
            {isStreaming ? 'Halt Stream' : 'Initialize'}
          </button>
        </div>
      </header>

      {/* MAIN DASHBOARD */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
        
        {/* LEFT: Live Transaction Gateway */}
        <div className="lg:col-span-4 flex flex-col h-[700px]">
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-[11px] text-slate-400 tracking-[0.2em] uppercase font-mono border-l-2 border-emerald-500 pl-3">
              Live Gateway Feed
            </h2>
            <span className="text-[10px] text-slate-600 font-mono">STATUS: {isStreaming ? <span className="text-emerald-500 animate-pulse">ACTIVE</span> : 'STANDBY'}</span>
          </div>
          
          <div className="overflow-y-auto flex-1 pr-2 space-y-2 custom-scrollbar">
            {logs.length === 0 && (
              <div className="h-full flex items-center justify-center border border-white/5 bg-[#0a0b10] border-dashed">
                <p className="text-slate-600 text-xs font-mono tracking-widest uppercase">Awaiting Data...</p>
              </div>
            )}
            
            {logs.map(log => (
              <div 
                key={log.id} 
                onClick={() => log.isFraud && setActiveThreat(log)}
                className={`p-4 border font-mono transition-all group ${
                  log.isFraud 
                    ? 'bg-[#1a0b10] border-red-900/50 cursor-pointer hover:border-red-500/50' 
                    : 'bg-[#0a0b10] border-white/5'
                }`}
              >
                <div className="flex justify-between items-start mb-3">
                  <span className="text-[10px] text-slate-500">{log.time}</span>
                  {log.isFraud ? (
                    <span className="text-[10px] tracking-wider font-bold text-red-500 border border-red-500/30 px-2 py-0.5 bg-red-500/10">BLOCKED</span>
                  ) : (
                    <span className="text-[10px] tracking-wider text-emerald-500">APPROVED</span>
                  )}
                </div>
                <div className="flex justify-between items-end">
                  <span className="text-xl text-white font-light">${log.amount.toFixed(2)}</span>
                  <div className="text-right">
                    <span className="block text-[9px] text-slate-600 uppercase tracking-widest mb-0.5">Risk Score</span>
                    <span className={`text-xs ${log.isFraud ? 'text-red-400' : 'text-slate-400'}`}>
                      {log.score.toFixed(4)}
                    </span>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* RIGHT: XAI Panel */}
        <div className="lg:col-span-8 flex flex-col h-[700px]">
          <h2 className="text-[11px] text-slate-400 tracking-[0.2em] uppercase font-mono border-l-2 border-emerald-500 pl-3 mb-4">
            Explainable AI Diagnostics
          </h2>

          <div className="bg-blue-500/5 border border-blue-500/20 p-4 mb-6 flex gap-4 items-start">
            <Info className="w-4 h-4 text-blue-400 shrink-0 mt-0.5" />
            <p className="text-[11px] text-slate-400 leading-relaxed font-mono">
              <strong className="text-blue-400 uppercase tracking-wider block mb-1">Data Privacy Protocol</strong> 
              Raw PII is mathematically obfuscated via Principal Component Analysis (PCA). Features V1-V28 represent anonymized latent vectors, ensuring strict GDPR compliance while maintaining analytical integrity.
            </p>
          </div>

          {!activeThreat ? (
            <div className="flex-1 border border-white/5 bg-[#0a0b10] flex flex-col items-center justify-center text-slate-600 font-mono text-xs uppercase tracking-widest relative overflow-hidden">
              <div className="absolute inset-0 bg-[radial-gradient(ellipse_at_center,_var(--tw-gradient-stops))] from-white/[0.02] to-transparent"></div>
              <ShieldAlert className="w-12 h-12 mb-4 opacity-20" strokeWidth={1} />
              <p>Monitoring latent space for anomalies</p>
            </div>
          ) : (
            <div className="flex-1 flex flex-col bg-[#0a0b10] border border-white/5 p-6 relative">
              <div className="absolute top-0 left-0 w-full h-1 bg-gradient-to-r from-red-600 to-transparent"></div>
              
              <div className="mb-8">
                <h3 className="text-red-500 font-mono tracking-widest uppercase text-sm mb-3 flex items-center gap-2">
                  <ShieldAlert className="w-4 h-4" /> Critical Anomaly Intercepted
                </h3>
                <p className="text-xs text-slate-400 font-mono leading-relaxed border-l border-red-900/50 pl-4 py-1">
                  Transaction of <span className="text-white">${activeThreat.amount.toFixed(2)}</span> flagged with risk score <span className="text-white">{activeThreat.score.toFixed(4)}</span> (Threshold: 4.84). <br/>
                  The deep autoencoder isolated the following latent features as mathematical drivers:
                </p>
              </div>

              <div className="flex-1 w-full min-h-[300px]">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={activeThreat.anomalies} layout="vertical" margin={{ top: 0, right: 30, left: 20, bottom: 0 }}>
                    <XAxis type="number" stroke="#334155" tick={{ fill: '#475569', fontSize: 10, fontFamily: 'monospace' }} />
                    <YAxis dataKey="feature" type="category" stroke="#334155" width={60} tick={{ fill: '#94a3b8', fontSize: 11, fontFamily: 'monospace' }} />
                    <Tooltip content={<CustomTooltip />} cursor={{fill: '#ffffff05'}} />
                    <Bar dataKey="error" fill="#ef4444" barSize={32} />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}