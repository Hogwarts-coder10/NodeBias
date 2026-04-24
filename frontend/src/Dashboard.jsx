import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { TABS } from './config';

export default function Dashboard() {
  const navigate = useNavigate();
  const [activeTab, setActiveTab] = useState('overview');
  const [time, setTime] = useState('--:--:--');
  const [sessionId] = useState(() => Math.random().toString(36).slice(2, 8).toUpperCase());
  
  // User Profile State
  const [currentUser, setCurrentUser] = useState({ name: 'Guest Engineer', avatar: 'G' });
  
  // Audit & Dataset State
  const [uploadedFile, setUploadedFile] = useState(null);
  const [csvHeaders, setCsvHeaders] = useState([]);
  const [detectedSensitive, setDetectedSensitive] = useState('');
  const [detectedTarget, setDetectedTarget] = useState('');
  
  const [auditRunning, setAuditRunning] = useState(false);
  const [auditHistory, setAuditHistory] = useState([]);
  const [lastResult, setLastResult] = useState(null);
  const [activeModel, setActiveModel] = useState('Random Forest');
  const [activeMitigation, setActiveMitigation] = useState('none');

  // Terminal State
  const [liveTermLog, setLiveTermLog] = useState([
    { type: 'INFO', msg: 'NodeBias Audit Engine v2.0 initialised' },
    { type: 'INFO', msg: 'Backend: Python Flask (Dynamic Hybrid Mitigation Router)' },
    { type: 'PASS', msg: 'Ready — upload a CSV and run audit' }
  ]);

  // Read the user from memory when the dashboard loads
  useEffect(() => {
    const savedUser = localStorage.getItem('nodebias_user');
    if (savedUser) {
      setCurrentUser(JSON.parse(savedUser));
    }
  }, []);

  // Clock Effect
  useEffect(() => {
    const timer = setInterval(() => {
      setTime(new Date().toLocaleTimeString('en-GB'));
    }, 1000);
    return () => clearInterval(timer);
  }, []);

  const handleLogout = () => {
    // Clear the fake session so you can demo it again from scratch
    localStorage.removeItem('nodebias_user');
    navigate('/'); 
  };

  // Dynamic Universal Dataset Parser
  const parseDatasetSchema = (file) => {
    const reader = new FileReader();
    reader.onload = (e) => {
      const text = e.target.result;
      const firstLine = text.slice(0, text.indexOf('\n')).replace('\r', '');
      const headers = firstLine.split(',').map(h => h.trim().replace(/^"|"$/g, '')).filter(Boolean);
      
      if (headers.length === 0) {
        setLiveTermLog(prev => [...prev, { type: 'FAIL', msg: `Failed to parse CSV schema. File might be empty.` }]);
        return;
      }

      setCsvHeaders(headers);
      setLiveTermLog(prev => [...prev, { type: 'INFO', msg: `Parsed dataset schema: ${headers.length} columns detected.` }]);

      const sensitiveRegex = /gender|sex|race|ethnic|age|nation|religion|marital|protect|minority|disability/i;
      const foundSensitive = headers.find(h => sensitiveRegex.test(h));
      
      const targetRegex = /target|label|class|readmit|outcome|status|fraud|default/i;
      const foundTarget = headers.find(h => targetRegex.test(h));

      if (foundSensitive) {
        setDetectedSensitive(foundSensitive);
        setLiveTermLog(prev => [...prev, { type: 'PASS', msg: `Auto-mapped sensitive attribute: '${foundSensitive}'` }]);
      } else {
        setDetectedSensitive(headers[0]); 
        setLiveTermLog(prev => [...prev, { type: 'WARN', msg: `Could not auto-map sensitive attribute. Defaulting to '${headers[0]}'.` }]);
      }

      if (foundTarget) {
        setDetectedTarget(foundTarget);
      } else {
        setDetectedTarget(headers[headers.length - 1]); 
      }
    };
    reader.readAsText(file.slice(0, 4096)); 
  };

  const handleFileUpload = (e) => {
    const file = e.target.files[0];
    if (file) {
      setUploadedFile(file);
      parseDatasetSchema(file);
    }
  };

  const handleDragOver = (e) => e.preventDefault();
  const handleDrop = (e) => {
    e.preventDefault();
    const file = e.dataTransfer.files[0];
    if (file) {
      setUploadedFile(file);
      parseDatasetSchema(file);
    }
  };

  const runAudit = async () => {
    if (auditRunning) return;
    if (!detectedSensitive || !detectedTarget) {
      setLiveTermLog(prev => [...prev, { type: 'FAIL', msg: `Missing target or sensitive column selections.` }]);
      return;
    }

    setAuditRunning(true);
    setLiveTermLog([]);

    const model = document.getElementById('cfg-model').value;
    const mitigation = document.getElementById('cfg-mitigation').value;
    const backendUrl = 'http://localhost:5000/api/audit';

    setActiveModel(model);
    setActiveMitigation(mitigation);

    setLiveTermLog(prev => [
      ...prev,
      { type: 'INFO', msg: `Forwarding dataset to secure backend` },
      { type: 'INFO', msg: `Strategy: ${mitigation === 'reweighing' ? 'Algorithmic Reweighing' : 'Baseline'}` },
      { type: 'INFO', msg: `Training ${model} on-the-fly...` }
    ]);

    try {
      const formData = new FormData();
      if (!uploadedFile) throw new Error("Please upload a dataset first.");
      
      formData.append('dataset', uploadedFile);
      formData.append('modelType', model);
      formData.append('targetColumn', detectedTarget);
      formData.append('sensitiveColumn', detectedSensitive);
      formData.append('mitigation', mitigation);

      const response = await fetch(backendUrl, { 
        method: 'POST', 
        body: formData 
      });

      if (!response.ok) {
        const txt = await response.text();
        throw new Error(`Server HTTP ${response.status}: ${txt}`);
      }

      const data = await response.json();
      if (data.error) throw new Error(data.error);

      const pass = data.dir_after >= 0.8;
      const result = {
        model, 
        target: detectedTarget, 
        sensitive: detectedSensitive,
        dir_after: parseFloat(data.dir_after),
        disparity_gap_after: parseFloat(data.disparity_gap_after),
        group_rates_after: data.group_rates_after || {},
        features_used: data.features_used,
        status: data.status,
        strategy_used: data.strategy_used,
        ai_summary: data.ai_summary, 
        ts: new Date(),
        pass
      };

      setLastResult(result);
      setAuditHistory(prev => [...prev, result]);
      
      setLiveTermLog(prev => [
        ...prev, 
        { type: 'INFO', msg: `Audit Complete. DIR: ${result.dir_after.toFixed(3)}` },
        { type: pass ? 'PASS' : 'FAIL', msg: `Status: ${result.status}` }
      ]);

      setActiveTab('overview');

    } catch (err) {
      console.error(err);
      setLiveTermLog(prev => [
        ...prev,
        { type: 'WARN', msg: `Error: ${err.message}` }
      ]);
    } finally {
      setAuditRunning(false);
    }
  };

  const exportReport = () => {
    const report = {
      generated: new Date().toISOString(),
      engine: 'NodeBias Backend 2.0',
      audits: auditHistory,
      last_result: lastResult
    };
    const blob = new Blob([JSON.stringify(report, null, 2)], { type: 'application/json' });
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = 'nodebias_v2_audit_report.json';
    a.click();
  };

  return (
    <div className="app">
      {/* Header */}
      <header className="header">
        <a className="logo" href="#">
          <div className="logo-mark">N</div>
          <span className="logo-wordmark">Node<em>Bias</em></span>
          <span className="logo-version">v2.0</span>
        </a>
        <div className="header-right">
          <div className="status-chip">
            <span className="pulse-dot"></span>
            AUDIT ENGINE v2.0 &nbsp;·&nbsp; <span>{time}</span>
          </div>
          
          {/* --- USER PROFILE BADGE --- */}
          <div style={{ display: 'flex', alignItems: 'center', gap: '12px', background: 'var(--surface)', padding: '6px 16px 6px 6px', borderRadius: '100px', border: '1px solid var(--border)', marginLeft: '10px' }}>
            <div style={{ width: '28px', height: '28px', borderRadius: '50%', background: 'linear-gradient(135deg, var(--blue), var(--green))', color: '#000', display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: '14px', fontWeight: 'bold' }}>
              {currentUser.avatar}
            </div>
            <div style={{ display: 'flex', flexDirection: 'column' }}>
              <span style={{ fontSize: '13px', fontWeight: 'bold', color: 'var(--text)', lineHeight: '1.2' }}>{currentUser.name}</span>
              <span style={{ fontSize: '10px', color: 'var(--green)', fontFamily: 'var(--mono)' }}>Authenticated</span>
            </div>
          </div>

          <button className="export-btn" onClick={exportReport} style={{ marginLeft: '10px' }}>↓ EXPORT JSON</button>
          
          <button 
            className="export-btn" 
            onClick={handleLogout} 
            style={{ borderColor: 'var(--red-dim)', color: 'var(--red)', marginLeft: '10px' }}
          >
            ⏻ LOGOUT
          </button>
        </div>
      </header>

      {/* Hero Score Board */}
      <section className="hero">
        <div className="hero-meta">
          <div className="hero-row"><span className="hero-label">Dataset</span><span className="hero-val">{uploadedFile ? uploadedFile.name : '—'}</span></div>
          <div className="hero-row"><span className="hero-label">Features</span><span className="hero-val">{lastResult ? lastResult.features_used : '—'}</span></div>
          <div className="hero-row"><span className="hero-label">Sensitive attr.</span><span className="hero-val">{lastResult ? lastResult.sensitive : '—'}</span></div>
        </div>
        <div className="hero-divider"></div>
        <div className="hero-score">
          <div className="hero-score-label">Disparate Impact Ratio</div>
          <div className={`hero-score-number ${lastResult && !lastResult.pass ? 'fail' : ''}`}>
            {lastResult ? lastResult.dir_after.toFixed(3) : '—'}
          </div>
          <div>
            <span className={`hero-badge ${lastResult ? (lastResult.pass ? '' : 'fail') : 'pending'}`}>
              {lastResult ? (lastResult.pass ? '✓ PASS' : '✗ FAIL') : '◌ AWAITING AUDIT'}
            </span>
          </div>
        </div>
        <div className="hero-divider"></div>
        <div className="hero-meta">
          <div className="hero-row"><span className="hero-label">Model</span><span className="hero-val">{lastResult ? lastResult.model : '—'}</span></div>
          <div className="hero-row"><span className="hero-label">Strategy</span><span className="hero-val">{lastResult ? lastResult.strategy_used : '—'}</span></div>
          <div className="hero-row"><span className="hero-label">Status</span><span className={`hero-val ${lastResult && lastResult.pass ? 'ok' : 'err'}`}>{lastResult ? (lastResult.pass ? 'FAIR' : 'BIASED') : '—'}</span></div>
        </div>
      </section>

      {/* Navigation */}
      <nav className="nav">
        {TABS.map(tab => (
          <button 
            key={tab} 
            className={`nav-btn ${activeTab === tab ? 'active' : ''}`}
            onClick={() => setActiveTab(tab)}
          >
            {tab.toUpperCase()}
          </button>
        ))}
      </nav>

      {/* ─── TAB: OVERVIEW ─── */}
      <div className={`section ${activeTab === 'overview' ? 'active' : ''}`}>
        <div className="grid-4 gap-b">
          <div className="stat-box">
            <div className="stat-label">DIR Score</div>
            <div className={`stat-val ${lastResult?.pass ? 'green' : (lastResult ? 'red' : 'amber')}`}>{lastResult ? lastResult.dir_after.toFixed(3) : '—'}</div>
            <div className="stat-sub">RUN AN AUDIT TO START</div>
          </div>
          <div className="stat-box">
            <div className="stat-label">Demographic Gap</div>
            <div className={`stat-val ${lastResult?.pass ? 'amber' : (lastResult ? 'red' : 'amber')}`}>{lastResult ? `${lastResult.disparity_gap_after}%` : '—'}</div>
            <div className="stat-sub">AWAITING DATA</div>
          </div>
          <div className="stat-box">
             <div className="stat-label">Features Used</div>
             <div className="stat-val blue">{lastResult ? lastResult.features_used : '—'}</div>
             <div className="stat-sub">AUTO-DETECTED</div>
          </div>
          <div className="stat-box">
             <div className="stat-label">Verdict</div>
             <div className={`stat-val ${lastResult?.pass ? 'green' : (lastResult ? 'red' : '')}`}>{lastResult ? (lastResult.pass ? 'PASS' : 'FAIL') : '—'}</div>
             <div className="stat-sub">{lastResult ? lastResult.status.toUpperCase() : 'PENDING'}</div>
          </div>
        </div>

        <div className="terminal">
          <div className="term-bar">
            <div className="term-dots">
              <div className="term-dot" style={{background:'#FF5F57'}}></div>
              <div className="term-dot" style={{background:'#FFBD2E'}}></div>
              <div className="term-dot" style={{background:'#28CA41'}}></div>
            </div>
            <span style={{marginLeft:'10px'}}>nodebias v2.0 — live output</span>
          </div>
          <div className="term-body">
            {liveTermLog.map((log, i) => (
              <div className="term-line" key={i}>
                <span className="t-time">{time}</span>
                <span className={log.type === 'PASS' ? 't-pass' : log.type === 'FAIL' ? 't-fail' : log.type === 'WARN' ? 't-warn' : 't-info'}>{log.type}</span>
                <span className="t-msg">{log.msg}</span>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* ─── TAB: RUN AUDIT ─── */}
      <div className={`section ${activeTab === 'audit' ? 'active' : ''}`}>
        <div className="card gap-b">
          <div className="card-title">Upload Dataset</div>
          <div 
            className={`upload-zone ${uploadedFile ? 'loaded' : ''}`}
            onDragOver={handleDragOver} 
            onDrop={handleDrop}
            onClick={() => document.getElementById('file-upload').click()}
          >
            <input type="file" id="file-upload" className="hidden" accept=".csv" onChange={handleFileUpload} style={{display: 'none'}} />
            <div className="upload-icon">{uploadedFile ? '✓' : '⬆'}</div>
            <div className="upload-text" style={{color: uploadedFile ? 'var(--green)' : ''}}>
              {uploadedFile ? uploadedFile.name : 'Drop your CSV dataset here to begin'}
            </div>
          </div>

          <div className="form-row mt-4" style={{marginTop: '20px'}}>
            <div>
              <label className="form-label">Model Type</label>
              <select className="form-select" id="cfg-model" defaultValue={activeModel}>
                <option value="Logistic Regression">Logistic Regression</option>
                <option value="Random Forest">Random Forest</option>
                <option value="Decision Tree">Decision Tree</option>
              </select>
            </div>
            <div>
              <label className="form-label">Mitigation Strategy</label>
              <select className="form-select" id="cfg-mitigation" defaultValue={activeMitigation}>
                <option value="none">Baseline (No Mitigation)</option>
                <option value="reweighing">Algorithmic Reweighing</option>
              </select>
            </div>
          </div>

          <div className="form-row">
            <div>
              <label className="form-label">
                Target Column
                {csvHeaders.length > 0 && <span style={{marginLeft: '8px', fontSize: '10px', color: 'var(--amber)'}}>(Auto-Mapped)</span>}
              </label>
              <select 
                className="form-select" 
                value={detectedTarget} 
                onChange={(e) => setDetectedTarget(e.target.value)}
                disabled={csvHeaders.length === 0}
                style={{ borderColor: detectedTarget && csvHeaders.length > 0 ? 'var(--amber)' : 'var(--border)' }}
              >
                {csvHeaders.length === 0 ? (
                  <option value="">Awaiting dataset upload...</option>
                ) : (
                  csvHeaders.map(h => <option key={`target-${h}`} value={h}>{h}</option>)
                )}
              </select>
            </div>

            <div>
              <label className="form-label">
                Sensitive Attribute
                {csvHeaders.length > 0 && <span style={{marginLeft: '8px', fontSize: '10px', color: 'var(--blue)'}}>(Auto-Mapped)</span>}
              </label>
              <select 
                className="form-select" 
                value={detectedSensitive} 
                onChange={(e) => setDetectedSensitive(e.target.value)}
                disabled={csvHeaders.length === 0}
                style={{ borderColor: detectedSensitive && csvHeaders.length > 0 ? 'var(--blue)' : 'var(--border)' }}
              >
                {csvHeaders.length === 0 ? (
                  <option value="">Awaiting dataset upload...</option>
                ) : (
                  csvHeaders.map(h => <option key={`sens-${h}`} value={h}>{h}</option>)
                )}
              </select>
            </div>
          </div>
          
          <div className="btn-row">
            <button className="btn btn-green" onClick={runAudit} disabled={auditRunning || csvHeaders.length === 0}>
              {auditRunning ? 'RUNNING...' : '▶ RUN AUDIT'}
            </button>
          </div>
        </div>
      </div>

      {/* ─── TAB: MODELS ─── */}
      <div className={`section ${activeTab === 'models' ? 'active' : ''}`}>
        <div className="card gap-b">
          <div className="card-title">Model Registry — Backend 2.0</div>
          <div className="model-grid" style={{ gridTemplateColumns: '1fr 1fr 1fr' }}>
            <div className="model-card active">
              <div className="mc-name">Logistic Regression</div>
              <div className="mc-type">MOMENTUM OPTIMIZER · BCE LOSS · 10,000 EPOCHS</div>
              <div className={`mc-score ${lastResult && lastResult.model === 'Logistic Regression' ? (lastResult.pass ? 'mc-pass' : 'mc-fail') : 'mc-na'}`}>
                {lastResult && lastResult.model === 'Logistic Regression' ? lastResult.dir_after.toFixed(3) : '—'}
              </div>
              <div className={`mc-status ${lastResult && lastResult.model === 'Logistic Regression' ? (lastResult.pass ? 'pass' : 'fail') : 'na'}`}>
                {lastResult && lastResult.model === 'Logistic Regression' ? (lastResult.pass ? '✓ PASS' : '✗ FAIL') : 'AWAITING AUDIT'}
              </div>
            </div>
            <div className="model-card">
              <div className="mc-name">Random Forest</div>
              <div className="mc-type">ENSEMBLE · 10 TREES · MAX DEPTH 14 · MIN SPLIT 100</div>
              <div className={`mc-score ${lastResult && lastResult.model === 'Random Forest' ? (lastResult.pass ? 'mc-pass' : 'mc-fail') : 'mc-na'}`}>
                {lastResult && lastResult.model === 'Random Forest' ? lastResult.dir_after.toFixed(3) : '—'}
              </div>
              <div className={`mc-status ${lastResult && lastResult.model === 'Random Forest' ? (lastResult.pass ? 'pass' : 'fail') : 'na'}`}>
                {lastResult && lastResult.model === 'Random Forest' ? (lastResult.pass ? '✓ PASS' : '✗ FAIL') : 'AWAITING AUDIT'}
              </div>
            </div>
            <div className="model-card">
              <div className="mc-name">Decision Tree</div>
              <div className="mc-type">SINGLE TREE · MAX DEPTH 14 · SUBSAMPLE 5K</div>
              <div className={`mc-score ${lastResult && lastResult.model === 'Decision Tree' ? (lastResult.pass ? 'mc-pass' : 'mc-fail') : 'mc-na'}`}>
                {lastResult && lastResult.model === 'Decision Tree' ? lastResult.dir_after.toFixed(3) : '—'}
              </div>
              <div className={`mc-status ${lastResult && lastResult.model === 'Decision Tree' ? (lastResult.pass ? 'pass' : 'fail') : 'na'}`}>
                {lastResult && lastResult.model === 'Decision Tree' ? (lastResult.pass ? '✓ PASS' : '✗ FAIL') : 'AWAITING AUDIT'}
              </div>
            </div>
          </div>
        </div>

        <div className="card">
          <div className="card-title">Mitigation Strategy</div>
          <div className="grid-3">
            <div className={`mit-card ${activeMitigation === 'none' ? 'selected' : ''}`}>
              <div className="mit-icon">🚫</div>
              <div className="mit-name">Baseline (No Mitigation)</div>
              <div className="mit-desc">RAW PREDICTIONS</div>
            </div>
            <div className={`mit-card ${activeMitigation === 'reweighing' ? 'selected' : ''}`}>
              <div className="mit-icon">⚖</div>
              <div className="mit-name">Algorithmic Reweighing</div>
              <div className="mit-desc">SAMPLE WEIGHT ADJUSTMENT</div>
            </div>
          </div>
        </div>
      </div>

      {/* ─── TAB: EXPLAINABILITY ─── */}
      <div className={`section ${activeTab === 'explain' ? 'active' : ''}`}>
        
        {/* 👇 NEW GEMINI AI SUMMARY CARD 👇 */}
        <div className="card gap-b" style={{ borderTop: '2px solid var(--amber)', background: 'linear-gradient(180deg, var(--amber-dim) 0%, transparent 100%)' }}>
          <div className="card-title" style={{ color: 'var(--amber)' }}>✨ AI Executive Summary (Powered by Gemini)</div>
          {lastResult && lastResult.ai_summary ? (
            <p style={{ fontSize: '1.1rem', lineHeight: '1.6', color: 'var(--text)', margin: 0 }}>
              {lastResult.ai_summary}
            </p>
          ) : (
            <span className="t-dim">Run an audit to generate an AI explanation.</span>
          )}
        </div>

        <div className="grid-2 gap-b">
          <div className="card">
            <div className="card-title">Data Pipeline Architecture</div>
            <div className="pipe-step" style={{ background: 'var(--blue-dim)', borderColor: 'rgba(77,159,255,0.25)' }}>
              <span className="pipe-num" style={{ color: 'var(--blue)' }}>STEP 1 · INPUT</span>
              React form → Vite Proxy → Python Flask
            </div>
            <div className="pipe-arrow">↓</div>
            <div className="pipe-step" style={{ background: 'var(--amber-dim)', borderColor: 'rgba(255,184,0,0.25)' }}>
              <span className="pipe-num" style={{ color: 'var(--amber)' }}>STEP 2 · CLEAN & ENCODE</span>
              Handle NaNs · .cat.codes for objects
            </div>
            <div className="pipe-arrow">↓</div>
            <div className="pipe-step" style={{ background: 'var(--purple-dim)', borderColor: 'rgba(139,92,246,0.25)' }}>
              <span className="pipe-num" style={{ color: 'var(--purple)' }}>STEP 3 · MITIGATION</span>
              Check for active mitigation bounds (e.g., Reweighing sample weights)
            </div>
            <div className="pipe-arrow">↓</div>
            <div className="pipe-step" style={{ background: 'var(--red-dim)', borderColor: 'rgba(255,77,109,0.2)' }}>
              <span className="pipe-num" style={{ color: 'var(--red)' }}>STEP 4 · TRAIN</span>
              On-the-fly model training (LR, RF, or DT)
            </div>
            <div className="pipe-arrow">↓</div>
            <div className="pipe-step" style={{ background: 'var(--green-dim)', borderColor: 'rgba(0,245,160,0.3)' }}>
              <span className="pipe-num" style={{ color: 'var(--green)' }}>STEP 5 · AUDIT</span>
              Predictions grouped by sensitive attribute → DIR = min÷max
            </div>
          </div>

          <div className="card">
            <div className="card-title">API Response Snapshot</div>
            {lastResult ? (
              <div className="tree-box" style={{ fontSize: '11px', lineHeight: '1.8' }}>
                <span style={{ color: 'var(--text-dim)' }}>{'{'}</span><br/>
                &nbsp;&nbsp;<span className="feat">"strategy_used"</span>: <span className="val">"{lastResult.strategy_used}"</span>,<br/>
                &nbsp;&nbsp;<span className="feat">"features_used"</span>: <span className="val">{lastResult.features_used}</span>,<br/>
                &nbsp;&nbsp;<span className="feat">"disparity_gap_after"</span>: <span className="val">{lastResult.disparity_gap_after}</span>,<br/>
                &nbsp;&nbsp;<span className="feat">"dir_after"</span>: <span className="val">{lastResult.dir_after.toFixed(3)}</span>,<br/>
                &nbsp;&nbsp;<span className="feat">"status"</span>: <span className="val">"{lastResult.status}"</span><br/>
                <span style={{ color: 'var(--text-dim)' }}>{'}'}</span>
              </div>
            ) : (
              <div className="term-line"><span className="t-dim">Run an audit to view real-time payload.</span></div>
            )}
          </div>
        </div>
      </div>

      {/* ─── TAB: GLASSBOXML ─── */}
      <div className={`section ${activeTab === 'glassbox' ? 'active' : ''}`}>
        <div className="card gap-b" style={{ borderTop: '2px solid var(--purple)' }}>
          <div className="card-title">GlassBoxML — Engine Status</div>
          <p style={{ fontSize: '13px', color: 'var(--text-muted)', lineHeight: '1.75', margin: 0 }}>
            GlassBoxML exposes the internal routing of the Python backend. Backend 2.0 trains models <strong style={{ color: 'var(--text)' }}>on-the-fly</strong> — no serialised .pkl files, no stale weights.
          </p>
        </div>

        <div className="grid-3 gap-b">
          <div className="card" style={{ borderTop: '2px solid var(--blue)' }}>
            <div className="card-title" style={{ color: 'var(--blue)' }}>glassboxml.core</div>
            <div style={{ fontSize: '12px', color: 'var(--text-muted)', lineHeight: '1.7' }}>
              <div style={{ marginBottom: '10px' }}><code style={{ color: 'var(--amber)', fontFamily: 'var(--mono)' }}>train_test_split()</code> — Zero data leakage.</div>
              <div><code style={{ color: 'var(--amber)', fontFamily: 'var(--mono)' }}>Momentum</code> — Custom gradient-descent optimiser.</div>
            </div>
          </div>
          <div className="card" style={{ borderTop: '2px solid var(--amber)' }}>
            <div className="card-title" style={{ color: 'var(--amber)' }}>preprocessing</div>
            <div style={{ fontSize: '12px', color: 'var(--text-muted)', lineHeight: '1.7' }}>
              <code style={{ color: 'var(--amber)', fontFamily: 'var(--mono)' }}>StandardScaler</code> — fit-transform pipeline that centres features to μ=0, σ=1 in a single pass.
            </div>
          </div>
          <div className="card" style={{ borderTop: '2px solid var(--green)' }}>
            <div className="card-title" style={{ color: 'var(--green)' }}>mitigation</div>
            <div style={{ fontSize: '12px', color: 'var(--text-muted)', lineHeight: '1.7' }}>
              <code style={{ color: 'var(--green)', fontFamily: 'var(--mono)' }}>compute_sample_weight</code> — Dynamically adjusts training penalization based on privileged class distributions.
            </div>
          </div>
        </div>
      </div>

      {/* ─── TAB: LOG ─── */}
      <div className={`section ${activeTab === 'log' ? 'active' : ''}`}>
         <div className="terminal">
          <div className="term-bar">
            <span style={{marginLeft:'10px'}}>Audit History</span>
          </div>
          <div className="term-body">
             {auditHistory.length === 0 ? (
               <div className="term-line"><span className="t-dim">No audits run yet.</span></div>
             ) : (
               auditHistory.map((h, i) => (
                 <div className="term-line" key={i}>
                   <span className="t-time">{h.ts.toLocaleTimeString()}</span>
                   <span className="t-info">[{h.model}]</span>
                   <span className="t-msg">Strategy: {h.strategy_used} | DIR: <strong style={{color: h.pass ? 'var(--green)' : 'var(--red)'}}>{h.dir_after.toFixed(3)}</strong></span>
                 </div>
               ))
             )}
          </div>
        </div>
      </div>

    </div>
  );
}