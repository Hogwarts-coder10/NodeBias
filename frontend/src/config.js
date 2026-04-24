export const TABS = ['overview', 'audit', 'models', 'explain', 'glassbox', 'log'];

export const FULL_LOG_STATIC = [
  { type: 'INFO', msg: 'Session initialized' },
  { type: 'INFO', msg: 'NodeBias Backend 2.0 — Dynamic On-The-Fly Mitigation' },
  { type: 'INFO', msg: 'Node.js server listening on port 3000' },
  { type: 'INFO', msg: 'Python Flask engine listening on port 5000' },
  { type: 'INFO', msg: 'mitigation_engine.py loaded — GlassBoxML models ready' },
  { type: 'INFO', msg: 'Supported: Logistic Regression (Momentum, BCE, 10k epochs)' },
  { type: 'INFO', msg: 'Supported: Random Forest (10 trees, depth 14, min_split 100)' },
  { type: 'INFO', msg: 'Supported: Decision Tree (depth 14, subsampled to 5k records)' },
  { type: 'WARN', msg: 'No .pkl serialisation — models train live on uploaded data' },
  { type: 'INFO', msg: 'DIR threshold: 0.800 (EEOC 80% rule)' },
];

export const PRE_LOG_STEPS = [
  { type: 'INFO', msg: 'Standardizing nulls: ?, NA, N/A → NaN' },
  { type: 'INFO', msg: 'Binarising target column dynamically...' },
  { type: 'INFO', msg: 'Auto-encoding categorical columns via .cat.codes' },
  { type: 'INFO', msg: 'Auto-detecting safe numerical features...' },
  { type: 'INFO', msg: 'StandardScaler: fit_transform complete' }
];