import React from 'react';
import { useNavigate } from 'react-router-dom';

export default function Landing() {
  const navigate = useNavigate();

  return (
    <div className="app" style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', minHeight: '100vh', paddingTop: '12vh', overflowX: 'hidden' }}>
      
      {/* --- HERO SECTION --- */}
      <div className="section active anim-d1" style={{ maxWidth: '850px', marginBottom: '4rem', textAlign: 'center' }}>
        <div style={{ display: 'inline-block', padding: '8px 16px', background: 'var(--green-dim)', border: '1px solid rgba(0,245,160,0.2)', borderRadius: '100px', color: 'var(--green)', fontFamily: 'var(--mono)', fontSize: '12px', marginBottom: '24px', letterSpacing: '1px' }}>
          NODEBIAS AUDIT ENGINE v2.0 IS LIVE
        </div>
        <h1 style={{ fontSize: '5.5rem', margin: '0 0 1.5rem 0', fontFamily: 'var(--font)', letterSpacing: '-2.5px', textShadow: '0 0 40px rgba(77,159,255,0.2)', lineHeight: '1.1' }}>
          Node<em style={{ color: 'var(--green)', fontStyle: 'normal' }}>Bias</em>
        </h1>
        <p style={{ fontSize: '1.25rem', color: 'var(--text-muted)', maxWidth: '600px', margin: '0 auto 2.5rem auto', lineHeight: '1.6' }}>
          An enterprise-grade fairness evaluation engine built for clinical healthcare networks. Detect demographic bias in patient predictions, apply algorithmic reweighing, and generate AI-powered explainability reports
        </p>
        <div style={{ display: 'flex', gap: '16px', justifyContent: 'center' }}>
          <button className="btn btn-green" onClick={() => navigate('/login')} style={{ padding: '14px 32px', fontSize: '14px' }}>
            LAUNCH DASHBOARD
          </button>
          <button className="btn-ghost" onClick={() => window.open('https://github.com/Hogwarts-coder10/nodebias', '_blank')} style={{ padding: '14px 32px', fontSize: '14px', background: 'var(--surface)', border: '1px solid var(--border)', color: 'var(--text)', borderRadius: '6px', cursor: 'pointer' }}>
            VIEW SOURCE
          </button>
        </div>
      </div>

      {/* --- CORE CAPABILITIES (NEW) --- */}
      <div className="section active anim-d2" style={{ maxWidth: '1050px', width: '100%', marginBottom: '6rem', padding: '0 2rem' }}>
        <h3 style={{ textAlign: 'center', fontFamily: 'var(--mono)', color: 'var(--blue)', fontSize: '14px', letterSpacing: '2px', marginBottom: '3rem' }}>ENGINE ARCHITECTURE</h3>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))', gap: '24px' }}>
          
          <div className="card" style={{ padding: '2rem', background: 'linear-gradient(145deg, rgba(255,255,255,0.03), rgba(0,0,0,0.2))' }}>
            <div style={{ color: 'var(--green)', fontSize: '24px', marginBottom: '16px' }}>⚙️</div>
            <h4 style={{ fontSize: '1.2rem', marginBottom: '12px' }}>GlassBoxML Baseline</h4>
            <p className="t-dim" style={{ fontSize: '0.95rem', lineHeight: '1.6' }}>Custom-built math engine featuring a proprietary Momentum optimizer. Trains live without opaque .pkl serialization.</p>
          </div>

          <div className="card" style={{ padding: '2rem', background: 'linear-gradient(145deg, rgba(255,255,255,0.03), rgba(0,0,0,0.2))' }}>
            <div style={{ color: 'var(--blue)', fontSize: '24px', marginBottom: '16px' }}>⚖️</div>
            <h4 style={{ fontSize: '1.2rem', marginBottom: '12px' }}>Hybrid Mitigation</h4>
            <p className="t-dim" style={{ fontSize: '0.95rem', lineHeight: '1.6' }}>On-the-fly algorithmic reweighing powered by Sklearn. Dynamically penalizes privileged class advantages to repair Disparate Impact Ratios.</p>
          </div>

          <div className="card" style={{ padding: '2rem', background: 'linear-gradient(145deg, rgba(255,255,255,0.03), rgba(0,0,0,0.2))' }}>
            <div style={{ color: 'var(--purple)', fontSize: '24px', marginBottom: '16px' }}>✨</div>
            <h4 style={{ fontSize: '1.2rem', marginBottom: '12px' }}>Gemini Explainability</h4>
            <p className="t-dim" style={{ fontSize: '0.95rem', lineHeight: '1.6' }}>Direct integration with Google's Gemini API to translate complex probability distributions into executive-ready fairness summaries.</p>
          </div>

        </div>
      </div>

      {/* --- WORKFLOW PIPELINE --- */}
      <div className="section active anim-d3" style={{ width: '100%', maxWidth: '1050px', padding: '0 2rem 6rem 2rem' }}>
        <h3 style={{ textAlign: 'center', fontFamily: 'var(--mono)', color: 'var(--text-muted)', fontSize: '12px', letterSpacing: '2px', marginBottom: '3rem' }}>EXECUTION PIPELINE</h3>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))', gap: '2rem' }}>
          <div style={{ position: 'relative' }}>
            <div style={{ position: 'absolute', top: '12px', left: '-20px', width: '2px', height: '100%', background: 'linear-gradient(to bottom, var(--border), transparent)' }}></div>
            <div style={{ color: 'var(--text-dim)', fontFamily: 'var(--mono)', fontSize: '14px', marginBottom: '10px' }}>01 / INGESTION</div>
            <h4 style={{ margin: '0 0 10px 0', fontSize: '1.1rem' }}>Data Upload</h4>
            <p className="t-dim" style={{ fontSize: '0.95rem', lineHeight: '1.5' }}>Upload raw CSV datasets. Engine dynamically encodes categoricals and normalizes vectors.</p>
          </div>
          <div style={{ position: 'relative' }}>
            <div style={{ position: 'absolute', top: '12px', left: '-20px', width: '2px', height: '100%', background: 'linear-gradient(to bottom, var(--border), transparent)' }}></div>
            <div style={{ color: 'var(--blue)', fontFamily: 'var(--mono)', fontSize: '14px', marginBottom: '10px' }}>02 / CONFIGURATION</div>
            <h4 style={{ margin: '0 0 10px 0', fontSize: '1.1rem' }}>Mitigation Strategy</h4>
            <p className="t-dim" style={{ fontSize: '0.95rem', lineHeight: '1.5' }}>Toggle between pure transparent baseline models and sample-reweighed models.</p>
          </div>
          <div style={{ position: 'relative' }}>
            <div style={{ position: 'absolute', top: '12px', left: '-20px', width: '2px', height: '100%', background: 'linear-gradient(to bottom, var(--border), transparent)' }}></div>
            <div style={{ color: 'var(--amber)', fontFamily: 'var(--mono)', fontSize: '14px', marginBottom: '10px' }}>03 / PROCESSING</div>
            <h4 style={{ margin: '0 0 10px 0', fontSize: '1.1rem' }}>Transparent Execution</h4>
            <p className="t-dim" style={{ fontSize: '0.95rem', lineHeight: '1.5' }}>Pass data through Logistic Regression, Random Forest, or Decision Trees.</p>
          </div>
          <div style={{ position: 'relative' }}>
            <div style={{ color: 'var(--green)', fontFamily: 'var(--mono)', fontSize: '14px', marginBottom: '10px' }}>04 / EVALUATION</div>
            <h4 style={{ margin: '0 0 10px 0', fontSize: '1.1rem' }}>Fairness Audit</h4>
            <p className="t-dim" style={{ fontSize: '0.95rem', lineHeight: '1.5' }}>Calculate Disparate Impact Ratios and demographic disparity gaps.</p>
          </div>
        </div>
      </div>

      {/* --- FOOTER --- */}
      <footer style={{ width: '100%', borderTop: '1px solid var(--border)', padding: '3rem 2rem', textAlign: 'center', background: '#03050a' }}>
        <div style={{ maxWidth: '1050px', margin: '0 auto', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
            <div className="logo-mark" style={{ width: '32px', height: '32px', fontSize: '16px', margin: 0 }}>N</div>
            <span style={{ fontFamily: 'var(--font)', fontWeight: 'bold', fontSize: '1.1rem' }}>NodeBias</span>
          </div>
          <div style={{ color: 'var(--text-dim)', fontSize: '13px' }}>
            Built for Club Evaluation • v2.0 Architecture
          </div>
        </div>
      </footer>
    </div>
  );
}