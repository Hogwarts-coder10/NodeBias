import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';


export default function Login() {
  const navigate = useNavigate();
  const [loading, setLoading] = useState(false);
  const [googleLoading, setGoogleLoading] = useState(false);

  const handleLogin = (e) => {
    e.preventDefault();
    setLoading(true);
    setTimeout(() => {
      navigate('/dashboard');
    }, 1200); 
  };

  const handleGoogleLogin = () => {
    setGoogleLoading(true);
    // Simulate OAuth delay
    navigate('/auth/google')
  };

  return (
    <div className="app section active anim-d1" style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', minHeight: '100vh' }}>
      
      <div className="card" style={{ width: '100%', maxWidth: '420px', padding: '3rem 2.5rem', position: 'relative', overflow: 'hidden' }}>
        
        {/* Decorative Top Bar */}
        <div style={{ position: 'absolute', top: 0, left: 0, right: 0, height: '3px', background: 'linear-gradient(90deg, var(--blue), var(--green))' }}></div>

        <div style={{ textAlign: 'center', marginBottom: '2.5rem' }}>
          <div className="logo-mark" style={{ margin: '0 auto 20px auto', width: '48px', height: '48px', fontSize: '22px' }}>N</div>
          <h2 style={{ margin: '0 0 8px 0', fontFamily: 'var(--font)', fontSize: '1.5rem', letterSpacing: '-0.5px' }}>Access NodeBias</h2>
          <p className="t-dim" style={{ fontSize: '0.9rem' }}>Authenticate to initialize the audit engine.</p>
        </div>

        {/* --- GOOGLE LOGIN BUTTON --- */}
        <button 
          onClick={handleGoogleLogin}
          disabled={googleLoading || loading}
          style={{ 
            width: '100%', 
            display: 'flex', 
            alignItems: 'center', 
            justifyContent: 'center', 
            gap: '12px',
            padding: '12px', 
            background: 'white', 
            color: '#000', 
            border: 'none', 
            borderRadius: '6px', 
            fontWeight: 'bold',
            fontSize: '14px',
            cursor: (googleLoading || loading) ? 'not-allowed' : 'pointer',
            transition: 'opacity 0.2s',
            opacity: (googleLoading || loading) ? 0.7 : 1,
            marginBottom: '1.5rem'
          }}
        >
          {googleLoading ? (
            <span className="spinner" style={{ borderColor: 'rgba(0,0,0,0.2)', borderTopColor: '#000', width: '16px', height: '16px' }}></span>
          ) : (
            <svg viewBox="0 0 24 24" width="18" height="18" xmlns="http://www.w3.org/2000/svg">
              <g transform="matrix(1, 0, 0, 1, 27.009001, -39.238998)">
                <path fill="#4285F4" d="M -3.264 51.509 C -3.264 50.719 -3.334 49.969 -3.454 49.239 L -14.754 49.239 L -14.754 53.749 L -8.284 53.749 C -8.574 55.229 -9.424 56.479 -10.684 57.329 L -10.684 60.329 L -6.824 60.329 C -4.564 58.239 -3.264 55.159 -3.264 51.509 Z"/>
                <path fill="#34A853" d="M -14.754 63.239 C -11.514 63.239 -8.804 62.159 -6.824 60.329 L -10.684 57.329 C -11.764 58.049 -13.134 58.489 -14.754 58.489 C -17.884 58.489 -20.534 56.379 -21.484 53.529 L -25.464 53.529 L -25.464 56.619 C -23.494 60.539 -19.444 63.239 -14.754 63.239 Z"/>
                <path fill="#FBBC05" d="M -21.484 53.529 C -21.734 52.809 -21.864 52.039 -21.864 51.239 C -21.864 50.439 -21.724 49.669 -21.484 48.949 L -21.484 45.859 L -25.464 45.859 C -26.284 47.479 -26.754 49.299 -26.754 51.239 C -26.754 53.179 -26.284 54.999 -25.464 56.619 L -21.484 53.529 Z"/>
                <path fill="#EA4335" d="M -14.754 43.989 C -12.984 43.989 -11.404 44.599 -10.154 45.789 L -6.734 42.369 C -8.804 40.429 -11.514 39.239 -14.754 39.239 C -19.444 39.239 -23.494 41.939 -25.464 45.859 L -21.484 48.949 C -20.534 46.099 -17.884 43.989 -14.754 43.989 Z"/>
              </g>
            </svg>
          )}
          {googleLoading ? 'AUTHENTICATING...' : 'Continue with Google'}
        </button>

        {/* --- DIVIDER --- */}
        <div style={{ display: 'flex', alignItems: 'center', margin: '1.5rem 0' }}>
          <div style={{ flex: 1, height: '1px', background: 'var(--border)' }}></div>
          <span style={{ padding: '0 12px', fontSize: '12px', color: 'var(--text-dim)', fontFamily: 'var(--mono)' }}>OR SYSTEM LOGIN</span>
          <div style={{ flex: 1, height: '1px', background: 'var(--border)' }}></div>
        </div>

        <form onSubmit={handleLogin}>
          <div className="form-row" style={{ display: 'block', marginBottom: '1.5rem' }}>
            <label className="form-label" style={{ color: 'var(--text)' }}>Engineer ID</label>
            <input 
              className="form-input" 
              type="text" 
              placeholder="v.karthik@nodebias.local" 
              required 
              style={{ width: '100%', boxSizing: 'border-box', background: 'rgba(0,0,0,0.2)', padding: '12px 16px' }} 
            />
          </div>
          
          <div className="form-row" style={{ display: 'block', marginBottom: '2.5rem' }}>
            <label className="form-label" style={{ color: 'var(--text)' }}>Access Token</label>
            <input 
              className="form-input" 
              type="password" 
              placeholder="••••••••" 
              required 
              style={{ width: '100%', boxSizing: 'border-box', background: 'rgba(0,0,0,0.2)', padding: '12px 16px', letterSpacing: '4px' }} 
            />
          </div>

          <button 
            type="submit" 
            className="btn btn-green" 
            style={{ width: '100%', justifyContent: 'center', padding: '14px 0', fontSize: '13px' }}
            disabled={loading || googleLoading}
          >
            {loading ? <span className="spinner" style={{ marginRight: '8px' }}></span> : null}
            {loading ? 'VERIFYING CREDENTIALS...' : 'ENTER DASHBOARD'}
          </button>
        </form>

        <div style={{ textAlign: 'center', marginTop: '2rem' }}>
          <button 
            className="btn-ghost" 
            style={{ background: 'none', border: 'none', fontFamily: 'var(--mono)', fontSize: '10px', cursor: 'pointer', textDecoration: 'underline', color: 'var(--text-muted)' }}
          >
            REQUEST ACCESS PROVISIONING
          </button>
        </div>
      </div>
    </div>
  );
}