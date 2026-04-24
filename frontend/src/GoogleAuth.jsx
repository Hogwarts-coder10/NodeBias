import React from 'react';
import { useNavigate } from 'react-router-dom';

export default function GoogleAuth() {
  const navigate = useNavigate();

  // The fake profiles we will show on the selection screen
  const accounts = [
    { name: "End User", email: "User@nodebias.local", avatar: "U" },
    { name: "Club Evaluator", email: "judge@university.edu", avatar: "C" }
  ];

  const handleSelectAccount = (account) => {
    // 1. Save the fake user to the browser's local storage
    localStorage.setItem('nodebias_user', JSON.stringify(account));
    
    // 2. Simulate the redirect back to our app's dashboard
    navigate('/dashboard');
  };

  return (
    <div style={{ backgroundColor: '#f0f2f5', minHeight: '100vh', display: 'flex', justifyContent: 'center', alignItems: 'center', fontFamily: 'Arial, sans-serif' }}>
      <div style={{ backgroundColor: 'white', padding: '40px 40px', borderRadius: '8px', boxShadow: '0 2px 10px rgba(0,0,0,0.1)', width: '100%', maxWidth: '450px', textAlign: 'center' }}>
        
        {/* Fake Google Logo */}
        <div style={{ display: 'flex', justifyContent: 'center', marginBottom: '16px' }}>
           <svg viewBox="0 0 24 24" width="48" height="48" xmlns="http://www.w3.org/2000/svg">
              <g transform="matrix(1, 0, 0, 1, 27.009001, -39.238998)">
                <path fill="#4285F4" d="M -3.264 51.509 C -3.264 50.719 -3.334 49.969 -3.454 49.239 L -14.754 49.239 L -14.754 53.749 L -8.284 53.749 C -8.574 55.229 -9.424 56.479 -10.684 57.329 L -10.684 60.329 L -6.824 60.329 C -4.564 58.239 -3.264 55.159 -3.264 51.509 Z"/>
                <path fill="#34A853" d="M -14.754 63.239 C -11.514 63.239 -8.804 62.159 -6.824 60.329 L -10.684 57.329 C -11.764 58.049 -13.134 58.489 -14.754 58.489 C -17.884 58.489 -20.534 56.379 -21.484 53.529 L -25.464 53.529 L -25.464 56.619 C -23.494 60.539 -19.444 63.239 -14.754 63.239 Z"/>
                <path fill="#FBBC05" d="M -21.484 53.529 C -21.734 52.809 -21.864 52.039 -21.864 51.239 C -21.864 50.439 -21.724 49.669 -21.484 48.949 L -21.484 45.859 L -25.464 45.859 C -26.284 47.479 -26.754 49.299 -26.754 51.239 C -26.754 53.179 -26.284 54.999 -25.464 56.619 L -21.484 53.529 Z"/>
                <path fill="#EA4335" d="M -14.754 43.989 C -12.984 43.989 -11.404 44.599 -10.154 45.789 L -6.734 42.369 C -8.804 40.429 -11.514 39.239 -14.754 39.239 C -19.444 39.239 -23.494 41.939 -25.464 45.859 L -21.484 48.949 C -20.534 46.099 -17.884 43.989 -14.754 43.989 Z"/>
              </g>
            </svg>
        </div>
        <h2 style={{ margin: '0 0 10px 0', fontSize: '24px', fontWeight: '400', color: '#202124' }}>Choose an account</h2>
        <p style={{ margin: '0 0 30px 0', fontSize: '16px', color: '#5f6368' }}>to continue to <strong style={{color: '#202124'}}>NodeBias Audit Engine</strong></p>

        {/* Account List */}
        <div style={{ borderTop: '1px solid #dadce0', textAlign: 'left' }}>
          {accounts.map((acc, index) => (
            <div 
              key={index} 
              onClick={() => handleSelectAccount(acc)}
              style={{ padding: '12px 24px', borderBottom: '1px solid #dadce0', display: 'flex', alignItems: 'center', cursor: 'pointer', transition: 'background-color 0.2s' }}
              onMouseOver={(e) => e.currentTarget.style.backgroundColor = '#f8f9fa'}
              onMouseOut={(e) => e.currentTarget.style.backgroundColor = 'transparent'}
            >
              <div style={{ width: '40px', height: '40px', borderRadius: '50%', backgroundColor: index === 0 ? '#1967d2' : '#1e8e3e', color: 'white', display: 'flex', justifyContent: 'center', alignItems: 'center', fontSize: '18px', fontWeight: 'bold', marginRight: '16px' }}>
                {acc.avatar}
              </div>
              <div style={{ flex: 1 }}>
                <div style={{ fontSize: '14px', fontWeight: '500', color: '#3c4043' }}>{acc.name}</div>
                <div style={{ fontSize: '12px', color: '#5f6368' }}>{acc.email}</div>
              </div>
            </div>
          ))}
        </div>
        <p style={{ marginTop: '30px', fontSize: '12px', color: '#5f6368', textAlign: 'left', lineHeight: '1.5' }}>
          To continue, Google will share your name, email address, and profile picture with NodeBias Audit Engine. Before using this app, you can review NodeBias Audit Engine's <span style={{color: '#1a73e8', cursor:'pointer'}}>privacy policy</span> and <span style={{color: '#1a73e8', cursor:'pointer'}}>terms of service</span>.
        </p>
      </div>
    </div>
  );
}