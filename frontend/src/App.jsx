import React from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import Landing from './Landing';
import Login from './Login';
import Dashboard from './dashboard';
import GoogleAuth from './GoogleAuth';

export default function App() {
  return (
    <Router>
      <div className="bg-grid"></div>
      <div className="bg-glow bg-glow-1"></div>
      <div className="bg-glow bg-glow-2"></div>
      <div className="bg-noise"></div>
      <div className="bg-scan"></div>
      
      <Routes>
        <Route path="/" element={<Landing />} />
        <Route path="/login" element={<Login />} />
        <Route path="/auth/google" element={<GoogleAuth/>} />
        <Route path="/dashboard" element={<Dashboard />} />
        {/* Fallback route */}
        <Route path="*" element={<Navigate to="/" />} />
      </Routes>
    </Router>
  );
}