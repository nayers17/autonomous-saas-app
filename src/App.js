// src/App.js

import React from 'react';
import SentimentAnalysis from './SentimentAnalysis';
import CodeGeneration from './CodeGeneration';
import LeadGeneration from './LeadGeneration';
import './App.css';

function App() {
  return (
    <div className="App">
      <header>
        <h1>Autonomous SaaS Application</h1>
      </header>
      <main>
        <SentimentAnalysis />
        <CodeGeneration />
        <LeadGeneration />
      </main>
    </div>
  );
}

export default App;
