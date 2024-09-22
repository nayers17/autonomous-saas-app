// src/SentimentAnalysis.js

import React, { useState } from 'react';
import axios from 'axios';

function SentimentAnalysis() {
  const [text, setText] = useState('');
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const analyzeSentiment = async () => {
    setLoading(true);
    setError(null);
    try {
      const response = await axios.post('http://127.0.0.1:8000/sentiment-analysis/', {
        text: text,
      });
      setResult(response.data);
    } catch (err) {
      setError('Error analyzing sentiment.');
    }
    setLoading(false);
  };

  return (
    <div className="service-container">
      <h2>Sentiment Analysis</h2>
      <textarea
        value={text}
        onChange={(e) => setText(e.target.value)}
        rows="4"
        cols="50"
        placeholder="Enter text here..."
      />
      <br />
      <button onClick={analyzeSentiment} disabled={loading}>
        {loading ? 'Analyzing...' : 'Analyze Sentiment'}
      </button>
      {result && (
        <div className="result">
          <p><strong>Sentiment:</strong> {result.sentiment}</p>
          <p><strong>Confidence:</strong> {result.confidence}</p>
        </div>
      )}
      {error && <p className="error">{error}</p>}
    </div>
  );
}

export default SentimentAnalysis;
