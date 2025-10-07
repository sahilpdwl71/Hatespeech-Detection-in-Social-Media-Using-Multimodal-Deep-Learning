import React from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import './ResultPage.css';

const Result = () => {
  const location = useLocation();
  const navigate = useNavigate();
  const { result, method, heatmap_image, heatmap_text } = location.state || {};

  return (
    <div className="result-page">
      <div className="result-container">
        <h2 className="result-title">
          <span style={{ color: '#1a2e35' }}>Detection </span>
          <span style={{ color: '#ec4755' }}>Result</span>
        </h2>

        <div className="result-boxs">
          <p><strong>PREDICTION:</strong> {result}</p>
       
        </div>

        <div className="graph-placeholder">
          {heatmap_image ? (
            <img
              src={heatmap_image}
              alt="GradCAM Heatmap"
              className="heatmap-img"
            />
          ) : (
            <p>Graph will appear here (Coming Soon)</p>
          )}
        </div>

        <div className="result-buttons">
          <button className="primary-btn" onClick={() => navigate('/upload')}>
            Try Again
          </button>
          <button
            className="secondary-btn"
            onClick={() => (window.location.href = 'http://localhost:3000')}
          >
            Home
          </button>
        </div>
      </div>
    </div>
  );
};

export default Result;
