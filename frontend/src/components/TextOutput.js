import React from "react";
import { useLocation, useNavigate } from "react-router-dom";
import Plot from "react-plotly.js";
import "./TextOutput.css";

const TextOutput = () => {
  const location = useLocation();
  const navigate = useNavigate();
  const { result, model, top_features } = location.state || {};

  const words = top_features?.map((f) => f.word) || [];
  const shapValues = top_features?.map((f) => f.shap_value || f.importance) || [];

  return (
    <div className="textoutput-page">
      <div className="textoutput-container">
        <h2 className="textoutput-title">
          <span style={{ color: "#1a2e35" }}>Detection </span>
          <span style={{ color: "#ec4755" }}>Result</span>
        </h2>

        <div className="output-box">
          <p><strong>MODEL USED:</strong> {model?.toUpperCase()}</p>
          <p><strong>PREDICTION:</strong> {result}</p>
        </div>

        {top_features && (
          <div className="graph-placeholder">
            <Plot
              data={[
                {
                  type: "waterfall",
                  x: words,
                  y: shapValues,
                  text: shapValues.map((v) => v.toFixed(3)),
                  textposition: "outside",
                  increasing: { marker: { color: "green" } },
                  decreasing: { marker: { color: "red" } },
                },
              ]}
              layout={{
                title: "SHAP Waterfall Plot",
                autosize: true,
                showlegend: false,
                waterfallgap: 0.2,
                margin: { l: 40, r: 20, t: 50, b: 100 },
                xaxis: {
                  automargin: true,
                  tickangle: -30, 
                },
                yaxis: {
                  automargin: true,
                },
              }}
              config={{
                displayModeBar: true, 
                scrollZoom: false, 
              }}
              style={{ width: "100%", height: "450px" }}
              useResizeHandler
            />
          </div>
        )}

        <div className="output-buttons">
          <button className="primary-btn" onClick={() => navigate("/text-detection")}>
            Try Again
          </button>
          <button
            className="secondary-btn"
            onClick={() => (window.location.href = "http://localhost:3000")}
          >
            Home
          </button>
        </div>
      </div>
    </div>
  );
};

export default TextOutput;