import React, { useState } from "react";
import { useNavigate } from "react-router-dom";
import axios from "axios";
import "./TextDetection.css";

const TextDetection = () => {
  const [inputText, setInputText] = useState("");
  const [selectedModel, setSelectedModel] = useState("svm");
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false); 
  const navigate = useNavigate();

  const handleSubmit = async () => {
    if (!inputText.trim()) {
      setError("Please enter some text first.");
      return;
    }
    setError("");
    setLoading(true); 

    let apiUrl;
    if (selectedModel === "svm") {
      apiUrl = "https://svm-aqud.onrender.com/predict";
    } else if (selectedModel === "bert") {
      apiUrl = "https://bert-vc6f.onrender.com/predict";
    } else {
      apiUrl = "https://lstm-v2c0.onrender.com/predict"; 
    }

    try {
      const requestData = selectedModel === "bert" 
        ? { text: inputText, include_shap: true } 
        : { text: inputText };

      const response = await axios.post(apiUrl, requestData);

      navigate("/text-output", {
        state: {
          result: response.data.prediction,
          model: selectedModel,
          top_features:
            response.data.top_contributing_words || 
            response.data.word_importance || 
            response.data.top_features,
        },
      });
    } catch (err) {
      setError("Error connecting to the server. Try again.");
    } finally {
      setLoading(false); 
    }
  };

  const handleClearText = () => {
    setInputText("");
    setError("");
  };

  return (
    <div className="textdetect-page">
      <div className="textdetect-container">
        <h2 className="textdetect-title">
          <span style={{ color: "#1a2e35" }}>Text </span>
          <span style={{ color: "#ec4755" }}>Detection</span>
        </h2>

        <div className="text-input-wrapper">
          <textarea
            className="text-input"
            rows="5"
            placeholder="Enter text here..."
            value={inputText}
            onChange={(e) => setInputText(e.target.value)}
          ></textarea>
        </div>

        <select
          className="model-dropdown"
          value={selectedModel}
          onChange={(e) => setSelectedModel(e.target.value)}
        >
          <option value="svm">SVM Model</option>
          <option value="lstm">LSTM Model</option>
          <option value="bert">BERT Model</option>
        </select>

        {error && <p className="error-message">{error}</p>}

        <div className="button-group">
          <button className="secondary-btn" onClick={handleClearText}>
            Clear
          </button>
          <button
            className="primary-btn"
            onClick={handleSubmit}
            disabled={loading} 
          >
            {loading ? (
              <div className="loader"></div>
            ) : (
              "Submit"
            )}
          </button>
        </div>
      </div>
    </div>
  );
};

export default TextDetection;