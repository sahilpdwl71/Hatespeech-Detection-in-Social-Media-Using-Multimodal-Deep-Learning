import React from "react";
import { useNavigate } from "react-router-dom";
import "./GetStarted.css";

const GetStarted = () => {
  const navigate = useNavigate();

  const handleTextDetection = () => {
    navigate("/text-detection");
  };

  const handleImageDetection = () => {
    navigate("/upload");
  };

  return (
    <div className="getstarted-page">
      <div className="getstarted-container">
       

        <div className="detection-buttons">
          <button className="detect-btn" onClick={handleTextDetection}>
            TEXT BASED DETECTION
          </button>
          <button className="detect-btn" onClick={handleImageDetection}>
            IMAGE BASED DETECTION
          </button>
        </div>
      </div>
    </div>
  );
};

export default GetStarted;
