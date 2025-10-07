import React, { useState, useEffect } from 'react';
import { useDropzone } from 'react-dropzone';
import { useNavigate } from 'react-router-dom';
import axios from 'axios';
import './UploadPage.css';

const UploadPage = () => {
  const navigate = useNavigate();
  const [file, setFile] = useState(null);
  const [error, setError] = useState('');
  const [preview, setPreview] = useState(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);

  const { getRootProps, getInputProps } = useDropzone({
    onDrop: (acceptedFiles) => {
      const uploadedFile = acceptedFiles[0];
      setFile(uploadedFile);
      setPreview(URL.createObjectURL(uploadedFile));
      setResult(null);
    },
    accept: 'image/*',
  });

  useEffect(() => {
    return () => {
      if (preview) URL.revokeObjectURL(preview);
    };
  }, [preview]);

  const handleUpload = async () => {
    if (!file) {
      setError('Please upload an image first.');
      return;
    }

    const formData = new FormData();
    formData.append('image', file);

    try {
      setLoading(true);
      setError('');
      setResult(null);

      const response = await axios.post(
        'https://clip-8rgq.onrender.com/predict_image',
        formData,
        {
          headers: { 'Content-Type': 'multipart/form-data' },
        }
      );

        const { prediction, method, heatmap_text, heatmap_image } = response.data;
         navigate('/result', {
        state: {
          result: prediction.toUpperCase(),
          method: method,
          heatmap_image : heatmap_image,
          heatmap_text: heatmap_text,
        },
      });
    } catch (err) {
      setError(err.response?.data?.error || 'Error uploading image. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  const handleClear = () => {
    setFile(null);
    setPreview(null);
    setError('');
    setResult(null);
  };

  return (
    <div className="upload-page">
      <div className="upload-container">
        <h2 className="upload-title">
          <span style={{ color: '#1a2e35' }}>Upload </span>
          <span style={{ color: '#ec4755' }}>Image</span>
        </h2>

        <div {...getRootProps()} className="dropzone">
          <input {...getInputProps()} />
          {preview ? (
            <img src={preview} alt="Uploaded" className="preview-image" />
          ) : (
            <p className="text-sm text-tertiary">
              Drag & drop your image here, or click to select from gallery
            </p>
          )}
        </div>
        {error && <p className="upload-error">{error}</p>}

        <div className="upload-buttons">
          {file && (
            <button onClick={handleClear} className="primary-btn secondary-btn">
              Clear
            </button>
          )}
          <button
            onClick={handleUpload}
            className="primary-btn"
            disabled={loading}
          >
            {loading ? <div className="loader"></div> : 'Submit'}
          </button>
        </div>

      </div>
    </div>
  );
};

export default UploadPage;
