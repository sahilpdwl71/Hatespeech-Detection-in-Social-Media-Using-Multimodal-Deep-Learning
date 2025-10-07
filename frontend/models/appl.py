from flask import Flask, request, jsonify
import joblib
from flask_cors import CORS
import re
import numpy as np
import shap
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import logging
import os
from supabase import create_client, Client



# Initialize Flask App
app = Flask(__name__)
CORS(app)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Global Variables
model = None
tokenizer = None
label_reverse = {0: "Normal", 1: "Offensive", 2: "HateSpeech"}
max_length = 50  

# Supabase credentials
SUPABASE_URL = "https://fbtvsatqimjqyxwwrksk.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImZidHZzYXRxaW1qcXl4d3dya3NrIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTU2MjgxMDgsImV4cCI6MjA3MTIwNDEwOH0.5hnMGGYm6BL7x8dVN_h3NV33Qdy9evRLc4EAQqBz2zY"
BUCKET_NAME = "model"

# Init Supabase client
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

def load_models():
    global model, tokenizer
    try:
        # Download model file as bytes
        model_bytes = supabase.storage.from_(BUCKET_NAME).download("bilstm_hatexplain_3class.h5")
        tokenizer_bytes = supabase.storage.from_(BUCKET_NAME).download("tokenizer.pkl")

        # ✅ Load tokenizer directly from bytes
        tokenizer = joblib.load(io.BytesIO(tokenizer_bytes))

        # ❌ Keras doesn’t support loading .h5 from raw bytes directly
        # ✅ Workaround: wrap bytes in BytesIO and use `tf.keras.models.load_model`
        with io.BytesIO(model_bytes) as f:
            model = tf.keras.models.load_model(f)

        logger.info("Models loaded successfully from Supabase")
        return True

    except Exception as e:
        logger.error(f"Error loading models from Supabase: {str(e)}")
        return False


# Preprocessing
def mild_clean_text(text):
    """Clean and preprocess text input"""
    if not isinstance(text, str):
        return ''
    if not text.strip():
        return ''
    
    text = text.lower()
    text = re.sub(r"http\S+", "", text)  
    text = re.sub(r"@\w+", "", text)     
    text = re.sub(r"#\w+", "", text)      
    text = re.sub(r"\s+", " ", text).strip()  
    return text

def vectorize_texts(texts):
    """Convert texts to padded sequences"""
    if not texts or not isinstance(texts, list):
        return np.array([]), []
    
    cleaned = [mild_clean_text(t) for t in texts]
    if not any(cleaned):  
        return np.array([]), []
    
    try:
        seq = tokenizer.texts_to_sequences(cleaned)
        padded = pad_sequences(seq, maxlen=max_length, padding='post', truncating='post')
        return padded, seq
    except Exception as e:
        logger.error(f"Error in vectorize_texts: {str(e)}")
        return np.array([]), []

def predict_hate_bilstm(text):
    """Make prediction on single text"""
    if not text or not isinstance(text, str):
        raise ValueError("Invalid text input")
    
    padded, seq = vectorize_texts([text])
    if padded.size == 0:
        raise ValueError("Text could not be processed")
    
    try:
        pred = model.predict([padded], verbose=0)
        label_idx = np.argmax(pred[0])
        confidence = float(np.max(pred[0]))
        
        return {
            'label': label_reverse[label_idx],
            'scores': pred[0].tolist(),
            'confidence': confidence,
            'sequence': seq[0] if seq else [],
            'padded': padded
        }
    except Exception as e:
        logger.error(f"Error in prediction: {str(e)}")
        raise


# SHAP Explainability
def explain_text_with_shap(text, nsamples=50):
    """Generate SHAP explanations for text prediction"""
    try:
        pred_result = predict_hate_bilstm(text)
        prediction = pred_result['label']
        scores = pred_result['scores']
        seq = pred_result['sequence']
        padded = pred_result['padded']
        
        if not seq or len([x for x in seq if x != 0]) == 0: 
            return prediction, scores, []
        
        def predict_fn(x):
            if isinstance(x, list):
                x = np.array(x)
            if len(x.shape) == 1:
                x = x.reshape(1, -1)
            elif len(x.shape) == 3:  
                x = x.reshape(x.shape[0], -1)
            
    
            if x.shape[1] != max_length:
                if x.shape[1] < max_length:
                    x = np.pad(x, ((0, 0), (0, max_length - x.shape[1])), mode='constant')
                else:
                    x = x[:, :max_length]
            
            return model.predict(x, verbose=0)
        
        
        background = np.zeros((1, max_length))
        
        
        explainer = shap.KernelExplainer(predict_fn, background)
        
        if len(padded.shape) == 1:
            padded = padded.reshape(1, -1)
      
        shap_values = explainer.shap_values(padded, nsamples=nsamples)
        
        predicted_class = int(np.argmax(scores))
        
        vals = None
        if isinstance(shap_values, list):
            if len(shap_values) > predicted_class:
                vals = np.array(shap_values[predicted_class])
                if len(vals.shape) > 1:
                    vals = vals.flatten()
            else:
                logger.warning(f"SHAP values list length {len(shap_values)} <= predicted class {predicted_class}")
                return prediction, scores, []
        elif isinstance(shap_values, np.ndarray):
            if len(shap_values.shape) == 3:  
                vals = shap_values[0, :, predicted_class] if shap_values.shape[2] > predicted_class else shap_values[0, :, 0]
            elif len(shap_values.shape) == 2:  
                vals = shap_values[0]
            elif len(shap_values.shape) == 1:  
                vals = shap_values
            else:
                logger.warning(f"Unexpected SHAP values shape: {shap_values.shape}")
                return prediction, scores, []
        else:
            logger.warning(f"Unexpected SHAP values type: {type(shap_values)}")
            return prediction, scores, []
        
        if vals is None:
            logger.warning("Could not extract SHAP values")
            return prediction, scores, []
        
      
        vals = np.array(vals).flatten()
        
        
        non_zero_indices = [i for i, idx in enumerate(seq) if idx != 0 and i < len(vals)]
        
        if not non_zero_indices:
            return prediction, scores, []
        
       
        if not hasattr(tokenizer, 'word_index'):
            logger.error("Tokenizer missing word_index")
            return prediction, scores, []
        
        index_word = {v: k for k, v in tokenizer.word_index.items()}
        tokens = []
        token_values = []
        
        for i in non_zero_indices:
            idx = seq[i]
            word = index_word.get(idx, "<OOV>")
            tokens.append(word)
            try:
                val = float(vals[i]) if i < len(vals) else 0.0
                token_values.append(val)
            except (TypeError, ValueError) as e:
                logger.warning(f"Could not convert SHAP value to float: {vals[i]}, error: {e}")
                token_values.append(0.0)
        
        if not tokens:
            return prediction, scores, []
        
        merged_tokens, merged_values = merge_bigrams(tokens, token_values)
        
        top_tokens = sorted(
            zip(merged_tokens, merged_values),
            key=lambda x: abs(x[1]),
            reverse=True
        )[:10]
        
        return prediction, scores, [{"word": w, "shap_value": round(v, 4)} for w, v in top_tokens]
        
    except Exception as e:
        logger.error(f"Error in SHAP explanation: {str(e)}")
        try:
            fallback_result = predict_hate_bilstm(text)
            return fallback_result['label'], fallback_result['scores'], []
        except Exception as fallback_error:
            logger.error(f"Fallback prediction also failed: {str(fallback_error)}")
            return "Normal", [1.0, 0.0, 0.0], []

def merge_bigrams(tokens, values):
    """Merge adjacent tokens with similar SHAP values"""
    if len(tokens) != len(values) or len(tokens) == 0:
        return tokens, values
    
    merged_tokens = []
    merged_values = []
    i = 0
    
    while i < len(tokens):
        current_val = values[i]
        
       
        if (i < len(tokens) - 1 and 
            abs(current_val) > 1e-6 and 
            abs(values[i + 1]) > 1e-6 and  
            np.sign(current_val) == np.sign(values[i + 1])):  
            
        
            merged_tokens.append(f"{tokens[i]} {tokens[i + 1]}")
            merged_values.append(current_val + values[i + 1])
            i += 2
        else:
            merged_tokens.append(tokens[i])
            merged_values.append(current_val)
            i += 1
    
    return merged_tokens, merged_values

# Flask Routes
@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None,
        "tokenizer_loaded": tokenizer is not None
    })

@app.route('/predict', methods=['POST'])
def predict():
    """Main prediction endpoint with SHAP explanations"""
    try:
        if model is None or tokenizer is None:
            return jsonify({"error": "Models not loaded. Please restart the server."}), 500
        
    
        data = request.get_json(force=True)
        if not data or "text" not in data:
            return jsonify({"error": "Please provide a 'text' field in JSON body"}), 400
        
        text = data.get("text", "").strip()
        if not text:
            return jsonify({"error": "Text field cannot be empty"}), 400
        
        if len(text) > 1000: 
            return jsonify({"error": "Text too long. Maximum 1000 characters allowed."}), 400
        
        nsamples = data.get("nsamples", 50)
        if not isinstance(nsamples, int) or nsamples < 10 or nsamples > 200:
            nsamples = 50
        
        prediction, scores, top_features = explain_text_with_shap(text, nsamples)
        
        return jsonify({
            "text": text,
            "prediction": prediction,
            "scores": [round(s, 4) for s in scores],
            "confidence": round(max(scores), 4),
            "top_features": top_features,
            "nsamples_used": nsamples
        })
        
    except ValueError as ve:
        logger.error(f"Validation error: {str(ve)}")
        return jsonify({"error": str(ve)}), 400
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return jsonify({"error": "An error occurred during prediction"}), 500

@app.route('/predict_batch', methods=['POST'])
def predict_batch():
    """Batch prediction endpoint (without SHAP for performance)"""
    try:
        if model is None or tokenizer is None:
            return jsonify({"error": "Models not loaded"}), 500
        
        data = request.get_json(force=True)
        if not data or "texts" not in data:
            return jsonify({"error": "Please provide a 'texts' field with list of texts"}), 400
        
        texts = data.get("texts", [])
        if not isinstance(texts, list) or len(texts) == 0:
            return jsonify({"error": "texts must be a non-empty list"}), 400
        
        if len(texts) > 100:  
            return jsonify({"error": "Batch size too large. Maximum 100 texts allowed."}), 400
        
        
        results = []
        for i, text in enumerate(texts):
            try:
                if not isinstance(text, str):
                    results.append({"error": f"Text at index {i} is not a string"})
                    continue
                
                pred_result = predict_hate_bilstm(text)
                results.append({
                    "text": text,
                    "prediction": pred_result['label'],
                    "scores": [round(s, 4) for s in pred_result['scores']],
                    "confidence": round(pred_result['confidence'], 4)
                })
            except Exception as e:
                results.append({"error": f"Error processing text at index {i}: {str(e)}"})
        
        return jsonify({"results": results})
        
    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}")
        return jsonify({"error": "An error occurred during batch prediction"}), 500


# Application Startu
def startup():
    """Load models at application startup"""
    if not load_models():
        logger.error("Failed to load models. Server may not function properly.")
        return False
    return True


# Run App
if __name__ == '__main__':
    
    if startup():
        print(" Models loaded successfully!")
        print(" Server running on http://127.0.0.1:5003")
        print("Endpoints available:")
        print("   POST /predict - Single text prediction with SHAP")
        print("   POST /predict_batch - Batch prediction (no SHAP)")
        print("   GET /health - Health check")
        app.run(debug=True, port=5003, host='127.0.0.1')
    else:
        print("Failed to load models. Please check model files exist:")
      