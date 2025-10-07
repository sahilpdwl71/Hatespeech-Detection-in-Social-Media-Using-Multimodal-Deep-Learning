from flask import Flask, request, jsonify
import joblib
from scipy.sparse import hstack, csr_matrix
from flask_cors import CORS
import re
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import shap
import numpy as np
import logging
import os
from typing import List, Dict, Optional
from supabase import create_client, Client
import io

# ================================
# Initialize Flask App
# ================================
app = Flask(__name__)
CORS(app)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SUPABASE_URL = "https://fbtvsatqimjqyxwwrksk.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImZidHZzYXRxaW1qcXl4d3dya3NrIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTU2MjgxMDgsImV4cCI6MjA3MTIwNDEwOH0.5hnMGGYm6BL7x8dVN_h3NV33Qdy9evRLc4EAQqBz2zY"
BUCKET_NAME = "model"
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# Global Variables
model = None
word_vectorizer = None
char_vectorizer = None
stop_words = None
lemmatizer = None

label_mapping = {0: "normal", 1: "offensive", 2: "hatespeech"}

# NLTK Downloads
def download_nltk_data():
    try:
        nltk.download('stopwords', quiet=True)
        nltk.download('wordnet', quiet=True)
        nltk.download('omw-1.4', quiet=True)
        logger.info("✅ NLTK data downloaded successfully")
        return True
    except Exception as e:
        logger.error(f" Error downloading NLTK data: {str(e)}")
        return False

def load_from_supabase(filename: str):
    try:
        logger.info(f"⬇️ Downloading {filename} from Supabase...")
        response = supabase.storage.from_(BUCKET_NAME).download(filename)
        if not response:
            raise FileNotFoundError(f"{filename} not found in Supabase bucket '{BUCKET_NAME}'")
        return joblib.load(io.BytesIO(response))
    except Exception as e:
        logger.error(f"❌ Error loading {filename} from Supabase: {str(e)}")
        raise

# Model Loading
def load_models():
    global model, word_vectorizer, char_vectorizer, stop_words, lemmatizer

    try:
        model = load_from_supabase("svm_hater_speech_model.pkl")
        word_vectorizer = load_from_supabase("words_vectorizer.pkl")
        char_vectorizer = load_from_supabase("chars_vectorizer.pkl")

        if not download_nltk_data():
            raise Exception("Failed to download NLTK data")

        stop_words = set(stopwords.words('english'))
        lemmatizer = WordNetLemmatizer()

        logger.info("✅ All models and components loaded successfully from Supabase")
        return True
    except Exception as e:
        logger.error(f"❌ Error loading models: {str(e)}")
        return False


# Text Preprocessing
def clean_text(text: str, remove_stopwords: bool = True) -> str:
    if not isinstance(text, str):
        return ''
    if not text.strip():
        return ''
    try:
        text = text.lower()
        text = re.sub(r"http\S+|www\S+|https\S+", "", text)
        text = re.sub(r"[^a-zA-Z\s]", "", text)

        words = text.split()
        processed_words = []
        for word in words:
            if len(word) > 1:
                lemmatized = lemmatizer.lemmatize(word)
                if not remove_stopwords or lemmatized not in stop_words:
                    processed_words.append(lemmatized)
        return " ".join(processed_words)
    except Exception as e:
        logger.error(f"Error in text cleaning: {str(e)}")
        return text.lower()

def clean_text_for_shap(text: str) -> str:
    return clean_text(text, remove_stopwords=False)

def vectorize_text(texts: List[str]) -> Optional[np.ndarray]:
    try:
        word_features = word_vectorizer.transform(texts)
        char_features = char_vectorizer.transform(texts)
        return hstack([word_features, char_features])
    except Exception as e:
        logger.error(f"Error in vectorization: {str(e)}")
        return None

# Prediction
def predict_single_text(text: str) -> Dict:
    cleaned = clean_text(text)
    features = vectorize_text([cleaned])
    prediction = model.predict(features)[0]

    probabilities = None
    try:
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(features)[0].tolist()
        elif hasattr(model, 'decision_function'):
            scores = model.decision_function(features)
            scores = np.array(scores).flatten()
            exp_scores = np.exp(scores - np.max(scores))
            probabilities = (exp_scores / exp_scores.sum()).tolist()
    except Exception as e:
        logger.warning(f"Could not get prediction probabilities: {str(e)}")

    return {
        'prediction': int(prediction),
        'label': label_mapping.get(prediction, "unknown"),
        'probabilities': probabilities,
        'cleaned_text': cleaned
    }

# SHAP EXPLAINABILITY 
def explain_text_with_shap_magic(text: str, max_features: int = 10) -> List[Dict]:
    """
    Uses multiple approaches to ensure we get meaningful SHAP values
    """
    try:
        cleaned_text = clean_text_for_shap(text)
        if not cleaned_text:
            return []

        
        word_features = word_vectorizer.transform([cleaned_text])
        full_features = vectorize_text([cleaned_text])  
        
        logger.info(f"Analyzing text: '{cleaned_text}'")
        logger.info(f"Word features shape: {word_features.shape}")
        
        if hasattr(model, 'coef_') and model.coef_ is not None:
            try:
                logger.info("Trying LinearExplainer (Method 1)...")
                
                class WordOnlyModel:
                    def __init__(self, original_model, word_vectorizer, char_vectorizer):
                        self.original_model = original_model
                        self.word_vectorizer = word_vectorizer
                        self.char_vectorizer = char_vectorizer
                    
                    def predict_proba(self, word_features_only):
                        dummy_text = " ".join(["dummy"] * word_features_only.shape[0])
                        char_features = self.char_vectorizer.transform([dummy_text] * word_features_only.shape[0])
                        combined = hstack([word_features_only, char_features])
                        
                        if hasattr(self.original_model, 'predict_proba'):
                            return self.original_model.predict_proba(combined)
                        else:
                            
                            scores = self.original_model.decision_function(combined)
                            if len(scores.shape) == 1:
                                scores = scores.reshape(-1, 1)
                            exp_scores = np.exp(scores - np.max(scores, axis=1, keepdims=True))
                            return exp_scores / exp_scores.sum(axis=1, keepdims=True)
                
                word_model = WordOnlyModel(model, word_vectorizer, char_vectorizer)
                
                
                background_size = min(10, word_features.shape[0])
                background = np.zeros((background_size, word_features.shape[1]))
                
                explainer = shap.LinearExplainer(word_model.predict_proba, background)
                shap_values = explainer.shap_values(word_features)
                
              
                predicted_class = model.predict(full_features)[0]
                logger.info(f"Predicted class: {predicted_class} ({label_mapping[predicted_class]})")
                
               
                if isinstance(shap_values, list) and len(shap_values) > predicted_class:
                    class_shap_values = shap_values[predicted_class][0]
                elif isinstance(shap_values, np.ndarray):
                    if len(shap_values.shape) == 3:  
                        class_shap_values = shap_values[0, :, predicted_class]
                    else:
                        class_shap_values = shap_values[0]
                else:
                    raise Exception("Unexpected SHAP values format")
                
                
                try:
                    word_feature_names = list(word_vectorizer.get_feature_names_out())
                except AttributeError:
                    word_feature_names = list(word_vectorizer.get_feature_names())
                
                word_features_dense = word_features.toarray()[0]
                non_zero_indices = np.where(word_features_dense != 0)[0]
                
                logger.info(f"🔍 Found {len(non_zero_indices)} non-zero features")
                
                if len(non_zero_indices) > 0:
                    feature_pairs = []
                    for idx in non_zero_indices:
                        word = word_feature_names[idx]
                        shap_val = float(class_shap_values[idx])
                        if abs(shap_val) > 1e-8:  
                            feature_pairs.append((word, shap_val))
                            logger.info(f"📈 {word}: {shap_val:.6f}")
                    
                    if feature_pairs:
                        sorted_pairs = sorted(feature_pairs, key=lambda x: abs(x[1]), reverse=True)[:max_features]
                        result = [{"word": w, "shap_value": round(s, 4)} for w, s in sorted_pairs]
                        logger.info(f"LinearExplainer success! Found {len(result)} features")
                        return result
                
            except Exception as e:
                logger.warning(f"LinearExplainer failed: {str(e)}")
        
        logger.info("Trying Enhanced KernelExplainer (Method 2)...")
        
        dense_features = full_features.toarray()
        
        np.random.seed(42)  
        background = np.random.normal(0, 0.001, (5, dense_features.shape[1]))
        
        def enhanced_predict_fn(x):
            """Enhanced prediction function that works better with SHAP"""
            try:
                if len(x.shape) == 1:
                    x = x.reshape(1, -1)
                
                x_sparse = csr_matrix(x)
                
                if hasattr(model, 'predict_proba'):
                    probs = model.predict_proba(x_sparse)
                    logger.debug(f"Predict proba shape: {probs.shape}")
                    return probs
                else:
                    scores = model.decision_function(x_sparse)
                    logger.debug(f"Decision scores shape: {scores.shape}")
                    
                    if len(scores.shape) == 1:
                        
                        scores = scores.reshape(-1, 1)
                        
                        prob_pos = 1 / (1 + np.exp(-scores))
                        return np.column_stack([1 - prob_pos, prob_pos])
                    else:
                       
                        exp_scores = np.exp(scores - np.max(scores, axis=1, keepdims=True))
                        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
                        return probs
                        
            except Exception as e:
                logger.error(f"Error in prediction function: {str(e)}")
                n_classes = len(label_mapping)
                n_samples = x.shape[0]
                return np.ones((n_samples, n_classes)) / n_classes
        
        test_pred = enhanced_predict_fn(dense_features)
        logger.info(f"Test prediction shape: {test_pred.shape}, values: {test_pred[0]}")
        
       
        explainer = shap.KernelExplainer(enhanced_predict_fn, background)
        
        logger.info(" Computing SHAP values...")
        shap_values = explainer.shap_values(dense_features, nsamples=100, l1_reg="aic")
       
        predicted_class = model.predict(full_features)[0]
        logger.info(f"Predicted class: {predicted_class}")
        
        
        if isinstance(shap_values, list):
            if len(shap_values) > predicted_class:
                class_shap_values = np.array(shap_values[predicted_class][0]).flatten()
            else:
                class_shap_values = np.array(shap_values[0][0]).flatten()
        else:
            class_shap_values = np.array(shap_values[0]).flatten()
        
        logger.info(f"SHAP values range: [{np.min(class_shap_values):.6f}, {np.max(class_shap_values):.6f}]")
        
        
        try:
            word_feature_names = list(word_vectorizer.get_feature_names_out())
        except AttributeError:
            word_feature_names = list(word_vectorizer.get_feature_names())
        
        word_feature_count = len(word_feature_names)
        word_shap_values = class_shap_values[:word_feature_count]
        word_features_dense = dense_features[0][:word_feature_count]
        
        non_zero_indices = np.where(word_features_dense != 0)[0]
        logger.info(f" Found {len(non_zero_indices)} non-zero word features")
        
        if len(non_zero_indices) == 0:
            return []
        
        threshold = max(1e-6, np.std(word_shap_values) * 0.1)  
        feature_pairs = []
        
        for idx in non_zero_indices:
            word = word_feature_names[idx]
            shap_val = float(word_shap_values[idx])
            if abs(shap_val) > threshold:
                feature_pairs.append((word, shap_val))
                logger.info(f" {word}: {shap_val:.6f}")
        
        if not feature_pairs:
            logger.warning("No significant SHAP values found, lowering threshold...")
            for idx in non_zero_indices:
                word = word_feature_names[idx]
                shap_val = float(word_shap_values[idx])
                feature_pairs.append((word, shap_val))
        
        sorted_pairs = sorted(feature_pairs, key=lambda x: abs(x[1]), reverse=True)[:max_features]
        result = [{"word": w, "shap_value": round(s, 4)} for w, s in sorted_pairs]
        
        logger.info(f"KernelExplainer success! Found {len(result)} features")
        return result
        
    except Exception as e:
        logger.error(f"All SHAP methods failed: {str(e)}")
        
        logger.info(" Trying Fallback Method (Model Coefficients)...")
        try:
            if hasattr(model, 'coef_') and model.coef_ is not None:
                cleaned_text = clean_text_for_shap(text)
                word_features = word_vectorizer.transform([cleaned_text])
                
                full_features = vectorize_text([cleaned_text])
                predicted_class = model.predict(full_features)[0]
                
                if len(model.coef_.shape) > 1:
                    coefficients = model.coef_[predicted_class]
                else:
                    coefficients = model.coef_[0]
                
                try:
                    word_feature_names = list(word_vectorizer.get_feature_names_out())
                except AttributeError:
                    word_feature_names = list(word_vectorizer.get_feature_names())
                
                word_coef = coefficients[:len(word_feature_names)]
                word_features_dense = word_features.toarray()[0]
                non_zero_indices = np.where(word_features_dense != 0)[0]
                
                feature_pairs = []
                for idx in non_zero_indices:
                    word = word_feature_names[idx]
                    importance = float(word_coef[idx] * word_features_dense[idx])
                    feature_pairs.append((word, importance))
                
                sorted_pairs = sorted(feature_pairs, key=lambda x: abs(x[1]), reverse=True)[:max_features]
                result = [{"word": w, "shap_value": round(s, 4)} for w, s in sorted_pairs]
                
                logger.info(f"Fallback method success! Found {len(result)} features")
                return result
        except Exception as fallback_error:
            logger.error(f"Fallback method also failed: {str(fallback_error)}")
        
        return []

# Flask Routes
@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None,
        "word_vectorizer_loaded": word_vectorizer is not None,
        "char_vectorizer_loaded": char_vectorizer is not None
    })

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)
        text = data.get("text", "").strip()
        max_features = min(max(1, data.get("max_features", 10)), 50)

        if not text:
            return jsonify({"error": "Text cannot be empty"}), 400

        logger.info(f"Processing prediction for: '{text[:100]}...'")
        
        pred_result = predict_single_text(text)
        logger.info(f"Getting SHAP explanations...")
        shap_explanations = explain_text_with_shap_magic(text, max_features)

        response = {
            "text": text,
            "cleaned_text": pred_result['cleaned_text'],
            "prediction": pred_result['label'],
            "prediction_id": pred_result['prediction'],
            "top_contributing_words": shap_explanations
        }
        if pred_result['probabilities']:
            response["probabilities"] = [round(p, 4) for p in pred_result['probabilities']]

        logger.info(f"Prediction complete: {pred_result['label']}")
        return jsonify(response)
        
    except Exception as e:
        logger.error(f" Prediction error: {str(e)}")
        return jsonify({"error": "Prediction failed"}), 500

@app.route('/predict_batch', methods=['POST'])
def predict_batch():
    try:
        data = request.get_json(force=True)
        texts = data.get("texts", [])
        if len(texts) > 50:  
            return jsonify({"error": "Batch too large, max 50 texts"}), 400
            
        results = []
        for i, text in enumerate(texts):
            try:
                result = predict_single_text(text)
                results.append(result)
            except Exception as e:
                logger.error(f"Error processing text {i}: {str(e)}")
                results.append({"error": f"Failed to process text {i}"})
        
        return jsonify({"results": results})
    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}")
        return jsonify({"error": "Batch prediction failed"}), 500

# Run App
def startup():
    return load_models()

if __name__ == '__main__':
    print("🚀 Loading SVM Hate Speech Detection Server from Supabase... ")
    if startup():
        print("✅ Models loaded successfully!")
        print("✨ SHAP Magic is ready to work!")
        print("🌐 Server running on http://127.0.0.1:5002")
        print("Endpoints:")
        print("   POST /predict - Single text with SHAP ✨")
        print("   POST /predict_batch - Batch prediction")
        print("   GET /health - Health check")
        app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 10000)))
    else:
        print("❌ Failed to load models.")