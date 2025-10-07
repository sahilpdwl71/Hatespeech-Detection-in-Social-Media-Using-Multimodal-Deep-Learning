from flask import Flask, request, jsonify
import torch
import shap
import numpy as np
import matplotlib
matplotlib.use('Agg')  
import matplotlib.pyplot as plt
import base64
from io import BytesIO
from flask_cors import CORS
import logging
from typing import List, Dict, Optional, Tuple, Any
from transformers import BertTokenizer, BertForSequenceClassification
from supabase import create_client, Client
import os
import tempfile
import json

SUPABASE_URL = "https://fbtvsatqimjqyxwwrksk.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImZidHZzYXRxaW1qcXl4d3dya3NrIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTU2MjgxMDgsImV4cCI6MjA3MTIwNDEwOH0.5hnMGGYm6BL7x8dVN_h3NV33Qdy9evRLc4EAQqBz2zY"
BUCKET_NAME = "model"
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# Initialize Flask App
app = Flask(__name__)
CORS(app)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Global Variables
model = None
tokenizer = None
device = None
explainer = None

label_mapping = {0: "Normal", 1: "Offensive", 2: "HateSpeech"}
reverse_labels = {0: "Normal", 1: "Offensive", 2: "HateSpeech"}
label_colors = {"Normal": "green", "Offensive": "orange", "HateSpeech": "red"}

def download_file_from_supabase(file_name: str) -> bytes:
    """Download file from Supabase Storage"""
    try:
        data = supabase.storage.from_(BUCKET_NAME).download(file_name)
        return data
    except Exception as e:
        logger.error(f"Failed to download {file_name}: {str(e)}")
        raise


# Model Loading
def load_models():
    global model, tokenizer, device, explainer

    try:
        logger.info("Downloading model files from Supabase...")

        # Required files
        files = {
            "config.json": download_file_from_supabase("bert_model/config.json"),
            "model.safetensors": download_file_from_supabase("bert_model/model.safetensors"),
            "tokenizer_config.json": download_file_from_supabase("bert_model/tokenizer_config.json"),
            "vocab.txt": download_file_from_supabase("bert_model/vocab.txt")
        }

        # Create a temporary directory to hold them
        with tempfile.TemporaryDirectory() as tmpdir:
            for fname, data in files.items():
                fpath = os.path.join(tmpdir, fname)
                with open(fpath, "wb") as f:
                    f.write(data)

            # Load tokenizer and model from temp dir
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            logger.info(f"Using device: {device}")

            logger.info("Loading BERT model...")
            model = BertForSequenceClassification.from_pretrained(tmpdir)
            tokenizer = BertTokenizer.from_pretrained(tmpdir)

            model.to(device)
            model.eval()

            logger.info("Initializing SHAP explainer...")
            try:
                explainer = shap.Explainer(model, tokenizer)
                logger.info("SHAP explainer initialized successfully")
            except Exception as shap_error:
                logger.warning(f"SHAP explainer initialization failed: {str(shap_error)}")
                explainer = None

        logger.info("✅ Models and tokenizer loaded successfully from Supabase")
        return True

    except Exception as e:
        logger.error(f"Error loading models from Supabase: {str(e)}")
        return False


# Prediction Functions
def predict_with_probabilities(text: str) -> Tuple[int, np.ndarray]:
    """Get prediction and probabilities for a text"""
    try:
        inputs = tokenizer(text, 
                          return_tensors="pt", 
                          padding=True, 
                          truncation=True, 
                          max_length=128).to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            probabilities = torch.softmax(outputs.logits, dim=1)
            pred = torch.argmax(outputs.logits, dim=1).item()

        return pred, probabilities.cpu().numpy()

    except Exception as e:
        logger.error(f"Error in prediction: {str(e)}")
        raise

def simple_predict_text(text: str) -> str:
    """Simple prediction function without probabilities"""
    try:
        inputs = tokenizer(text, 
                          return_tensors="pt", 
                          padding=True, 
                          truncation=True, 
                          max_length=128).to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            pred = torch.argmax(outputs.logits, dim=1).item()

        return reverse_labels[pred]

    except Exception as e:
        logger.error(f"Error in simple prediction: {str(e)}")
        raise


# BERT WORD IMPORTANCE
def enhanced_word_importance_analysis(text: str, max_words: int = 10) -> Dict[str, Any]:
    """
    Analyzes word importance by systematically removing words and observing prediction changes
    """
    try:
        logger.info(f"Starting word importance analysis for: '{text[:50]}...'")
        
        
        baseline_pred, baseline_probs = predict_with_probabilities(text)
        baseline_label = reverse_labels[baseline_pred]
        baseline_confidence = baseline_probs[0][baseline_pred]

        logger.info(f" Baseline prediction: {baseline_label} (confidence: {baseline_confidence:.4f})")

        words = text.split()
        if len(words) == 0:
            return {
                "baseline_prediction": baseline_label,
                "baseline_confidence": float(baseline_confidence),
                "baseline_probabilities": baseline_probs[0].tolist(),
                "word_importance": [],
                "error": "No words found in text"
            }

        word_importance = []
        word_effects = []

        
        for i, word in enumerate(words):
            try:
                
                masked_text = ' '.join(words[:i] + words[i+1:])

                if masked_text.strip():
                    masked_pred, masked_probs = predict_with_probabilities(masked_text)
                    masked_label = reverse_labels[masked_pred]
                    masked_confidence = masked_probs[0][baseline_pred]

                    
                    importance = float(baseline_confidence - masked_confidence)
                    word_importance.append((word, importance))

                    
                    effects = {
                        'word': word,
                        'importance': importance,
                        'normal_change': float(masked_probs[0][0] - baseline_probs[0][0]),
                        'offensive_change': float(masked_probs[0][1] - baseline_probs[0][1]),
                        'hatespeech_change': float(masked_probs[0][2] - baseline_probs[0][2]),
                        'new_prediction': masked_label,
                        'confidence_change': float(masked_confidence - baseline_confidence)
                    }
                    word_effects.append(effects)

                    logger.debug(f"'{word}': importance={importance:.4f}, new_pred={masked_label}")

            except Exception as word_error:
                logger.warning(f"Error analyzing word '{word}': {str(word_error)}")
                continue

        word_importance.sort(key=lambda x: abs(x[1]), reverse=True)
        
        top_words = []
        for word, importance in word_importance[:max_words]:
            top_words.append({
                "word": word,
                "importance": round(importance, 4),
                "effect": "strengthens" if importance > 0 else "weakens"
            })

        logger.info(f"Word importance analysis complete. Found {len(word_importance)} words")

        return {
            "baseline_prediction": baseline_label,
            "baseline_confidence": round(float(baseline_confidence), 4),
            "baseline_probabilities": {
                "Normal": round(float(baseline_probs[0][0]), 4),
                "Offensive": round(float(baseline_probs[0][1]), 4),
                "HateSpeech": round(float(baseline_probs[0][2]), 4)
            },
            "word_importance": top_words,
            "word_effects": word_effects[:max_words],
            "total_words_analyzed": len(words)
        }

    except Exception as e:
        logger.error(f"Word importance analysis failed: {str(e)}")
        return {
            "error": f"Analysis failed: {str(e)}",
            "baseline_prediction": None,
            "word_importance": []
        }

def create_importance_visualization(word_importance: List[Tuple[str, float]], 
                                   text: str, 
                                   baseline_label: str) -> Optional[str]:
    """
    Create visualization for word importance and return as base64 encoded image
    """
    try:
        if len(word_importance) == 0:
            return None

        words, importances = zip(*word_importance[:10])  
        colors = ['red' if imp > 0 else 'blue' for imp in importances]

        plt.figure(figsize=(12, 6))
        bars = plt.bar(range(len(words)), importances, color=colors, alpha=0.7)
        
        plt.xlabel('Words')
        plt.ylabel(f'Importance for {baseline_label} prediction')
        plt.title(f'Word Importance Analysis\n"{text[:50]}..."')
        plt.xticks(range(len(words)), words, rotation=45, ha='right')
        plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)

        for bar, imp in zip(bars, importances):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + (0.01 if height > 0 else -0.02),
                    f'{imp:+.3f}', ha='center', va='bottom' if height > 0 else 'top')

        plt.tight_layout()

        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
        buffer.seek(0)
        plot_data = buffer.getvalue()
        buffer.close()
        plt.close()

        plot_base64 = base64.b64encode(plot_data).decode()
        return plot_base64

    except Exception as e:
        logger.error(f"Error creating visualization: {str(e)}")
        return None

def get_shap_explanation(text: str) -> Optional[Dict]:
    """
    Get SHAP explanations if explainer is available
    """
    try:
        if explainer is None:
            logger.warning("SHAP explainer not available")
            return None

        inputs = tokenizer(text, 
                          return_tensors="pt", 
                          padding=True, 
                          truncation=True, 
                          max_length=128,
                          return_attention_mask=True).to(device)

        shap_values = explainer(inputs)
        

        if hasattr(shap_values, 'values') and hasattr(shap_values, 'data'):
            
            tokens = shap_values.data[0]  
            values = shap_values.values[0]  
            
           
            pred_class = np.argmax(np.sum(values, axis=0))
            
            
            class_values = values[:, pred_class] if len(values.shape) > 1 else values
            
            token_importance = []
            for token, shap_val in zip(tokens, class_values):
                if token not in ['[CLS]', '[SEP]', '[PAD]']:
                    token_importance.append({
                        "token": str(token),
                        "shap_value": float(shap_val)
                    })
            
            return {
                "predicted_class": reverse_labels[pred_class],
                "token_importance": token_importance
            }
        
        return None

    except Exception as e:
        logger.error(f"Error getting SHAP explanation: {str(e)}")
        return None

# ================================
# Flask Routes
# ================================
@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None,
        "tokenizer_loaded": tokenizer is not None,
        "device": str(device) if device else None,
        "shap_available": explainer is not None,
        "model_type": "BERT"
    })

@app.route('/predict', methods=['POST'])
def predict():
    """Single text prediction with word importance analysis"""
    try:
        data = request.get_json(force=True)
        text = data.get("text", "").strip()
        max_words = min(max(1, data.get("max_words", 10)), 20)
        include_visualization = data.get("include_visualization", False)
        include_shap = data.get("include_shap", False)

        if not text:
            return jsonify({"error": "Text cannot be empty"}), 400

        logger.info(f"Processing prediction for: '{text[:100]}...'")
        
        
        analysis_result = enhanced_word_importance_analysis(text, max_words)
        
        if "error" in analysis_result:
            return jsonify(analysis_result), 500

       
        response = {
            "text": text,
            "prediction": analysis_result["baseline_prediction"],
            "confidence": analysis_result["baseline_confidence"],
            "probabilities": analysis_result["baseline_probabilities"],
            "word_importance": analysis_result["word_importance"],
            "total_words_analyzed": analysis_result["total_words_analyzed"]
        }

        if include_visualization:
            word_tuples = [(w["word"], w["importance"]) for w in analysis_result["word_importance"]]
            viz_base64 = create_importance_visualization(
                word_tuples, 
                text, 
                analysis_result["baseline_prediction"]
            )
            if viz_base64:
                response["visualization"] = f"data:image/png;base64,{viz_base64}"

        if include_shap:
            shap_result = get_shap_explanation(text)
            if shap_result:
                response["shap_explanation"] = shap_result

        logger.info(f"Prediction complete: {analysis_result['baseline_prediction']}")
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return jsonify({"error": "Prediction failed", "details": str(e)}), 500

@app.route('/predict_batch', methods=['POST'])
def predict_batch():
    """Batch prediction for multiple texts"""
    try:
        data = request.get_json(force=True)
        texts = data.get("texts", [])
        max_words = min(max(1, data.get("max_words", 5)), 10)
        
        if len(texts) > 20:  
            return jsonify({"error": "Batch too large, max 20 texts"}), 400
            
        results = []
        for i, text in enumerate(texts):
            try:
                logger.info(f"Processing batch item {i+1}/{len(texts)}")
                
                pred, probs = predict_with_probabilities(text.strip())
                label = reverse_labels[pred]
                confidence = float(probs[0][pred])
                
                result = {
                    "text": text,
                    "prediction": label,
                    "confidence": round(confidence, 4),
                    "probabilities": {
                        "Normal": round(float(probs[0][0]), 4),
                        "Offensive": round(float(probs[0][1]), 4),
                        "HateSpeech": round(float(probs[0][2]), 4)
                    }
                }
                results.append(result)
                
            except Exception as e:
                logger.error(f"Error processing batch text {i}: {str(e)}")
                results.append({
                    "text": text,
                    "error": f"Failed to process: {str(e)}"
                })
        
        predictions = [r.get("prediction") for r in results if "prediction" in r]
        summary = {
            "total_processed": len(results),
            "successful": len(predictions),
            "failed": len(results) - len(predictions),
            "prediction_distribution": {
                label: predictions.count(label) for label in set(predictions)
            } if predictions else {}
        }
        
        return jsonify({
            "results": results,
            "summary": summary
        })
        
    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}")
        return jsonify({"error": "Batch prediction failed", "details": str(e)}), 500

@app.route('/analyze', methods=['POST'])
def analyze_detailed():
    """Detailed analysis endpoint with full word importance breakdown"""
    try:
        data = request.get_json(force=True)
        text = data.get("text", "").strip()
        
        if not text:
            return jsonify({"error": "Text cannot be empty"}), 400

        logger.info(f" Starting detailed analysis for: '{text[:100]}...'")
        
        
        analysis_result = enhanced_word_importance_analysis(text, max_words=20)
        
        if "error" in analysis_result:
            return jsonify(analysis_result), 500

        
        word_tuples = [(w["word"], w["importance"]) for w in analysis_result["word_importance"]]
        viz_base64 = create_importance_visualization(
            word_tuples, 
            text, 
            analysis_result["baseline_prediction"]
        )

        response = {
            **analysis_result,
            "analysis_type": "detailed",
            "visualization": f"data:image/png;base64,{viz_base64}" if viz_base64 else None
        }

        
        shap_result = get_shap_explanation(text)
        if shap_result:
            response["shap_explanation"] = shap_result

        logger.info("Detailed analysis complete")
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Detailed analysis error: {str(e)}")
        return jsonify({"error": "Analysis failed", "details": str(e)}), 500

@app.route('/model_info', methods=['GET'])
def model_info():
    """Get information about the loaded model"""
    try:
        info = {
            "model_type": "BERT for Sequence Classification",
            "model_loaded": model is not None,
            "tokenizer_loaded": tokenizer is not None,
            "device": str(device) if device else None,
            "shap_available": explainer is not None,
            "labels": reverse_labels,
            "max_sequence_length": 128
        }
        
        if model is not None:
            info["model_config"] = {
                "num_labels": model.config.num_labels,
                "hidden_size": model.config.hidden_size,
                "num_attention_heads": model.config.num_attention_heads,
                "num_hidden_layers": model.config.num_hidden_layers
            }
        
        return jsonify(info)
        
    except Exception as e:
        logger.error(f"Error getting model info: {str(e)}")
        return jsonify({"error": "Failed to get model info"}), 500


# Run App
def startup(model_path: str = "./bert_hatespeech_model"):
    """Startup function to load all models"""
    return load_models(model_path)

if __name__ == '__main__':
    logger.info("Starting Flask server...")
    if load_models():
        print("BERT model loaded successfully from Supabase!")
        print("Server running on http://127.0.0.1:5004")
        app.run(debug=True, port=5004, host='127.0.0.1')
    else:
        print("❌ Failed to load BERT model from Supabase.")
        