from flask import Flask, request, jsonify
import os
import cv2
import torch
from flask_cors import CORS
from pyngrok import ngrok
import torch.nn as nn
import re
import requests
import base64
from transformers import CLIPModel, CLIPProcessor
from PIL import Image
from clarifai.client.model import Model
import asyncio
import threading
import io
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter
import clip as openai_clip
from PIL import Image as PILImage
from clarifai_grpc.channel.clarifai_channel import ClarifaiChannel
from clarifai_grpc.grpc.api import service_pb2, service_pb2_grpc
from clarifai_grpc.grpc.api import resources_pb2
from clarifai_grpc.grpc.api.status import status_code_pb2

# INITIALIZE FLASK
app = Flask(__name__)
CORS(app)


# API KEYS & GLOBAL VARIABLES

GOOGLE_API_KEY = "AIzaSyAThUa2Mi97AKr3ASw9OBWXHDRsrUBiP18"
CLARIFAI_PAT = "1d7c443db4f34ed0a6e75e9b92400a82"
USER_ID = 'clarifai'
APP_ID = 'main'
MODEL_ID = 'hate-symbol-detection'
MODEL_VERSION_ID = 'bcf5a7776bee4a8da0abc8781faa760c'
USER_IDT = 'salesforce'
APP_IDT = 'blip'
MODEL_IDT = 'general-english-image-caption-blip'
MODEL_VERSION_IDT = 'cdb690f13e62470ea6723642044f95e4'
SUPABASE_URL = "https://fbtvsatqimjqyxwwrksk.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImZidHZzYXRxaW1qcXl4d3dya3NrIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTU2MjgxMDgsImV4cCI6MjA3MTIwNDEwOH0.5hnMGGYm6BL7x8dVN_h3NV33Qdy9evRLc4EAQqBz2zY"
SUPABASE_BUCKET = "model"
MODEL_FILENAME = "clip_mlp_epoch5.pt"

# Initialize Supabase client
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

def download_model_from_supabase():
    """Download model file from Supabase Storage and return local path."""
    try:
        response = supabase.storage.from_(SUPABASE_BUCKET).download(MODEL_FILENAME)
        if response is None:
            raise FileNotFoundError(f"Model {MODEL_FILENAME} not found in Supabase bucket {SUPABASE_BUCKET}")

        # Save temporarily
        tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pt")
        tmp_file.write(response)
        tmp_file.close()
        print(f"Model downloaded to {tmp_file.name}")
        return tmp_file.name
    except Exception as e:
        raise RuntimeError(f"Failed to download model from Supabase: {e}")


# CLIP + MLP MODEL SETUP (Path 1)

class CLIP_MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.classifier = nn.Sequential(
            nn.Linear(512 + 512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 2)
        )

    def forward(self, input_ids, attention_mask, pixel_values):
        with torch.no_grad():
            outputs = self.clip(input_ids=input_ids, attention_mask=attention_mask, pixel_values=pixel_values)
        combined = torch.cat([outputs.text_embeds, outputs.image_embeds], dim=1)
        return self.classifier(combined)

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Download model from Supabase
model_path = download_model_from_supabase()

# Load model
clip_model = CLIP_MLP().to(device)
clip_model.load_state_dict(torch.load(model_path, map_location=device))
clip_model.eval()

processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")


# GOOGLE VISION OCR (for Path 1)

def detect_meme_text(image_path):
    """Extract text using Google Vision OCR"""
    with open(image_path, "rb") as image_file:
        image_content = base64.b64encode(image_file.read()).decode('UTF-8')

    request_body = {
        "requests": [{
            "image": {"content": image_content},
            "features": [{"type": "TEXT_DETECTION"}]
        }]
    }

    response = requests.post(
        f"https://vision.googleapis.com/v1/images:annotate?key={GOOGLE_API_KEY}",
        json=request_body
    )

    if response.status_code == 200:
        result = response.json()
        try:
            detected_text = result['responses'][0]['fullTextAnnotation']['text']
            return detected_text.strip()
        except KeyError:
            return ""
    else:
        return ""

def remove_special_characters(text):
    text = re.sub(r'[^A-Za-z0-9\s]', '', text)
    text = re.sub(r'\n+', ' ', text)
    return text.strip()


# GOOGLE SAFE SEARCH + CLARIFAI (Path 2)

def google_safe_search(image_path):
    """Check if image is hateful using Google Vision Safe Search"""
    with open(image_path, "rb") as image_file:
        image_content = base64.b64encode(image_file.read()).decode('UTF-8')

    request_body = {
        "requests": [{
            "image": {"content": image_content},
            "features": [{"type": "SAFE_SEARCH_DETECTION"}]
        }]
    }

    response = requests.post(
        f"https://vision.googleapis.com/v1/images:annotate?key={GOOGLE_API_KEY}",
        json=request_body
    )

    if response.status_code == 200:
        result = response.json()
        safe_search = result['responses'][0]['safeSearchAnnotation']
        hate_likelihood = safe_search.get("violence", "VERY_UNLIKELY")
        racy_likelihood = safe_search.get("racy", "VERY_UNLIKELY")
        hateful_levels = ["LIKELY", "VERY_LIKELY"]

        return hate_likelihood in hateful_levels or racy_likelihood in hateful_levels
    return False

def run_in_thread_with_loop(func, *args):
    """Run async function in a new thread with its own event loop"""
    result = [None]
    exception = [None]

    def target():
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            result[0] = loop.run_until_complete(func(*args))
        except Exception as e:
            exception[0] = e
        finally:
            loop.close()

    thread = threading.Thread(target=target)
    thread.start()
    thread.join()

    if exception[0]:
        raise exception[0]
    return result[0]

async def async_clarifai_hate_symbol(image_path):
    """Async version of Clarifai hate symbol detection"""
    with open(image_path, 'rb') as image_file:
        image_data = image_file.read()

    model_url = "https://clarifai.com/clarifai/main/models/hate-symbol-detection"
    model = Model(url=model_url, pat=CLARIFAI_PAT)
    response = model.predict_by_bytes(image_data)
    regions = response.outputs[0].data.regions
    return bool(regions)

def clarifai_hate_symbol(image_path):
    """Check if image contains hate symbols using Clarifai Hate Symbol Model"""
    try:
        return run_in_thread_with_loop(async_clarifai_hate_symbol, image_path)
    except Exception as e:
        print(f"Clarifai error: {e}")
        return False

def clarifai_hate_symbol_sync(image_path):
    """Synchronous version using requests directly to Clarifai API"""
    try:
        with open(image_path, 'rb') as image_file:
            image_data = image_file.read()

        # Clarifai API URL for model inference
        url = "https://api.clarifai.com/v2/models/hate-symbol-detection/outputs"

        headers = {
            "Authorization": f"Key {CLARIFAI_PAT}",
            "Content-Type": "application/json",
        }

        data = {
            "user_app_id": {
                "user_id": USER_ID,
                "app_id": APP_ID,
            },
            "inputs": [{
                "data": {
                    "image": {
                        "base64": base64.b64encode(image_data).decode('utf-8')  # Convert image to base64
                    }
                }
            }]
        }

        response = requests.post(url, headers=headers, json=data)

        if response.status_code == 200:
            result = response.json()
            regions = result.get('outputs', [{}])[0].get('data', {}).get('regions', [])

            return bool(regions)
        else:
            print(f"Clarifai API error: {response.status_code} - {response.text}")
            return False

    except Exception as e:
        print(f"Clarifai sync error: {e}")
        return False

# CLASSIFICATION FUNCTIONS

def classify_with_clip(image_path):
    """Path 1: CLIP + Google Vision OCR"""
    ocr_text = detect_meme_text(image_path)
    print(f"OCR Text Detected (Path 1): {ocr_text}")

    image = Image.open(image_path).convert("RGB")
    inputs = processor(
        text=[ocr_text],
        images=image,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=77
    )
    print(f"Tokenized text input: {inputs['input_ids']}")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        output = clip_model(**inputs)
        logits = output[0]
        confidence = torch.max(torch.softmax(logits, dim=-1))
        if confidence < 0.7:
            predicted_class = 1
        else:
            predicted_class = torch.argmax(logits, dim=-1).item()


    label_map = {0: "non-hateful", 1: "hateful"}
    print("Prediction:", label_map[predicted_class])
    return label_map[predicted_class]

def classify_with_google_clarifai(image_path):
    """Path 2: Google Safe Search + Clarifai"""
    if google_safe_search(image_path) or clarifai_hate_symbol_sync(image_path):
        return "hateful"
    return "non-hateful"



# ADDED FOR GRADCAM
# Load OpenAI CLIP (RN50) once for GradCAM
_gradcam_device = "cuda" if torch.cuda.is_available() else "cpu"
_gradcam_model, _gradcam_preprocess = openai_clip.load("RN50", device=_gradcam_device, jit=False)

def _normalize(x: np.ndarray) -> np.ndarray:
    x = x - x.min()
    if x.max() > 0:
        x = x / x.max()
    return x

def _getAttMap(img, attn_map, blur=True):
    if blur:
        attn_map = gaussian_filter(attn_map, 0.02 * max(img.shape[:2]))
    attn_map = _normalize(attn_map)
    cmap = plt.get_cmap('jet')
    attn_map_c = np.delete(cmap(attn_map), 3, 2)
    attn_map = 1 * (1 - attn_map**0.7).reshape(attn_map.shape + (1,)) * img + \
               (attn_map**0.7).reshape(attn_map.shape + (1,)) * attn_map_c
    return attn_map

class _Hook:
    def __init__(self, module: nn.Module):
        self.data = None
        self.hook = module.register_forward_hook(self._save)

    def _save(self, module, input, output):
        self.data = output
        output.requires_grad_(True)
        output.retain_grad()

    def __enter__(self): return self
    def __exit__(self, exc_type, exc_value, exc_tb): self.hook.remove()
    def activation(self) -> torch.Tensor: return self.data
    def gradient(self) -> torch.Tensor: return self.data.grad

def _gradCAM(model: nn.Module, input_tensor: torch.Tensor, target: torch.Tensor, layer: nn.Module) -> torch.Tensor:
    if input_tensor.grad is not None:
        input_tensor.grad.data.zero_()

    requires_grad = {}
    for name, p in model.named_parameters():
        requires_grad[name] = p.requires_grad
        p.requires_grad_(False)

    with _Hook(layer) as hook:
        output = model(input_tensor)
        output.backward(target)

        grad = hook.gradient().float()
        act = hook.activation().float()
        alpha = grad.mean(dim=(2, 3), keepdim=True)
        cam = torch.sum(act * alpha, dim=1, keepdim=True)
        cam = torch.clamp(cam, min=0)

    cam = F.interpolate(cam, input_tensor.shape[2:], mode='bicubic', align_corners=False)

    # Unfreeze params
    for name, p in model.named_parameters():
        p.requires_grad_(requires_grad[name])

    return cam

def _image_to_base64(arr: np.ndarray) -> str:
    """arr in [0,1] RGB"""
    fig = plt.figure(figsize=(5, 5))
    plt.imshow(arr)
    plt.axis('off')
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    b64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    return f"data:image/png;base64,{b64}"

def _load_img_for_overlay(img_path: str, resize: int = None) -> np.ndarray:
    image = PILImage.open(img_path).convert("RGB")
    if resize is not None:
        image = image.resize((resize, resize))
    return np.asarray(image).astype(np.float32) / 255.0

def generate_gradcam_heatmap(image_path: str, text_prompt: str, blur=True) -> str:
    """Returns base64 PNG heatmap overlay for the given image/text."""
    if not text_prompt or text_prompt.strip() == "":
        text_prompt = "the main subject of the image"
    image_input = _gradcam_preprocess(PILImage.open(image_path)).unsqueeze(0).to(_gradcam_device)
    image_np = _load_img_for_overlay(image_path, _gradcam_model.visual.input_resolution)
    text_input = openai_clip.tokenize([text_prompt]).to(_gradcam_device)

    attn_map = _gradCAM(
        _gradcam_model.visual,
        image_input,
        _gradcam_model.encode_text(text_input).float(),
        getattr(_gradcam_model.visual, "layer4")
    )
    attn_map = attn_map.squeeze().detach().cpu().numpy()
    overlay = _getAttMap(image_np, attn_map, blur=blur)
    return _image_to_base64(overlay)


async def get_blip_caption(image_path: str) -> str:
    """Generate image caption using Clarifai BLIP model via gRPC."""
    try:
        channel = ClarifaiChannel.get_grpc_channel()
        stub = service_pb2_grpc.V2Stub(channel)
        metadata = (("authorization", f"Key {CLARIFAI_PAT}"),)
        userDataObject = resources_pb2.UserAppIDSet(user_id=USER_IDT, app_id=APP_IDT)

        with open(image_path, "rb") as f:
            image_bytes = f.read()

        response = stub.PostModelOutputs(
            service_pb2.PostModelOutputsRequest(
                user_app_id=userDataObject,
                model_id=MODEL_IDT,
                version_id=MODEL_VERSION_IDT,
                inputs=[
                    resources_pb2.Input(
                        data=resources_pb2.Data(
                            image=resources_pb2.Image(base64=image_bytes)
                        )
                    )
                ]
            ),
            metadata=metadata
        )

        if response.status.code != status_code_pb2.SUCCESS:
            print("Clarifai error:", response.status)
            return ""

        output = response.outputs[0]
        return output.data.text.raw.strip()

    except Exception as e:
        print(f"BLIP caption error: {e}")
        return ""

# 7. FLASK ROUTE
@app.route('/predict_image', methods=['POST'])
def predict_image():
    if 'image' not in request.files:
        return jsonify({"error": "No image provided"}), 400

    image = request.files['image']
    save_path = f"temp_{image.filename}"
    image.save(save_path)

    try:

        ocr_text = detect_meme_text(save_path)
        print(f"OCR Text Detected (Route): {ocr_text}")
        word_count = len(ocr_text.split())
        if word_count > 5:
            prediction = classify_with_clip(save_path)
            method = "Path 1 (CLIP + Google Vision OCR)"

            heatmap_text = remove_special_characters(ocr_text) if ocr_text else ""
            heatmap_b64 = generate_gradcam_heatmap(save_path, heatmap_text)

        else:
            prediction = classify_with_google_clarifai(save_path)
            method = "Path 2 (Google Safe Search + Clarifai)"

            caption = run_in_thread_with_loop(get_blip_caption, save_path)
            print(f"BLIP Caption (Path 2): {caption}")
            heatmap_text = caption
            heatmap_b64 = generate_gradcam_heatmap(save_path, heatmap_text)

        return jsonify({
            "method": method,
            "prediction": prediction,
            "heatmap_text": heatmap_text,
            "heatmap_image": heatmap_b64
        })

    except Exception as e:
        print(f"Error in prediction: {e}")
        return jsonify({"error": str(e)}), 500

    finally:
        if os.path.exists(save_path):
            os.remove(save_path)


# 8. RUN APP

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 10000)))
