# Explainable Hate Speech Detection in Social Media Using Multimodal Deep Learning  

> MSc Data Science Dissertation Project (University of Leicester, 2025) developed by: Nikhil Ayyappan Nair, Abhishek Kumar Pal, Sahil Dinesh Padwal. [REMOVE IF NOT DESIRED]

---

## Project Description  

The proliferation of social media has created unprecedented challenges in content moderation, with **hate speech posing a significant threat** to online communities. This project designs, implements, and evaluates a **multimodal system** for hate speech detection across **text and memes**.  

The research adopts a **tiered methodology**:  
- Establishing a **traditional baseline** using **SVM** (TF-IDF + char n-grams).  
- Progressing to **deep learning** models (**BiLSTM, fine-tuned BERT**) for text.  
- Extending to **multimodal detection** with **CLIP + OCR** for hateful memes.  

Beyond accuracy, the system integrates **explainability** (SHAP, Grad-CAM) and **fairness audits** (gender counterfactual testing). The final models are deployed in a **web application** as a proof-of-concept bridging research and practice.  

---

## Features  
- **Text Models**: SVM, BiLSTM, fine-tuned BERT  
- **Image/Meme Model**: CLIP + OCR for multimodal hate detection  
- **Fairness Audits**: Counterfactual testing for gender bias  
- **Explainability**: SHAP (text) & Grad-CAM (images)
- **Deployment**: Flask APIs + React Dashboard + Next.js landing site  

---

## Project Structure  

```
├── SafeScan-main/                # Core ML artefacts + Flask helpers
│   └── frontend/models/          # SVM, BiLSTM, BERT, CLIP runners
│
├── reacty-main/                  # React Dashboard (UI + charts, uploads)
├── nexty-main/                   # Next.js Landing Site (presentation layer)
│
├── Supplementary_Materials_Group_Indigo/   # Dissertation research artefacts
│   ├── Text Models/ (SVM, BiLSTM, BERT notebooks)
│   ├── Image Model/ (CLIP + OCR + Grad-CAM notebooks)
│   ├── Preprocessing/ (dataset merge, cleaning)
│   └── Fairness notebooks
│
├── bert-main/                    # Minimal BERT runner
├── lstm-main/                    # Minimal LSTM runner
├── clip-main/                    # Minimal CLIP runner
└── HateSpeech_Detection_Final_Report_Group_Indigo.docx
```

---

## Datasets  

- **Text Corpus**: Merged HateXplain + Hate Speech & Offensive Tweets (~44,931 samples)  
  - Normal: 12,896 | Offensive: 24,670 | Hate Speech: 7,365  
- **Image Corpus**: Facebook Hateful Memes (~10,000 memes)  
  - Binary labels: hateful / non-hateful  

---

## Deployment  

### React Dashboard  
```bash
cd reacty-main
npm install
npm start
```
Runs at: [http://localhost:6000](http://localhost:6000)  

### Next.js Landing Site  
```bash
cd nexty-main
npm install
npm run dev
```
Runs at: [http://localhost:3000](http://localhost:3000)  

### Flask APIs  
```bash
cd bert-main
pip install -r requirements.txt
python appk.py
```
[REPEAT Similar for `lstm-main`, `clip-main`, or scripts under `SafeScan-main/frontend/models/`]

---

## Technologies Used

### Machine Learning & NLP
- **Python 3.9+**
- **PyTorch** (`torch`), **Transformers** (`transformers`) for BERT
- **TensorFlow** — BiLSTM training in notebooks
- **SHAP** (`shap`) for explainability
- **OpenCV** (`opencv-python`) and **Pillow** (`pillow`) for image handling
- **Google Cloud Vision** (`google-cloud-vision`) for OCR (present)
- **NLTK** (`nltk`) for text preprocessing
- **Matplotlib** (`matplotlib`) for plots

### Web & APIs
- **Flask** + **Flask‑CORS** for model inference APIs
- **React**
- **Next.js (TypeScript)** landing site
- **Charts**: React‑Plotly.js, ApexCharts / react‑apexcharts
- **SEO/Meta**: react‑helmet, next‑seo

### Notebooks & Experimentation
Jupyter Notebooks

### Dev & Ops
pyngrok

---



