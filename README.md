%<div align="center">

# 🛰️ OrbitalClaim

### *AI-Powered Satellite Disaster Damage Assessment*

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![React](https://img.shields.io/badge/React-18.2-61DAFB.svg)](https://reactjs.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**Automated insurance claim processing powered by satellite imagery, deep learning, and conversational AI.**

[Features](#-features) • [Architecture](#-architecture) • [Quick Start](#-quick-start) • [API Reference](#-api-reference) • [Model Weights](#-model-weights)

</div>

---

## 🎯 Overview

**OrbitalClaim** is an end-to-end AI system that revolutionizes disaster damage assessment for insurance claims. By combining **Google Earth Engine's Sentinel-2 satellite imagery**, **xView2 damage detection models**, and **Gemini 2.5 conversational AI**, it delivers:

- ⚡ **Real-time damage assessment** from space (10m resolution)
- 🤖 **Natural language interface** for claim processing
- 📊 **Automated decisions** with confidence scores and cost estimates
- 🎨 **Interactive UI** with before/after image comparison
- 📄 **Comprehensive reports** ready for policyholders

---

## ✨ Features

### 🛰️ **Satellite Intelligence**
- **Live Sentinel-2 Imagery**: Fetches pre/post disaster satellite data via Google Earth Engine
- **AlphaEarth Embeddings**: 64-dimensional AI-powered satellite features
- **Cloud-Optimized**: Automatically selects clearest images (<30% cloud cover)
- **10m Resolution**: High-precision RGB imagery (B4, B3, B2 bands)

### 🤖 **AI-Powered Analysis**
- **Deep Learning Models**: xView2-trained ResNet34 U-Net for damage classification
- **5-Class Damage Detection**: No Building, No Damage, Minor, Major, Destroyed
- **Gemini 2.5 Integration**: Function-calling chatbot with natural language understanding
- **Building Segmentation**: Optional Keras-based footprint detection

### 💼 **Insurance Automation**
- **Automated Decisions**: APPROVE / REJECT / MANUAL REVIEW recommendations
- **Priority Levels**: HIGH / MEDIUM / LOW based on damage severity
- **Cost Estimation**: Repair cost ranges from $5K-$300K+ based on damage
- **Confidence Scoring**: Percentage-based reliability metrics

### 🎨 **Modern Web Interface**
- **React + TypeScript**: Fast, responsive single-page application
- **Interactive Sliders**: Drag-to-compare before/after satellite imagery
- **Map Picker**: Click-to-select coordinates with Leaflet integration
- **Image Upload**: Manual analysis support for custom before/after images
- **Dark Mode**: Professional UI with Tailwind CSS

### 📊 **Reporting & Communication**
- **PDF/HTML Reports**: Downloadable assessments with all metrics
- **Email Integration**: One-click policyholder notifications
- **Damage Heatmaps**: Color-coded visualization of affected areas
- **JSON Export**: Structured data for downstream processing

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      Frontend (React)                        │
│  • Chat Interface  • Image Comparison  • Map Picker          │
└───────────────────────────┬─────────────────────────────────┘
                            │ REST API
┌───────────────────────────▼─────────────────────────────────┐
│                    Backend (Flask)                           │
│  • /api/chat  • /api/analyze-image  • /api/sample-analysis  │
└───────────────────────────┬─────────────────────────────────┘
                            │
        ┌───────────────────┼───────────────────┐
        │                   │                   │
┌───────▼────────┐ ┌────────▼────────┐ ┌───────▼────────┐
│  Gemini 2.5    │ │  Google Earth   │ │  xView2 Models │
│  Function Call │ │  Engine API     │ │  (ResNet34)    │
│  Chatbot       │ │  Sentinel-2     │ │  Damage CNN    │
└────────────────┘ └─────────────────┘ └────────────────┘
```

### **Tech Stack**

| Layer | Technologies |
|-------|-------------|
| **Frontend** | React 18, TypeScript, Vite, Tailwind CSS, Leaflet, Axios |
| **Backend** | Flask, Flask-CORS, Pillow, NumPy, OpenCV |
| **AI/ML** | PyTorch, Keras, xView2 Models, Gemini 2.5 Flash |
| **Data** | Google Earth Engine, Sentinel-2, AlphaEarth Embeddings |

---

## 🚀 Quick Start

### **Prerequisites**
- Python 3.8+
- Node.js 16+
- Google Cloud Project (for Earth Engine)
- Gemini API Key

### **1. Clone & Install**

```bash
# Clone repository
git clone https://github.com/XBastille/project.git
cd project

# Install backend dependencies
pip install flask flask-cors pillow earthengine-api google-genai torch torchvision opencv-python numpy

# Install frontend dependencies
cd frontend
npm install
cd ..
```

### **2. Authenticate Services**

```bash
# Google Earth Engine
earthengine authenticate

# Set Gemini API Key (Windows)
set GEMINI_API_KEY=your-api-key-here

# Or (Linux/Mac)
export GEMINI_API_KEY='your-api-key-here'
```

### **3. Setup Model Weights**

> **⚠️ IMPORTANT: Model Weights Assembly**
> 
> The pre-trained model weights are split into five files due to size constraints. You must combine them before running the application:
> 
> ```bash
> # Navigate to model directory
> cd model/weights
> 
> # Combine split files
> cat split-weights-* > weights.tar.gz
> 
> # Extract combined archive
> tar -xzf weights.tar.gz
> 
> # Verify weights/weights/ directory exists with model files
> ls weights/
> ```
> 
> **Expected structure:**
> ```
> model/
> └── weights/
>     └── weights/
>         ├── res34_cls2_0_tuned_best
>         ├── dpn92_loc_0_tuned_best
>         ├── senet154_loc_2_tuned_best
>         └── ... (24 total model files)
> ```

### **4. Launch Application**

```bash
# Option A: Unified startup (recommended)
python start.py

# Option B: Manual startup
# Terminal 1 - Backend
cd backend
python server.py

# Terminal 2 - Frontend
cd frontend
npm run dev
```

### **5. Access Application**

- **Frontend UI**: http://localhost:3000
- **Backend API**: http://localhost:5000

---

## 📖 Usage

### **Chat-Based Assessment**

```
User: Assess hurricane damage in Fort Myers, Florida at coordinates 
      26.6406°N, 81.8723°W, before date 2022-08-15, after date 2022-10-05

AI: I'll fetch Sentinel-2 satellite imagery for those coordinates...
    [Analyzing pre/post disaster images...]
    
    ✅ Assessment Complete:
    • Damage Level: Major Damage
    • Confidence: 87.3%
    • Decision: APPROVE
    • Priority: HIGH
    • Estimated Cost: $75,000 - $150,000
    
    Satellite analysis detected significant structural damage with 1,247 
    affected building pixels. Pre-disaster imagery confirms property was 
    intact. Recommendation: Expedite claim approval.
```

### **Image Upload Analysis**

1. Click **Before** button → Upload pre-disaster image
2. Click **After** button → Upload post-disaster image
3. System analyzes with:
   - Gemini Vision API (visual description)
   - Building segmentation (optional)
   - xView2 damage detection (pixel-level classification)
4. Receive comprehensive damage assessment

### **Map Picker**

1. Click **Map** button in composer
2. Click location on interactive map
3. Coordinates auto-inserted into message
4. Chatbot uses coordinates to fetch satellite imagery

---

## 🧪 Testing

### **Real Disaster Tests**

```bash
cd model
python test_real_disasters.py
```

**Test Scenarios:**
- ✅ Hurricane Ian (Fort Myers, FL - Sept 2022)
- ✅ Nepal Earthquake (Kathmandu - April 2015)
- ✅ Camp Fire (Paradise, CA - Nov 2018)
- ✅ Control (San Francisco - No Disaster)

**Output:** `model/test_outputs/` contains satellite imagery and damage maps.

---

## 🧠 Model Weights

The system uses **24 pre-trained xView2 models** for ensemble damage detection:

| Model | Architecture | Purpose |
|-------|--------------|---------|
| `res34_cls2_0_tuned_best` | ResNet34 U-Net | Primary damage classifier |
| `dpn92_loc_0_tuned_best` | DPN92 U-Net | Building localization |
| `senet154_loc_2_tuned_best` | SENet154 U-Net | Fine-grained damage |

**Location:** `model/weights/weights/`

> **Note:** Weights are split into 5 files. Combine using:
> ```bash
> cat split-weights-* > weights.tar.gz
> ```

---

## 📁 Project Structure

```
project/
├── backend/                 # Flask API server
│   ├── server.py           # Main API endpoints
│   └── requirements.txt    # Python dependencies
├── frontend/               # React web application
│   ├── src/
│   │   ├── components/     # React components
│   │   │   ├── ChatInterface.tsx
│   │   │   ├── Composer.tsx
│   │   │   ├── ImageComparisonSlider.tsx
│   │   │   └── MapPicker.tsx
│   │   ├── App.tsx
│   │   └── main.tsx
│   ├── package.json
│   └── vite.config.ts
├── model/                  # AI/ML components
│   ├── chatbot.py          # Gemini-powered chatbot
│   ├── models.py           # Neural network architectures
│   ├── inference/
│   │   ├── damage_detector.py    # Core damage detection
│   │   └── process_claim.py      # End-to-end pipeline
│   ├── weights/            # Pre-trained model weights
│   ├── chat_outputs/       # Chatbot generated results
│   └── test_outputs/       # Test scenario outputs
├── start.py                # Unified startup script
└── README.md
```

---

## 🔧 Configuration

### **Environment Variables**

```bash
# Required
GEMINI_API_KEY=your-gemini-api-key

# Optional
GEE_PROJECT=your-google-cloud-project  # For Earth Engine
```

### **Backend Settings** (`backend/server.py`)
- **Port:** 5000 (configurable in `app.run()`)
- **CORS:** Enabled for all origins
- **Debug Mode:** Disabled for production

### **Frontend Settings** (`frontend/vite.config.ts`)
- **Port:** 3000
- **API Proxy:** Forwards `/api/*` to `http://localhost:5000`

---

## 🤝 Contributing

We welcome contributions! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🆘 Troubleshooting

### **Earth Engine Authentication Failed**
```bash
earthengine authenticate --quiet
```

### **Model Weights Not Found**
Ensure you've combined the split files:
```bash
cd model/weights
cat split-weights-* > weights.tar.gz
tar -xzf weights.tar.gz
```

### **Frontend Not Loading**
Check that backend is running on port 5000:
```bash
curl http://localhost:5000/api/health
```

### **No Satellite Images Found**
- Check date ranges (±30 days search window)
- Verify coordinates are valid
- Try increasing cloud threshold

---

## 🔗 Resources

- [xView2 Challenge](https://xview2.org/) - Building damage assessment dataset
- [Google Earth Engine](https://earthengine.google.com/) - Satellite imagery platform
- [Sentinel-2](https://sentinel.esa.int/web/sentinel/missions/sentinel-2) - ESA satellite mission
- [Gemini API](https://ai.google.dev/) - Google's multimodal AI

---

## 📦 Project Backup

**Full project backup with model weights and datasets:**

🔗 **[Download from Google Drive](#)**  
*Replace `#` with your actual Google Drive link*

> **Backup includes:**
> - Complete source code
> - Pre-trained model weights (combined)
> - Sample test datasets
> - Configuration files

---

<div align="center">

**Built with ❤️ using Satellite AI, Deep Learning, and Modern Web Technologies**

[![GitHub](https://img.shields.io/badge/GitHub-Repository-black.svg)](https://github.com/XBastille/project)

*For questions or support, please open an issue on GitHub.*

</div>
