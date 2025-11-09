# AlphaEarth Insurance AI

AI-powered disaster damage assessment using satellite imagery and conversational AI.

## 🚀 Quick Start

```bash
# 1. Install dependencies
pip install earthengine-api google-genai torch torchvision opencv-python numpy

# 2. Authenticate
earthengine authenticate
export GEMINI_API_KEY='your-key-here'

# 3. Run chatbot
cd model
python chatbot.py
```

## 🤖 Features

✅ **AI Chatbot**: Natural conversation with Gemini 2.0 Flash  
✅ **Satellite Data**: Real-time Sentinel-2 imagery from Google Earth Engine  
✅ **Damage Detection**: Pre-trained xView2 models (ResNet34, DPN92, SENet154)  
✅ **Automated Claims**: APPROVE/REJECT/REVIEW decisions  
✅ **Function Calling**: AI automatically triggers satellite assessment  

## 💬 Example Chat

```
You: I need help with Hurricane Ian damage in Fort Myers

AI: I can help assess that. Hurricane Ian hit Fort Myers in Sept 2022.
    I'll fetch satellite imagery to analyze the damage. One moment...

🛰️  [Fetching satellite data...]

AI: Based on satellite analysis:
    - Damage Level: Major Damage
    - Decision: APPROVE
    - Confidence: 85%
    - Priority: HIGH
    
    Your claim is approved for expedited processing!
```

## 📁 Structure

```
model/
├── chatbot.py              # 🤖 Main entry point
├── test_real_disasters.py  # 🧪 Testing
├── inference/
│   ├── damage_detector.py  # Damage analysis
│   └── process_claim.py    # Manual mode
├── weights/weights/        # 24 pre-trained models
├── chat_outputs/           # Chatbot results
└── test_outputs/           # Test results
```

## 🎯Ready

✅ **Automated Damage Claims** from Satellite Imagery  
✅ **Natural Language** risk explanation via Gemini  
✅ **Real Disasters** tested (Hurricane Ian, Camp Fire)  
✅ **Minimal & Dynamic** - No training required  
✅ **Production Ready** - Complete end-to-end system
