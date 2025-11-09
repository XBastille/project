"""
Backend API Server for AlphaEarth Insurance AI
Connects React frontend with chatbot and damage assessment system
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import sys
from pathlib import Path
import base64
import io
import os
from PIL import Image
import numpy as np
import cv2
import traceback
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add model directory to path
model_dir = Path(__file__).parent.parent / 'model'
sys.path.insert(0, str(model_dir))

from chatbot import InsuranceClaimChatbot

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})

# Initialize chatbot
chatbot = None

def get_chatbot():
    """Lazy initialization of chatbot"""
    global chatbot
    if chatbot is None:
        chatbot = InsuranceClaimChatbot()
    return chatbot


@app.route('/api/chat', methods=['POST'])
def chat():
    """Handle chat messages from frontend"""
    try:
        data = request.json
        message = data.get('message', '')
        
        if not message:
            return jsonify({'error': 'No message provided'}), 400
        
        logger.info(f"Received message: {message[:100]}...")
        
        bot = get_chatbot()
        response = bot.chat(message)
        
        # Check if response contains assessment data
        # Extract assessment if present in bot's last function call result
        assessment = None
        before_img_url = None
        after_img_url = None
        seg_img_url = None
        
        if hasattr(bot, '_last_assessment'):
            assessment = bot._last_assessment
            
            # Load saved images from chat_outputs if they exist
            if assessment and 'saved_to' in assessment:
                import glob
                from pathlib import Path
                
                # Extract the path pattern
                saved_pattern = assessment['saved_to']
                # Convert to absolute path
                model_dir = Path(__file__).parent.parent / 'model'
                output_dir = model_dir / 'chat_outputs'
                
                # Find the most recent files for this assessment
                all_files = sorted(output_dir.glob('*.png'), key=lambda x: x.stat().st_mtime, reverse=True)
                
                # Get the 3 most recent (before, after, damage)
                if len(all_files) >= 3:
                    for file in all_files[:3]:
                        img = Image.open(file)
                        buffer = io.BytesIO()
                        img.save(buffer, format='PNG')
                        img_b64 = f"data:image/png;base64,{base64.b64encode(buffer.getvalue()).decode()}"
                        
                        if 'before' in file.name:
                            before_img_url = img_b64
                        elif 'damage' in file.name:
                            seg_img_url = img_b64
                        elif 'after' in file.name:
                            after_img_url = img_b64
            
            bot._last_assessment = None
        
        logger.info("Response generated successfully")
        
        return jsonify({
            'response': response,
            'assessment': assessment,
            'beforeImageUrl': before_img_url,
            'afterImageUrl': after_img_url,
            'segmentationImageUrl': seg_img_url
        })
    
    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({'error': str(e), 'details': traceback.format_exc()}), 500


@app.route('/api/analyze-image', methods=['POST'])
def analyze_image():
    """Analyze uploaded damage images (before/after) using Gemini Vision and building segmentation"""
    try:
        if 'before' not in request.files or 'after' not in request.files:
            return jsonify({'error': 'Both before and after images required'}), 400
        
        before_file = request.files['before']
        after_file = request.files['after']
        
        # Read images
        before_bytes = before_file.read()
        after_bytes = after_file.read()
        before_image = Image.open(io.BytesIO(before_bytes))
        after_image = Image.open(io.BytesIO(after_bytes))
        
        # Convert to numpy arrays
        before_array = np.array(before_image.convert('RGB'))
        after_array = np.array(after_image.convert('RGB'))
        
        # Step 1: Get building segmentation using Gradio API
        logger.info("Getting building segmentation...")
        segmentation_mask = None
        try:
            from gradio_client import Client as GradioClient, handle_file
            import tempfile
            
            # Save after image to temp file
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
                after_image.save(tmp.name)
                tmp_path = tmp.name
            
            # Call building segmentation API
            seg_client = GradioClient("gunayk3/building_footprint_segmentation")
            seg_result = seg_client.predict(img=handle_file(tmp_path), api_name="/predict")
            
            # Load segmentation result
            if seg_result and os.path.exists(seg_result):
                seg_mask = cv2.imread(seg_result, cv2.IMREAD_GRAYSCALE)
                segmentation_mask = (seg_mask > 128).astype(np.uint8)
                logger.info(f"Building segmentation complete: {segmentation_mask.sum()} building pixels")
            
            # Clean up
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
                
        except Exception as e:
            logger.warning(f"Building segmentation failed: {e}")
        
        # Step 2: Use Gemini Vision for damage analysis
        from google import genai
        from google.genai import types
        
        try:
            # Initialize Gemini client
            api_key = "AIzaSyARncEnZVsr878AGaSPM3X5Nj3zIpuXlS4"
            client = genai.Client(api_key=api_key)
            
            # Analyze after image with Gemini
            contents = [
                types.Content(
                    role="user",
                    parts=[
                        types.Part.from_bytes(
                            mime_type="image/jpeg",
                            data=after_bytes,
                        ),
                        types.Part.from_text(
                            text="Analyze this disaster damage image. Describe what type of disaster damage you see (hurricane, flood, fire, earthquake, etc.), the severity of damage, and what structures or areas are affected. Be specific about visible damage patterns."
                        ),
                    ],
                ),
            ]
            
            generate_content_config = types.GenerateContentConfig(
                thinking_config=types.ThinkingConfig(thinking_budget=0),
                temperature=0.7,
            )
            
            gemini_response = client.models.generate_content(
                model="gemini-2.0-flash-exp",
                contents=contents,
                config=generate_content_config,
            )
            
            vision_analysis = gemini_response.text
            
        except Exception as e:
            logger.warning(f"Gemini vision analysis failed: {e}")
            vision_analysis = "Vision analysis unavailable"
        
        # Step 3: Run xView2 damage detection
        bot = get_chatbot()
        result = bot.detector.assess(before_array, after_array)
        
        # Format assessment response
        assessment = {
            'location': 'Uploaded Images',
            'damage_level': result['damage_level'],
            'confidence': f"{result['confidence']:.1f}%",
            'decision': result['decision'],
            'priority': result['priority'],
            'damage_probabilities': {
                'no_building': f"{result['damage_probabilities'][0]:.1%}",
                'no_damage': f"{result['damage_probabilities'][1]:.1%}",
                'minor_damage': f"{result['damage_probabilities'][2]:.1%}",
                'major_damage': f"{result['damage_probabilities'][3]:.1%}",
                'destroyed': f"{result['damage_probabilities'][4]:.1%}"
            }
        }
        
        # Step 4: Save images with base64 encoding for frontend
        before_buffer = io.BytesIO()
        after_buffer = io.BytesIO()
        seg_buffer = io.BytesIO()
        before_image.save(before_buffer, format='JPEG')
        after_image.save(after_buffer, format='JPEG')
        
        # Create segmentation visualization
        damage_class_map = np.argmax(result['damage_map'], axis=2).astype(np.uint8)
        damage_heatmap = cv2.applyColorMap(damage_class_map * 50, cv2.COLORMAP_JET)
        damage_heatmap_rgb = cv2.cvtColor(damage_heatmap, cv2.COLOR_BGR2RGB)
        seg_image = Image.fromarray(damage_heatmap_rgb)
        seg_image.save(seg_buffer, format='JPEG')
        
        before_b64 = f"data:image/jpeg;base64,{base64.b64encode(before_buffer.getvalue()).decode()}"
        after_b64 = f"data:image/jpeg;base64,{base64.b64encode(after_buffer.getvalue()).decode()}"
        seg_b64 = f"data:image/jpeg;base64,{base64.b64encode(seg_buffer.getvalue()).decode()}"
        
        # Combine results
        response = f"**Visual Analysis:**\n{vision_analysis}\n\n"
        response += f"**Damage Detection Results:**\n"
        response += f"Detected {result['damage_level'].lower()} with {result['confidence']:.1f}% confidence. "
        response += f"Recommendation: {result['decision']}. Priority level: {result['priority']}."
        
        if segmentation_mask is not None:
            building_pixels = segmentation_mask.sum()
            response += f"\n\n**Building Analysis:**\n{building_pixels:,} building pixels detected."
        
        return jsonify({
            'response': response,
            'assessment': assessment,
            'beforeImageUrl': before_b64,
            'afterImageUrl': after_b64,
            'segmentationImageUrl': seg_b64
        })
    
    except Exception as e:
        logger.error(f"Error in image analysis: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({'error': str(e), 'details': traceback.format_exc()}), 500


@app.route('/api/sample-analysis', methods=['POST'])
def sample_analysis():
    """Run analysis on sample disaster (Hurricane Ian)"""
    try:
        bot = get_chatbot()
        
        # Trigger sample analysis for Hurricane Ian
        message = "Assess damage from Hurricane Ian in Fort Myers, Florida at coordinates 26.6406° N, 81.8723° W, before date 2022-09-25, after date 2022-09-29"
        response = bot.chat(message)
        
        # Extract assessment if available
        assessment = None
        if hasattr(bot, '_last_assessment'):
            assessment = bot._last_assessment
            bot._last_assessment = None
        
        return jsonify({
            'response': response,
            'assessment': assessment
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({'status': 'healthy'})


if __name__ == '__main__':
    print("=" * 70)
    print("Starting AlphaEarth Insurance API Server...")
    print("API running on: http://localhost:5000")
    print("=" * 70)
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
