"""
AlphaEarth Insurance AI Chatbot
Uses Gemini with function calling to chat with users and assess disaster damage
"""

import os
import sys
from pathlib import Path
import json
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add model to path
model_dir = Path(__file__).parent
sys.path.insert(0, str(model_dir))

from google import genai
from google.genai import types

# Import our damage assessment system
from inference.damage_detector import DamageDetector

# Google Earth Engine for satellite fetching
try:
    import ee
    HAS_GEE = True
except ImportError:
    print("[WARNING] earthengine-api not installed")
    HAS_GEE = False


class InsuranceClaimChatbot:
    """
    AI Chatbot for insurance claim processing
    - Chats with user to gather disaster information
    - Extracts location, dates automatically
    - Calls satellite damage assessment tool when ready
    - Explains results in natural language
    """
    
    def __init__(self):
        """Initialize chatbot with Gemini and damage detector"""
        logger.info("Initializing InsuranceClaimChatbot...")
        
        # Initialize Gemini
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable not set")
        
        self.client = genai.Client(api_key=api_key)
        self.model = "gemini-2.0-flash-exp"
        logger.info(f"Gemini client initialized with model: {self.model}")
        
        # Initialize damage detector
        logger.info("Loading damage detector model...")
        self.detector = DamageDetector(use_gpu=False)
        logger.info("Damage detector ready")
        
        # Initialize GEE
        if HAS_GEE:
            try:
                logger.info("Initializing Google Earth Engine...")
                ee.Initialize(project="liquid-galaxy-469409-g5")
                self.gee_ready = True
                logger.info("Google Earth Engine initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize GEE: {e}")
                self.gee_ready = False
        else:
            logger.warning("Google Earth Engine not available - earthengine-api not installed")
            self.gee_ready = False
        
        # Conversation history
        self.history = []
        
        # Store last assessment for API access
        self._last_assessment = None
        
        # System prompt
        self.system_instruction = """You are an AI insurance claims assistant for OrbitalClaim.

Your role:
1. Chat with users about disaster damage claims (hurricanes, earthquakes, wildfires, floods)
2. Extract key information: location (city/coordinates), disaster type, and dates (before/after)
3. When you have enough information, call the assess_disaster_damage function
4. Explain assessment results in natural language with confidence scores and risk assessment

Guidelines:
- Be conversational and helpful
- Ask clarifying questions if information is missing
- Explain that you use satellite imagery to assess damage
- If user mentions a known disaster (e.g., "Hurricane Ian"), you can infer dates
- Be empathetic - these are real disasters affecting real people
- Always provide confidence percentages when explaining damage assessment
- Explain the reasoning behind auto-approve, manual review, or auto-deny decisions
- Use natural language like: "Based on satellite analysis, I detected [damage level] with [X]% confidence. The affected area shows [details]. This claim is recommended for [decision] with [priority] priority."

Information needed:
- Location: city name or coordinates (lat, lon)
- Before date: date before disaster occurred (YYYY-MM-DD)
- After date: date after disaster occurred (YYYY-MM-DD)
- Location name: human-readable description

Once you have this info, call assess_disaster_damage automatically."""
    
    def assess_disaster_damage(self, location_name: str, latitude: float, longitude: float, 
                              before_date: str, after_date: str) -> dict:
        """
        Assess disaster damage using satellite imagery
        
        Args:
            location_name: Human-readable location (e.g., "Fort Myers, Florida")
            latitude: Latitude coordinate
            longitude: Longitude coordinate
            before_date: Date before disaster (YYYY-MM-DD)
            after_date: Date after disaster (YYYY-MM-DD)
        
        Returns:
            Assessment results with damage level, decision, confidence
        """
        logger.info("=" * 80)
        logger.info(f"ASSESSING DISASTER DAMAGE")
        logger.info(f"Location: {location_name}")
        logger.info(f"Coordinates: {latitude}, {longitude}")
        logger.info(f"Before: {before_date}, After: {after_date}")
        logger.info("=" * 80)
        
        if not self.gee_ready:
            logger.error("Google Earth Engine not available!")
            return {
                "error": "Google Earth Engine not available",
                "message": "Cannot fetch satellite imagery. Please check GEE authentication."
            }
        
        try:
            # Fetch satellite imagery
            from datetime import datetime, timedelta
            import numpy as np
            import cv2
            import io
            
            def fetch_image(lat, lon, date):
                logger.info(f"Fetching satellite image for date: {date}")
                point = ee.Geometry.Point([lon, lat])
                area = point.buffer(1000)
                
                target_date = datetime.strptime(date, '%Y-%m-%d')
                start_date = (target_date - timedelta(days=30)).strftime('%Y-%m-%d')
                end_date = (target_date + timedelta(days=30)).strftime('%Y-%m-%d')
                
                logger.info(f"Searching Sentinel-2 imagery from {start_date} to {end_date}")
                logger.info(f"Location: ({lat}, {lon}), Buffer: 1000m")
                
                collection = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
                    .filterBounds(area) \
                    .filterDate(start_date, end_date) \
                    .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 30))
                
                count = collection.size().getInfo()
                logger.info(f"Found {count} images with <30% cloud cover")
                
                if count == 0:
                    logger.warning("No images with <30% cloud cover, trying <50%...")
                    collection = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
                        .filterBounds(area) \
                        .filterDate(start_date, end_date) \
                        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 50))
                    count = collection.size().getInfo()
                    logger.info(f"Found {count} images with <50% cloud cover")
                
                if count == 0:
                    logger.error(f"No satellite imagery available for {date}")
                    return None
                
                logger.info("Selecting clearest image...")
                image = collection.sort('CLOUDY_PIXEL_PERCENTAGE').first()
                
                logger.info("Extracting RGB bands (B4, B3, B2)...")
                # Use getRegion instead of sampleRectangle for better handling
                rgb = image.select(['B4', 'B3', 'B2'])
                
                # Get thumbnail instead - more reliable
                thumbnail_url = rgb.getThumbURL({
                    'region': area,
                    'dimensions': '1024x1024',
                    'format': 'png'
                })
                
                logger.info(f"Downloading from: {thumbnail_url[:50]}...")
                
                import requests
                response = requests.get(thumbnail_url)
                
                if response.status_code != 200:
                    logger.error(f"Failed to download image: {response.status_code}")
                    return None
                
                # Convert to numpy array
                from PIL import Image as PILImage
                img_pil = PILImage.open(io.BytesIO(response.content))
                arr = np.array(img_pil.convert('RGB'))
                
                logger.info(f"Image shape: {arr.shape}")
                logger.info("Resizing to 1024x1024...")
                arr = cv2.resize(arr, (1024, 1024))
                logger.info(f"Image ready: {arr.shape}, dtype: {arr.dtype}")
                return arr
            
            logger.info("\n--- Fetching BEFORE image ---")
            before_img = fetch_image(latitude, longitude, before_date)
            
            logger.info("\n--- Fetching AFTER image ---")
            after_img = fetch_image(latitude, longitude, after_date)
            
            if before_img is None or after_img is None:
                logger.error("Failed to fetch satellite imagery!")
                return {
                    "error": "No satellite imagery available",
                    "message": f"Could not find clear satellite images for {location_name} on the specified dates."
                }
            
            # Run damage assessment
            logger.info("\n--- Running xView2 damage detection ---")
            result = self.detector.assess(before_img, after_img)
            logger.info(f"Detection complete: {result['damage_level']} ({result['confidence']:.1f}% confidence)")
            
            # Save results
            logger.info("\n--- Saving outputs ---")
            output_dir = model_dir / 'chat_outputs'
            output_dir.mkdir(exist_ok=True)
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            safe_name = location_name.replace(' ', '_').replace(',', '')
            
            cv2.imwrite(str(output_dir / f"{safe_name}_{timestamp}_before.png"),
                       cv2.cvtColor(before_img, cv2.COLOR_RGB2BGR))
            cv2.imwrite(str(output_dir / f"{safe_name}_{timestamp}_after.png"),
                       cv2.cvtColor(after_img, cv2.COLOR_RGB2BGR))
            
            damage_class_map = np.argmax(result['damage_map'], axis=2).astype(np.uint8)
            damage_heatmap = cv2.applyColorMap(damage_class_map * 50, cv2.COLORMAP_JET)
            cv2.imwrite(str(output_dir / f"{safe_name}_{timestamp}_damage.png"), damage_heatmap)
            
            logger.info(f"Saved outputs to: {output_dir}")
            logger.info(f"  - {safe_name}_{timestamp}_before.png")
            logger.info(f"  - {safe_name}_{timestamp}_after.png")
            logger.info(f"  - {safe_name}_{timestamp}_damage.png")
            
            # Build assessment result
            assessment_result = {
                "location": location_name,
                "coordinates": f"{latitude}, {longitude}",
                "before_date": before_date,
                "after_date": after_date,
                "damage_level": result['damage_level'],
                "confidence": f"{result['confidence']:.1f}%",
                "decision": result['decision'],
                "priority": result['priority'],
                "num_buildings": result['num_buildings'],
                "damage_probabilities": {
                    "no_building": f"{result['damage_probabilities'][0]:.1%}",
                    "no_damage": f"{result['damage_probabilities'][1]:.1%}",
                    "minor_damage": f"{result['damage_probabilities'][2]:.1%}",
                    "major_damage": f"{result['damage_probabilities'][3]:.1%}",
                    "destroyed": f"{result['damage_probabilities'][4]:.1%}"
                },
                "satellite_source": "Sentinel-2 (10m resolution)",
                "saved_to": f"model/chat_outputs/{safe_name}_{timestamp}_*.png"
            }
            
            # Store for API access
            self._last_assessment = assessment_result
            
            logger.info("\n" + "=" * 80)
            logger.info("ASSESSMENT COMPLETE")
            logger.info(f"Decision: {result['decision']} | Priority: {result['priority']}")
            logger.info("=" * 80 + "\n")
            
            return assessment_result
            
        except Exception as e:
            logger.error(f"ERROR in assess_disaster_damage: {str(e)}")
            logger.error(f"Traceback:", exc_info=True)
            return {
                "error": str(e),
                "message": f"Failed to assess damage: {str(e)}"
            }
    
    def chat(self, user_message: str) -> str:
        """
        Chat with user and process insurance claims
        
        Args:
            user_message: User's message
        
        Returns:
            AI assistant's response
        """
        # Add user message to history
        self.history.append(types.Content(
            role="user",
            parts=[types.Part.from_text(text=user_message)]
        ))
        
        # Define function declaration for tool calling
        assess_function = types.FunctionDeclaration(
            name="assess_disaster_damage",
            description="Assess building damage from a disaster using satellite imagery. Call this when you have location coordinates and before/after dates.",
            parameters={
                "type": "object",
                "properties": {
                    "location_name": {
                        "type": "string",
                        "description": "Human-readable location name (e.g., 'Fort Myers, Florida')"
                    },
                    "latitude": {
                        "type": "number",
                        "description": "Latitude coordinate"
                    },
                    "longitude": {
                        "type": "number",
                        "description": "Longitude coordinate"
                    },
                    "before_date": {
                        "type": "string",
                        "description": "Date before disaster in YYYY-MM-DD format"
                    },
                    "after_date": {
                        "type": "string",
                        "description": "Date after disaster in YYYY-MM-DD format"
                    }
                },
                "required": ["location_name", "latitude", "longitude", "before_date", "after_date"]
            }
        )
        
        tools = [types.Tool(function_declarations=[assess_function])]
        
        # Generate response with tool calling enabled
        config = types.GenerateContentConfig(
            system_instruction=self.system_instruction,
            tools=tools,
            temperature=0.7
        )
        
        response = self.client.models.generate_content(
            model=self.model,
            contents=self.history,
            config=config
        )
        
        # Handle function calls
        if response.candidates[0].content.parts:
            for part in response.candidates[0].content.parts:
                # Check for function call
                if hasattr(part, 'function_call') and part.function_call:
                    function_call = part.function_call
                    
                    if function_call.name == "assess_disaster_damage":
                        # Extract arguments
                        args = function_call.args
                        
                        logger.info("\n🤖 AI is calling assess_disaster_damage function")
                        logger.info(f"Arguments: {dict(args)}")
                        
                        # Call our function
                        result = self.assess_disaster_damage(
                            location_name=args.get('location_name'),
                            latitude=float(args.get('latitude')),
                            longitude=float(args.get('longitude')),
                            before_date=args.get('before_date'),
                            after_date=args.get('after_date')
                        )
                        
                        # Add function response to history
                        self.history.append(types.Content(
                            role="model",
                            parts=[part]
                        ))
                        
                        self.history.append(types.Content(
                            role="user",
                            parts=[types.Part.from_function_response(
                                name="assess_disaster_damage",
                                response=result
                            )]
                        ))
                        
                        # Generate final response with function result
                        final_response = self.client.models.generate_content(
                            model=self.model,
                            contents=self.history,
                            config=types.GenerateContentConfig(
                                system_instruction=self.system_instruction,
                                temperature=0.7
                            )
                        )
                        
                        assistant_message = final_response.text
                        
                        # Add to history
                        self.history.append(types.Content(
                            role="model",
                            parts=[types.Part.from_text(text=assistant_message)]
                        ))
                        
                        return assistant_message
        
        # Regular text response (no function call)
        assistant_message = response.text
        
        # Add to history
        self.history.append(types.Content(
            role="model",
            parts=[types.Part.from_text(text=assistant_message)]
        ))
        
        return assistant_message


def main():
    """Interactive chat interface - for testing only"""
    print("=" * 80)
    print("ALPHAEARTH INSURANCE AI CHATBOT")
    print("=" * 80)
    print("\nI can help you assess disaster damage using satellite imagery!")
    print("Just tell me about the disaster - location, type, and when it happened.\n")
    print("Type 'quit' to exit\n")
    print("-" * 80)
    
    # Initialize chatbot
    try:
        bot = InsuranceClaimChatbot()
    except ValueError as e:
        print(f"\nError: {e}")
        print("\nPlease set GEMINI_API_KEY environment variable:")
        print("  export GEMINI_API_KEY='your-api-key-here'")
        return
    
    if not bot.gee_ready:
        print("\nWARNING: Google Earth Engine not initialized")
        print("Satellite imagery features may not work. Run: earthengine authenticate\n")
    
    # Chat loop
    while True:
        try:
            user_input = input("\nYou: ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() in ['quit', 'exit', 'bye']:
                print("\nGoodbye! Stay safe!")
                break
            
            # Get AI response
            response = bot.chat(user_input)
            print(f"\nAI: {response}")
            
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"\nError: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
