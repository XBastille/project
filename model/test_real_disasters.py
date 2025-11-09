"""
TEST REAL DISASTERS - Fetch satellite imagery and run damage assessment

This test fetches ACTUAL satellite data from Google Earth Engine for known disasters
and runs the damage detection model on them.

NO LOCAL IMAGES - Everything fetched from GEE like in 01_test_gee_access.py and 02_fetch_sentinel2.py
"""

import sys
from pathlib import Path
import numpy as np
import cv2

# Add model to path
model_dir = Path(__file__).parent
sys.path.insert(0, str(model_dir))

from inference.damage_detector import DamageDetector

# Google Earth Engine
try:
    import ee
    HAS_GEE = True
except ImportError:
    print("[ERROR] earthengine-api not installed. Run: pip install earthengine-api")
    sys.exit(1)


def initialize_gee():
    """Initialize Google Earth Engine (same as 01_test_gee_access.py)"""
    print("[INIT] Initializing Google Earth Engine...")
    try:
        try:
            ee.Initialize(project="liquid-galaxy-469409-g5")
            print("[SUCCESS] Connected to GEE with project")
        except:
            ee.Initialize(opt_url='https://earthengine-highvolume.googleapis.com')
            print("[SUCCESS] Connected to GEE (high-volume endpoint)")
        return True
    except Exception as e:
        print(f"[ERROR] Failed to connect: {e}")
        print("\nRun: earthengine authenticate")
        return False


def fetch_sentinel2_image(lat, lon, date, days_range=30, cloud_threshold=30):
    """
    Fetch Sentinel-2 imagery from GEE (same as 02_fetch_sentinel2.py)
    
    Args:
        lat, lon: coordinates
        date: YYYY-MM-DD
        days_range: search window
        cloud_threshold: max cloud %
    
    Returns:
        numpy array (1024, 1024, 3) RGB image or None
    """
    from datetime import datetime, timedelta
    
    # Create point and area
    point = ee.Geometry.Point([lon, lat])
    area = point.buffer(1000)  # 1km radius
    
    # Date range
    target_date = datetime.strptime(date, '%Y-%m-%d')
    start_date = (target_date - timedelta(days=days_range)).strftime('%Y-%m-%d')
    end_date = (target_date + timedelta(days=days_range)).strftime('%Y-%m-%d')
    
    # Fetch collection
    collection = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
        .filterBounds(area) \
        .filterDate(start_date, end_date) \
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', cloud_threshold))
    
    count = collection.size().getInfo()
    print(f"   [SEARCH] {start_date} to {end_date}, found {count} images")
    
    if count == 0:
        print(f"   [WARNING] No images found, trying higher cloud threshold...")
        collection = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
            .filterBounds(area) \
            .filterDate(start_date, end_date) \
            .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 50))
        count = collection.size().getInfo()
        if count == 0:
            return None
    
    # Get least cloudy
    image = collection.sort('CLOUDY_PIXEL_PERCENTAGE').first()
    
    # Get metadata
    image_date = ee.Date(image.get('system:time_start')).format('YYYY-MM-dd').getInfo()
    cloud_cover = image.get('CLOUDY_PIXEL_PERCENTAGE').getInfo()
    print(f"   [SUCCESS] {image_date} (cloud: {cloud_cover:.1f}%)")
    
    # Get RGB bands
    rgb = image.select(['B4', 'B3', 'B2']).sampleRectangle(region=area)
    
    # Convert to numpy
    try:
        arr = None
        for i, band in enumerate(['B4', 'B3', 'B2']):
            band_data = np.array(rgb.get(band).getInfo())
            if arr is None:
                arr = np.zeros((band_data.shape[0], band_data.shape[1], 3))
            arr[:, :, i] = band_data
        
        # Resize to 1024x1024 and normalize
        arr = cv2.resize(arr, (1024, 1024))
        arr = np.clip(arr * 0.0001 * 255, 0, 255).astype(np.uint8)
        
        return arr
    except Exception as e:
        print(f"   [ERROR] Failed to download: {e}")
        return None


def test_disaster(name, lat, lon, before_date, after_date, expected_damage):
    """
    Test damage detection on a real disaster
    
    Args:
        name: disaster name
        lat, lon: coordinates
        before_date: YYYY-MM-DD before disaster
        after_date: YYYY-MM-DD after disaster
        expected_damage: what we expect to see (for verification)
    """
    print(f"\n{'='*80}")
    print(f"TESTING: {name}")
    print(f"{'='*80}")
    print(f"Location: ({lat}, {lon})")
    print(f"Expected: {expected_damage}")
    print(f"")
    
    # Fetch BEFORE image
    print(f"[BEFORE] Fetching satellite imagery for {before_date}...")
    before_img = fetch_sentinel2_image(lat, lon, before_date)
    
    if before_img is None:
        print(f"[FAILED] Could not fetch BEFORE image for {name}")
        return None
    
    # Fetch AFTER image
    print(f"\n[AFTER] Fetching satellite imagery for {after_date}...")
    after_img = fetch_sentinel2_image(lat, lon, after_date)
    
    if after_img is None:
        print(f"[FAILED] Could not fetch AFTER image for {name}")
        return None
    
    # Save images to model/test_outputs
    output_dir = Path(__file__).parent / 'test_outputs'
    output_dir.mkdir(exist_ok=True)
    
    safe_name = name.replace(' ', '_').replace(',', '')
    cv2.imwrite(str(output_dir / f"{safe_name}_before.png"), 
                cv2.cvtColor(before_img, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(output_dir / f"{safe_name}_after.png"), 
                cv2.cvtColor(after_img, cv2.COLOR_RGB2BGR))
    print(f"\n[SAVED] Images saved to test_outputs/")
    
    # Run damage assessment
    print(f"\n[ASSESS] Running damage detection model...")
    detector = DamageDetector(use_gpu=False)
    result = detector.assess(before_img, after_img)
    
    # Save results
    damage_class_map = np.argmax(result['damage_map'], axis=2).astype(np.uint8)
    damage_heatmap = cv2.applyColorMap(damage_class_map * 50, cv2.COLORMAP_JET)
    cv2.imwrite(str(output_dir / f"{safe_name}_damage.png"), damage_heatmap)
    
    # Print results
    print(f"\n{'='*80}")
    print(f"RESULTS: {name}")
    print(f"{'='*80}")
    print(f"Damage Level:    {result['damage_level']}")
    print(f"Confidence:      {result['confidence']:.1f}%")
    print(f"Decision:        {result['decision']}")
    print(f"Priority:        {result['priority']}")
    print(f"Buildings:       {result['num_buildings']} pixels")
    print(f"Expected:        {expected_damage}")
    print(f"")
    print(f"Damage Probabilities:")
    labels = ['No Building', 'No Damage', 'Minor', 'Major', 'Destroyed']
    for i, (label, prob) in enumerate(zip(labels, result['damage_probabilities'])):
        print(f"  {label:15s}: {prob:.1%}")
    print(f"{'='*80}\n")
    
    return result


def main():
    """Run tests on known disasters"""
    
    print(f"\n{'#'*80}")
    print(f"# REAL DISASTER TEST - AlphaEarth Insurance AI")
    print(f"# Fetching ACTUAL satellite data from Google Earth Engine")
    print(f"{'#'*80}\n")
    
    # Initialize GEE
    if not initialize_gee():
        print("[STOP] Cannot proceed without GEE")
        return
    
    print("\n[INFO] Will test on known disasters with confirmed dates")
    print("[INFO] Fetching satellite imagery may take a few moments...\n")
    
    # Test scenarios with VERIFIED coordinates and dates from web search
    disasters = [
        {
            'name': 'Hurricane Ian - Fort Myers FL',
            'lat': 26.6406,
            'lon': -81.8723,
            'before_date': '2022-08-15',  # Before hurricane (mid-August)
            'after_date': '2022-10-05',   # After hurricane (early Oct)
            'expected': 'APPROVE - Major/Destroyed (Sept 28 landfall)'
        },
        {
            'name': 'Nepal Earthquake - Kathmandu',
            'lat': 27.7172,
            'lon': 85.3240,
            'before_date': '2015-03-15',  # Before earthquake (mid-March)
            'after_date': '2015-05-15',   # After earthquake (mid-May)
            'expected': 'APPROVE - Major Damage (April 25 earthquake 7.8M)'
        },
        {
            'name': 'Camp Fire - Paradise CA',
            'lat': 39.7596,
            'lon': -121.6219,
            'before_date': '2018-10-15',  # Before fire (mid-October)
            'after_date': '2018-11-20',   # After fire (late Nov)
            'expected': 'APPROVE - Destroyed (Nov 8-25 fire)'
        },
        {
            'name': 'Control - San Francisco (No Disaster)',
            'lat': 37.7749,
            'lon': -122.4194,
            'before_date': '2024-01-15',
            'after_date': '2024-06-15',
            'expected': 'REJECT - No Damage'
        }
    ]
    
    results = []
    
    for disaster in disasters:
        try:
            result = test_disaster(
                name=disaster['name'],
                lat=disaster['lat'],
                lon=disaster['lon'],
                before_date=disaster['before_date'],
                after_date=disaster['after_date'],
                expected_damage=disaster['expected']
            )
            if result:
                results.append({
                    'name': disaster['name'],
                    'result': result,
                    'expected': disaster['expected']
                })
        except Exception as e:
            print(f"[ERROR] Test failed for {disaster['name']}: {e}")
            import traceback
            traceback.print_exc()
    
    # Summary
    print(f"\n{'#'*80}")
    print(f"# TEST SUMMARY")
    print(f"{'#'*80}\n")
    
    if results:
        print(f"Completed {len(results)}/{len(disasters)} tests\n")
        for r in results:
            print(f"{r['name']:45s} => {r['result']['decision']:15s} ({r['result']['damage_level']})")
            print(f"{'':45s}    Expected: {r['expected']}")
            print()
    else:
        print("[WARNING] No tests completed successfully")
        print("Check your GEE authentication and internet connection")
    
    print(f"\n[INFO] All satellite images and damage maps saved to: model/test_outputs/")
    print(f"[INFO] Review the before/after/damage images to verify model performance\n")


if __name__ == "__main__":
    main()
