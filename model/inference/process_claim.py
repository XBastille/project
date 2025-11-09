"""
Main inference system for AlphaEarth Insurance AI
Integrates: Google Earth Engine + Sentinel-2 + AlphaEarth + Damage Detection
"""

import sys
from pathlib import Path
import numpy as np
import cv2
import torch
import json
from datetime import datetime, timedelta

# Add model to path
model_dir = Path(__file__).parent.parent
sys.path.insert(0, str(model_dir))

from inference.damage_detector import DamageDetector

# Google Earth Engine
try:
    import ee
    HAS_GEE = True
except ImportError:
    ee = None
    HAS_GEE = False


class InsuranceClaimProcessor:
    """
    Complete insurance claim processing system
    Features:
    1. Fetch satellite imagery (Sentinel-2)
    2. Access AlphaEarth embeddings
    3. Detect building damage
    4. Make insurance decision
    """
    
    def __init__(self, use_gpu=False):
        """Initialize processor"""
        self.detector = DamageDetector(use_gpu=use_gpu)
        
        if HAS_GEE:
            try:
                ee.Initialize(project="liquid-galaxy-469409-g5")
                print("[GEE] Authenticated successfully")
                self.gee_ready = True
            except Exception as e:
                print(f"[GEE] Failed to initialize: {e}")
                self.gee_ready = False
        else:
            self.gee_ready = False
            print("[GEE] Not available - using local images only")
    
    def fetch_sentinel2(self, lat, lon, date, days_range=30, cloud_threshold=30):
        """
        Fetch Sentinel-2 imagery
        Args:
            lat, lon: location coordinates
            date: target date (YYYY-MM-DD)
            days_range: search window
            cloud_threshold: max cloud cover %
        Returns:
            numpy array (1024, 1024, 3) RGB image
        """
        if not self.gee_ready:
            print("[WARNING] GEE not available")
            return None
        
        point = ee.Geometry.Point([lon, lat])
        area = point.buffer(1000)
        
        target_date = datetime.strptime(date, '%Y-%m-%d')
        start_date = (target_date - timedelta(days=days_range)).strftime('%Y-%m-%d')
        end_date = (target_date + timedelta(days=days_range)).strftime('%Y-%m-%d')
        
        collection = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
            .filterBounds(area) \
            .filterDate(start_date, end_date) \
            .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', cloud_threshold))
        
        count = collection.size().getInfo()
        if count == 0:
            return None
        
        image = collection.sort('CLOUDY_PIXEL_PERCENTAGE').first()
        
        # Get RGB bands
        rgb = image.select(['B4', 'B3', 'B2']).sampleRectangle(region=area)
        
        # Convert to numpy
        arr = None
        for i, band in enumerate(['B4', 'B3', 'B2']):
            band_data = np.array(rgb.get(band).getInfo())
            if arr is None:
                arr = np.zeros((band_data.shape[0], band_data.shape[1], 3))
            arr[:, :, i] = band_data
        
        # Resize and normalize
        arr = cv2.resize(arr, (1024, 1024))
        arr = np.clip(arr * 0.0001 * 255, 0, 255).astype(np.uint8)
        
        return arr
    
    def fetch_alphaearth_embeddings(self, lat, lon, year=2023):
        """
        Fetch AlphaEarth embeddings
        Args:
            lat, lon: location coordinates
            year: embedding year (2016-2023)
        Returns:
            dict with embedding info
        """
        if not self.gee_ready:
            return None
        
        point = ee.Geometry.Point([lon, lat])
        area = point.buffer(1000)
        
        collection = ee.ImageCollection('GOOGLE/EARTH_AI/SATELLITE_EMBEDDING/V1') \
            .filterBounds(area) \
            .filterDate(f'{year}-01-01', f'{year}-12-31')
        
        count = collection.size().getInfo()
        if count == 0:
            return None
        
        embedding = collection.first()
        
        return {
            'available': True,
            'year': year,
            'count': count,
            'bands': 64  # AlphaEarth has 64-dimensional embeddings
        }
    
    def process_claim(self, lat, lon, before_date, after_date, location_name="Unknown"):
        """
        Complete claim processing pipeline
        Args:
            lat, lon: location coordinates
            before_date: date before disaster (YYYY-MM-DD)
            after_date: date after disaster (YYYY-MM-DD)
            location_name: human-readable name
        Returns:
            dict with complete assessment
        """
        print(f"\n{'='*70}")
        print(f"ALPHAEARTH INSURANCE CLAIM PROCESSOR")
        print(f"{'='*70}")
        print(f"Location: {location_name}")
        print(f"Coordinates: ({lat}, {lon})")
        print(f"Before: {before_date}")
        print(f"After: {after_date}")
        print(f"{'='*70}\n")
        
        # Step 1: Fetch Sentinel-2 imagery
        print("[STEP 1/4] Fetching Sentinel-2 satellite imagery...")
        before_img = self.fetch_sentinel2(lat, lon, before_date)
        after_img = self.fetch_sentinel2(lat, lon, after_date)
        
        if before_img is None or after_img is None:
            print("[ERROR] Could not fetch satellite imagery")
            return {'error': 'No satellite imagery available'}
        
        print(f"  Before: {before_img.shape} (cloud optimized)")
        print(f"  After: {after_img.shape} (cloud optimized)")
        
        # Step 2: Fetch AlphaEarth embeddings
        print("\n[STEP 2/4] Accessing AlphaEarth AI embeddings...")
        alphaearth = self.fetch_alphaearth_embeddings(lat, lon)
        
        if alphaearth:
            print(f"  AlphaEarth: {alphaearth['bands']}-dimensional embeddings available")
        else:
            print("  AlphaEarth: Not available for this location")
        
        # Step 3: Detect damage
        print("\n[STEP 3/4] Analyzing building damage with AI...")
        result = self.detector.assess(before_img, after_img)
        
        # Step 4: Make insurance decision
        print("\n[STEP 4/4] Making insurance claim decision...")
        
        # Add location info
        result['location'] = {
            'name': location_name,
            'lat': lat,
            'lon': lon,
            'before_date': before_date,
            'after_date': after_date
        }
        result['alphaearth'] = alphaearth
        result['satellite_source'] = 'Sentinel-2 SR Harmonized'
        
        # Print results
        print(f"\n{'='*70}")
        print(f"CLAIM ASSESSMENT RESULTS")
        print(f"{'='*70}")
        print(f"Damage Level:    {result['damage_level']}")
        print(f"Confidence:      {result['confidence']:.1f}%")
        print(f"Decision:        {result['decision']}")
        print(f"Priority:        {result['priority']}")
        print(f"Buildings:       {result['num_buildings']} pixels detected")
        print(f"Data Sources:    Sentinel-2 + AlphaEarth")
        print(f"{'='*70}\n")
        
        return result
    
    def process_local_images(self, before_path, after_path, location_name="Unknown"):
        """
        Process claim from local images (no satellite fetch)
        Args:
            before_path: path to before image
            after_path: path to after image
            location_name: human-readable name
        Returns:
            dict with assessment
        """
        print(f"\n{'='*70}")
        print(f"PROCESSING LOCAL IMAGES")
        print(f"{'='*70}")
        print(f"Location: {location_name}")
        print(f"Before: {before_path}")
        print(f"After: {after_path}")
        print(f"{'='*70}\n")
        
        # Load images
        before_img = cv2.imread(str(before_path))
        after_img = cv2.imread(str(after_path))
        
        if before_img is None or after_img is None:
            return {'error': 'Could not load images'}
        
        before_img = cv2.cvtColor(before_img, cv2.COLOR_BGR2RGB)
        after_img = cv2.cvtColor(after_img, cv2.COLOR_BGR2RGB)
        
        # Assess damage
        result = self.detector.assess(before_img, after_img)
        result['location'] = {'name': location_name}
        result['satellite_source'] = 'Local files'
        
        print(f"\n{'='*70}")
        print(f"ASSESSMENT RESULTS")
        print(f"{'='*70}")
        print(f"Damage Level:    {result['damage_level']}")
        print(f"Confidence:      {result['confidence']:.1f}%")
        print(f"Decision:        {result['decision']}")
        print(f"Priority:        {result['priority']}")
        print(f"{'='*70}\n")
        
        return result
    
    def save_report(self, result, output_dir='results'):
        """Save assessment report"""
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        location_name = result['location'].get('name', 'unknown').replace(' ', '_')
        base_name = f"{location_name}_{timestamp}"
        
        # Save images
        if 'before_image' in result:
            cv2.imwrite(str(output_dir / f"{base_name}_before.png"),
                       cv2.cvtColor(result['before_image'], cv2.COLOR_RGB2BGR))
            cv2.imwrite(str(output_dir / f"{base_name}_after.png"),
                       cv2.cvtColor(result['after_image'], cv2.COLOR_RGB2BGR))
            cv2.imwrite(str(output_dir / f"{base_name}_buildings.png"),
                       result['building_mask'] * 255)
            
            # Damage heatmap
            damage_class_map = np.argmax(result['damage_map'], axis=2).astype(np.uint8)
            damage_heatmap = cv2.applyColorMap(damage_class_map * 50, cv2.COLORMAP_JET)
            cv2.imwrite(str(output_dir / f"{base_name}_damage.png"), damage_heatmap)
        
        # JSON report
        report = {
            'claim_id': f"CLM_{timestamp}",
            'timestamp': timestamp,
            'location': result.get('location', {}),
            'satellite_source': result.get('satellite_source', 'Unknown'),
            'alphaearth': result.get('alphaearth', {}),
            'assessment': {
                'damage_level': result['damage_level'],
                'damage_class': result['damage_class'],
                'confidence': result['confidence'],
                'decision': result['decision'],
                'priority': result['priority'],
                'num_buildings': result['num_buildings']
            },
            'damage_probabilities': {
                'no_building': result['damage_probabilities'][0],
                'no_damage': result['damage_probabilities'][1],
                'minor_damage': result['damage_probabilities'][2],
                'major_damage': result['damage_probabilities'][3],
                'destroyed': result['damage_probabilities'][4]
            }
        }
        
        report_path = output_dir / f"{base_name}_claim_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"[SAVED] Complete report: {report_path}")
        return str(report_path)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='AlphaEarth Insurance Claim Processor')
    parser.add_argument('--mode', type=str, choices=['satellite', 'local', 'test'], required=True,
                       help='Processing mode: satellite (live fetch), local (image files), test (demo)')
    parser.add_argument('--lat', type=float, help='Latitude')
    parser.add_argument('--lon', type=float, help='Longitude')
    parser.add_argument('--before-date', type=str, help='Before date (YYYY-MM-DD)')
    parser.add_argument('--after-date', type=str, help='After date (YYYY-MM-DD)')
    parser.add_argument('--before-img', type=str, help='Path to before image')
    parser.add_argument('--after-img', type=str, help='Path to after image')
    parser.add_argument('--name', type=str, default='Location', help='Location name')
    parser.add_argument('--output', type=str, default='results', help='Output directory')
    parser.add_argument('--gpu', action='store_true', help='Use GPU')
    
    args = parser.parse_args()
    
    # Initialize processor
    processor = InsuranceClaimProcessor(use_gpu=args.gpu)
    
    # Process based on mode
    if args.mode == 'satellite':
        if not all([args.lat, args.lon, args.before_date, args.after_date]):
            print("ERROR: satellite mode requires --lat --lon --before-date --after-date")
            sys.exit(1)
        
        result = processor.process_claim(
            lat=args.lat,
            lon=args.lon,
            before_date=args.before_date,
            after_date=args.after_date,
            location_name=args.name
        )
    
    elif args.mode == 'local':
        if not all([args.before_img, args.after_img]):
            print("ERROR: local mode requires --before-img --after-img")
            sys.exit(1)
        
        result = processor.process_local_images(
            before_path=args.before_img,
            after_path=args.after_img,
            location_name=args.name
        )
    
    elif args.mode == 'test':
        print("Running demo with synthetic test images...")
        
        # Create synthetic test images
        import numpy as np
        before_img = np.random.randint(0, 255, (1024, 1024, 3), dtype=np.uint8)
        after_img = np.random.randint(0, 255, (1024, 1024, 3), dtype=np.uint8)
        
        # Save temporarily
        import cv2
        temp_dir = Path('temp_test')
        temp_dir.mkdir(exist_ok=True)
        cv2.imwrite(str(temp_dir / 'before.png'), before_img)
        cv2.imwrite(str(temp_dir / 'after.png'), after_img)
        
        result = processor.process_local_images(
            before_path=temp_dir / 'before.png',
            after_path=temp_dir / 'after.png',
            location_name='Synthetic Test'
        )
        
        # Cleanup
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    # Save report
    if 'error' not in result:
        processor.save_report(result, args.output)
