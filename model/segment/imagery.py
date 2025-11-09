import numpy as np
import segment.utils as utils
from keras.models import load_model
import matplotlib.pyplot as plt

# Load model
model = load_model("building_segmentation_model.keras")

def segment(img):
    """Segment buildings from image"""
    mask = utils.predict(model, img)
    mask = mask[..., 0]
    
    # Set masked regions to blue with 70% transparency
    blue_color = np.array([255, 0, 0], dtype=np.uint8)  # Red color
    transparency = 0.7
    
    masked_image = img.copy()
    masked_image[mask == 1] = masked_image[mask == 1] * (1 - transparency) +  transparency * blue_color
    
    return masked_image

# CLI usage example:
# from segment.imagery import segment
# import cv2
# img = cv2.imread('image.png')
# result = segment(img)
# cv2.imwrite('segmented.png', result) 