import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from pathlib import Path

# ---------------------------------- Setup ---------------------------------- #

# Constants
SEGMENT_COLOR = {
    'top': 0,
    'middle': 80,
    'bottom': 160,
    'bg_top': 255,
    'bg_middle': 225,
    'bg_bottom': 190
}

# Path to GOALS dataset root directory
DATA_ROOT_DIR = Path('../data/GOALS/')
DATA_SUB_DIRS = ['Train', 'Validation', 'Test']
NEW_LAYER_MASK_DIR = 'Layer_Masks++'

# Create directories for saving the images
for dir in DATA_SUB_DIRS:
    Path.mkdir(DATA_ROOT_DIR / dir / 'Layer_Masks++', parents=True, exist_ok=True)

# ---------------------------------- Functions ---------------------------------- #

def get_middle_seed(image):
    """
    Returns the seed point for the middle segment background (assumes it is initially
    the same color as bg_top)
    """
    height, _ = image.shape
    for i in range(height):
        if image[i, 0] == SEGMENT_COLOR['middle'] and image[i+1, 0] == SEGMENT_COLOR['bg_top']:
            return (0, i+1)
        
# ---------------------------------- Main ----------------------------------- #

def main():
    # Loop over all images
    for dir in DATA_SUB_DIRS:
        for img_path in Path(DATA_ROOT_DIR / dir).glob('Layer_Masks/*.png'):
            # Load the image
            image = cv.imread(str(img_path), cv.IMREAD_GRAYSCALE)

            # Fill the bottom background segment
            height, _ = image.shape
            cv.floodFill(image, None, (0, height-1), SEGMENT_COLOR['bg_bottom'])

            # Get seed point for middle segment background and fill it
            middle_seed = get_middle_seed(image)
            cv.floodFill(image, None, middle_seed, SEGMENT_COLOR['bg_middle'])

            # < Leave top segment as is >

            # Save the image
            cv.imwrite(str(DATA_ROOT_DIR / dir / NEW_LAYER_MASK_DIR / img_path.name), image)

if __name__ == '__main__':
    main()



