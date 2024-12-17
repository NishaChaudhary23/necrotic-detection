# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import openslide
import numpy as np
import cv2
import matplotlib.pyplot as plt

# Load the WSI image
wsi_path = '/home/varnit/Documents/necrotic-ai/raw-data/op-18-22.svs'
slide = openslide.OpenSlide(wsi_path)

# Read the whole-slide image
wsi_image = slide.read_region((0, 0), 0, slide.level_dimensions[0])

dims = slide.level_dimensions

#Get a thumbnail of the image and visualize
#slide_thumbnail = slide.get_thumbnail((slide.dimensions[0] / 256, slide.dimensions[1] / 256))
#slide_thumbnail.show()

#By how much are levels downsampled from the original image?
factors = slide.level_downsamples
print("Each level is downsampled by an amount of: ", factors)

#Copy an image from a level
level3_dim = dims[2]
#Give pixel coordinates (top left pixel in the original large image)
#Also give the level number (for level 3 we are providing a valueof 2)

#Size of your output image
#the output would be a RGBA image (Not, RGB)
level3_img = slide.read_region((0,0), 2, level3_dim) #Pillow object, mode=RGBA

#Convert the image to RGB
level3_img_RGB = level3_img.convert('RGB')
level3_img_RGB.show()

#Convert the image into numpy array for processing
level3_img_np = np.array(level3_img_RGB)
plt.imshow(level3_img_np)

#Return the best level for displaying the given downsample.
SCALE_FACTOR = 32
best_level = slide.get_best_level_for_downsample(SCALE_FACTOR)
#Here it returns the best level to be 2 (third level)
#If you change the scale factor to 2, it will suggest the best level to be 0 (our 1st level)
#############################################################################################

# Convert the image to grayscale
gray_image = cv2.cvtColor(level3_img_np, cv2.COLOR_RGBA2GRAY)

# Display the grayscale image using Matplotlib
plt.figure(figsize=(8, 8))
plt.title('Grayscale Image')
plt.imshow(gray_image, cmap='gray')
plt.axis('off')
plt.show()

cv2.imwrite('/home/varnit/Documents/necrotic-ai/segmentation-output/op-18-22-gray.png', gray_image)
cv2.imwrite('/home/varnit/Documents/necrotic-ai/segmentation-output/op-18-22-rgb.png', level3_img_np)


# Apply Otsu's thresholding to create a binary mask
_, binary_mask = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Invert the binary mask
inverted_mask = cv2.bitwise_not(binary_mask)

# Save the segmented image
cv2.imwrite('/home/varnit/Documents/necrotic-ai/segmentation-output/op-18-22segmented_image_otsu.png', inverted_mask)

# Display the inverted segmented image using Matplotlib
plt.imshow(inverted_mask, cmap='gray')
plt.title('Segmented Image')
plt.show()

# Now, let's proceed to step 4, tiling of the tissue regions

# Apply the binary mask to the RGB image
# First, ensure the mask is the same shape as the RGB image
binary_mask_3d = np.repeat(binary_mask[:, :, np.newaxis], 3, axis=2)
masked_rgb_image = cv2.bitwise_and(level3_img_np, binary_mask_3d)

# Define the tile size
tile_size = 256

# Function to extract tiles from the masked RGB image
def extract_rgb_tiles(masked_rgb_image, binary_mask, tile_size):
    tiles = []
    for i in range(0, binary_mask.shape[0], tile_size):
        for j in range(0, binary_mask.shape[1], tile_size):
            # Extract the tile from the binary mask to check if it contains tissue
            binary_tile = binary_mask[i:i+tile_size, j:j+tile_size]
            if np.any(binary_tile):
                # Extract the corresponding tile from the RGB image
                rgb_tile = masked_rgb_image[i:i+tile_size, j:j+tile_size]
                tiles.append((i, j, rgb_tile))
    return tiles

# Extract RGB tiles from the masked RGB image
rgb_tiles = extract_rgb_tiles(masked_rgb_image, binary_mask, tile_size)
print(f"Number of tiles containing tissue: {len(rgb_tiles)}")

# Specify the save path
save_path = '/home/varnit/Documents/necrotic-ai/segmentation-output/op-18-22-tiles'

# Save the RGB tiles
for idx, (i, j, rgb_tile) in enumerate(rgb_tiles):
    tile_image_path = f'{save_path}/rgb_tile_{idx}_{i}_{j}.png'
    cv2.imwrite(tile_image_path, cv2.cvtColor(rgb_tile, cv2.COLOR_RGB2BGR))
    print(f"RGB Tile {idx} saved at {tile_image_path}")
















