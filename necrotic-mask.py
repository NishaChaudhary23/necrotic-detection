#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 21 15:04:52 2024

@author: varnit
"""

# Import the required libraries
import openslide
from openslide.deepzoom import DeepZoomGenerator
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.color import rgb2hed, hed2rgb
from skimage.exposure import rescale_intensity
from skimage.filters import threshold_otsu
from PIL import Image

# Load the WSI image
wsi_path = '/Users/nishachaudhary/Documents/others/TA/Necrotic-detection/necrotic-ai/op-18-22.svs'
slide = openslide.OpenSlide(wsi_path)

# Read the level 3 image from the WSI
level3_dim = slide.level_dimensions[2]
level3_img = slide.read_region((0,0), 2, level3_dim)

# Convert the image to RGB
level3_img_RGB = level3_img.convert('RGB')

# Convert the image into numpy array for processing
level3_img_np = np.array(level3_img_RGB)

fig, ax = plt.subplots(figsize=(4, 4), dpi=300)
ax.imshow(level3_img_np)  # Convert BGR to RGB for displaying
#ax.set_title('Overlayed Image with Contours')
ax.axis('off')
# Save the figure
save_path_level3_img_RGB = '/Users/nishachaudhary/Documents/others/TA/Necrotic-detection/necrotic-ai/op18-22-np.png'
fig.savefig(save_path_level3_img_RGB, dpi=300, bbox_inches='tight', pad_inches=0)
cv2.imwrite('/Users/nishachaudhary/Documents/others/TA/Necrotic-detection/necrotic-ai/op18-22-np.png', level3_img_np)

# Convert the RGB image to grayscale
gray_image = cv2.cvtColor(level3_img_np, cv2.COLOR_RGB2GRAY)

# Apply Otsu's thresholding to create a binary mask
_, binary_mask = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Invert the binary mask
inverted_mask = cv2.bitwise_not(binary_mask)

# Remove the alpha channel if present to get an RGB image
if level3_img_np.shape[2] == 4:
    level3_img_np = level3_img_np[:, :, :3]

# Perform color deconvolution to separate the stains
hed = rgb2hed(level3_img_np)

# Rescale the Hematoxylin channel to a range suitable for thresholding
hematoxylin = rescale_intensity(hed[:, :, 0], out_range=(0, 1))

# Threshold the Hematoxylin channel to segment the lightly stained tissue
thresh = threshold_otsu(hematoxylin)
lightly_stained_mask = hematoxylin > thresh

# Post-processing: Clean the mask
kernel = np.ones((3,3), np.uint8)
lightly_stained_mask_cleaned = cv2.morphologyEx(lightly_stained_mask.astype(np.uint8), cv2.MORPH_OPEN, kernel, iterations=1)

# Apply the segmented region mask to the binary mask of the tissue
combined_mask = cv2.bitwise_and(inverted_mask, inverted_mask, mask=lightly_stained_mask_cleaned)

min_val = np.min(combined_mask)
max_val = np.max(combined_mask)

# If you have the 'hematoxylin' channel already calculated from the deconvolution step
thresh_value = threshold_otsu(hematoxylin)

print(f"Minimum value in combined_mask: {min_val}")
print(f"Maximum value in combined_mask: {max_val}")
print(f"Otsu's threshold value: {thresh_value}")


# Save the combined mask image
save_path_combined_mask = '/Users/nishachaudhary/Documents/others/TA/Necrotic-detection/necrotic-ai/combined_mask.png'
cv2.imwrite(save_path_combined_mask, combined_mask)

# Display the combined mask
plt.figure(figsize=(10, 10))
plt.imshow(combined_mask, cmap='gray')
plt.title('Combined Mask of Lightly Stained Tissue')
plt.axis('off')
plt.show()

#print(f"Combined mask saved at: {save_path_combined_mask}")


# Find contours of the combined mask
contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Fill the contours onto the original image using a black color
for contour in contours:
    cv2.drawContours(level3_img_np, [contour], -1, (0, 0, 0), thickness=-1)  # Fill contour with black color

# Convert the result to BGR for saving using OpenCV functions
overlayed_image_filled_contours = cv2.cvtColor(level3_img_np, cv2.COLOR_RGB2BGR)


# Since the contour has been drawn directly onto the level3_img_np, we don't need to blend it
# Simply convert to BGR if needed for saving/displaying using OpenCV functions
overlayed_image_with_filled_contours = cv2.cvtColor(level3_img_np, cv2.COLOR_RGB2BGR)

# Display the overlayed image with contours
fig, ax = plt.subplots(figsize=(4, 4), dpi=300)
ax.imshow(cv2.cvtColor(overlayed_image_with_filled_contours, cv2.COLOR_BGR2RGB))  # Convert BGR to RGB for displaying
ax.set_title('Overlayed Image with Contours')
ax.axis('off')

# Save the figure
save_path_overlayed_image_with_contours = '/Users/nishachaudhary/Documents/others/TA/Necrotic-detection/necrotic-ai/overlayed_image_with_filled_contours.png'
fig.savefig(save_path_overlayed_image_with_contours, dpi=300, bbox_inches='tight', pad_inches=0)

overlayed_image_with_filled_contours_np = np.array(overlayed_image_with_filled_contours)
cv2.imwrite('/Users/nishachaudhary/Documents/others/TA/Necrotic-detection/necrotic-ai/overlay_with_filled_contours.png', overlayed_image_with_filled_contours)
plt.show()

# Convert the mask to a color overlay (e.g., green color)
mask_color = [0, 255, 0]  # green color
colored_mask = np.zeros_like(level3_img_np, dtype=np.uint8)
for i in range(3):
    colored_mask[:,:,i] = combined_mask * mask_color[i]

# Overlay the colored mask on the original image
overlayed_image = cv2.addWeighted(level3_img_np, 1.0, colored_mask, 0.7, 0)

# Save the overlayed image
save_path_overlayed_image = '/Users/nishachaudhary/Documents/others/TA/Necrotic-detection/necrotic-ai/overlayed_image-g.png'
cv2.imwrite(save_path_overlayed_image, cv2.cvtColor(overlayed_image, cv2.COLOR_RGB2BGR))

# Display the overlayed image
plt.figure(figsize=(8, 8))
plt.imshow(overlayed_image)
plt.title('Overlayed Image')
plt.axis('off')
plt.show()



##################################################################################################################################



# Load the WSI and create the DeepZoomGenerator object
wsi_path = '/home/varnit/Documents/necrotic-ai/raw-data/op-18-22.svs'
slide = openslide.OpenSlide(wsi_path)
dz = DeepZoomGenerator(slide, tile_size=256, overlap=0, limit_bounds=False)

# Load the combined mask (already at level 3)
combined_mask_path = '/home/varnit/Documents/necrotic-ai/segmentation-output/combined_mask.png'
combined_mask = cv2.imread(combined_mask_path, cv2.IMREAD_GRAYSCALE)

# Verify that the mask is loaded correctly
if combined_mask is None:
    raise ValueError("Failed to load the combined mask.")

print(f"Mask Shape: {combined_mask.shape}")
print(f"Non-zero elements in Mask: {np.count_nonzero(combined_mask)}")

# Define the output directory for the tiles
tiles_output_dir = '/home/varnit/Documents/necrotic-ai/segmentation-output/tiles_level3'
os.makedirs(tiles_output_dir, exist_ok=True)

# Iterate through the tiles at level 3
tiles = dz.level_tiles[3]
tile_count = 0
for y in range(tiles[1]):
    for x in range(tiles[0]):
        # Get the tile coordinates at the current downsample level
        tile_mask = combined_mask[y * 256:(y + 1) * 256, x * 256:(x + 1) * 256]
        # If there's any segmentation within the tile, save the tile
        if np.any(tile_mask):
            tile = dz.get_tile(3, (x, y))
            tile_path = os.path.join(tiles_output_dir, f'tile_{x}_{y}_level3.png')
            tile.save(tile_path)
            tile_count += 1

print(f"Tiles from level 3 saved to {tiles_output_dir}")
print(f"Number of tiles saved: {tile_count}")









# Define the tile size
tile_size = 256

# Function to extract tiles from the masked RGB image
def extract_segmented_tiles(masked_rgb_image, combined_mask, tile_size):
    tiles = []
    for i in range(0, combined_mask.shape[0], tile_size):
        for j in range(0, combined_mask.shape[1], tile_size):
            # Extract the tile from the combined mask to check if it contains segmented tissue
            mask_tile = combined_mask[i:i+tile_size, j:j+tile_size]
            if np.any(mask_tile):  # If any part of the tile is segmented
                # Extract the corresponding tile from the RGB image
                rgb_tile = masked_rgb_image[i:i+tile_size, j:j+tile_size]
                tiles.append((i, j, rgb_tile))
    return tiles

# Apply the combined mask to the level3_img_np to get the segmented regions
masked_rgb_image = cv2.bitwise_and(level3_img_np, level3_img_np, mask=combined_mask.astype(np.uint8))

# Extract tiles from the masked RGB image
segmented_tiles = extract_segmented_tiles(masked_rgb_image, combined_mask, tile_size)

# Specify the save path
save_path_tiles = '/home/varnit/Documents/necrotic-ai/segmentation-output/segmented_tiles'

# Save the tiles that contain segmented regions
for idx, (i, j, rgb_tile) in enumerate(segmented_tiles):
    tile_image_path = f'{save_path_tiles}/segmented_tile_{idx}_{i}_{j}.png'
    cv2.imwrite(tile_image_path, rgb_tile)
    print(f"Segmented Tile {idx} saved at {tile_image_path}")

# Display the number of tiles containing segmented regions
print(f"Number of tiles containing segmented tissue: {len(segmented_tiles)}")









