#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 14 14:42:36 2022

@author: nisha
"""

#import required libraries and packages

import openslide
import numpy as np
import matplotlib.pyplot as plt
from openslide import open_slide, ImageSlide
from PIL import Image

#import .svs file
slide_a= open_slide('/Users/nishachaudhary/Documents/nisha/WSI-digiscan/11Feb.-digiscan-65slides/OSMF with OSCC/152-19.svs')
slide_props = slide_a.properties

print("Vendor is:", slide_props['openslide.vendor'])
print("Pixel size of X in um is:", slide_props['openslide.mpp-x'])
print("Pixel size of Y in um is:", slide_props['openslide.mpp-y'])

#Objective used to capture the image
objective = float(slide_a.properties[openslide.PROPERTY_NAME_OBJECTIVE_POWER])
print("The objective power is: ", objective)

# get slide dimensions for the level 0 - max resolution level
slide_dims = slide_a.dimensions
print(slide_dims)

#Getting a thumbnail of the image and visualize
slide_thumb_600 = slide_a.get_thumbnail(size=(600, 600))
slide_thumb_600.show()

#Convert thumbnail to numpy array
slide_thumb_600_np = np.array(slide_thumb_600)
plt.figure(figsize=(8,8))
plt.imshow(slide_thumb_600_np)

#Getting slide dims at each level, whole slide images store information
#as pyramid at various levels
dims = slide_a.level_dimensions

num_levels = len(dims)
print("Number of levels in this image are:", num_levels)

print("Dimensions of various levels in this image are:", dims)

#By how much are levels downsampled from the original image?
factors = slide_a.level_downsamples
print("Each level is downsampled by an amount of: ", factors)

#Copy an image from a level
level3_dim = dims[2]
#Give pixel coordinates (top left pixel in the original large image)
#Also give the level number (for level 3 we are providing a valueof 2)
#Size of your output image
#the output would be a RGBA image (Not, RGB)
level3_img = slide_a.read_region((0,0), 2, level3_dim) #Pillow object, mode=RGBA

#Convert the image to RGB
level3_img_RGB = level3_img.convert('RGB')
level3_img_RGB.show()

#Convert the image into numpy array for processing
level3_img_np = np.array(level3_img_RGB)
plt.imshow(level3_img_np)

#Return the best level for displaying the given downsample.
SCALE_FACTOR = 32
best_level = slide_a.get_best_level_for_downsample(SCALE_FACTOR)
#Here it returns the best level to be 2 (third level)
#If you change the scale factor to 2, it will suggest the best level to be 0 (our 1st level)
#################################

#Generating tiles

from openslide.deepzoom import DeepZoomGenerator
#Generate object for tiles using the DeepZoomGenerator
tiles = DeepZoomGenerator(slide_a, tile_size=300, overlap=0, limit_bounds=False)
#Here, we have divided our svs into tiles of size 300 with no overlap. 

#The tiles object also contains data at many levels. 
#To check the number of levels
print("The number of levels in the tiles object are: ", tiles.level_count)

print("The dimensions of data in each level are: ", tiles.level_dimensions)

#Total number of tiles in the tiles object
print("Total number of tiles = : ", tiles.tile_count)

#How many tiles at a specific level?
level_num = 11
print("Tiles shape at level ", level_num, " is: ", tiles.level_tiles[level_num])
print("This means there are ", tiles.level_tiles[level_num][0]*tiles.level_tiles[level_num][1], " total tiles in this level")

#Dimensions of the tile (tile size) for a specific tile from a specific layer
tile_dims = tiles.get_tile_dimensions(11, (3,4)) 
#Provide deep zoom level and address (column, row)

#Tile count at the highest resolution level (level 16 in our tiles)
tile_count_in_large_image = tiles.level_tiles[16] 
#233 x 215 (59621/256 = 233 with no overlap pixels)

#Check tile size for some random tile
tile_dims = tiles.get_tile_dimensions(16, (120,140))

#Last tiles may not have full 256x256 dimensions as our large image is not exactly divisible by 256
tile_dims = tiles.get_tile_dimensions(16, (232, 214))

##########################################################################################
single_tile = tiles.get_tile(16, (500, 500)) #Provide deep zoom level and address (column, row)
single_tile_RGB = single_tile.convert('RGB')
single_tile_RGB.show()

###### Saving each tile to local directory
cols, rows = tiles.level_tiles[16]

import os
tile_dir = "/home/nisha/Documents/WSI-DIGISCAN/original_tiles/38-21a-300by300/"
for row in range(rows):
    for col in range(cols):
        tile_name = os.path.join(tile_dir, '%d_%d' % (col, row))
        print("Now saving tile with title: ", tile_name)
        temp_tile = tiles.get_tile(16, (col, row))
        temp_tile_RGB = temp_tile.convert('RGB')
        temp_tile_np = np.array(temp_tile_RGB)
        plt.imsave(tile_name + ".png", temp_tile_np)






