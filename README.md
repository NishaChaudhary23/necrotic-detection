# Necrotic Tissue Detection

This repository contains Python scripts for processing Whole Slide Images (WSIs) to detect and classify necrotic tissue. The workflow includes tissue mask generation, necrotic region segmentation, patch extraction, and CNN model training for necrotic tissue classification.

---

## **Scripts Overview**

1. **`complete_tissue_mask.py`**  
   - Generates binary tissue masks from Whole Slide Images (WSIs).

2. **`necrotic_mask.py`**  
   - Segments necrotic regions from the tissue masks.

3. **`wsi_tiling.py`**  
   - Extracts 256x256 patches from the necrotic tissue regions.

4. **`cnn_training.py`**  
   - Trains a convolutional neural network (CNN) to classify patches as necrotic or non-necrotic.

---

## **Setup Instructions**

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/NishaChaudhary23/necrotic-detection.git
   cd necrotic-detection
