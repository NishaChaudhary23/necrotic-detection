# Necrotic Tissue Detection

This repository contains Python scripts for processing Whole Slide Images (WSIs) to detect and classify necrotic tissue. The workflow includes tissue mask generation, necrotic region segmentation, patch extraction, and CNN model training for necrotic tissue classification.

---

## **Scripts Overview**

1. **`complete-tissue-mask.py`**  
   - Generates binary tissue masks from Whole Slide Images (WSIs).

2. **`necrotic-mask.py`**  
   - Segments necrotic regions from the tissue masks.

3. **`wsi-tiling.py`**  
   - Extracts 256x256 patches from the necrotic tissue regions.

4. **`model-training.py`**  
   - Trains a convolutional neural network (DenseNet161) to classify patches as necrotic or non-necrotic.

---

## **Setup Instructions**

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/NishaChaudhary23/necrotic-detection.git
   cd necrotic-detection
