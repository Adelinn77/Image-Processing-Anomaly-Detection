# Image Processing Anomaly Detection

A Visual Anomaly Detection System built in Python that relies on statistical modeling (Z-score) and classical image processing techniques to identify defects in images.

## Overview

This project implements an interactive pipeline for anomaly detection. It works by "learning" what a normal image looks like using a set of fault-free ("good") training images. It computes a **Golden Template** consisting of the pixel-wise mean and standard deviation across all normal images.

When evaluating a new test image, the system compares it against the golden template. It calculates a Z-score for each pixel based on how far its intensity deviates from the expected mean, normalized by the expected standard deviation. Pixels with a Z-score above a certain threshold are flagged as anomalous.

## Key Features

- **Interactive Pipeline**: A CLI menu combined with a `tkinter` file dialog to interactively select datasets and test images.
- **Golden Template Generation**: Computes mean and standard deviation matrices from normal samples.
- **Statistical Anomaly Detection**: Uses Z-score maps to detect significant deviations.
- **Custom Image Processing Implementations**: `algorithms.py` includes low-level implementations of several operations:
  - Grayscale conversions.
  - Gaussian kernel generation and 2D convolution (using sliding windows).
  - Morphological operations: Erosion, Dilation, Opening, Closing.
  - Histogram computation and Otsu's thresholding.
  - Heatmap lookup table (LUT) generation for visualizations.

## File Structure

- `pipeline.py`: The main entry point. Orchestrates the interactive CLI, model training (golden template generation), user prompts, and visualizes the results using OpenCV.
- `algorithms.py`: The core math and image processing library for the project. Contains functions for filtering, thresholding, morphology, and anomaly scoring.
- `data_loader.py`: Utility functions for loading the normal training images and test images from the file system.
- `documentation.pdf` / `DOC.pdf`: In-depth project documentation and theoretical background.

## Dependencies

- Python 3.x
- `numpy`
- `opencv-python` (`cv2`)
- `scipy`
- `matplotlib`
- `tkinter` (usually comes pre-installed with standard Python distributions)

## How to Run

1. **Prepare your datasets**: The pipeline expects datasets to be located in a `DATASETS` folder located one directory above the project directory. The expected structure is:
   ```text
   ../DATASETS/
   └── <Dataset_Name>/
       ├── train/
       │   └── good/
       │       ├── img1.png
       │       ├── img2.png
       │       └── ...
       └── test/
           └── ... (test images, the file picker will let you browse)
   ```

2. **Execute the pipeline**:
   ```bash
   python pipeline.py
   ```

3. **Follow the interactive prompts**:
   - Select a dataset from the available options.
   - The system will automatically process the `train/good/` folder to build the golden template.
   - Select option `1` to pick a test image using the pop-up file dialog.
   - The results will be displayed in 4 OpenCV windows: Original Test Image, Mean Template, Heatmap, and the Anomaly Mask. Press `q` or `ESC` to close the windows and choose another image.