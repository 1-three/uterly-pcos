# Uterly PCOS Detection Model

## Overview

This repository contains the specialized machine learning model used in Uterly's women's health platform for detecting Polycystic Ovary Syndrome (PCOS) from ultrasound images. The model utilizes advanced computer vision techniques and deep learning to analyze ovarian ultrasound scans and provide explainable AI results.

![PCOS Detection Example](https://ibb.co/VW1TkjjD)
![PCOS Detection Example](https://ibb.co/Hf7GJwYH)
## Features

- **4-Channel Image Processing**: Combines RGB channels with edge detection for improved feature recognition
- **EfficientNetB3 Architecture**: Leverages transfer learning from a powerful image classification backbone
- **Explainable AI**: Includes Grad-CAM heatmap visualization to highlight areas contributing to the diagnosis
- **High Accuracy**: Achieves reliable classification between normal and PCOS ultrasound scans
- **Interactive Interface**: Gradio-based demo for easy testing and visualization

## Technical Details

### Input Processing

The model processes ultrasound images using:
- Resizing to 224x224 pixels
- RGB channel extraction
- Sobel filter edge detection as a 4th channel
- Normalization and data augmentation

### Model Architecture

- **Backbone**: EfficientNetB3 with pretrained ImageNet weights
- **Edge Channel Path**: Separate convolutional layers for edge feature extraction
- **Feature Fusion**: Combination of RGB and edge features
- **Classification Head**: Dropout regularization and dense layers for final prediction

### Grad-CAM Visualization

The model includes a Grad-CAM implementation that:
- Identifies regions most influential to the classification decision
- Generates heatmaps overlaid on the original image
- Provides visual explanation of diagnostic factors

## Requirements

- Python 3.8+
- TensorFlow 2.x
- OpenCV
- NumPy
- Matplotlib
- Pandas
- Gradio
- Scikit-learn
- Seaborn

## Installation

```bash
# Clone repository
git clone https://github.com/uterly/pcos-detection.git
cd pcos-detection

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Running the Demo

```bash
# Run with pre-trained model
python pcos_detection.py --demo
```

### Training a New Model

```bash
# Run full training pipeline
python pcos_detection.py --train
```

### Using the Model in Production

```python
# Example code
from pcos_detection import preprocess_single_image, predict_image

# Load image and predict
processed_image = preprocess_single_image("patient_scan.jpg")
results, predicted_class, confidence = predict_image(model, "patient_scan.jpg", label_to_index)

# Generate heatmap
original_img, heatmap_overlay, _, _ = generate_heatmap_overlay("patient_scan.jpg", model, label_to_index)
```

## Model Performance

| Metric | Value |
|--------|-------|
| Accuracy | 94.2% |
| Precision | 92.8% |
| Recall | 95.1% |
| F1 Score | 93.9% |

## Limitations

- The model is trained primarily on a specific dataset and may require fine-tuning for different ultrasound equipment
- Performance can vary based on image quality and proper positioning during ultrasound
- Intended as an assistive tool for healthcare professionals, not as a replacement for medical diagnosis


## License

This model is released under the MIT License.
