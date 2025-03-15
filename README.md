# Fire Detection and Segmentation using Computer Vision

## Overview
This project focuses on fire detection and segmentation using various computer vision techniques and deep learning approaches. It utilizes OpenCV for classical image processing techniques and placeholders for deep learning-based segmentation methods. The dataset consists of fire-related video footage, which is preprocessed and analyzed using multiple segmentation techniques.

## Dataset
- The dataset consists of videos stored in `/kaggle/input/firesense/fire/pos` under https://www.kaggle.com/datasets/chrisfilo/firesense 
- The project processes the first 10 video files from this directory.
- Sample video paths:
  - `/kaggle/input/firesense/fire/pos/posVideo10.869.avi`
  - `/kaggle/input/firesense/fire/neg/negsVideo5.862.avi`
  - `/kaggle/input/firesense/fire/neg/negsVideo11.1073.avi`

## Features Implemented
### 1. Preprocessing
- Extracts the first frame from each video.
- Converts images to RGB format for visualization.

### 2. Image Filtering
- **Median Filtering**: Reduces noise while preserving edges.
- **Anisotropic Diffusion**: Smoothens images while maintaining edges.
- **Non-Local Means Denoising**: Reduces noise while preserving details.

### 3. Fire Segmentation Techniques
####  **HSV Color-Based Segmentation**
- Converts the image to HSV color space.
- Identifies fire-like regions using predefined color thresholds.

####  **Motion-Based Segmentation**
- Computes the frame difference to detect moving fire.
- Uses thresholding and contour detection to highlight motion areas.

####  **Watershed Segmentation**
- Uses morphological operations and distance transformation.
- Labels foreground and background regions before applying the watershed algorithm.

#### 🤖 **Deep Learning-Based Segmentation (Placeholder for U-Net Model)**
- A placeholder for integrating a deep learning model like U-Net for more accurate fire segmentation.

### 4. Connected Component Analysis (CCA)
- Identifies and labels connected regions in thresholded images.
- Filters out small regions to focus on meaningful fire segments.

## Visualization
- Displays the original frame alongside different segmentation results.
- Uses Matplotlib to plot results in a structured manner.

## Dependencies
Ensure the following Python libraries are installed:
```bash
pip install numpy pandas matplotlib opencv-python scikit-image
```

## How to Run
1. Upload the dataset to Kaggle.
2. Run the provided notebook in a Kaggle environment.
3. The segmented outputs will be displayed as images.

## Future Enhancements
- Implement deep learning-based fire detection using U-Net or other CNN architectures.
- Integrate real-time fire detection using OpenCV and webcam feeds.
- Improve motion detection using optical flow techniques.

