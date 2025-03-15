# Fire Detection and Segmentation using Computer Vision

## Overview
This project focuses on fire detection and segmentation using various computer vision techniques and deep learning approaches. It utilizes OpenCV for classical image processing techniques and placeholders for deep learning-based segmentation methods. The dataset consists of fire-related video footage, which is preprocessed and analyzed using multiple segmentation techniques.

## Dataset
- The dataset consists of videos obtained from: https://www.kaggle.com/datasets/chrisfilo/firesense 
- The project processes the first 10 video files from this directory.
- Sample video paths:
  - `/kaggle/input/firesense/fire/pos/posVideo10.869.avi`
  - `/kaggle/input/firesense/fire/neg/negsVideo5.862.avi`
  - `/kaggle/input/firesense/fire/neg/negsVideo11.1073.avi`

## Features Implemented

### 1. Preprocessing
- Extracts the first frame from each video.
- Converts images to RGB format for visualization.
- Resizes frames to a standard resolution for consistency.

#### Code Structure:
```python
cap = cv2.VideoCapture(video_path)
success, frame = cap.read()
if success:
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.resize(frame, (width, height))
```

### 2. Image Filtering
Image filtering techniques are used to reduce noise while preserving essential features.

#### **Median Filtering**
- Reduces salt-and-pepper noise by replacing each pixel with the median value of neighboring pixels.
- Helps in retaining edges.

**Mathematical formulation:**
\[
I_{filtered}(x,y) = \text{median} (I(x-k,y-k), \dots, I(x+k,y+k))
\]

```python
filtered_frame = cv2.medianBlur(frame, ksize=5)
```

#### **Anisotropic Diffusion**
- Reduces noise while preserving edges by using an iterative PDE-based approach.
- Utilizes the Perona-Malik equation:

\[
\frac{\partial I}{\partial t} = \nabla \cdot (c(x,y,t) \nabla I)
\]

where \(c(x,y,t)\) is the diffusion coefficient, which controls the smoothing effect.

```python
filtered_frame = anisotropic_diffusion(frame, niter=10, kappa=50, gamma=0.1)
```

#### **Non-Local Means Denoising**
- Reduces noise while preserving fine details.
- Uses a weighted average of pixels based on similarity.

```python
filtered_frame = cv2.fastNlMeansDenoisingColored(frame, None, 10, 10, 7, 21)
```

### 3. Fire Segmentation Techniques

#### **HSV Color-Based Segmentation**
- Converts the image to HSV color space.
- Identifies fire-like regions using predefined thresholds in the Hue, Saturation, and Value channels.

```python
hsv_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
lower_bound = np.array([0, 50, 50])
upper_bound = np.array([35, 255, 255])
mask = cv2.inRange(hsv_frame, lower_bound, upper_bound)
```

#### **Motion-Based Segmentation**
- Computes the frame difference to detect moving fire.
- Uses thresholding and contour detection to highlight motion areas.

**Mathematical formulation:**
\[
D(x,y) = |I_t(x,y) - I_{t-1}(x,y)|
\]
where \(I_t(x,y)\) is the intensity at time \(t\), and \(I_{t-1}(x,y)\) is the previous frame.

```python
diff = cv2.absdiff(prev_frame, current_frame)
gray_diff = cv2.cvtColor(diff, cv2.COLOR_RGB2GRAY)
_, thresh = cv2.threshold(gray_diff, 30, 255, cv2.THRESH_BINARY)
```

#### **Watershed Segmentation**
- Uses morphological operations and distance transformation to refine segmentation.
- Labels foreground and background regions before applying the watershed algorithm.

**Mathematical formulation (distance transform):**
\[
D(x,y) =
\begin{cases}
    0, & \text{if } I(x,y) = 0 \\
    \min(D(x',y')+1), & \text{otherwise}
\end{cases}
\]

```python
dist_transform = cv2.distanceTransform(binary_mask, cv2.DIST_L2, 5)
_, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
markers = cv2.watershed(frame, markers)
```


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

## 5. FINAL EVALUATION & REPORT

### Quantitative Evaluation of Segmentation Performance

| Segmentation Method      | IoU    | Dice Coefficient | Pixel Accuracy | Precision | Recall  | F1-Score |
|-------------------------|--------|-----------------|---------------|----------|--------|---------|
| Watershed              | 0.4821 | 0.6124          | 0.7843        | 0.6520   | 0.7156 | 0.6823  |
| Region Growing + CCA   | 0.3892 | 0.5416          | 0.6738        | -        | -      | -       |

### Strengths & Weaknesses of Approaches

| Method                  | Strengths                                   | Weaknesses                                      |
|-------------------------|--------------------------------------------|------------------------------------------------|
| Watershed Segmentation | Best boundary detection, high recall       | Computationally expensive, may over-segment small noise regions |
| Region Growing + CCA   | Simple implementation                      | Highly dependent on seed point                 |

