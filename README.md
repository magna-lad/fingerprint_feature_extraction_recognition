# Hybrid Fingerprint Recognition System

This project implements a multi-modal biometric authentication system that fuses **Geometric Graph Analysis** (using Minutiae topology) with **Deep Texture Analysis** (using CNNs).

## 📂 Project Structure

The project is divided into processing, feature extraction, and model training modules.
### Project Architecture
![Demo Image](https://github.com/magna-lad/fingerprint_feature_extraction_recognition/blob/main/utils/architecture.png)

### 1. Core Execution
*   **`main.py`**
    *   **Purpose:** The entry point for data preprocessing.
    *   **Function:** Reads raw images, applies Gabor filtering, extracts skeletons/minutiae, and caches the results to `processed_data.pkl`.
*   **`run_final_pipeline.py`**
    *   **Purpose:** The main training and evaluation script.
    *   **Function:** Loads the cached data, creates graph pairs, trains the XGBoost model, trains the CNN ensemble, and calculates the final fusion score.

### 2. Preprocessing & Minutiae Extraction
*   **`reader.py`**: Iterates through the specific directory structure (`User/Hand/Finger`) to load images.
*   **`minutia_loader.py`**: Handles ROI segmentation, normalization, and masking.
*   **`skeleton_maker.py`**:
    *   Computes ridge orientation fields.
    *   Estimates ridge frequency.
    *   Applies Gabor filters.
    *   Performs skeletonization (thinning).
*   **`minutiaeExtractor.py`**: Uses the Crossing Number method on the skeleton to find ridge endings and bifurcations.
*   **`minutiae_filter.py`**: filtering logic to remove false minutiae near image borders or in high-noise areas.

### 3. Geometric Branch (Graph + XGBoost)
*   **`graph_minutiae.py`**:
    *   Converts minutiae points into a graph using Delaunay Triangulation.
    *   Handles dataset splitting (Train/Val/Test) disjoint by user.
*   **`xgboost_feature_extractor.py`**:
    *   Extracts handcrafted features from graph pairs (Edge length histograms, angle histograms, spatial density).

### 4. Visual Branch (Deep Learning)
*   **`cnn_model.py`**:
    *   **DeeperCNN:** A ResNet-based Siamese network for embedding extraction.
    *   **FingerprintTextureDataset:** Handles core-alignment and data augmentation.

### 5. Utilities
*   **`load_save.py`**: Manages the loading and saving of the `processed_data.pkl` file in the `biometric_cache` directory.

---

## ⚙️ Installation

### Prerequisites
*   Python 3.8+
*   CUDA-capable GPU (recommended for CNN training)

### Install Dependencies
```bash
pip install numpy opencv-python matplotlib tqdm torch torchvision torchaudio
pip install xgboost scikit-learn scipy scikit-image torch-geometric pandas
```

# 🧠 Hybrid Fingerprint Matching Pipeline

This project implements a **hybrid fingerprint matching system** that combines **geometric (minutiae-graph + XGBoost)** and **visual (CNN ensemble)** similarity scores using a **dynamic weighted score-level fusion** strategy, commonly used in multibiometric systems for improved robustness and accuracy.

---

## 🚀 Usage

### 1. Data Setup

Organize your dataset in the following structure:

```
Dataset_Root/
├── 001/
│ ├── L/ # Left hand images
│ └── R/ # Right hand images
├── 002/
│ ├── L/
│ └── R/
...
```


Each subject directory (e.g., `001`, `002`) contains `L/` and `R/` folders for left and right hand fingerprint images respectively, which is a common organization pattern in fingerprint datasets .

---

### 2. Preprocessing (Cache Generation)

This step is **computationally expensive**, so it is designed to be run once to generate a cache file for faster repeated training.

1. Open `main.py`.
2. Set the dataset root path:
```
data_dir = r"/path/to/your/dataset"
```

3. Run:
```
python main.py
```


**Output:**

- A file named `processed_data.pkl` will be created inside a `biometric_cache/` folder, acting as a preprocessed dataset cache similar to typical cached feature representations in ML pipelines.

---

### 3. Training & Inference

1. Open `run_final_pipeline.py`.
2. Point `DATA_FILE` to the cache file:

```
DATA_FILE = 'biometric_cache/processed_data.pkl'
```

3. Run the full pipeline:

```
python run_final_pipeline.py
```


This script trains the models and evaluates them on the validation/test split using ROC/AUC metrics, which are standard for biometric verification tasks .

---

## 📊 Pipeline Logic

### Geometric Score (\(P_{xgb}\))

- Fingerprint **minutiae** are extracted and converted into graph-based representations, a classical approach in minutiae-based fingerprint recognition.
- An **XGBoost** classifier estimates geometric similarity between a pair of minutiae graphs, leveraging gradient-boosted decision trees for discriminative scoring.

### Visual Score (\(P_{cnn}\))

- A **\(96 \times 96\)** patch centered at the fingerprint **core** is cropped as the visual region of interest, consistent with ROI-based CNN fingerprint processing.
- An **ensemble of 3 CNNs** modules predicts similarity.

### Fusion Score

The final hybrid score is computed via weighted score-level fusion:

\[
\text{Score} = \alpha \cdot P_{cnn} + (1 - \alpha) \cdot P_{xgb}
\]

- The fusion strategy uses a **weighted sum**, a widely used rule in score-level fusion for multibiometric systems.
- The weight \(\alpha\) is **dynamically optimized** on the validation set to maximize performance (e.g., AUC), aligning with common practices in adaptive fusion schemes.

---

## 📉 Outputs
![Demo Image](https://github.com/magna-lad/fingerprint_feature_extraction_recognition/blob/main/utils/Results.png)

Running `run_final_pipeline.py` produces:

- **Console:**
- AUC scores for:
 - XGBoost-only (geometric)
 - CNN-only (visual)
 - Hybrid (fused) model, reflecting the improvement from fusion .
- **Plot:**
- `kaggle_final_results.png` containing ROC curves for all three models.
- **Models:**
- Best CNN weights saved as:
 - `cnn_v0.pth`
 - `cnn_v1.pth`
 - `cnn_v2.pth`
- These files can be reused for inference or fine-tuning, following standard PyTorch model serialization practice.

---

## 📁 Folder & File Overview

| Path / File                      | Description                                                        |
|----------------------------------|--------------------------------------------------------------------|
| `Dataset_Root/`                  | Root folder containing subject-wise fingerprint images.   |
| `Dataset_Root/<ID>/L/`, `R/`     | Left/right hand fingerprint images for subject `<ID>`.    |
| `biometric_cache/processed_data.pkl` | Cached preprocessed dataset for fast training runs |
| `kaggle_final_results.png`       | ROC curve visualization of XGBoost, CNN, and Hybrid models.       |
| `cnn_v0.pth`, `cnn_v1.pth`, ...  | Saved CNN model checkpoints for later reuse.                      |

---

## 🛠 Dependencies

Install required dependencies :

```
pip install torch torchvision xgboost numpy pandas scikit-learn matplotlib
```


Depending on your environment, additional packages for data loading, image processing, or augmentation may be needed (e.g., `opencv-python`, `tqdm`).
