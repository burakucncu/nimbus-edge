# ☁️ Nimbus-Edge: Real-Time On-Board Cloud Detection for Satellite Imagery

Nimbus-Edge is a lightweight, high-performance deep learning pipeline designed for **on-board cloud segmentation** in Earth observation satellites. Built with a focus on resource-constrained edge computing environments, the model achieves real-time processing speeds without compromising the structural integrity of cloud boundary detection.

## 🚀 Key Highlights
* **Blazing Fast Inference:** Achieves **~198 FPS** (5.05 ms/patch) on Apple Silicon (MPS), proving its readiness for Edge AI devices (e.g., NVIDIA Jetson, ARM-based flight computers).
* **Smart Dataset Engineering:** Implements dynamic thresholding for raw 16-bit imagery, effectively resolving label saturation and local dynamic thresholding traps.
* **Edge-Ready:** Exports trained PyTorch weights directly to ONNX format for framework-agnostic deployment.

## 🧠 Dataset Optimization & Engineering
Handling raw, atmospherically uncorrected 16-bit satellite imagery required a robust data pipeline to generate accurate ground truth masks automatically:
* **Noise Floor Integration (`max_val < 0.02`):** Successfully eliminated False Positives over dark water bodies and sensor noise.
* **Relaxed Dynamic Thresholding (`max_val * 0.45`):** Captured fuzzy, translucent cloud boundaries (like cirrus clouds) instead of over-segmenting just the bright optical centers.
* **Class Balance:** Processed a full satellite scene into 866 standardized patches (256x256), achieving an optimal **30/70 class balance** (262 cloudy positive samples vs. 604 clear negative samples).

## 📊 Model Architecture & Training
* **Architecture:** `LightweightUNet` (Optimized for minimal parameter count while retaining spatial context).
* **Input:** 4-Channel 16-bit Multi-spectral Patches.
* **Convergence:** Trained from scratch for 20 epochs, converging smoothly to a final Binary Cross-Entropy Loss of **0.135**. 
* **Validation:** Qualitative analysis demonstrated high spatial correlation, accurately predicting both dense cloud centers and diffuse boundaries without triggering on empty ocean patches.

## ⏱️ Benchmarking Results
Benchmarking was performed using a standard 100-iteration inference loop on an Apple M-Series chip via PyTorch MPS backend.

| Metric | Result |
| :--- | :--- |
| **Hardware / Backend** | Apple Silicon / `mps` |
| **Average Inference Time** | 5.05 ms per patch |
| **Processing Speed** | 198.15 FPS |
| **Throughput per second** | ~13 Megapixels/sec |

## 🛠️ Getting Started

### Prerequisites
```bash
pip install torch torchvision rasterio numpy matplotlib onnx
```

### 1. Data Preprocessing
Extracts patches from raw `.tif` scenes and generates dynamic ground truth masks:
```bash
python3 src/preprocess.py
python3 src/check_data.py  # Verifies dataset balance
```

### 2. Model Training
Trains the LightweightUNet architecture on the generated dataset:
```bash
python3 src/train.py
```

### 3. Evaluation & Edge Export
Visualize the qualitative results and export to ONNX for edge deployment:
```bash
python3 src/gallery.py       # Visual truth vs. prediction comparison
python3 src/export_onnx.py   # Generates models/nimbus_model_v1.onnx
python3 src/benchmark.py     # Runs FPS and latency tests
```

---
*Developed by Burak Üçüncü — Geomatics Engineering | Remote Sensing & Edge AI*