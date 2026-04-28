# 🌍 Nimbus-Edge: Satellite Imagery Cloud Detection

**Nimbus-Edge** is a Deep Learning project built on **Transfer Learning** and **Continual Learning** architectures, designed to detect clouds in satellite imagery (Optical & NIR) with high precision.

It is specifically engineered to overcome the physical inconsistencies and **Domain Shift** challenges that arise when working with data from different geographical regions and various sensor bit-depths (e.g., uint16 vs. int16).

## 🚀 Purpose and Architecture

Unlike traditional thresholding or basic pixel-based algorithms, this project leverages a pre-trained **ResNet34 (ImageNet)** model. The model has been meticulously **Fine-Tuned** based on the true physical reflectance values of satellite imagery.

### 🧠 Core Features
* **Physical Data Normalization:** Instead of using the classic 8-bit (0-255) RGB scale, the imagery is normalized using a 1/10,000 reflectance scaling rule tailored for 16-bit remote sensing physics. This successfully filters out ocean glare and salt-and-pepper noise.
* **Domain Shift Resolution:** The model avoids Overfitting to a single region. By blending bright ocean scenes (e.g., Florida coast - uint16) with darker urban/sea textures (e.g., Mediterranean coast - int16) into a **Continual Learning** pool, it achieves true global Generalization.
* **Class Imbalance Control:** The data preprocessing pipeline automatically drops 95% of empty patches (pure ocean/land) to provide the model with a balanced and effective training dataset.
* **On-the-fly CRS Resolver:** Projection mismatches between vector masks (.gpkg/.shp) and raster satellite images (.tif) are automatically detected and reprojected on-the-fly during the rasterization phase.

## 🛠️ Tech Stack
* **Deep Learning:** PyTorch, Segmentation Models PyTorch (SMP)
* **Geospatial & Remote Sensing:** Rasterio, GeoPandas, GDAL
* **Data Processing:** NumPy, OpenCV

---

## 📂 Directory Structure

```text
nimbus-edge/
├── data/
│   ├── raw/               # Original .tif satellite images and .gpkg masks
│   ├── images/            # Extracted Golden Patches (256x256) for training
│   └── masks/             # Corresponding Binary Masks for the patches
├── models/                # Trained .pt (PyTorch) model weights
├── output/                # Final generated cloud masks from inference
└── src/                   # Source code
    ├── rasterize_mask.py  # Converts vector polygons to raster masks (CRS-aware)
    ├── preprocess_golden.py # Slices images into patches (Supports Continual Learning)
    ├── debug_pipeline.py  # Data pool and physical scale diagnostic tool (Sanity Check)
    ├── train.py           # ResNet34 Fine-Tuning pipeline
    └── analyze_large.py   # End-to-end inference on massive, unseen satellite scenes
```

---

## ⚙️ Pipeline and Usage

The project features a fully automated pipeline, from data preparation to full-scene inference.

### 1. Data Preparation (Rasterization)
Convert hand-drawn polygons (vector masks created in QGIS/ArcGIS) into raster (0-1) masks that perfectly match the spatial resolution and CRS of the reference satellite image:

```bash
python3 src/rasterize_mask.py
```

### 2. Golden Dataset Extraction (Patching)
Slice massive satellite images and their corresponding masks into 256x256 patches suitable for the model's input size. *(Includes a safe-naming strategy to prevent overwriting existing datasets during Continual Learning)*:

```bash
python3 src/preprocess_golden.py
```

### 3. Diagnostics (Sanity Check)
Before training, verify the data type (int16/uint16), maximum reflectance values, and the total number of cloud pixels in the training pool:

```bash
python3 src/debug_pipeline.py
```

### 4. Training (Fine-Tuning)
Train the model using local hardware acceleration (MPS/CUDA):

```bash
python3 src/train.py
```

### 5. Large Scene Inference
Deploy the trained model to analyze a massive, previously unseen `.tif` scene to calculate the true cloud coverage percentage. Results are saved in the `output/` folder as a `.tif` (for GIS software) and a `.jpg` Binary Mask (for quick previews):

```bash
python3 src/analyze_large.py
```

---
*Developed by Burak Üçüncü — Geomatics Engineering | Remote Sensing & Edge AI*