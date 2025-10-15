# PBAPS (NeurIPS 2025)

**PBAPS: Open-Vocabulary Part Segmentation via Progressive and Boundary-Aware Strategy**

---

## 🌍 Overview

PBAPS proposes a **Progressive and Boundary-Aware Part Segmentation** framework for open-vocabulary settings. It integrates hierarchical decomposition and adaptive boundary refinement, leveraging DINOv2 features and visual prototypes to achieve precise and semantically aligned part segmentation.

This repository provides:
- Full inference pipeline for **open-vocabulary part segmentation (OVPS)**
- Scripts for **visual prototype generation**
- Generated **part and object prototypes**


---

## 🧩 Conda Environment Setup

```bash
conda create -n PBAPS python=3.8
conda activate PBAPS
```

### Install dependencies

```bash
install pydensecrf https://github.com/lucasb-eyer/pydensecrf
pip install git+https://github.com/lucasb-eyer/pydensecrf.git

# Install other required packages
pip install -r requirements.txt

If there is an error: bug for cannot import name 'autocast' from 'torch', please refer to the website:
https://github.com/pesser/stable-diffusion/issues/14
```

---

## 🗂️ Dataset Preparation

### **PascalPart116**

```bash
gdown https://drive.google.com/uc?id=1QF0BglrcC0teKqx15vP8qJNakGgCWaEH
tar -xzf PascalPart116.tar.gz
find datasets/PascalPart116/images/val/ -name '._*' -delete
find datasets/PascalPart116/ -name '._*' -delete
```

### **ADE20KPart234**

```bash
gdown https://drive.google.com/uc?id=1EBVPW_tqzBOQ_DC6yLcouyxR7WrctRKi
tar -xzf ADE20KPart234.tar.gz
```

### **PartImageNet**

1. Download the file **`LOC_synset_mapping.txt`** from [this link](https://image-net.org).
2. Download **PartImageNet_Seg** from the official [PartImageNet](https://github.com/TACJu/PartImageNet) and extract it.

> For more details, please refer to the official [OV-PARTS](https://github.com/InternRobotics/OV_PARTS).

---

## 🎨 Visual Prototype Generation

We use **stable diffusion**, **SAM**, and **DINOv2** to generate visual prototypes for objects and parts.

### Step 1. Generate image and attention maps

```bash
python generate_prototype/1_diffus_generate_img_attention.py
```

### Step 2. Generate masks and extract features with SAM + DINOv2

```bash
python generate_prototype/2_SAM_DINOv2_mask_feature.py
```

### Step 3. Aggregate part features into visual prototypes

```bash
python generate_prototype/3_generate_visualprototype.py

# You can also use these two scripts to get the object prototype and the optimized part prototype
# python generate prototype/object_prototype.py
# python generate prototype/part_prototype_aggregate.py
```


### Pre-generated Prototypes

We provide generated part prototypes for direct testing: [Download from Google Drive](https://drive.google.com/file/d/1bXAaymlCkBiyztoIZl2LdO0f6B9CIoH3/view?usp=sharing)

---

## 🚀 Inference (Testing)

You can directly use our generated visual prototypes to perform inference.

### Step 1. Update configuration paths

Edit the paths in **`run_PBAPS.py`**:
```python
IMG_DIR = "/path/to/datasets/PascalPart116/images/val"
GT_DIR  = "/path/to/datasets/PascalPart116/annotations_detectron2_part/val"
PROTO_ROOT = "/path/to/visual_prototypes/PascalPart116"
SAVE_DIR = "./results/PascalPart116"
```

### Step 2. Run PBAPS inference

```bash
python model/run_PBAPS.py
```

This script will:
- Perform hierarchical decomposition (object → parts → sub-parts)
- Apply **Boundary-Aware Refinement (BAR)** at each layer
- Generate: Part segmentation visualizations (`.jpg`), Label maps (`.png`), Evaluation metrics (`metrics.csv`)




---

## 🙏 Acknowledgement

We would like to express our sincere gratitude to the open-source projects and their contributors, including: [OV-PARTS](https://github.com/InternRobotics/OV_PARTS), [OVDiff](https://github.com/karazijal/ovdiff), [DiffuMask](https://github.com/weijiawu/DiffuMask), [DINOv2](https://github.com/facebookresearch/dinov2), and [SAM](https://github.com/facebookresearch/segment-anything).


