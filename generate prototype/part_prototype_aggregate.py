import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoImageProcessor, AutoModel
import os
import numpy as np
from sklearn.cluster import KMeans
from tqdm import tqdm
import logging
import traceback

# ========== Path Configuration ==========
ROOT_DIR = "/data_16T/zjp/LXL/OVPS/PBAPS/Visual Prototype/PascalPart116"
LOG_PATH = os.path.join(ROOT_DIR, "part_prototype_merge_log.txt")

# ========== Category Mapping ==========
category_to_id = {
    'aeroplane body': 0, 'aeroplane stern': 1, 'aeroplane wing': 2, 'aeroplane tail': 3, 'aeroplane engine': 4, 'aeroplane wheel': 5,
    'bicycle wheel': 6, 'bicycle saddle': 7, 'bicycle handlebar': 8, 'bicycle chainwheel': 9, 'bicycle headlight': 10,
    'bird wing': 11, 'bird tail': 12, 'bird head': 13, 'bird eye': 14, 'bird beak': 15, 'bird torso': 16, 'bird neck': 17, 'bird leg': 18, 'bird foot': 19,
    'bottle body': 20, 'bottle cap': 21,
    'bus wheel': 22, 'bus headlight': 23, 'bus front': 24, 'bus side': 25, 'bus back': 26, 'bus roof': 27, 'bus mirror': 28, 'bus license plate': 29, 'bus door': 30, 'bus window': 31,
    'car wheel': 32, 'car headlight': 33, 'car front': 34, 'car side': 35, 'car back': 36, 'car roof': 37, 'car mirror': 38, 'car license plate': 39, 'car door': 40, 'car window': 41,
    'cat tail': 42, 'cat head': 43, 'cat eye': 44, 'cat torso': 45, 'cat neck': 46, 'cat leg': 47, 'cat nose': 48, 'cat paw': 49, 'cat ear': 50,
    'cow tail': 51, 'cow head': 52, 'cow eye': 53, 'cow torso': 54, 'cow neck': 55, 'cow leg': 56, 'cow ear': 57, 'cow muzzle': 58, 'cow horn': 59,
    'dog tail': 60, 'dog head': 61, 'dog eye': 62, 'dog torso': 63, 'dog neck': 64, 'dog leg': 65, 'dog nose': 66, 'dog paw': 67, 'dog ear': 68, 'dog muzzle': 69,
    'horse tail': 70, 'horse head': 71, 'horse eye': 72, 'horse torso': 73, 'horse neck': 74, 'horse leg': 75, 'horse ear': 76, 'horse muzzle': 77, 'horse hoof': 78,
    'motorbike wheel': 79, 'motorbike saddle': 80, 'motorbike handlebar': 81, 'motorbike headlight': 82,
    'person head': 83, 'person eye': 84, 'person torso': 85, 'person neck': 86, 'person leg': 87, 'person foot': 88, 'person nose': 89, 'person ear': 90, 'person eyebrow': 91, 'person mouth': 92, 'person hair': 93, 'person lower arm': 94, 'person upper arm': 95, 'person hand': 96,
    'pottedplant pot': 97, 'pottedplant plant': 98,
    'sheep tail': 99, 'sheep head': 100, 'sheep eye': 101, 'sheep torso': 102, 'sheep neck': 103, 'sheep leg': 104, 'sheep ear': 105, 'sheep muzzle': 106, 'sheep horn': 107,
    'train headlight': 108, 'train head': 109, 'train front': 110, 'train side': 111, 'train back': 112, 'train roof': 113, 'train coach': 114,
    'tvmonitor screen': 115
}

id_to_category = {v: k for k, v in category_to_id.items()}

# ========== Hierarchy Definitions ==========
root_object_mapping = {
    "root": ["aeroplane", "bicycle", "bird", "bottle", "bus", "car", "cat", "cow", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "train", "tvmonitor"]
}

object_part1_mapping = {
    "aeroplane": ["aeroplane body", "aeroplane stern", "aeroplane wing", "aeroplane tail", "aeroplane engine", "aeroplane wheel"],
    "bicycle": ["bicycle wheel", "bicycle saddle", "bicycle handlebar", "bicycle chainwheel"],
    "bird": ["bird tail", "bird head", "bird torso", "bird neck", "bird leg", "bird foot"],
    "bottle": ["bottle body", "bottle cap"],
    "bus": ["bus wheel", "bus front", "bus side", "bus back", "bus roof", "bus mirror", "bus license plate"],
    "car": ["car wheel", "car front", "car side", "car back", "car roof", "car license plate"],
    "cat": ["cat tail", "cat head", "cat torso", "cat neck", "cat leg", "cat paw"],
    "cow": ["cow tail", "cow head", "cow torso", "cow neck", "cow leg"],
    "dog": ["dog tail", "dog head", "dog torso", "dog neck", "dog leg", "dog paw"],
    "horse": ["horse tail", "horse head", "horse torso", "horse neck", "horse leg", "horse hoof"],
    "motorbike": ["motorbike wheel", "motorbike handlebar", "motorbike headlight"],
    "person": ["person head", "person torso", "person neck", "person leg", "person foot", "person lower arm", "person upper arm", "person hand"],
    "pottedplant": ["pottedplant pot", "pottedplant plant"],
    "sheep": ["sheep tail", "sheep head", "sheep torso", "sheep neck", "sheep leg"],
    "train": ["train head", "train back", "train coach"],
    "tvmonitor": ["tvmonitor screen"]
}

part1_part2_mapping = {
    "bird head": ["bird eye", "bird beak"],
    "bus front": ["bus headlight", "bus window"],
    "bus side": ["bus door", "bus window"],
    "car side": ["car mirror", "car door", "car window"],
    "cat head": ["cat eye", "cat nose", "cat ear"],
    "cow head": ["cow eye", "cow ear", "cow muzzle", "cow horn"],
    "dog head": ["dog eye", "dog ear", "dog muzzle"],
    "horse head": ["horse eye", "horse ear", "horse muzzle"],
    "person head": ["person eye", "person nose", "person ear", "person eyebrow", "person mouth", "person hair"],
    "sheep head": ["sheep eye", "sheep ear", "sheep muzzle", "sheep horn"],
    "train head": ["train front", "train side", "train roof"]
}

part2_part3_mapping = {
    "dog muzzle": ["dog nose"],
    "train coach": ["train side", "train roof"]
}

part3_part4_mapping = {
    "train front": ["train headlight"]
}

# ========== Utility Functions ==========
def load_prototypes(folder_path):
    """Load all prototypes (.npy files) from a given folder."""
    if not os.path.exists(folder_path):
        return []
    files = sorted([f for f in os.listdir(folder_path) if f.endswith(".npy")])
    protos = []
    for f in files:
        try:
            arr = np.load(os.path.join(folder_path, f)).reshape(-1)
            protos.append(arr)
        except Exception as e:
            logging.warning(f"⚠️ Failed to load {f}: {e}")
    return protos


def save_prototypes(folder_path, prototypes, start_idx):
    """Save prototypes to a directory, sequentially numbered from start_idx."""
    os.makedirs(folder_path, exist_ok=True)
    for i, p in enumerate(prototypes):
        idx = start_idx + i
        out_path = os.path.join(folder_path, f"{idx}.npy")
        np.save(out_path, p)


def cluster_and_merge(parent, children, root_dir, ratio=1/3):
    """Cluster all child prototypes and merge them into the parent category."""
    parent_path = os.path.join(root_dir, parent)
    child_paths = [os.path.join(root_dir, c) for c in children]

    # Collect child prototypes
    all_child_protos = []
    for c, p in zip(children, child_paths):
        protos = load_prototypes(p)
        all_child_protos.extend(protos)
        logging.info(f"Child {c}: loaded {len(protos)} prototypes")

    if len(all_child_protos) == 0:
        logging.warning(f"❌ {parent}: no child prototypes found, skipping")
        return

    all_child_protos = np.stack(all_child_protos)
    num_clusters = max(1, len(all_child_protos) // 3)
    logging.info(f"→ {parent}: clustering {len(all_child_protos)} → {num_clusters}")

    # KMeans clustering
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10).fit(all_child_protos)
    cluster_centers = kmeans.cluster_centers_

    # Get the highest existing prototype index
    os.makedirs(parent_path, exist_ok=True)
    existing = [f for f in os.listdir(parent_path) if f.endswith('.npy')]
    max_idx = 0
    for fname in existing:
        name = os.path.splitext(fname)[0]
        if name.isdigit():
            max_idx = max(max_idx, int(name))

    # Save new prototypes
    start_idx = max_idx + 1
    save_prototypes(parent_path, cluster_centers, start_idx)
    logging.info(f"✅ {parent}: added {len(cluster_centers)} prototypes, indices {start_idx}-{start_idx + len(cluster_centers) - 1}")


# ========== Main Process ==========
def merge_hierarchy(root_dir, mappings):
    """Iteratively merge prototypes from lower-level parts to higher-level categories."""
    for i, mapping in enumerate(mappings[::-1], start=1):  # Start from the lowest layer
        logging.info(f"\n=== Merging Level {i} ===")
        for parent, children in tqdm(mapping.items()):
            cluster_and_merge(parent, children, root_dir)


if __name__ == "__main__":
    logging.basicConfig(filename=LOG_PATH, level=logging.INFO, format="%(asctime)s - %(message)s")
    logging.info("====== Hierarchical Prototype Aggregation Started ======")

    hierarchy_mappings = [part1_part2_mapping, part2_part3_mapping, part3_part4_mapping]

    # Execute from lowest to highest level
    merge_hierarchy(ROOT_DIR, hierarchy_mappings)

    logging.info("====== Aggregation Completed ======")
    print("✅ Hierarchical prototype aggregation completed. Log saved to:", LOG_PATH)

