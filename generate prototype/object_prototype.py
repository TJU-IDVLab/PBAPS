import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from transformers import AutoImageProcessor, AutoModel
import os
import numpy as np
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_softmax
from sklearn.cluster import KMeans
from tqdm import tqdm
import logging
import traceback

# ================== Logging Configuration ==================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("object_prototype_generation.log"), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# ================== Category Mapping ==================
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

# Build object-to-parts mapping
object_to_parts = {}
for part in category_to_id.keys():
    obj = part.split()[0]
    object_to_parts.setdefault(obj, []).append(part)


# ================== Feature Collection ==================
def collect_part_features(part_dirs, prototype_root):
    """Collect all feature vectors (.npy) for the given parts."""
    all_features = []
    for part in part_dirs:
        part_path = os.path.join(prototype_root, part)
        if not os.path.exists(part_path):
            logger.warning(f"Part folder not found: {part_path}, skipping.")
            continue

        for filename in os.listdir(part_path):
            if filename.endswith('.npy'):
                try:
                    if not filename.split('.')[0].isdigit():
                        logger.warning(f"Invalid file name format: {filename}, skipping.")
                        continue
                    feature = np.load(os.path.join(part_path, filename))
                    all_features.append(feature)
                except Exception as e:
                    logger.error(f"Failed to read {filename}: {e}")
                    continue

    if not all_features:
        return None
    return np.array(all_features)


# ================== Utility ==================
def get_max_prototype_index(save_dir):
    """Get the highest prototype index currently in the directory."""
    if not os.path.exists(save_dir):
        return 0
    max_idx = 0
    for filename in os.listdir(save_dir):
        if filename.endswith('.npy'):
            try:
                idx = int(filename.split('.')[0])
                max_idx = max(max_idx, idx)
            except ValueError:
                continue
    return max_idx


# ================== Prototype Generation ==================
def generate_object_prototypes(prototype_root, overwrite=False):
    """Generate object-level visual prototypes by clustering all part-level features."""
    progress_file = os.path.join(prototype_root, "object_prototype_progress.txt")
    completed_objects = set()

    if os.path.exists(progress_file) and not overwrite:
        with open(progress_file, 'r', encoding='utf-8') as f:
            completed_objects = set(line.strip() for line in f if line.strip())
        logger.info(f"{len(completed_objects)} objects already processed.")

    for obj_name, parts in tqdm(object_to_parts.items(), desc="Generating object-level prototypes"):
        if obj_name in completed_objects and not overwrite:
            logger.info(f"Object {obj_name} already processed, skipping.")
            continue

        logger.info(f"Processing object: {obj_name} | Parts: {parts}")
        try:
            # Step 1. Collect all part features
            all_features = collect_part_features(parts, prototype_root)
            if all_features is None or len(all_features) == 0:
                logger.warning(f"No valid features found for {obj_name}, skipping.")
                continue

            total_features = len(all_features)
            logger.info(f"Collected {total_features} features for {obj_name}")

            # Step 2. Determine cluster count
            n_clusters = max(1, total_features // 2)
            logger.info(f"Clustering into {n_clusters} prototypes.")

            # Step 3. Run KMeans
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            kmeans.fit(all_features)
            cluster_centers = kmeans.cluster_centers_

            # Step 4. Save prototypes
            save_dir = os.path.join(prototype_root, obj_name)
            os.makedirs(save_dir, exist_ok=True)
            start_idx = get_max_prototype_index(save_dir) + 1

            for i, center in enumerate(cluster_centers):
                save_path = os.path.join(save_dir, f"{start_idx + i}.npy")
                np.save(save_path, center)

            logger.info(f"Generated {n_clusters} prototypes for {obj_name}, saved to {save_dir}")

            # Step 5. Record progress
            with open(progress_file, 'a', encoding='utf-8') as f:
                f.write(f"{obj_name}\n")

        except Exception as e:
            logger.error(f"Error processing {obj_name}: {e}")
            logger.error(traceback.format_exc())
            continue

    logger.info("All object prototypes successfully generated.")


# ================== Entry Point ==================
if __name__ == "__main__":
    prototype_root = "/data_16T/zjp/LXL/OVPS/PBAPS/Visual Prototype/PascalPart116"

    generate_object_prototypes(
        prototype_root=prototype_root,
        overwrite=False  # Set True to regenerate existing prototypes
    )
