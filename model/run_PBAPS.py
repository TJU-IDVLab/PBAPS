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
import os
import torch
import numpy as np
from sklearn.cluster import KMeans
from tqdm import tqdm
import logging
from multiprocessing import Pool, cpu_count
import traceback
import random
import csv
from scipy.ndimage import binary_dilation
from evaluation import compute_miou_and_biou


IMG_DIR = "/data_16T/zjp/LXL/OVPS/PBAPS/PascalPart116/images/val"
GT_DIR  = "/data_16T/zjp/LXL/OVPS/PBAPS/PascalPart116/annotations_detectron2_part/val"
PROTO_ROOT = "/data_16T/zjp/LXL/OVPS/PBAPS/Visual Prototype/PascalPart116"
SAVE_DIR = "/data_16T/zjp/LXL/OVPS/PBAPS/test"
os.makedirs(SAVE_DIR, exist_ok=True)
PRED_LABEL_DIR = os.path.join(SAVE_DIR, "pred_label_png")
os.makedirs(PRED_LABEL_DIR, exist_ok=True)

THRESHOLD_OBJECT = 0.8   # object threshold (for background)
THRESHOLD_CHILD = [0.9, 0.9, 0.9]  # for deeper layers: require child best-sim >= this to replace parent label
DEVICE = "cuda:9"  
SIM_BATCH = 8192
ALPHA = 0.45

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
    'tvmonitor screen': 115}

id_to_category = {v: k for k, v in category_to_id.items()}

root_object_mapping = {
    "root":["aeroplane","bicycle","bird","bottle","bus","car","cat","cow","dog","horse","motorbike","person","pottedplant","sheep","train","tvmonitor"]
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

part1_adjacency={
    "aeroplane body": ["aeroplane stern", "aeroplane wing"],
    "aeroplane wing": ["aeroplane engine"],
    "bicycle wheel": ["bicycle handlebar"],
    "bird torso": ["bird tail", "bird neck","bird leg"],
    "bird head": ["bird neck"],
    "bird leg": ["bird foot"],
    "bottle body": ["bottle cap"],
    "bus side": ["bus wheel", "bus front"],
    "car side": ["car wheel", "car front","car roof"],
    "car front": ["car roof"],
    "cat head": ["cat neck"],
    "cat torso": ["cat tail", "cat neck","cat leg"],
    "cat leg": ["cat paw"],
    "cow torso": ["cow tail", "cow head", "cow neck", "cow leg"],
    "cow head": ["cow neck"],
    "dog torso": ["dog tail", "dog neck","dog leg"],
    "dog head": ["dog neck"],
    "dog leg": ["dog paw"],
    "horse torso": ["horse tail", "horse neck","horse leg"],
    "horse neck": ["horse head"],
    "horse leg": ["horse hoof"],
    "person neck": ["person head", "person torso"],
    "person leg": ["person torso", "person foot"],
    "person lower arm": ["person upper arm", "person hand"],
    "person torso": ["person upper arm"],
    "pottedplant pot": ["pottedplant plant"],
    "sheep torso": ["sheep tail", "sheep neck","sheep leg"],
    "sheep head": ["sheep neck"],
    "train head": ["train coach"]
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

part2_adjacency={
    "car mirror": ["car door", "car window"],
    "car door": ["car window"],
    "train front": ["train side", "train roof"],
    "train roof": ["train side"]
}

part2_part3_mapping = {
    "dog muzzle": ["dog nose"],
    "train coach": ["train side", "train roof"]
}

part3_adjacency={
    "train roof": ["train side"]
}

part3_part4_mapping = {
    "train front": ["train headlight"]
}

# full hierarchy list in order (from higher to lower)
HIERARCHY = [object_part1_mapping, part1_part2_mapping, part2_part3_mapping, part3_part4_mapping]



def extract_features_with_sliding_window(image, processor, model, device="cuda:9", window_size=224, stride=112):
    w, h = image.size
    if min(h, w) < window_size:
        new_size = (window_size, window_size)  # PIL  (W, H)
        image = image.resize(new_size, Image.Resampling.BICUBIC)
        window = image
        inputs = processor(images=window, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            patch_features = outputs.last_hidden_state[:, 1:, :].permute(0, 2, 1).view(1, 768, 16, 16)  
            per_pixel_feature = F.interpolate(
                patch_features,   # [1, 768, H, W]
                size=(h, w),
                mode='bilinear',
                align_corners=False
            ).permute(0, 2, 3, 1)                        # [1, orig_H, orig_W, 768]
        return per_pixel_feature
    
    
    image_np = np.array(image)  # [H, W, 3]
    H, W, _ = image_np.shape

    per_pixel_feature = torch.zeros((1, H, W, 768), device=device)  

    counts = torch.zeros_like(per_pixel_feature)
    
    def process_window(x_start, y_start, x_end, y_end):
        window = image.crop((x_start, y_start, x_end, y_end))  
        inputs = processor(images=window, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            patch_features = outputs.last_hidden_state[:, 1:, :].permute(0, 2, 1).view(1, 768, 16, 16)  
            window_feature = F.interpolate(patch_features, size=(y_end - y_start, x_end - x_start), mode="bilinear", align_corners=False)
            window_feature = window_feature.permute(0, 2, 3, 1)  # [1, H', W', 768]
        per_pixel_feature[:, y_start:y_end, x_start:x_end] += window_feature
        counts[:, y_start:y_end, x_start:x_end] += 1
    
  
    for y in range(0, H - window_size + 1, stride):
        for x in range(0, W - window_size + 1, stride):
            process_window(x, y, x + window_size, y + window_size)
    

    if (H - y) % stride != 0 and (H - y) > 0:
        y_start = H - window_size
        for x in range(0, W - window_size + 1, stride):
            x_end = min(x + window_size, W)
            process_window(x, y_start, x_end, H)
   
    if (W - x) % stride != 0 and (W - x) > 0:
        x_start = W - window_size
        for y in range(0, H - window_size + 1, stride):
            y_end = min(y + window_size, H)
            process_window(x_start, y, W, y_end)
    
    if (H - y) % stride != 0 and (H - y) > 0 and (W - x) % stride != 0 and (W - x) > 0:
        process_window(W - window_size, H - window_size, W, H)

    counts[counts == 0] = 1  
    per_pixel_feature /= counts

    return per_pixel_feature


def load_prototypes_for_category(proto_root, category):
    folder = os.path.join(proto_root, category)
    if not os.path.isdir(folder):
        return None
    files = sorted([f for f in os.listdir(folder) if f.endswith('.npy')])
    vecs = []
    for f in files:
        try:
            v = np.load(os.path.join(folder, f))
            v = np.asarray(v).reshape(-1)
            vecs.append(v.astype(np.float32))
        except Exception:
            continue
    if len(vecs) == 0:
        return None
    mat = np.stack(vecs, axis=0)
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    mat = mat / norms
    return mat

def batched_cosine_similarity(pixels, proto_mat, batch_size=SIM_BATCH):
    P, D = pixels.shape
    M, _ = proto_mat.shape
    sims = np.zeros((P, M), dtype=np.float32)
    for i in range(0, P, batch_size):
        j = min(P, i + batch_size)
        sims[i:j] = np.matmul(pixels[i:j], proto_mat.T)
    return sims

def random_color_map(classes):
    colors = {}
    rnd = random.Random(42)
    for c in classes:
        colors[c] = tuple([int(x) for x in rnd.sample(range(50, 230), 3)])  # avoid extremes
    colors["background"] = (0,0,0)
    return colors


# ---------------- hierarchical decomposition helpers ----------------
def decompose_children_for_parent(pred_map_ids, parent_name, child_names, pixels_flat, W, proto_cache, threshold_child=THRESHOLD_CHILD):
    """
    - pred_map_ids: numpy array H x W where parent pixels currently have id category_to_id[parent]
    - parent_name: string (e.g., "bottle")
    - child_names: list of strings (e.g., ["bottle body", "bottle cap"])
    - pixels_flat: flattened features shape [H*W, D], normalized
    - proto_cache: dict for caching loaded prototypes {category_name: np.array or None}
    return: updated pred_map_ids (in-place modification acceptable)
    """
    H, W = pred_map_ids.shape
    parent_id = category_to_id.get(parent_name, None)
    if parent_id is None:
        return pred_map_ids  # parent not in category map

    # find pixels belonging to this parent
    ys, xs = np.nonzero(pred_map_ids == parent_id)
    if len(ys) == 0:
        return pred_map_ids

    idx_flat = ys * W + xs
    sub_pixels = pixels_flat[idx_flat]  # [N, D]

    # prepare child prototypes (cache)
    proto_list = []
    parts_list = []
    for c in child_names:
        if c not in proto_cache:
            proto_cache[c] = load_prototypes_for_category(PROTO_ROOT, c)
        mat = proto_cache[c]
        if mat is None:
            continue
        proto_list.append(mat)
        parts_list.extend([c]*mat.shape[0])
    if len(proto_list) == 0:
        return pred_map_ids

    proto_children = np.concatenate(proto_list, axis=0)  # [M, D]
    sims_sub = batched_cosine_similarity(sub_pixels, proto_children, batch_size=SIM_BATCH)  # [N, M]
    best_idx = np.argmax(sims_sub, axis=1)
    best_scores = np.max(sims_sub, axis=1)
    # translate best_idx->child name
    best_child_names = np.array([parts_list[i] for i in best_idx])

    # assign: only replace if best_score >= threshold_child
    for (y, x, cname, score) in zip(ys, xs, best_child_names, best_scores):
        if score >= threshold_child and cname in category_to_id:
            pred_map_ids[y, x] = category_to_id[cname]
        # else: keep pred_map_ids[y, x] as parent_id

    return pred_map_ids


def boundary_aware_refinement_v7(pred_map_ids, pixels_flat, proto_cache, category_to_id,
                                 proto_root, adjacency_dict, H, W,
                                 lam=0.1, gamma=0.8, alpha=0.7):
    def normalize_vec(v):
        return v / (np.linalg.norm(v) + 1e-8)

    def cosine_similarity_matrix(A, B):
        return np.matmul(A, B.T)

    pred_refined = pred_map_ids.copy()

    for partA, neighbors in adjacency_dict.items():
        if partA not in category_to_id:
            continue
        idA = category_to_id[partA]
        maskA = (pred_refined == idA)
        if not maskA.any():
            continue

        if partA not in proto_cache:
            proto_cache[partA] = load_prototypes_for_category(proto_root, partA)
        protoA = proto_cache.get(partA)
        if protoA is None:
            continue

        for partB in neighbors:
            if partB not in category_to_id:
                continue
            idB = category_to_id[partB]
            maskB = (pred_refined == idB)
            if not maskB.any():
                continue

            if partB not in proto_cache:
                proto_cache[partB] = load_prototypes_for_category(proto_root, partB)
            protoB = proto_cache.get(partB)
            if protoB is None:
                continue

            mask_union = maskA | maskB
            idx_union = np.nonzero(mask_union.ravel())[0]
            if idx_union.size == 0:
                continue
            f_union = pixels_flat[idx_union]

            simA = cosine_similarity_matrix(f_union, protoA)
            simB = cosine_similarity_matrix(f_union, protoB)
            scoreA = np.max(simA, axis=1)
            scoreB = np.max(simB, axis=1)

            D = np.abs(scoreA - scoreB)
            if D.max() - D.min() < 1e-8:
                D_norm = np.zeros_like(D)
            else:
                D_norm = (D - D.min()) / (D.max() - D.min())

            ambiguous_local = (D_norm <= lam)
            ambiguous_mask = np.zeros((H, W), dtype=bool)
            ambiguous_mask.ravel()[idx_union] = ambiguous_local

            detA_mask = maskA & (~ambiguous_mask)
            detB_mask = maskB & (~ambiguous_mask)

            idx_detA = np.nonzero(detA_mask.ravel())[0]
            idx_detB = np.nonzero(detB_mask.ravel())[0]
            idx_amb = np.nonzero(ambiguous_mask.ravel())[0]

            if len(idx_detA) == 0 or len(idx_detB) == 0 or len(idx_amb) == 0:
                continue

            f_detA = pixels_flat[idx_detA]
            f_detB = pixels_flat[idx_detB]
            f_amb = pixels_flat[idx_amb]

            def context_aggregate(f_m, f_det):
                sims = np.matmul(f_m, f_det.T)
                sims_exp = np.exp(sims)
                weights = sims_exp / (np.sum(sims_exp, axis=1, keepdims=True) + 1e-8)
                ctx = np.matmul(weights, f_det)
                return ctx

            cA = context_aggregate(f_amb, f_detA)
            cB = context_aggregate(f_amb, f_detB)

            f_tilde = gamma * f_amb + (1 - gamma) * 0.5 * (cA + cB)
            f_tilde = f_tilde / (np.linalg.norm(f_tilde, axis=1, keepdims=True) + 1e-8)

            qA = normalize_vec(np.mean(f_detA, axis=0))
            qB = normalize_vec(np.mean(f_detB, axis=0))

            pA_adapt_all = []
            for pa in protoA:
                pa_adapt = normalize_vec(alpha * pa + (1 - alpha) * qA)
                pA_adapt_all.append(pa_adapt)
            pA_adapt_all = np.stack(pA_adapt_all, axis=0)

            pB_adapt_all = []
            for pb in protoB:
                pb_adapt = normalize_vec(alpha * pb + (1 - alpha) * qB)
                pB_adapt_all.append(pb_adapt)
            pB_adapt_all = np.stack(pB_adapt_all, axis=0)

            simA_ref = cosine_similarity_matrix(f_tilde, pA_adapt_all)
            simB_ref = cosine_similarity_matrix(f_tilde, pB_adapt_all)
            scoreA_ref = np.max(simA_ref, axis=1)
            scoreB_ref = np.max(simB_ref, axis=1)

            new_labels = np.where(scoreA_ref >= scoreB_ref, idA, idB)

            refined_flat = pred_refined.ravel()
            refined_flat[idx_amb] = new_labels
            pred_refined = refined_flat.reshape(H, W)

    return pred_refined



if __name__ == "__main__":
    
    
    processor = AutoImageProcessor.from_pretrained("/data_16T/zjp/LXL/OVPS/DiffuMask/dinov2-base/", local_files_only=True)
    model = AutoModel.from_pretrained("/data_16T/zjp/LXL/OVPS/DiffuMask/dinov2-base/", local_files_only=True).to(DEVICE)
    model.eval()

    # ----- prepare object prototypes -----
    objects = root_object_mapping["root"]
    proto_dict = {obj: load_prototypes_for_category(PROTO_ROOT, obj) for obj in objects}
    # concat available protos into proto_all with label_index
    proto_all_list = []
    label_index = []
    for obj in objects:
        mat = proto_dict.get(obj)
        if mat is None:
            continue
        proto_all_list.append(mat)
        label_index.extend([obj]*mat.shape[0])
    if len(proto_all_list) == 0:
        raise RuntimeError("No object prototypes found under PROTO_ROOT.")
    proto_all = np.concatenate(proto_all_list, axis=0)

    # gather all part1 names for visualization colors
    object_allpart_mapping = object_part1_mapping  # name compatibility
    all_part = set()
    for k, v in object_allpart_mapping.items():
        all_part.update(v)
    # also include deeper-level parts
    for mapping in [part1_part2_mapping, part2_part3_mapping, part3_part4_mapping]:
        for k, vs in mapping.items():
            all_part.update(vs)
    all_part = sorted(list(all_part))
    part_color_map = random_color_map(all_part)

    # proto cache for parts (avoid repeated I/O)
    proto_cache = {}

    # metrics accumulation
    metrics_rows = []
    total_miou = []
    total_biou = []

    img_list = sorted([f for f in os.listdir(IMG_DIR) if f.endswith(".jpg")])
    print("Start processing", len(img_list), "images")

    for img_name in tqdm(img_list):
        try:
            img_path = os.path.join(IMG_DIR, img_name)
            image = Image.open(img_path).convert("RGB")
            # 1) extract features (use your function)
            feat = extract_features_with_sliding_window(image, processor, model, device=DEVICE)
            if isinstance(feat, torch.Tensor):
                feat = feat.squeeze(0).cpu().numpy()
            else:
                feat = np.asarray(feat).squeeze(0)
            H, W, D = feat.shape
            pixels = feat.reshape(-1, D).astype(np.float32)
            pixels = pixels / (np.linalg.norm(pixels, axis=1, keepdims=True) + 1e-8)

            # 2) object assignment (as before)
            sims = batched_cosine_similarity(pixels, proto_all, batch_size=SIM_BATCH)
            best_idx = np.argmax(sims, axis=1)
            best_score = np.max(sims, axis=1)
            best_label = np.array([label_index[i] for i in best_idx])
            best_label[best_score < THRESHOLD_OBJECT] = "background"
            obj_map = best_label.reshape(H, W)

            # Initialize final prediction map with background
            pred_part_id_map = np.full((H, W), fill_value=255, dtype=np.uint8)  # store integer part ids, background=255

            # 3) FIRST: for every detected object, perform full decomposition into part1 (no threshold)
            for obj in objects:
                mask = (obj_map == obj)
                if not mask.any():
                    continue
                parts = object_part1_mapping.get(obj, [])
                if len(parts) == 0:
                    continue
                # load parts prototypes (cache)
                proto_dict_parts = {}
                for p in parts:
                    if p not in proto_cache:
                        proto_cache[p] = load_prototypes_for_category(PROTO_ROOT, p)
                    proto_dict_parts[p] = proto_cache[p]
                if all(v is None for v in proto_dict_parts.values()):
                    continue
                ys, xs = np.nonzero(mask)
                idx_flat = ys * W + xs
                sub_pixels = pixels[idx_flat]
                # build proto_children and parts_list
                proto_list = []
                parts_list = []
                for p in parts:
                    mat = proto_dict_parts.get(p)
                    if mat is None:
                        continue
                    proto_list.append(mat)
                    parts_list.extend([p]*mat.shape[0])
                if len(proto_list) == 0:
                    continue
                proto_children = np.concatenate(proto_list, axis=0)
                sims_sub = batched_cosine_similarity(sub_pixels, proto_children, batch_size=SIM_BATCH)
                best_idx_sub = np.argmax(sims_sub, axis=1)
                # assign every pixel to its best child part (full decomposition)
                best_child_names = np.array([parts_list[i] for i in best_idx_sub])
                for (yy, xx, cname) in zip(ys, xs, best_child_names):
                    if cname in category_to_id:
                        pred_part_id_map[yy, xx] = category_to_id[cname]
                    else:
                        pred_part_id_map[yy, xx] = 255
            pred_part_id_map = boundary_aware_refinement_v7(pred_part_id_map, pixels, proto_cache, category_to_id,
                                                    PROTO_ROOT, part1_adjacency, H, W)

            # 4) DEEPER LAYERS: iterative from part1->part2, part2->part3, ...
            # Mappings to apply in order:
            deeper_maps = [part1_part2_mapping, part2_part3_mapping, part3_part4_mapping]
            adjacency_maps = [part2_adjacency, part3_adjacency, None]  
            for mapping, adjacency, thresh in zip(deeper_maps, adjacency_maps, THRESHOLD_CHILD):
                # for each parent in mapping keys, do selective replacement
                for parent, children in mapping.items():
                    # parent here is a part name (string) e.g., "bird head"
                    # only proceed if parent exists in category_to_id (else skip)
                    parent_id = category_to_id.get(parent, None)
                    if parent_id is None:
                        continue
                    # ensure there are child names
                    if not children:
                        continue
                    # apply selective decomposition
                    pred_part_id_map = decompose_children_for_parent(
                        pred_part_id_map, parent, children, pixels, W, proto_cache, threshold_child=thresh
                    )
                # BAR 
                if adjacency is not None:
                    pred_part_id_map = boundary_aware_refinement_v7(
                        pred_part_id_map, pixels, proto_cache, category_to_id,
                        PROTO_ROOT, adjacency, H, W
                    )

            # 5) visualization: overlay parts on original image (only foreground parts)
            vis = np.array(image).astype(np.float32)
            overlay = np.zeros_like(vis, dtype=np.float32)
            combined_mask = np.zeros((H, W), dtype=bool)
            for p in all_part:
                pid = category_to_id.get(p, None)
                if pid is None:
                    continue
                pmask = (pred_part_id_map == pid)
                if pmask.any():
                    overlay[pmask] = part_color_map[p]
                    combined_mask |= pmask
            blended = vis.copy()
            blended[combined_mask] = ((1 - ALPHA) * vis[combined_mask] + ALPHA * overlay[combined_mask])
            blended = blended.astype(np.uint8)

            # save visualization and predicted label png
            save_vis_path = os.path.join(SAVE_DIR, img_name)
            Image.fromarray(blended).save(save_vis_path)

            pred_label_png_path = os.path.join(PRED_LABEL_DIR, os.path.splitext(img_name)[0] + ".png")
            Image.fromarray(pred_part_id_map).save(pred_label_png_path)

            # 6) compute metrics with GT if exists
            gt_path = os.path.join(GT_DIR, os.path.splitext(img_name)[0] + ".png")
            if os.path.exists(gt_path):
                gt = np.array(Image.open(gt_path))
                if gt.shape[:2] != pred_part_id_map.shape:
                    gt = np.array(Image.fromarray(gt).resize((W, H), resample=Image.NEAREST))
                num_classes = max(category_to_id.values()) + 1
                miou, mbiou = compute_miou_and_biou(gt, pred_part_id_map, num_classes=num_classes, ignore_index=255)
                total_miou.append(miou)
                total_biou.append(mbiou)
                metrics_rows.append([img_name, float(miou), float(mbiou)])
            else:
                metrics_rows.append([img_name, None, None])

        except Exception as e:
            print(f"Error on {img_name}: {e}")
            traceback.print_exc()
            metrics_rows.append([img_name, "error", "error"])
            continue

    # write metrics csv and print averages
    metrics_csv = os.path.join(SAVE_DIR, "metrics.csv")
    with open(metrics_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image", "miou", "biou"])
        for row in metrics_rows:
            writer.writerow(row)

    if len(total_miou) > 0:
        print("Average mIoU over images:", float(np.mean(total_miou)))
        print("Average bIoU over images:", float(np.mean(total_biou)))
    else:
        print("No GT found to compute metrics.")

    print("Saved visualizations to:", SAVE_DIR)
    print("Saved predicted label pngs to:", PRED_LABEL_DIR)
    print("Saved metrics csv to:", metrics_csv)
    
    


