import numpy as np
from utils import load_prototypes_for_category, batched_cosine_similarity
from BAR_refinement import boundary_aware_refinement_v7

# ---------------- hierarchical decomposition helpers ----------------
def decompose_children_for_parent(pred_map_ids, parent_name, child_names, pixels_flat, W, proto_cache, threshold_child=THRESHOLD_CHILD):
    """
    给定 pred_map_ids (H x W, uint8), 将 parent_name 区域内进一步分解为 child_names。
    - pred_map_ids: numpy array H x W where parent pixels currently have id category_to_id[parent]
    - parent_name: string (e.g., "bottle")
    - child_names: list of strings (e.g., ["bottle body", "bottle cap"])
    - pixels_flat: flattened features shape [H*W, D], normalized
    - proto_cache: dict for caching loaded prototypes {category_name: np.array or None}
    返回: updated pred_map_ids (in-place modification acceptable)
    规则：对于每个父像素，若其与某子原型的最佳相似度 >= threshold_child，则替换为该子部件 id；否则保留父 id.
    """
    H, W = pred_map_ids.shape
    parent_id = category_to_id.get(parent_name, None)
    if parent_id is None:
        return pred_map_ids  # parent not in category map (可能是 object name)，跳过

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

