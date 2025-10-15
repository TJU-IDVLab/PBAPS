import numpy as np
from utils import load_prototypes_for_category


def boundary_aware_refinement_v7(
    pred_map_ids,
    pixels_flat,
    proto_cache,
    category_to_id,
    proto_root,
    adjacency_dict,
    H,
    W,
    lam=0.1,
    gamma=0.7,
    alpha=0.5
):
    """
    Boundary-Aware Refinement (BAR) v7

    This module refines segmentation boundaries between adjacent parts
    by identifying ambiguous boundary regions and adaptively optimizing
    prototype representations.

    Process Overview:
        1. Detect adjacent parts using the adjacency dictionary.
        2. For each part pair (A, B):
           - Detect ambiguous regions where classification scores are similar.
           - Enhance ambiguous features via contextual aggregation.
           - Adapt each global prototype with local (region) prototypes.
           - Reclassify ambiguous pixels by comparing similarity to adapted prototypes.

    Args:
        pred_map_ids (np.ndarray): [H, W] predicted part ID map.
        pixels_flat (np.ndarray): [H*W, D] flattened pixel features (normalized).
        proto_cache (dict): cache of loaded prototypes for each category.
        category_to_id (dict): mapping from category name to integer ID.
        proto_root (str): directory of prototype vectors.
        adjacency_dict (dict): adjacency map between parts, e.g., {"car door": ["car window"]}.
        H, W (int): image height and width.
        lam (float): ambiguity threshold; smaller → fewer ambiguous pixels.
        gamma (float): feature mixing ratio for context enhancement.
        alpha (float): blending factor between global and local prototypes.

    Returns:
        np.ndarray: refined segmentation map (same shape as pred_map_ids).
    """

    def normalize_vec(v):
        """Normalize a vector to unit length."""
        return v / (np.linalg.norm(v) + 1e-8)

    def cosine_similarity_matrix(A, B):
        """Compute cosine similarity matrix between two feature sets."""
        return np.matmul(A, B.T)

    pred_refined = pred_map_ids.copy()

    for partA, neighbors in adjacency_dict.items():
        if partA not in category_to_id:
            continue
        idA = category_to_id[partA]
        maskA = (pred_refined == idA)
        if not maskA.any():
            continue

        # Load all global prototypes for partA
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

            # ---------- (1) Combined Region ----------
            mask_union = maskA | maskB
            idx_union = np.nonzero(mask_union.ravel())[0]
            if idx_union.size == 0:
                continue
            f_union = pixels_flat[idx_union]

            # ---------- (2) Classification Scores ----------
            simA = cosine_similarity_matrix(f_union, protoA)
            simB = cosine_similarity_matrix(f_union, protoB)
            scoreA = np.max(simA, axis=1)
            scoreB = np.max(simB, axis=1)

            # ---------- (3) Ambiguous Region Detection ----------
            D = np.abs(scoreA - scoreB)
            if D.max() - D.min() < 1e-8:
                D_norm = np.zeros_like(D)
            else:
                D_norm = (D - D.min()) / (D.max() - D.min())

            ambiguous_local = (D_norm <= lam)
            ambiguous_mask = np.zeros((H, W), dtype=bool)
            ambiguous_mask.ravel()[idx_union] = ambiguous_local

            # Confident regions
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

            # ---------- (4) Contextual Enhancement ----------
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

            # ---------- (5) Adaptive Prototype Optimization ----------
            qA = normalize_vec(np.mean(f_detA, axis=0))
            qB = normalize_vec(np.mean(f_detB, axis=0))

            # Fuse each global prototype individually with local prototype
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

            # ---------- (6) Ambiguous Region Reclassification ----------
            simA_ref = cosine_similarity_matrix(f_tilde, pA_adapt_all)
            simB_ref = cosine_similarity_matrix(f_tilde, pB_adapt_all)
            scoreA_ref = np.max(simA_ref, axis=1)
            scoreB_ref = np.max(simB_ref, axis=1)

            new_labels = np.where(scoreA_ref >= scoreB_ref, idA, idB)

            refined_flat = pred_refined.ravel()
            refined_flat[idx_amb] = new_labels
            pred_refined = refined_flat.reshape(H, W)

    return pred_refined
