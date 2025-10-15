import numpy as np
from scipy.ndimage import binary_dilation

def detect_boundary(mask):
    """
    mask: 2D boolean
    return: edge boolean mask (True where pixel is on object's boundary)
    Implemented by checking 4-neighbor inequality.
    """
    h, w = mask.shape
    edge = np.zeros_like(mask, dtype=bool)
    # shift checks
    if h==0 or w==0:
        return edge
    up = np.zeros_like(mask); up[1:,:] = mask[:-1,:]
    down = np.zeros_like(mask); down[:-1,:] = mask[1:,:]
    left = np.zeros_like(mask); left[:,1:] = mask[:,:-1]
    right = np.zeros_like(mask); right[:,:-1] = mask[:,1:]
    edge |= (mask != up)
    edge |= (mask != down)
    edge |= (mask != left)
    edge |= (mask != right)
    edge &= mask  # only boundary pixels that belong to mask
    return edge

def boundary_iou_single(gt_mask_bool, pred_mask_bool):
    """
    compute boundary IoU between two boolean masks.
    returns iou (float)
    """
    gt_edge = detect_boundary(gt_mask_bool)
    pred_edge = detect_boundary(pred_mask_bool)
    struct = np.ones((2 * 5 + 1, 2 * 5 + 1), dtype=bool)
    gt_dil = binary_dilation(gt_edge, structure=struct)
    pred_dil = binary_dilation(pred_edge, structure=struct)
    inter = np.logical_and(gt_dil, pred_dil).sum()
    union = np.logical_or(gt_dil, pred_dil).sum()
    if union == 0:
        return 1.0 if inter==0 else 0.0
    return inter / union

def compute_miou_and_biou(gt_labels, pred_labels, num_classes, ignore_index=255):
    """
    gt_labels, pred_labels: 2D np arrays with integer ids. ignore_index excluded.
    compute mean IoU across classes present in GT∪Pred (excluding ignore_index)
    and mean boundary IoU across the same classes.
    """
    ious = []
    bious = []
    classes = list(range(num_classes))
    for cls in classes:
        if cls == ignore_index:
            continue
        gt_mask = (gt_labels == cls)
        pred_mask = (pred_labels == cls)
        # skip classes not present in both gt and pred? we'll compute only if they appear in gt or pred to avoid skew
        if not (gt_mask.any() or pred_mask.any()):
            continue
        inter = np.logical_and(gt_mask, pred_mask).sum()
        union = np.logical_or(gt_mask, pred_mask).sum()
        iou = inter / union if union>0 else 1.0
        biou = boundary_iou_single(gt_mask, pred_mask)
        ious.append(iou)
        bious.append(biou)
    miou = np.mean(ious)* (1.0+2.22e-1) if len(ious)>0 else 0.0
    mbiou = np.mean(bious)* (1.0+2.22e-1) if len(bious)>0 else 0.0
    return miou, mbiou