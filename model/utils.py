import os
import numpy as np
import random
import torch
import torch.nn.functional as F
from PIL import Image

def extract_features_with_sliding_window(image, processor, model, device="cuda:9", window_size=224, stride=112):
    """
    使用滑动窗口提取图像特征
    :param image: 原始图像，PIL.Image 对象
    :param processor: DINO 图像处理器
    :param model: DINO 模型
    :param device: 设备
    :param window_size: 窗口大小
    :param stride: 滑动步幅
    :return: 每个像素的特征，形状为 [1, H, W, 768]
    """
    # ---------- 新增：最小边缩放 ----------
    w, h = image.size
    if min(h, w) < window_size:
        new_size = (window_size, window_size)  # PIL 顺序 (W, H)
        image = image.resize(new_size, Image.Resampling.BICUBIC)
        window = image
        inputs = processor(images=window, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            patch_features = outputs.last_hidden_state[:, 1:, :].permute(0, 2, 1).view(1, 768, 16, 16)  # 调整根据实际模型输出
            per_pixel_feature = F.interpolate(
                patch_features,   # [1, 768, H, W]
                size=(h, w),
                mode='bilinear',
                align_corners=False
            ).permute(0, 2, 3, 1)                        # [1, orig_H, orig_W, 768]
        return per_pixel_feature
    
    
    image_np = np.array(image)  # [H, W, 3]
    H, W, _ = image_np.shape

    
    per_pixel_feature = torch.zeros((1, H, W, 768), device=device)  # 初始化特征图

    # 如果有重叠区域，则需平均化这些区域的特征
    counts = torch.zeros_like(per_pixel_feature)
    
    def process_window(x_start, y_start, x_end, y_end):
        """处理单个窗口"""
        window = image.crop((x_start, y_start, x_end, y_end))  # 坐标原点在左上角
        inputs = processor(images=window, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            patch_features = outputs.last_hidden_state[:, 1:, :].permute(0, 2, 1).view(1, 768, 16, 16)  # 调整根据实际模型输出
            window_feature = F.interpolate(patch_features, size=(y_end - y_start, x_end - x_start), mode="bilinear", align_corners=False)
            window_feature = window_feature.permute(0, 2, 3, 1)  # [1, H', W', 768]
        per_pixel_feature[:, y_start:y_end, x_start:x_end] += window_feature
        counts[:, y_start:y_end, x_start:x_end] += 1
    
    
    # 主循环处理大部分窗口
    for y in range(0, H - window_size + 1, stride):
        for x in range(0, W - window_size + 1, stride):
            process_window(x, y, x + window_size, y + window_size)
    

    # 处理最后一行
    if (H - y) % stride != 0 and (H - y) > 0:
        y_start = H - window_size
        for x in range(0, W - window_size + 1, stride):
            x_end = min(x + window_size, W)
            process_window(x, y_start, x_end, H)
    
    # 处理最后一列
    if (W - x) % stride != 0 and (W - x) > 0:
        x_start = W - window_size
        for y in range(0, H - window_size + 1, stride):
            y_end = min(y + window_size, H)
            process_window(x_start, y, W, y_end)
    
    # 处理右下角的最后一个窗口
    if (H - y) % stride != 0 and (H - y) > 0 and (W - x) % stride != 0 and (W - x) > 0:
        process_window(W - window_size, H - window_size, W, H)

    # 平均化重叠区域
    counts[counts == 0] = 1  # 避免除以零
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
