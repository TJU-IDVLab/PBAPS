from typing import Optional, Union, Tuple, List, Callable, Dict
import torch
from diffusers import StableDiffusionPipeline
import torch.nn.functional as nnf
import numpy as np
import abc
import ptp_utils
import seq_aligner
import cv2
import json
import argparse
import multiprocessing as mp
import threading
from random import choice
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import torch
import torchvision.transforms as T
import cv2
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
import matplotlib.image as mpimg 
import matplotlib
from sklearn.decomposition import PCA
from transformers import AutoImageProcessor, AutoModel
import torch.nn as nn


MY_TOKEN = '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'
LOW_RESOURCE = False 
NUM_DIFFUSION_STEPS = 50
GUIDANCE_SCALE = 7.5
MAX_NUM_WORDS = 77

coco_category_list_check = {
    'aeroplane body': ['body'], 'aeroplane stern': ['stern'], 'aeroplane wing': ['wing'], 'aeroplane tail': ['tail'], 'aeroplane engine': ['engine'], 'aeroplane wheel': ['wheel'], 
    'bicycle wheel': ['wheel'], 'bicycle saddle': ['saddle'], 'bicycle handlebar': ['handle', 'bar'], 'bicycle chainwheel': ['chain', 'wheel'], 'bicycle headlight': ['head', 'light'], 
    'bird wing': ['wing'], 'bird tail': ['tail'], 'bird head': ['head'], 'bird eye': ['eye'], 'bird beak': ['beak'], 'bird torso': ['tor', 'so'], 'bird neck': ['neck'], 'bird leg': ['leg'], 'bird foot': ['foot'], 
    'bottle body': ['body'], 'bottle cap': ['cap'], 
    'bus wheel': ['wheel'], 'bus headlight': ['head', 'light'], 'bus front': ['front'], 'bus side': ['side'], 'bus back': ['back'], 'bus roof': ['roof'], 'bus mirror': ['mirror'], 'bus license plate': ['license', 'plate'], 'bus door': ['door'], 'bus window': ['window'], 
    'car wheel': ['wheel'], 'car headlight': ['head', 'light'], 'car front': ['front'], 'car side': ['side'], 'car back': ['back'], 'car roof': ['roof'], 'car mirror': ['mirror'], 'car license plate': ['license', 'plate'], 'car door': ['door'], 'car window': ['window'], 
    'cat tail': ['tail'], 'cat head': ['head'], 'cat eye': ['eye'], 'cat torso': ['tor', 'so'], 'cat neck': ['neck'], 'cat leg': ['leg'], 'cat nose': ['nose'], 'cat paw': ['paw'], 'cat ear': ['ear'], 
    'cow tail': ['tail'], 'cow head': ['head'], 'cow eye': ['eye'], 'cow torso': ['tor', 'so'], 'cow neck': ['neck'], 'cow leg': ['leg'], 'cow ear': ['ear'], 'cow muzzle': ['mu', 'zzle'], 'cow horn': ['horn'], 
    'dog tail': ['tail'], 'dog head': ['head'], 'dog eye': ['eye'], 'dog torso': ['tor', 'so'], 'dog neck': ['neck'], 'dog leg': ['leg'], 'dog nose': ['nose'], 'dog paw': ['paw'], 'dog ear': ['ear'], 'dog muzzle': ['mu', 'zzle'], 
    'horse tail': ['tail'], 'horse head': ['head'], 'horse eye': ['eye'], 'horse torso': ['tor', 'so'], 'horse neck': ['neck'], 'horse leg': ['leg'], 'horse ear': ['ear'], 'horse muzzle': ['mu', 'zzle'], 'horse hoof': ['hoof'], 
    'motorbike wheel': ['wheel'], 'motorbike saddle': ['saddle'], 'motorbike handlebar': ['handle', 'bar'], 'motorbike headlight': ['head', 'light'], 
    'person head': ['head'], 'person eye': ['eye'], 'person torso': ['tor', 'so'], 'person neck': ['neck'], 'person leg': ['leg'], 'person foot': ['foot'], 'person nose': ['nose'], 'person ear': ['ear'], 'person eyebrow': ['eyebrow'], 'person mouth': ['mouth'], 'person hair': ['hair'], 'person lower arm': ['lower', 'arm'], 'person upper arm': ['upper', 'arm'], 'person hand': ['hand'], 
    'pottedplant pot': ['pot'], 'pottedplant plant': ['plant'], 
    'sheep tail': ['tail'], 'sheep head': ['head'], 'sheep eye': ['eye'], 'sheep torso': ['tor', 'so'], 'sheep neck': ['neck'], 'sheep leg': ['leg'], 'sheep ear': ['ear'], 'sheep muzzle': ['mu', 'zzle'], 'sheep horn': ['horn'], 
    'train headlight': ['head', 'light'], 'train head': ['head'], 'train front': ['front'], 'train side': ['side'], 'train back': ['back'], 'train roof': ['roof'], 'train coach': ['coach'], 
    'tvmonitor screen': ['screen']}

coco_category_list_check2 = {
    'aeroplane body': ['aerop','lane','body'], 'aeroplane stern': ['aerop','lane','stern'], 'aeroplane wing': ['aerop','lane','wing'], 'aeroplane tail': ['aerop','lane','tail'], 'aeroplane engine': ['aerop','lane','engine'], 'aeroplane wheel': ['aerop','lane','wheel'], 
    'bicycle wheel': ['bicycle','wheel'], 'bicycle saddle': ['bicycle','saddle'], 'bicycle handlebar': ['bicycle','handle', 'bar'], 'bicycle chainwheel': ['bicycle','chain', 'wheel'], 'bicycle headlight': ['bicycle','head', 'light'], 
    'bird wing': ['bird','wing'], 'bird tail': ['bird','tail'], 'bird head': ['bird','head'], 'bird eye': ['bird','eye'], 'bird beak': ['bird','beak'], 'bird torso': ['bird','tor', 'so'], 'bird neck': ['bird','neck'], 'bird leg': ['bird','leg'], 'bird foot': ['bird','foot'], 
    'bottle body': ['bottle','body'], 'bottle cap': ['bottle','cap'], 
    'bus wheel': ['bus','wheel'], 'bus headlight': ['bus','head', 'light'], 'bus front': ['bus','front'], 'bus side': ['bus','side'], 'bus back': ['bus','back'], 'bus roof': ['bus','roof'], 'bus mirror': ['bus','mirror'], 'bus license plate': ['bus','license', 'plate'], 'bus door': ['bus','door'], 'bus window': ['bus','window'], 
    'car wheel': ['car','wheel'], 'car headlight': ['car','head', 'light'], 'car front': ['car','front'], 'car side': ['car','side'], 'car back': ['car','back'], 'car roof': ['car','roof'], 'car mirror': ['car','mirror'], 'car license plate': ['car','license', 'plate'], 'car door': ['car','door'], 'car window': ['car','window'], 
    'cat tail': ['cat','tail'], 'cat head': ['cat','head'], 'cat eye': ['cat','eye'], 'cat torso': ['cat','tor', 'so'], 'cat neck': ['cat','neck'], 'cat leg': ['cat','leg'], 'cat nose': ['cat','nose'], 'cat paw': ['cat','paw'], 'cat ear': ['cat','ear'], 
    'cow tail': ['cow','tail'], 'cow head': ['cow','head'], 'cow eye': ['cow','eye'], 'cow torso': ['cow','tor', 'so'], 'cow neck': ['cow','neck'], 'cow leg': ['cow','leg'], 'cow ear': ['cow','ear'], 'cow muzzle': ['cow','mu', 'zzle'], 'cow horn': ['cow','horn'], 
    'dog tail': ['dog','tail'], 'dog head': ['dog','head'], 'dog eye': ['dog','eye'], 'dog torso': ['dog','tor', 'so'], 'dog neck': ['dog','neck'], 'dog leg': ['dog','leg'], 'dog nose': ['dog','nose'], 'dog paw': ['dog','paw'], 'dog ear': ['dog','ear'], 'dog muzzle': ['dog','mu', 'zzle'], 
    'horse tail': ['horse','tail'], 'horse head': ['horse','head'], 'horse eye': ['horse','eye'], 'horse torso': ['horse','tor', 'so'], 'horse neck': ['horse','neck'], 'horse leg': ['horse','leg'], 'horse ear': ['horse','ear'], 'horse muzzle': ['horse','mu', 'zzle'], 'horse hoof': ['horse','hoof'], 
    'motorbike wheel': ['motorbike','wheel'], 'motorbike saddle': ['motorbike','saddle'], 'motorbike handlebar': ['motorbike','handle', 'bar'], 'motorbike headlight': ['motorbike','head', 'light'], 
    'person head': ['person','head'], 'person eye': ['person','eye'], 'person torso': ['person','tor', 'so'], 'person neck': ['person','neck'], 'person leg': ['person','leg'], 'person foot': ['person','foot'], 'person nose': ['person','nose'], 'person ear': ['person','ear'], 'person eyebrow': ['person','eyebrow'], 'person mouth': ['person','mouth'], 'person hair': ['person','hair'], 'person lower arm': ['person','lower', 'arm'], 'person upper arm': ['person','upper', 'arm'], 'person hand': ['person','hand'], 
    'pottedplant pot': ['pot','plant','ted','pot'], 'pottedplant plant': ['pot','plant','ted','plant'], 
    'sheep tail': ['sheep','tail'], 'sheep head': ['sheep','head'], 'sheep eye': ['sheep','eye'], 'sheep torso': ['sheep','tor', 'so'], 'sheep neck': ['sheep','neck'], 'sheep leg': ['sheep','leg'], 'sheep ear': ['sheep','ear'], 'sheep muzzle': ['sheep','mu', 'zzle'], 'sheep horn': ['sheep','horn'], 
    'train headlight': ['train','head', 'light'], 'train head': ['train','head'], 'train front': ['train','front'], 'train side': ['train','side'], 'train back': ['train','back'], 'train roof': ['train','roof'], 'train coach': ['train','coach'], 
    'tvmonitor screen': ['tv','monitor','screen']}

coco_category_to_id_v1 = {
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


coco_category_list = [
    'aeroplane body', 'aeroplane stern', 'aeroplane wing', 'aeroplane tail', 'aeroplane engine', 'aeroplane wheel', 
    'bicycle wheel', 'bicycle saddle', 'bicycle handlebar', 'bicycle chainwheel', 'bicycle headlight', 
    'bird wing', 'bird tail', 'bird head', 'bird eye', 'bird beak', 'bird torso', 'bird neck', 'bird leg', 'bird foot', 
    'bottle body', 'bottle cap', 
    'bus wheel', 'bus headlight', 'bus front', 'bus side', 'bus back', 'bus roof', 'bus mirror', 'bus license plate', 'bus door', 'bus window', 
    'car wheel', 'car headlight', 'car front', 'car side', 'car back', 'car roof', 'car mirror', 'car license plate', 'car door', 'car window', 
    'cat tail', 'cat head', 'cat eye', 'cat torso', 'cat neck', 'cat leg', 'cat nose', 'cat paw', 'cat ear', 
    'cow tail', 'cow head', 'cow eye', 'cow torso', 'cow neck', 'cow leg', 'cow ear', 'cow muzzle', 'cow horn', 
    'dog tail', 'dog head', 'dog eye', 'dog torso', 'dog neck', 'dog leg', 'dog nose', 'dog paw', 'dog ear', 'dog muzzle', 
    'horse tail', 'horse head', 'horse eye', 'horse torso', 'horse neck', 'horse leg', 'horse ear', 'horse muzzle', 'horse hoof', 
    'motorbike wheel', 'motorbike saddle', 'motorbike handlebar', 'motorbike headlight', 
    'person head', 'person eye', 'person torso', 'person neck', 'person leg', 'person foot', 'person nose', 'person ear', 'person eyebrow', 'person mouth', 'person hair', 'person lower arm', 'person upper arm', 'person hand', 
    'pottedplant pot', 'pottedplant plant', 
    'sheep tail', 'sheep head', 'sheep eye', 'sheep torso', 'sheep neck', 'sheep leg', 'sheep ear', 'sheep muzzle', 'sheep horn', 
    'train headlight', 'train head', 'train front', 'train side', 'train back', 'train roof', 'train coach', 
    'tvmonitor screen']

def init_models(device):
    
    dinov2_processor = AutoImageProcessor.from_pretrained("/data_16T/zjp/LXL/OVPS/PBAPS/dinov2-base/", local_files_only=True)
    dinov2_model = AutoModel.from_pretrained("/data_16T/zjp/LXL/OVPS/PBAPS/dinov2-base/", local_files_only=True).to(device)
    
    sam_checkpoint = "/data_16T/zjp/LXL/OVPS/PBAPS/sam_vit_b_01ec64.pth"
    sam = sam_model_registry["vit_b"](checkpoint=sam_checkpoint)
    sam.to(device=device)
    sam_predictor = SamPredictor(sam)
    
    return dinov2_processor, dinov2_model, sam_predictor

class VisualAttentionAnalyzer:
    def __init__(self, image_path, attention_scores_path, output_image_path, dinov2_processor, dinov2, sam, device="cuda:4"):
        self.image_path = image_path
        self.attention_scores_path = attention_scores_path
        self.output_image_path = output_image_path
        self.device = device
        self.image = None
        self.masks = None
        self.attention_scores_dict = None
        self.selected_coords = None
        self.dinov2_processor = dinov2_processor
        self.dinov2_model = dinov2
        self.sam_predictor = sam
        self.load_data()

    def load_data(self):
        self.image = Image.open(self.image_path)
        try:
            self.attention_scores_dict = np.load(self.attention_scores_path, allow_pickle=True).item()
        except:
            print(f"Failed to load {self.attention_scores_path}")

    def visualize_attention_map(self, category_index=23, threshold=0.7, num_points=5):
        aeroplane_attention = self.attention_scores_dict[category_index]
        aeroplane_attention = (aeroplane_attention - aeroplane_attention.min()) / (aeroplane_attention.max() - aeroplane_attention.min())
        
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        ax[0].imshow(self.image)
        ax[0].set_title('Original Image')
        ax[0].axis('off')

        ax[1].imshow(self.image)
        ax[1].imshow(aeroplane_attention, cmap='jet', alpha=0.5)
        ax[1].set_title('Attention Map')
        ax[1].axis('off')

        plt.tight_layout()
        plt.savefig(self.output_image_path)
        plt.close()  
        # print(f"Visualization saved to {self.output_image_path}")

        binary_attention = aeroplane_attention >= threshold
        coords = np.column_stack(np.where(binary_attention))

        if coords.shape[0] < num_points:
            print(f"Warning: Only {coords.shape[0]} points have attention scores >= {threshold}.")
            num_points = coords.shape[0]

        np.random.seed(42)  
        selected_indices = np.random.choice(coords.shape[0], size=num_points, replace=False)
        self.selected_coords = coords[selected_indices][:, [1, 0]]   # 调换横纵坐标

        plt.figure()
        plt.imshow(self.image)
        plt.scatter(self.selected_coords[:, 0], self.selected_coords[:, 1], c='red', s=100, marker='o') 
        plt.axis('off')
        plt.show()
        plt.savefig(self.output_image_path)
        plt.close()  

    def segment_with_sam(self, model_type="vit_b", sam_checkpoint="/data_16T/zjp/LXL/OVPS/PBAPS/sam_vit_b_01ec64.pth"):

        image_cv = cv2.imread(self.image_path)
        image_cv = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
        self.sam_predictor.set_image(image_cv)

        input_point = self.selected_coords
        input_label = np.ones(len(self.selected_coords), dtype=int)

        masks, scores, logits = self.sam_predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=False,
        )
        self.masks=masks
        plt.figure()
        plt.imshow(image_cv)
        self.show_mask(masks, plt.gca())
        self.show_points(input_point, input_label, plt.gca())
        plt.axis('off')
        plt.show()
        plt.savefig(self.output_image_path)
        plt.close()  

    def extract_features_with_dinov2(self, model_path="/data_16T/zjp/LXL/OVPS/PBAPS/dinov2-base/", feature_save_path=None):
        image_np = np.array(self.image)
        mask = self.masks[0]  
        masked_image_np = image_np.copy()
        masked_image_np[~mask] = 0  
        masked_image = Image.fromarray(masked_image_np)

        with torch.no_grad():
            inputs = self.dinov2_processor(images=masked_image, return_tensors="pt").to(self.device)
            outputs = self.dinov2_model(**inputs)
            image_features = outputs.last_hidden_state.mean(dim=1)    
            # cls_token_features = outputs.last_hidden_state[:, 0, :]


            if feature_save_path:
                torch.save(image_features.cpu(), feature_save_path)
                # print(f"Features saved to {feature_save_path}")

            return image_features.cpu().numpy()

    @staticmethod
    def show_mask(mask, ax, random_color=False):
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0) if random_color else np.array([30/255, 144/255, 255/255, 0.6])
        h, w = mask.shape[-2:]
        mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
        ax.imshow(mask_image)

    @staticmethod
    def show_points(coords, labels, ax, marker_size=375):
        pos_points = coords[labels == 1]
        neg_points = coords[labels == 0]
        ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
        ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)



def sub_processor(pid , class_list):
    try:
        text = 'processor %d' % pid
        print(text)
        
        dinov2_processor, dinov2_model, sam_predictor = init_models(f"cuda:{pid}")

        base_path="/data_16T/zjp/LXL/OVPS/PBAPS/Visual Prototype/PascalPart116/Data/All_Class"

        for class_one in class_list:
            class_dir = os.path.join(base_path, class_one)
            subfolders = ['mask','mask2','feature','feature2']
            for subfolder in subfolders:
                subfolder_path = os.path.join(class_dir, subfolder)
                if not os.path.exists(subfolder_path): os.makedirs(subfolder_path)
            image_folder=os.path.join(class_dir, 'image')    
            npy_folder=os.path.join(class_dir, 'npy') 
            npy2_folder=os.path.join(class_dir, 'npy2')        

            for image_file in os.listdir(image_folder):
                if image_file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):  
                    image_name, _ = os.path.splitext(image_file)
                    image_path = os.path.join(image_folder, image_file)
                    npy_file = f"{image_name}.npy"
                    npy_path = os.path.join(npy_folder, npy_file)
                    npy2_file = f"{image_name}.npy"
                    npy2_path = os.path.join(npy2_folder, npy_file)

                    if os.path.isfile(npy_path):
                        output_image_path = os.path.join(class_dir, 'mask', f"{image_name}.png")
                        feature_save_path = os.path.join(class_dir, 'feature', f"{image_name}.pt")

                        if not os.path.exists(output_image_path):
                            analyzer = VisualAttentionAnalyzer(
                                image_path=image_path,
                                attention_scores_path=npy_path,
                                output_image_path=output_image_path,
                                device=f"cuda:{pid}",
                                dinov2_processor=dinov2_processor, 
                                dinov2=dinov2_model, 
                                sam=sam_predictor
                            )
                            
                            try:
                                analyzer.visualize_attention_map(category_index=coco_category_to_id_v1[class_one], threshold=0.7, num_points=5)
                                analyzer.segment_with_sam()
                                features = analyzer.extract_features_with_dinov2(feature_save_path=feature_save_path)
                                
                                # print(f"Processed {image_file} and saved the result to {output_image_path}")
                            except Exception as e:
                                print(f"Failed to process {image_file}: {e}")
                    else:
                        print(f"Numpy file {npy_file} not found for image {image_file}")

                    if os.path.isfile(npy2_path):
                        output_image_path = os.path.join(class_dir, 'mask2', f"{image_name}.png")
                        feature_save_path = os.path.join(class_dir, 'feature2', f"{image_name}.pt")

                        if not os.path.exists(output_image_path):
                            analyzer = VisualAttentionAnalyzer(
                                image_path=image_path,
                                attention_scores_path=npy2_path,
                                output_image_path=output_image_path,
                                device=f"cuda:{pid}",
                                dinov2_processor=dinov2_processor, 
                                dinov2=dinov2_model, 
                                sam=sam_predictor
                            )
                            
                            try:
                                analyzer.visualize_attention_map(category_index=coco_category_to_id_v1[class_one], threshold=0.7, num_points=5)
                                analyzer.segment_with_sam()
                                features = analyzer.extract_features_with_dinov2(feature_save_path=feature_save_path)
                                # print(f"Processed {image_file} and saved the result to {output_image_path}")
                            except Exception as e:
                                print(f"Failed to process {image_file}: {e}")
                    else:
                        print(f"Numpy file {npy_file} not found for image {image_file}")
    except Exception as e:
                print(f"processor Failed to process {image_file}: {e}")
            
            
if __name__ == '__main__':
    
    
    result_dict = mp.Manager().dict()
    mp = mp.get_context("spawn")
    thread_num = 10
    processes = []
    per_thread_video_num = int(len(coco_category_list)/thread_num)

    print('Start inference')
    for i in range(thread_num):
        if i == thread_num - 1:
            sub_video_list = coco_category_list[i * per_thread_video_num:]
        else:
            sub_video_list = coco_category_list[i * per_thread_video_num: (i + 1) * per_thread_video_num]
        try:
            p = mp.Process(target=sub_processor, args=(i, sub_video_list))
            p.start()
            processes.append(p)
        except Exception as e:
                print(f"processor Failed {i}: {e}")


    for p in processes:
        p.join()

    result_dict = dict(result_dict)
    

# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9 /data_16T/zjp/envs/ovpp/bin/python "/data_16T/zjp/LXL/OVPS/PBAPS/Visual Prototype/2_SAM_DINOv2_mask_feature.py"

