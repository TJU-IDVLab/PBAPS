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


class AttentionControl(abc.ABC):
    
    def step_callback(self, x_t):
        return x_t
    
    def between_steps(self):
        return
    
    @property
    def num_uncond_att_layers(self):
        return self.num_att_layers if LOW_RESOURCE else 0
    
    @abc.abstractmethod
    def forward (self, attn, is_cross: bool, place_in_unet: str):
        raise NotImplementedError

    def __call__(self, attn, is_cross: bool, place_in_unet: str):
        if self.cur_att_layer >= self.num_uncond_att_layers:
            if LOW_RESOURCE:
                attn = self.forward(attn, is_cross, place_in_unet)
            else:
                h = attn.shape[0]
                attn[h // 2:] = self.forward(attn[h // 2:], is_cross, place_in_unet)
        self.cur_att_layer += 1
        if self.cur_att_layer == self.num_att_layers + self.num_uncond_att_layers:
            self.cur_att_layer = 0
            self.cur_step += 1
            self.between_steps()
        return attn
    
    def reset(self):
        self.cur_step = 0
        self.cur_att_layer = 0

    def __init__(self):
        self.cur_step = 0
        self.num_att_layers = -1
        self.cur_att_layer = 0

class EmptyControl(AttentionControl):
    
    def forward (self, attn, is_cross: bool, place_in_unet: str):
        return attn
    
    
class AttentionStore(AttentionControl):

    @staticmethod
    def get_empty_store():
        return {"down_cross": [], "mid_cross": [], "up_cross": [],
                "down_self": [],  "mid_self": [],  "up_self": []}

    def forward(self, attn, is_cross: bool, place_in_unet: str):
        key = f"{place_in_unet}_{'cross' if is_cross else 'self'}"
#         if attn.shape[1] <= 128 ** 2:  # avoid memory overhead
        self.step_store[key].append(attn)
        return attn

    def between_steps(self):
        if len(self.attention_store) == 0:
            self.attention_store = self.step_store
        else:
            for key in self.attention_store:
                for i in range(len(self.attention_store[key])):
                    self.attention_store[key][i] += self.step_store[key][i]
        self.step_store = self.get_empty_store()

    def get_average_attention(self):
        average_attention = {key: [item / self.cur_step for item in self.attention_store[key]] for key in self.attention_store}
        return average_attention


    def reset(self):
        super(AttentionStore, self).reset()
        self.step_store = self.get_empty_store()
        self.attention_store = {}

    def __init__(self):
        super(AttentionStore, self).__init__()
        self.step_store = self.get_empty_store()
        self.attention_store = {}



from PIL import Image

def aggregate_attention(attention_store: AttentionStore, res: int, from_where: List[str], is_cross: bool, select: int, prompts=None):
    out = []
    attention_maps = attention_store.get_average_attention()
    num_pixels = res ** 2
    for location in from_where:
        for item in attention_maps[f"{location}_{'cross' if is_cross else 'self'}"]:
            if item.shape[1] == num_pixels:
                cross_maps = item.reshape(len(prompts), -1, res, res, item.shape[-1])[select]
                out.append(cross_maps)
    out = torch.cat(out, dim=0)
    return out.cpu()


                        
def show_cross_attention(orignial_image,attention_store: AttentionStore, res: int, from_where: List[str], select: int = 0,out_put="./test_1.jpg",image_cnt=0,class_one=None,prompts=None , tokenizer=None,mask_diff=None):
    
    
    orignial_image = orignial_image.copy()
    show = True
    tokens = tokenizer.encode(prompts[select])
    decoder = tokenizer.decode
    attention_maps = aggregate_attention(attention_store, 16, from_where, True, select,prompts=prompts)
    attention_maps = [attention_maps.sum(0) / attention_maps.shape[0]]
    
    attention_maps_32 = aggregate_attention(attention_store, 32, from_where, True, select,prompts=prompts)
    attention_maps_32 = [attention_maps_32.sum(0) / attention_maps_32.shape[0]]
    
    attention_maps_64 = aggregate_attention(attention_store, 64, from_where, True, select,prompts=prompts)
    attention_maps_64 = [attention_maps_64.sum(0) / attention_maps_64.shape[0]]
    
    images = []
    attentions = []
    for idx, (attention_map, attention_map_36,attention_map_64) in enumerate(zip(attention_maps,attention_maps_32,attention_maps_64)):
        gt_kernel_final = np.zeros((512,512), dtype="float32")
        number_gt = 0
        for i in range(len(tokens)):
            class_current = decoder(int(tokens[i])) 

            if class_current not in coco_category_list_check[class_one]:
                continue
            image_16 = attention_map[:, :, i]
            image_16 = cv2.resize(image_16.numpy(), (512, 512), interpolation=cv2.INTER_CUBIC)
            image_16 = image_16 / image_16.max()
            
            image_32 = attention_map_36[:, :, i]
            image_32 = cv2.resize(image_32.numpy(), (512, 512), interpolation=cv2.INTER_CUBIC)
            image_32 = image_32 / image_32.max()
            
            image_64 = attention_map_64[:, :, i]
            image_64 = cv2.resize(image_64.numpy(), (512, 512), interpolation=cv2.INTER_CUBIC)
            image_64 = image_64 / image_64.max()
            
            image = (image_16 + image_32 + image_64) / 3
            

            
            gt_kernel_final += image.copy()
            number_gt += 1

        gt_kernel_final = gt_kernel_final/number_gt
        
        id_ = coco_category_to_id_v1[class_one]
        cam_dict = {}
        cam_dict[id_] = gt_kernel_final
        
        
        base_path="/data_16T/zjp/LXL/OVPS/PBAPS/Visual Prototype/PascalPart116"
        class_dir = os.path.join(base_path, class_one)
        npy_folder=os.path.join(class_dir, 'npy') 
        np.save(os.path.join(npy_folder, out_put.replace('png','npy')), cam_dict)


def show_cross_attention2(orignial_image,attention_store: AttentionStore, res: int, from_where: List[str], select: int = 0,out_put="./test_1.jpg",image_cnt=0,class_one=None,prompts=None , tokenizer=None,mask_diff=None):
    
    
    orignial_image = orignial_image.copy()
    show = True
    tokens = tokenizer.encode(prompts[select])
    decoder = tokenizer.decode
    attention_maps = aggregate_attention(attention_store, 16, from_where, True, select,prompts=prompts)
    attention_maps = [attention_maps.sum(0) / attention_maps.shape[0]]
    
    attention_maps_32 = aggregate_attention(attention_store, 32, from_where, True, select,prompts=prompts)
    attention_maps_32 = [attention_maps_32.sum(0) / attention_maps_32.shape[0]]
    
    attention_maps_64 = aggregate_attention(attention_store, 64, from_where, True, select,prompts=prompts)
    attention_maps_64 = [attention_maps_64.sum(0) / attention_maps_64.shape[0]]
    
    images = []
    attentions = []
    for idx, (attention_map, attention_map_36,attention_map_64) in enumerate(zip(attention_maps,attention_maps_32,attention_maps_64)):
        gt_kernel_final = np.zeros((512,512), dtype="float32")
        number_gt = 0
        for i in range(len(tokens)):
            class_current = decoder(int(tokens[i])) 

            if class_current not in coco_category_list_check2[class_one]:
                continue
            image_16 = attention_map[:, :, i]
            image_16 = cv2.resize(image_16.numpy(), (512, 512), interpolation=cv2.INTER_CUBIC)
            image_16 = image_16 / image_16.max()
            
            image_32 = attention_map_36[:, :, i]
            image_32 = cv2.resize(image_32.numpy(), (512, 512), interpolation=cv2.INTER_CUBIC)
            image_32 = image_32 / image_32.max()
            
            image_64 = attention_map_64[:, :, i]
            image_64 = cv2.resize(image_64.numpy(), (512, 512), interpolation=cv2.INTER_CUBIC)
            image_64 = image_64 / image_64.max()
            
            image = (image_16 + image_32 + image_64) / 3
            
            
            gt_kernel_final += image.copy()
            number_gt += 1

        gt_kernel_final = gt_kernel_final/number_gt
        
        id_ = coco_category_to_id_v1[class_one]
        cam_dict = {}
        cam_dict[id_] = gt_kernel_final
        
        base_path="/data_16T/zjp/LXL/OVPS/PBAPS/Visual Prototype/PascalPart116/Data/All_Class"
        class_dir = os.path.join(base_path, class_one)
        npy_folder=os.path.join(class_dir, 'npy2') 
        np.save(os.path.join(npy_folder, out_put.replace('png','npy')), cam_dict)

    
def run_and_display(prompts, controller, latent=None, run_baseline=False, generator=None,out_put = "",ldm_stable=None):
    if run_baseline:
        print("w.o. prompt-to-prompt")
        images, latent = run_and_display(prompts, EmptyControl(), latent=latent, run_baseline=False, generator=generator, out_put = out_put)
        print("with prompt-to-prompt")
        
    images_here, x_t = ptp_utils.text2image_ldm_stable(ldm_stable, prompts, controller, latent=latent, num_inference_steps=NUM_DIFFUSION_STEPS, guidance_scale=7, generator=generator, low_resource=LOW_RESOURCE)
    
    images, x_t = ptp_utils.text2image_ldm_stable(ldm_stable, prompts, controller, latent=latent, num_inference_steps=NUM_DIFFUSION_STEPS, guidance_scale=200, generator=generator, low_resource=LOW_RESOURCE)

    ptp_utils.view_images(images_here,out_put = out_put)
    return images_here, x_t



def sub_processor(pid , class_list):
    torch.cuda.set_device(pid)
    text = 'processor %d' % pid
    print(text)
    
    MY_TOKEN = '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'
    LOW_RESOURCE = False 
    NUM_DIFFUSION_STEPS = 50
    GUIDANCE_SCALE = 7.5
    MAX_NUM_WORDS = 77
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    ldm_stable = StableDiffusionPipeline.from_pretrained("LXL/OVPS/PBAPS/stable-diffusion-v1-4").to(device)
    # ldm_stable = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", use_auth_token=MY_TOKEN).to(device)
    tokenizer = ldm_stable.tokenizer

    number_per_class = 300   
    image_cnt = pid * (number_per_class*15)
    task = "AttentionStore"
    desc_id = {}
    for class_one in class_list:
        base_path="/data_16T/zjp/LXL/OVPS/PBAPS/Visual Prototype/PascalPart116/Data/All_Class"
        class_dir = os.path.join(base_path, class_one)
        
        if not os.path.exists(class_dir): os.makedirs(class_dir)
        subfolders = ['image', 'npy', 'npy2']
        for subfolder in subfolders:
            subfolder_path = os.path.join(class_dir, subfolder)
            if not os.path.exists(subfolder_path): os.makedirs(subfolder_path)
        image_folder=os.path.join(class_dir, 'image')    
        npy_folder=os.path.join(class_dir, 'npy') 
        npy2_folder=os.path.join(class_dir, 'npy2')         

        for rand in range(number_per_class):
            image_p = os.path.join(image_folder,"image_{}_{}.jpg".format(class_one,image_cnt+1))
            if os.path.exists(image_p): 
                image_cnt+=1
                continue
            print(image_p)
            g_cpu = torch.Generator().manual_seed(rand)

            prompts = ["Photo of a " + class_one]
            
            if task == "AttentionStore":
                controller = AttentionStore()
                image_cnt+=1
                image, x_t = run_and_display(prompts, controller, latent=None, run_baseline=False, generator=g_cpu,out_put = os.path.join(image_folder,"image_{}_{}.jpg".format(class_one,image_cnt)),ldm_stable=ldm_stable)
                # sheep head only consider head
                show_cross_attention(image[0].copy(),controller,res=32, from_where=("up", "down"),out_put = "image_{}_{}.png".format(class_one,image_cnt),image_cnt=image_cnt,class_one=class_one,prompts=prompts,tokenizer=tokenizer)
                # sheep head consider sheep and head
                show_cross_attention2(image[0].copy(),controller,res=32, from_where=("up", "down"),out_put = "image_{}_{}.png".format(class_one,image_cnt),image_cnt=image_cnt,class_one=class_one,prompts=prompts,tokenizer=tokenizer)
            
            

if __name__ == '__main__':
    
    
    result_dict = mp.Manager().dict()
    mp = mp.get_context("spawn")
    thread_num = 8
    processes = []
    per_thread_video_num = int(len(coco_category_list)/thread_num)

    print('Start inference')
    for i in range(thread_num):
        if i == thread_num - 1:
            sub_video_list = coco_category_list[i * per_thread_video_num:]
        else:
            sub_video_list = coco_category_list[i * per_thread_video_num: (i + 1) * per_thread_video_num]

        p = mp.Process(target=sub_processor, args=(i, sub_video_list))
        p.start()
        processes.append(p)


    for p in processes:
        p.join()

    result_dict = dict(result_dict)
    

# CUDA_VISIBLE_DEVICES=0,1,2,3,6,7,8,9 /data_16T/zjp/envs/ovpp/bin/python "/data_16T/zjp/LXL/OVPS/PBAPS/Visual Prototype/1_diffus_generate_img_attention.py"

