from class_graph import class_graph

import os
import torch
import numpy as np
from sklearn.cluster import KMeans
from tqdm import tqdm
import logging
from multiprocessing import Pool, cpu_count



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


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
data_root = "/data_16T/zjp/LXL/OVPS/PBAPS/Visual Prototype/PascalPart116/Data/All_Class"
output_root = "/data_16T/zjp/LXL/OVPS/PBAPS/Visual Prototype/PascalPart116/Category_feature2"

n_clusters = 8

def load_features(class_name):
    feature_dir = os.path.join(data_root, class_name, "feature2")
    features = []
    for feature_file in os.listdir(feature_dir):
        if feature_file.endswith(".pt"):
            feature_path = os.path.join(feature_dir, feature_file)
            try:
                feature = torch.load(feature_path)
                features.append(feature.numpy())
            except Exception as e:
                logging.error(f"Error loading {feature_path}: {e}")
    return np.vstack(features) if features else np.array([])

def get_descendants(class_name, part_matrix_index):
    graph = class_graph[class_name]
    matrix = graph["matrix"]
    part_index = graph["index"]
    
    descendants = []
    stack = [part_matrix_index]
    while stack:
        current = stack.pop()
        for i, relation in enumerate(matrix[current]):
            if relation > 0 and relation not in descendants:  
                    descendants.append(part_index[relation])
                    stack.append(relation)
    return descendants

def compute_category_feature(part_indices):
    all_features = []
    for part_index in part_indices:
        part_name = id_to_category[part_index] 
        features = load_features(part_name)
        if features.size > 0:
            all_features.append(features)

    if all_features:
        all_features = np.vstack(all_features)
        kmeans = KMeans(n_clusters=n_clusters, random_state=0)
        kmeans.fit(all_features)
        return kmeans.cluster_centers_
    else:
        return np.array([])

def save_category_feature(class_name, category_feature):
    output_dir = os.path.join(output_root, class_name)
    os.makedirs(output_dir, exist_ok=True)
    for i, feature in enumerate(category_feature):
            output_path = os.path.join(output_dir, f"feature_{i}.pt")
            torch.save(torch.tensor(feature), output_path)

def process_class(class_name):
    part_indices = class_graph[class_name]["index"][1:]  
    category_feature = compute_category_feature(part_indices)
    if category_feature.size > 0:
        save_category_feature(class_name, category_feature)
        logging.info(f"Processed class: {class_name}")
    else:
        logging.warning(f"No features found for class: {class_name}")

def process_part(class_name, index, part_index):
    logging.info(f"Processing part: {id_to_category[part_index]}")
    descendants = get_descendants(class_name, index + 1)
    part_indices = [part_index] + descendants
    logging.info(f"descendants: {part_indices}")
    category_feature = compute_category_feature(part_indices)
    if category_feature.size > 0:
        part_name = id_to_category[part_index]  
        save_category_feature(part_name, category_feature)
        # logging.info(f"Processed part: {part_name}")
    else:
        logging.warning(f"No features found for part: {part_name}")

def main():
    with Pool(cpu_count()) as pool:
        list(tqdm(pool.imap(process_class, class_graph.keys()), desc="Processing object categories", total=len(class_graph)))

    part_tasks = []
    for class_name in class_graph.keys():
        graph = class_graph[class_name]
        for index, part_index in enumerate(graph["index"][1:]):  
            part_tasks.append((class_name, index, part_index))

    with Pool(cpu_count()) as pool:
        list(tqdm(pool.starmap(process_part, part_tasks), desc="Processing part categories", total=len(part_tasks)))

if __name__ == "__main__":
    main()


# /data_16T/zjp/envs/ovpp/bin/python "/data_16T/zjp/LXL/OVPS/PBAPS/Visual Prototype/PascalPart116/Code/3_generate_visualprototype.py"