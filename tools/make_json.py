import json
import numpy as np
import pandas as pd
import cv2
import albumentations as A
from matplotlib import pyplot as plt
from albumentations.pytorch import ToTensorV2
import random
import copy

def make(source_json, out_json, category_num):
    
    base_dir = "/opt/ml/segmentation/semantic-segmentation-level2-cv-11/input/data/"
    json_dir = f"{base_dir}{source_json}"

    with open(json_dir, "r") as json_file:

        data = json.load(json_file)

    data_anns = data['annotations']
    ann_list = []
    img_id_list = [] 

    for ann in data_anns:

        if ann['category_id']  == category_num: #category
            ann_list.append(ann)
            img_id_list.append(ann['image_id'])

    img_ids = set(img_id_list)
    #=====================img anns 매치될때 씀=====================
    img_list = []
    data_imgs = data['images']
    for img in data_imgs:
        if img['id'] in img_ids:
            img_list.append(img)

    img_idx = 0
    ann_idx = 0
    new_datas = []
    new_data_ann = []
    while ann_idx < len(ann_list):

        if img_list[img_idx]['id'] == ann_list[ann_idx]['image_id']:

            new_ann = copy.deepcopy(ann_list[ann_idx])
            new_ann['image_id'] = img_idx
            new_ann['id'] = ann_idx
            new_ann['category_id'] = 1 #only clothing & background
            new_data_ann.append(new_ann)

            ann_idx += 1
        else:

            new_img = copy.deepcopy(img_list[img_idx])
            new_img['id'] = img_idx
            new_datas.append(new_img)

            img_idx += 1
            
            
    new_cat = data['categories'][-1]
    new_cat['id'] = 1

    last_json = {}
    last_json['images'] = new_datas
    last_json['annotations'] = new_data_ann
    last_json['categories'] = [new_cat]
    
    with open(f'{base_dir}{out_json}', 'w') as make_file:

        json.dump(last_json, make_file, indent='\t')


if __name__ == '__main__':
    pareser = argparse.ArgumentParser()
    parser.add_argument(
        "--original_json",
        type=str
        default='train_all.json'
    )
    parser.add_argument(
        "--extract_json",
        type=str
    )
    parser.add_argument(
        "--category_num",
        type=num
    )

#  Category number
# =====================
#  General transh : 1
#  Paper : 2
#  Paper pack : 3
#  Metal : 4
#  Glass : 5
#  Plastic : 6
#  Styrofoam : 7
#  Plasitc bag : 8
#  Battery : 9
#  Clothing : 10
args = parser.parse_args()
make(args.original_json, args.extract_json, args.category_num)
