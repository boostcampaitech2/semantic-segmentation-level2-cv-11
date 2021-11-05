import os
import random
import time
import json
import warnings 
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import cv2

import numpy as np
import pandas as pd
from tqdm import tqdm

# 전처리를 위한 라이브러리
from pycocotools.coco import COCO
import torchvision
import torchvision.transforms as transforms

import albumentations as A
from albumentations.pytorch import ToTensorV2

# 시각화를 위한 라이브러리
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
from matplotlib.patches import Patch
import webcolors

# seed 고정
random_seed = 21
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed) # if use multi-GPU
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(random_seed)
random.seed(random_seed)

dataset_path = "/opt/ml/segmentation/input/data"
anns_file_path = os.path.join(dataset_path, 'train_all.json')

with open(anns_file_path, 'r') as f:
    dataset = json.loads(f.read())

categories = dataset['categories']
anns = dataset['annotations']
imgs = dataset['images']
nr_cats = len(categories)
nr_annotations = len(anns)
nr_images = len(imgs)

# Load categories and super categories
cat_names = []
super_cat_names = []
super_cat_ids = {}
super_cat_last_name = ''
nr_super_cats = 0
for cat_it in categories:
    cat_names.append(cat_it['name'])
    super_cat_name = cat_it['supercategory']
    # Adding new supercat
    if super_cat_name != super_cat_last_name:
        super_cat_names.append(super_cat_name)
        super_cat_ids[super_cat_name] = nr_super_cats
        super_cat_last_name = super_cat_name
        nr_super_cats += 1

cat_histogram = np.zeros(nr_cats,dtype=int)
for ann in anns:
    cat_histogram[ann['category_id']-1] += 1
df = pd.DataFrame({'Categories': cat_names, 'Number of annotations': cat_histogram})
df = df.sort_values('Number of annotations', 0, False)
# category labeling 
sorted_temp_df = df.sort_index()

# background = 0 에 해당되는 label 추가 후 기존들을 모두 label + 1 로 설정
sorted_df = pd.DataFrame(["Backgroud"], columns = ["Categories"])
sorted_df = sorted_df.append(sorted_temp_df, ignore_index=True)
category_names = list(sorted_df.Categories)

def get_classname(classID, cats):
    for i in range(len(cats)):
        if cats[i]['id']==classID:
            return cats[i]['name']
    return "None"

class CustomDataLoader(Dataset):
    """COCO format"""
    def __init__(self, data_dir, mode = 'train', transform = None):
        super().__init__()
        self.mode = mode
        self.transform = transform
        self.coco = COCO(data_dir)
        
    def __getitem__(self, index: int):
        # dataset이 index되어 list처럼 동작
        image_id = self.coco.getImgIds(imgIds=index)
        image_infos = self.coco.loadImgs(image_id)[0]
        
        # cv2 를 활용하여 image 불러오기
        images = cv2.imread(os.path.join(dataset_path, image_infos['file_name']))
        images = cv2.cvtColor(images, cv2.COLOR_BGR2RGB).astype(np.float32)
        images /= 255.0
        
        if (self.mode in ('train', 'val')):
            ann_ids = self.coco.getAnnIds(imgIds=image_infos['id'])
            anns = self.coco.loadAnns(ann_ids)

            # Load the categories in a variable
            cat_ids = self.coco.getCatIds()
            cats = self.coco.loadCats(cat_ids)

            # masks : size가 (height x width)인 2D
            # 각각의 pixel 값에는 "category id" 할당
            # Background = 0
            masks = np.zeros((image_infos["height"], image_infos["width"]))
            anns = sorted(anns, key=lambda idx : idx['area'], reverse=True)
            for i in range(len(anns)):
                className = get_classname(anns[i]['category_id'], cats)
                pixel_value = category_names.index(className)
                masks[self.coco.annToMask(anns[i]) == 1] = pixel_value
            masks = masks.astype(np.int8)
                        
            # transform -> albumentations 라이브러리 활용
            if self.transform is not None:
                transformed = self.transform(image=images, mask=masks)
                images = transformed["image"]
                masks = transformed["mask"]
            return images, masks, image_infos
        
        if self.mode == 'test':
            # transform -> albumentations 라이브러리 활용
            if self.transform is not None:
                transformed = self.transform(image=images)
                images = transformed["image"]
            return images, image_infos
    
    def __len__(self) -> int:
        # 전체 dataset의 size를 return
        return len(self.coco.getImgIds())

def collate_fn(batch):
    return tuple(zip(*batch))


train_path = dataset_path + '/train.json'
val_path = dataset_path + '/val.json'

train_transform = A.Compose([
                            ToTensorV2()
                            ])
train_dataset = CustomDataLoader(data_dir=val_path, mode='val', transform=train_transform)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                           batch_size=1,
                                           shuffle=False,
                                           num_workers=4,
                                           collate_fn=collate_fn)


# start copy and paste
trash_imgs = []
img_dir = '/opt/ml/segmentation/mmseg_data/cp_val'
cnt = 0
for imgs, masks, image_infos in train_loader:
    image_infos = image_infos[0]
    imgs = imgs[0].numpy()
    masks = masks[0].numpy()
    objs = [int(i) for i in list(np.unique(masks))]
    # find class obj to copy and paste
    if 4 in objs:
        y, x = np.where(masks==4)
        trash_imgs.append((y, x, imgs, 4))
    if 1 in objs:
        y, x = np.where(masks==1)
        trash_imgs.append((y, x, imgs, 1))
    if 10 in objs:
        y, x = np.where(masks==10)
        trash_imgs.append((y, x, imgs, 10))
    if 6 in objs:
        y, x = np.where(masks==6)
        trash_imgs.append((y, x, imgs, 6))
    if 3 in objs:
        y, x = np.where(masks==3)
        trash_imgs.append((y, x, imgs, 6))
    
    # find no any class obj to copy and paste in img
    if all([False if obj in [1, 3, 4, 6, 10] else True for obj in objs]):
        for paste_y, paste_x, paste_imgs, paste_class in trash_imgs:
            min_x, min_y, max_x, max_y = min(paste_x), min(paste_y), max(paste_x), max(paste_y)
            # give offset to locate random position of img
            try:
                offset_x, offset_y = random.randrange(-min_x+1, 512-max_x-1), random.randrange(-min_y+1, 512-max_y-1)
                paste_y_offset = paste_y + offset_y
                paste_x_offset = paste_x + offset_x
            except:
                paste_y_offset = paste_y
                paste_x_offset = paste_x
            imgs[:, paste_y_offset, paste_x_offset] = paste_imgs[:, paste_y, paste_x]
            masks[paste_y_offset, paste_x_offset] = paste_class
            plt.imsave(os.path.join(img_dir[:-7], f"cp_only/val_{image_infos['id']:04}.jpg"), imgs.transpose([1,2,0]))
        trash_imgs = []
    plt.imsave(os.path.join(img_dir, f"img/{image_infos['id']:04}.jpg"), imgs.transpose([1,2,0]))
    cv2.imwrite(os.path.join(img_dir, f"ann/{image_infos['id']:04}.png"), masks)