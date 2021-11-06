import os
import warnings 
warnings.filterwarnings('ignore')

import copy
from torch.utils.data import Dataset
import cv2


import numpy as np
from pycocotools.coco import COCO


category_names = ['Backgroud', 'General trash', 'Paper', 'Paper pack', 'Metal', 'Glass', 'Plastic', 'Styrofoam', 'Plastic bag', 'Battery', 'Clothing']
dataset_path = './input/data'

def get_classname(classID, cats):
    for i in range(len(cats)):
        if cats[i]['id']==classID:
            return cats[i]['name']
    return "None"

class CustomDataLoader(Dataset):
    """COCO format"""
    def __init__(self, data_dir, mode = 'train', transform = None, preprocessing=None):
        super().__init__()
        self.mode = mode
        self.transform = transform
        self.coco = COCO(data_dir)
        self.preprocessing = preprocessing
        
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
            # General trash = 1, ... , Cigarette = 10
            anns = sorted(anns, key=lambda idx : idx['area'], reverse=True)
            for i in range(len(anns)):
                className = get_classname(anns[i]['category_id'], cats)
                pixel_value = category_names.index(className)
                masks[self.coco.annToMask(anns[i]) == 1] = pixel_value
            masks = masks.astype(np.int8)

            #==================
            #Add copy and paste
            #==================
            execute_ran = np.random.randint(10)
            
            if self.mode == 'train' and execute_ran < 7:

                cat_num = np.random.randint(10)     
                category=None
                if cat_num >= 7:
                    category='clothing'
                elif cat_num >= 5:
                    category='battery'
                elif cat_num == 4:
                    category='glass'
                elif cat_num == 3:
                    category='metal'
                elif cat_num <= 2:
                    category='paper pack'

                base_dir = f'{dataset_path}/add_imgs/{category}'
                img_list = os.listdir(base_dir)
                idx_num = np.random.randint(len(img_list))

                img = cv2.imread(os.path.join(base_dir, img_list[idx_num]))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                img2gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
                _, img2_mask = cv2.threshold(img2gray, 10, 255, cv2.THRESH_BINARY)
                mask_inv = cv2.bitwise_not(img2_mask)

                nimg = img.astype(np.float32)
                nimg /= 255.0

                real_mask = np.where(img2_mask==255,cat_num, img2_mask) #category 10

                new_img = cv2.bitwise_and(images, images, mask=mask_inv) + nimg
                new_mask = cv2.bitwise_and(masks, masks, mask=mask_inv) + real_mask
            else:
                new_img = images
                new_mask = masks
        
            # transform -> albumentations 라이브러리 활용
            if self.transform is not None:
                transformed = self.transform(image=new_img, mask=new_mask)
                images = transformed["image"]
                masks = transformed["mask"]
            if self.preprocessing:
                sample = self.preprocessing(image=images, mask=masks)
                images, masks = sample['image'], sample['mask']
                
            return images, masks, image_infos
        
        if self.mode == 'test':
            # transform -> albumentations 라이브러리 활용
            if self.transform is not None:
                transformed = self.transform(image=images)
                images = transformed["image"]
            if self.preprocessing:
                sample = self.preprocessing(image=images)
                images = sample['image']
            return images, image_infos
    
    def __len__(self) -> int:
        # 전체 dataset의 size를 return
        return len(self.coco.getImgIds())