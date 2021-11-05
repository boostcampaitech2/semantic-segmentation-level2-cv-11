import argparse
import os
import mmcv

from mmcv import Config
from mmseg.datasets import build_dataloader, build_dataset
from mmseg.models import build_segmentor
from mmseg.apis import single_gpu_test
from mmcv.runner import load_checkpoint
from mmcv.parallel import MMDataParallel

import pandas as pd
import numpy as np
import json
import torch

from tqdm import tqdm
import pickle

''' 
before start you have to change code mmsegmentation/mmseg/models/segmentors/encoder_decoder.py
to change return softmax score not argmax
'''
def main(args):
    models_path = args.ensemble_dir
    save_path = args.save_dir
    ensemble_models = os.listdir(models_path)
    ensemble_models = sorted(ensemble_models, reverse=True)
    ensemble_score = np.zeros((819, 11, 512, 512))

    print("start ensemble...")
    for number in ensemble_models:
        model_path = os.path.join(models_path, number)
        model, cfg = sorted(os.listdir(model_path), key=lambda x: x[-1])
        cfg = Config.fromfile(os.path.join(model_path, cfg))
        cfg.data.test.img_dir = '/opt/ml/segmentation/mmseg_data/test/img'
        cfg.work_dir = '/opt/ml/segmentation/baseline/mmsegmentation/ensemble_model/final/52'
        cfg.data.test.test_mode = True
        cfg.data.samples_per_gpu = 4
        cfg.data.workers_per_gpu = 4
        cfg.optimizer_config.grad_clip = dict(max_norm=35, norm_type=2)
        cfg.model.train_cfg = None

        checkpoint_path = os.path.join(cfg.work_dir, model)

        dataset = build_dataset(cfg.data.test)
        data_loader = build_dataloader(
                dataset,
                samples_per_gpu=4,
                workers_per_gpu=4,
                dist=False,
                shuffle=False)

        model = build_segmentor(cfg.model, test_cfg=cfg.get('test_cfg'))
        checkpoint = load_checkpoint(model, checkpoint_path, map_location='cpu')

        model.CLASSES = dataset.CLASSES
        model = MMDataParallel(model.cuda(), device_ids=[0])

        output = single_gpu_test(model, data_loader)
        with open(os.path.join(save_path, f'{number}.p'), 'wb') as file:
            pickle.dump(output, file)
        for x in range(819):
            ensemble_score[x] += output[x]
        

    
    # ensemble_score /= len(ensemble_models)
    ensemble_result = ensemble_score.argmax(axis=1)
    print("ensemble done!")

    submission = pd.read_csv('/opt/ml/segmentation/baseline/sample_submission.csv', index_col=None)
    json_dir = os.path.join("/opt/ml/segmentation/input/data/test.json")
    with open(json_dir, "r", encoding="utf8") as outfile:
        datas = json.load(outfile)
    input_size = 512
    output_size = 256
    bin_size = input_size // output_size

    print("make csv_file...")
    for image_id, predict in enumerate(ensemble_result):
        image_id = datas["images"][image_id]
        file_name = image_id["file_name"]
        
        temp_mask = []
        predict = predict.reshape(1, 512, 512)
        mask = predict.reshape((1, output_size, bin_size, output_size, bin_size)).max(4).max(2) # resize to 256*256
        temp_mask.append(mask)
        oms = np.array(temp_mask)
        oms = oms.reshape([oms.shape[0], output_size*output_size]).astype(int)

        string = oms.flatten()

        submission = submission.append({"image_id" : file_name, "PredictionString" : ' '.join(str(e) for e in string.tolist())}, 
                                    ignore_index=True)

    # submission.csv로 저장
    submission.to_csv(os.path.join(save_path, args.name), index=False)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ensemble_dir", type=str, default='./ensemble_model/final')
    # parser.add_argument("--weight", type=str, default=None)
    parser.add_argument("--save_dir", type=str, default='/opt/ml/segmentation/baseline/mmsegmentation/ensemble_result')
    parser.add_argument("--name", type=str, default='ensemble_result7.csv')
    args = parser.parse_args()
    main(args)