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

def main(args):
    cfg = Config.fromfile(args.config_dir)

    epoch = args.epoch

    # dataset config 수정
    cfg.data.test.pipeline[1]['img_scale'] = (512,512) # Resize
    cfg.data.test.test_mode = True

    cfg.data.samples_per_gpu = 4

    cfg.optimizer_config.grad_clip = dict(max_norm=35, norm_type=2)
    cfg.model.train_cfg = None

    # checkpoint path
    checkpoint_path = os.path.join(cfg.work_dir, f'{epoch}.pth')

    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
            dataset,
            samples_per_gpu=1,
            workers_per_gpu=cfg.data.workers_per_gpu,
            dist=False,
            shuffle=False)

    model = build_segmentor(cfg.model, test_cfg=cfg.get('test_cfg'))
    checkpoint = load_checkpoint(model, checkpoint_path, map_location='cpu')

    model.CLASSES = dataset.CLASSES
    model = MMDataParallel(model.cuda(), device_ids=[0])

    output = single_gpu_test(model, data_loader)
    
    # sample_submisson.csv 열기
    submission = pd.read_csv('/opt/ml/segmentation/baseline/sample_submission.csv', index_col=None)
    json_dir = os.path.join("/opt/ml/segmentation/input/data/test.json")
    with open(json_dir, "r", encoding="utf8") as outfile:
        datas = json.load(outfile)

    input_size = 512
    output_size = 256
    bin_size = input_size // output_size
            
    # PredictionString 대입
    for image_id, predict in enumerate(output):
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
    submission.to_csv(os.path.join(cfg.work_dir, f'submission_{epoch}.csv'), index=False)

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_dir",
        type=str,
        default=None
    )
    parser.add_argument(
        "--epoch",
        type=str,
        default='latest'
    )
    args = parser.parse_args()

    if args.config_dir is None:
        raise NameError('set config directory path')
    
    main(args)