import pandas as pd
import numpy as np
import argparse
import os
from collections import Counter
from tqdm import tqdm

def main(args):
    ensemble_path = args.ensemble_dir
    ensemble_list = os.listdir(ensemble_path)
    ensemble_list = sorted(ensemble_list)
    ensemble_weights = eval(args.weight)
    print(ensemble_list)

    ensembles = [pd.read_csv(os.path.join(ensemble_path, _)) for _ in ensemble_list]
    ensemble_cnt = len(ensembles)
    print("success load csv files")

    submission = pd.read_csv('/opt/ml/segmentation/baseline/sample_submission.csv', index_col=None)

    print("ensembling ...")
    for i in tqdm(range(819)):
        ensemble_result = []
        file_name = ensembles[0]['image_id'][i]
        ps = np.array([j['PredictionString'][i].split() for j in ensembles])
        for j in range(65536):
            vote = Counter(ps[:, j])
            for k in vote:
                score_idx = np.where(ps[:, j] == k)[0]
                vote[k] = [vote[k], sum([ensemble_weights[_] for _ in score_idx])/len(score_idx)]
            vote = sorted(vote.items(), key=lambda x : (x[1][0], x[1][1]), reverse=True)
            ensemble_result.append(vote[0][0])
            
        submission = submission.append({"image_id" : file_name, "PredictionString" : ' '.join(e for e in ensemble_result)}, 
                                        ignore_index=True)

    submission.to_csv(os.path.join(args.save_dir, args.name))

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ensemble_dir", type=str, default='./ensemble_csv')
    parser.add_argument("--weight", type=str, default=None)
    parser.add_argument("--save_dir", type=str, default='/opt/ml/segmentation/baseline/mmsegmentation/ensemble_result')
    parser.add_argument("--name", type=str, default='ensemble0.csv')
    args = parser.parse_args()

    main(args)
