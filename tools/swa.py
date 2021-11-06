import torch
import os
import argparse
from collections import OrderedDict

def main(args):
    swa_path = args.swa_dir
    swa_list = os.listdir(swa_path)
    print(swa_list)
    sample = torch.load(os.path.join(args.swa_dir,swa_list[0]))
    swa_len = len(swa_list)
    state_dict_list = []
    for swa in swa_list:
        model = torch.load(os.path.join(args.swa_dir,swa))
        state_dict = model['state_dict']
        state_dict_list.append(state_dict)
    
    result = OrderedDict()
    
    for k in state_dict_list[0]:
        temp = 0
        for i in range(swa_len):
            temp += state_dict_list[i][k]
        result[k] = temp
        result[k] =result[k]/float(swa_len)
    sample['state_dict'] = result
    torch.save(sample,'./swa.pth')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--swa_dir',type=str,default='./swa')
    args = parser.parse_args()

    main(args)