#!/usr/bin/python

import numpy as np
import csv, argparse, json
from pathlib import Path
from tqdm import tqdm
import core.config as conf
from core.dataset import dataset_from_scratch
from core.models import Net
import utils.utils as utils
# PyTorch libraries and modules
import torch
from torch.utils.data import DataLoader

base_path = conf.input['project_path']

test_file = base_path + 'data/csv/test_forward.csv'
json_path = base_path + 'data/csv/seq_lengths.json'


def main():
    scaler_path = conf.input['h5py_path'] + 'h5py_%s/feature_scaler.h5' % (args.info)
    normalize = args.normalize
    lr = args.lr
    epoch = args.epoch

    ## ----------- load json file with sequences' lengths
    f = open(json_path)
    lengths = json.load(f)

    ## ----------- Set device --------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    ## ------------- Get feature scaler -----------------
    if args.normalize:
        mean, std = utils.load_feature_scaler(scaler_path)
    else:
        mean = None
        std = None

    ## ---------- Data loaders -----------------
    d_test = dataset_from_scratch(test_file, 'test', normalize=normalize, mean=mean, std=std)
    dl_test = DataLoader(d_test, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)

    ## ---------- Load model weights -------------
    model = Net().to(device)
    model.load_state_dict(torch.load(base_path + 'ckpt/%s/%f/model_%03d.ckpt' % (args.info, lr, epoch),
                                     map_location=torch.device(device)))


    csv_out = Path(base_path + 'output/forward/%s/%f' % (args.info, lr))
    csv_out.mkdir(exist_ok=True, parents=True)

    model.eval()

    with open('%s/test_forward.csv' %csv_out, 'w') as file_write:
        writer = csv.writer(file_write)
        writer.writerow(['name', 'time', 'predicted x', 'confidence'])
        for count, image in enumerate(tqdm(dl_test)):
            name = image[2]
            name = ''.join(name)
            #print(name)
            cam = image[1]
            initial_time = float(image[3])
            tensor = image[0]
            output = model(tensor.to(device), cam.to(device))
            sequence_length = np.round(float(lengths[name[:-6]]), 2)
            for t in range(output.size(1)):
                current_time = initial_time + (t / conf.input['fps'])
                current_time = np.round(current_time, 2)
                if current_time < sequence_length:
                    position = float(output[0,t,0])
                    confidence = float(output[0,t,1])
                    writer.writerow([name, current_time, position, confidence])




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test set forward pass, specify arguments')
    parser.add_argument('--epoch', type=int, default=conf.training_param['epochs'], metavar='N',
                        help='number of epochs (default: %d)' % conf.training_param['epochs'])
    parser.add_argument('--lr', type=float, default=conf.training_param['learning_rate'], metavar='LR',
                        help='learning rate (default: %f)' % conf.training_param['learning_rate'])
    parser.add_argument('--normalize', default=True, action='store_true',
                        help='set True to normalize dataset with mean and std of train set')
    parser.add_argument('--info', type=str, default='default', metavar='S',
                        help='Add additional info for storing')
    args = parser.parse_args()
    main()
