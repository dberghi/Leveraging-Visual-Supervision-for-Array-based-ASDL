#!/usr/bin/python

import numpy as np
import argparse, random, glob, os
from pathlib import Path
import core.config as conf
from core.dataset import dataset_from_hdf5
from core.models import Net
from core.train_val_functions import train, val
import utils.utils as utils
# PyTorch libraries and modules
import torch
from torch.utils.data import DataLoader, Subset


base_path = conf.input['project_path']
dev_seqs_path = base_path + 'data/TragicTalkers/audio/development'


def main():
    train_file = conf.input['h5py_path'] + 'h5py_%s/development_dataset.h5' % (args.info)
    scaler_path = conf.input['h5py_path'] + 'h5py_%s/feature_scaler.h5' % (args.info)

    ## ---------- Experiment reproducibility --------
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    ## ------------ Set check point directory ----------------
    if args.fold_bool:
        ckpt_dir = Path(args.ckpt_dir + '%s/%f/%d' % (args.info, args.lr, args.fold))
    else:
        ckpt_dir = Path(args.ckpt_dir + '%s/%f' % (args.info, args.lr))
    ckpt_dir.mkdir(exist_ok=True, parents=True)

    ## --------------- Set device --------------------------
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(device)
    print(device, file=open('%s/log.txt' % ckpt_dir, "a"))

    ## ------------- divide the dataset in folds if fold_bool true ----------
    if args.fold_bool:
        tr = sorted(glob.glob(dev_seqs_path + '/*'))
        dataset_list = []
        for f in tr:
            dataset_list.append(os.path.basename(f))
        folded_dataset = utils.n_fold_generator(dataset_list, fold_num=5) # random seed ensures random consistency

    ## ------------- Get feature scaler -----------------
    if args.normalize:
        mean, std = utils.load_feature_scaler(scaler_path)
    else:
        mean = None
        std = None
    ## ------------- Data loaders -----------------
    full_train_set = dataset_from_hdf5(train_file, normalize=args.normalize, mean=mean, std=std)

    if args.fold_bool:
        sequences_list = []
        # load sequence names
        print('Loading sequence names from ds')
        print('Loading sequence names from ds', file=open('%s/log.txt' % ckpt_dir, "a"))
        for im in full_train_set:
            sequences_list.append(im[5].decode("utf-8"))

        #for fold in folded_dataset:
        print(folded_dataset[args.fold])
        print(folded_dataset[args.fold], file=open('%s/log.txt' % ckpt_dir, "a"))

        val_idx = [idx for idx, element in enumerate(sequences_list) if
                   utils.belong_to_val(element, folded_dataset[args.fold])]
        train_idx = [idx for idx, element in enumerate(sequences_list) if
                     not utils.belong_to_val(element, folded_dataset[args.fold])]
        d_val = Subset(full_train_set, val_idx)
        d_train = Subset(full_train_set, train_idx)

        dl_val = DataLoader(d_val, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
        dl_train = DataLoader(d_train, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)

    else: # use entire development set for training
        dl_train = DataLoader(full_train_set, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)


    #print('confidence loss mode: %s' % args.loss_mode)
    #print('confidence loss mode: %s' % args.loss_mode, file=open('%s/log.txt' % ckpt_dir, "a"))

    model = Net().to(device)

    ## ---------- Look for previous check points -------------
    if args.resume_training:
        ckpt_file = utils.get_latest_ckpt(ckpt_dir)
        if ckpt_file:
            model.load_state_dict(torch.load(ckpt_file))
            first_epoch = int(str(ckpt_file)[-8:-5])
            print('Resuming training from epoch %d' % first_epoch)
            print('Resuming training from epoch %d' % first_epoch, file=open('%s/log.txt' % ckpt_dir, "a"))
        else:
            print('No checkpoint found in "{}"...'.format(ckpt_dir))
            print('No checkpoint found in "{}"...'.format(ckpt_dir), file=open('%s/log.txt' % ckpt_dir, "a"))
            first_epoch = 1
    else:
        print('Resume training deactivated')
        print('Resume training deactivated', file=open('%s/log.txt' % ckpt_dir, "a"))
        first_epoch = 1


    opt = conf.training_param['optimizer']
    optimizer = opt(model.parameters(), lr=args.lr)

    if args.fold_bool:
        plot_dir = Path(base_path + 'output/training_plots/%s/%f/%d' % (args.info, args.lr, args.fold))
    else:
        plot_dir = Path(base_path + 'output/training_plots/%s/%f' % (args.info, args.lr))
    plot_dir.mkdir(exist_ok=True, parents=True)

    if first_epoch == 1:
        train_loss = []
        if args.fold_bool: val_loss = []
    else: # training resumed
        train_loss = torch.load('%s/train_loss.pt' % plot_dir)
        if args.fold_bool: val_loss = torch.load('%s/val_loss.pt' % plot_dir)

    ## --------------- TRAIN ------------------------
    for epoch in range(first_epoch, args.epochs+1):

        # update learning rate
        if epoch > 30:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= (0.90 ** (epoch - 30))
                print('Learning rate: %f' %param_group['lr'])

        train_loss.append(train(model, optimizer, dl_train, args, device, ckpt_dir, epoch))

        if args.fold_bool:
            print('Val forward pass...')
            val_loss.append(val(model, dl_val, args, device, ckpt_dir))

        # save train_loss and val_loss lists
        torch.save(train_loss, plot_dir / 'train_loss.pt')
        if args.fold_bool: torch.save(val_loss, plot_dir / 'val_loss.pt')





if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser(description='Parse arguments for training')
    parser.add_argument('--batch-size', type=int, default=conf.training_param['batch_size'], metavar='N',
                        help='input batch size for training (default: %d)' % conf.training_param['batch_size'])
    parser.add_argument('--epochs', type=int, default=conf.training_param['epochs'], metavar='N',
                        help='number of epochs to train (default: %d)' % conf.training_param['epochs'])
    parser.add_argument('--lr', type=float, default=conf.training_param['learning_rate'], metavar='LR',
                        help='learning rate (default: %f)' % conf.training_param['learning_rate'])
    #parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
    #                    help='SGD momentum (default: 0.5)')
    parser.add_argument('--normalize', default=True, action='store_true',
                        help='set True to normalize dataset with mean and std of train set')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=42, metavar='S',
                        help='random seed (default: 42)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--ckpt-dir', default=conf.input['project_path'] + 'ckpt/', help='path to save models')
    parser.add_argument('--resume-training', default=True, action='store_true',
                        help='resume training from latest checkpoint')
    parser.add_argument('--info', type=str, default='default', metavar='S',
                        help='Add additional info for storing')
    parser.add_argument('--fold', type=int, default=5, metavar='S',
                        help='number of folds used when fold-bool=True (default: 5)')
    parser.add_argument('--fold-bool', default=False, action='store_true',
                        help='set True if k-fold validation wanted, else the entire train set is used')
    args = parser.parse_args()

    main()
