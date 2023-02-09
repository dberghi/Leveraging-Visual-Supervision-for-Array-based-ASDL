#!/usr/bin/python

import numpy as np
import matplotlib.pyplot as plt
import argparse
#import core.config as conf
# PyTorch libraries and modules
import torch


#base_path = conf.input['project_path']

fold_num = 5

def main():
    lr = args.lr
    info = args.info
    #plot_path = base_path + 'output/training_plots/' + info
    plot_path = 'output/training_plots/' + info

    train_loss = []
    val_loss = []
    for fold in range(fold_num):
        train_loss.append(torch.load(plot_path + ('/%f/%d/train_loss.pt') %(lr,fold)))
        val_loss.append(torch.load(plot_path + ('/%f/%d/val_loss.pt') % (lr, fold)))

    train_l = np.asarray(train_loss)
    val_l = np.asarray(val_loss)
    # Average across folds
    train_l = np.mean(train_l, axis=0)
    val_l = np.mean(val_l, axis=0)

    train_loss = list(train_l)
    val_loss = list(val_l)

    torch.save(train_loss, plot_path + ('/%f/global_train_loss.pt') %(lr))
    torch.save(val_loss, plot_path + ('/%f/global_val_loss.pt') %(lr))

    for epoch in range(len(train_loss)):
        print('Ep %d: Train loss: %f' %(epoch+1, train_loss[epoch]), file=open(plot_path + ('/%f/log.txt') %(lr), "a"))
        print('Ep %d: Val loss: %f' %(epoch+1, val_loss[epoch]), file=open(plot_path + ('/%f/log.txt') %(lr), "a"))
        print('-------------------------', file=open(plot_path + ('/%f/log.txt') % (lr), "a"))

    # learning curves
    plt.figure(figsize=(8,5))
    plt.plot(range(1, len(train_loss)+1), train_loss, 'g', label='Training Loss')
    plt.plot(range(1, len(val_loss) + 1), val_loss, 'b', label='Val Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(plot_path + ('/%f/training_plot.pdf') %(lr))




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Merge folds, specify arguments')
    parser.add_argument('--lr', type=float, default=0.0001, metavar='LR',
                        help='learning rate (default: 0.0001)')
    parser.add_argument('--info', type=str, default='gcc_GT_GT', metavar='S',
                        help='Add additional info for storing (default: ours)')
    args = parser.parse_args()
    main()
