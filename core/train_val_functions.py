#!/usr/bin/python

import torch
import numpy as np



def loss_function(output, target_pos, pseudo_labels, speech_activity): # sum-squared error loss
    """
    output[batch,time,0]: predicted position x
    output[batch,tims,1]: predicted confidence c
    """
    loss = 0

    for batch in range(speech_activity.size(0)):
        confidence_loss = 0
        regression_loss = 0
        for tframe in range(speech_activity.size(1)):
            if np.int(speech_activity[batch,tframe]) == 1: # i.e active
                c_hat = 1
                confidence_loss = confidence_loss + ((output[batch, tframe, 1] - c_hat) ** 2)
                if np.int(pseudo_labels[batch,tframe]) == 1: # i.e. speech detected by teacher too
                    regression_loss = regression_loss + ((output[batch,tframe,0] - target_pos[batch,tframe]) ** 2)
            else: # np.int(speech_activity[batch,tframe]) == 0 i.e silent
                c_hat = 0
                confidence_loss = confidence_loss + ((output[batch, tframe, 1] - c_hat) ** 2)

        partial_loss = regression_loss + confidence_loss
        loss = loss + partial_loss
    return loss




def train(model, optimizer, dl_train, args, device, ckpt_dir, epoch):
    model.train()
    training_loss = 0

    for batch_idx, (features, cams, target_pos, pseudo_labels, speech_activity, sequence) in enumerate(dl_train):
        # pass data to gpu (or generally the target device)
        features, cams, target_pos, pseudo_labels, speech_activity = features.to(device), cams.to(
            device), target_pos.to(device), pseudo_labels.to(device), speech_activity.to(device)
        optimizer.zero_grad() # Clear the gradients if exists.
        output = model(features, cams)

        loss = loss_function(output, target_pos, pseudo_labels, speech_activity)
        loss.backward() # Backpropagate the losses
        optimizer.step() # Update Model parameters

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(features), len(dl_train.dataset),
                100. * batch_idx / len(dl_train), loss.item()/len(features)))
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(features), len(dl_train.dataset),
                       100. * batch_idx / len(dl_train), loss.item() / len(features)),
                file=open('%s/log.txt' % ckpt_dir, "a"))

        training_loss += loss.item()
        torch.save(model.state_dict(), ckpt_dir/'model_{:03d}.ckpt'.format(epoch))

    mean_batch_loss = training_loss/len(dl_train.dataset)*args.batch_size
    return  training_loss/len(dl_train.dataset) # mean_batch_loss


def val(model, dl_val, args, device, ckpt_dir):
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for data, cams, target_pos, pseudo_labels, speech_activity, sequence in dl_val:
            data, cams, target_pos, pseudo_labels, speech_activity = data.to(device), cams.to(device), target_pos.to(
                device), pseudo_labels.to(device), speech_activity.to(device)
            output = model(data, cams)

            # sum up batch loss
            val_loss += loss_function(output, target_pos, pseudo_labels, speech_activity).item()


    mean_val_batch_loss = val_loss / len(dl_val.dataset) * args.batch_size

    print('\nVal set: Average loss: {:.4f}\n'.format(val_loss / len(dl_val.dataset)))
    print('\nVal set: Average loss: {:.4f}\n'.format(val_loss / len(dl_val.dataset)), file=open('%s/log.txt' % ckpt_dir, "a"))
    return val_loss / len(dl_val.dataset) # mean_val_batch_loss
