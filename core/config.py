#!/usr/bin/python

import torch



input = {
    'project_path': '/home/davide/PycharmProjects/ASDL-CRNN/',
    'h5py_path': '/home/davide/PycharmProjects/ASDL-CRNN/data/', # might want to save h5py (and scaler) somewhere else.
    # choose from: 'GT_GT', 'GT_VAD', 'ASC_GT', 'ASC_VAD', 'ASC(s)_GT', 'ASC(s)_VAD', 'TalkNet_GT', 'TalkNet_VAD'
    'supervisory_condition': 'GT_GT',
    'fps': 30,
    'sr': 48000, # 48 kHz
    'frame_len_sec': 2, # seconds
    'frame_step_train': 1, #seconds
    'frame_step_test': 2, #seconds (no overlap)
}

training_param = {
    'optimizer': torch.optim.Adam,
    #'criterion': nn.CrossEntropyLoss,
    'learning_rate': 0.0001, # default if user does not provide a different lr to the parser
    'epochs': 50, # default if user does not provide a different number to the parser
    'batch_size': 32, # default if user does not provide a different size to the parser
    'frame_len_samples': input['frame_len_sec'] * input['sr'], # number of audio samples in 2 sec
}

spectrog_param = { # used for log mel spec, gcc-phat, or salsa
    'winlen': 512, # samples
    'hoplen': 100, # samples
    'numcep': 64, # number of cepstrum bins to return
    'n_fft': 512, #fft lenght
    'fmin': 40, #Hz
    'fmax': 24000 #Hz
}

