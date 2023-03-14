#!/usr/bin/python

import numpy as np
import soundfile as sf
import torch
import h5py, glob
from torch.utils.data import Dataset

import core.config as conf
import utils.utils as utils
import utils.salsa_feature_extraction as salsa_extraction



base_path = conf.input['project_path']
fps = conf.input['fps']


def read_audio_file(sequence, dev_or_test, rig, initial_time):
    # ==================== read audio file ===============================
    sequence_path = base_path + 'data/TragicTalkers/audio/' + dev_or_test + '/' + sequence + '/'

    if rig == '01':
        mic_ids = np.array(['01','02','03','04','05','06','07','08','09','10','11','12','13','14','15','16'])
    else:  # rig == '02'
        mic_ids = np.array(['23','24','25','26','27','28','29','30','31','32','33','34','35','36','37','38'])

    num_samples = conf.training_param['frame_len_samples'] # number of samples to produce an audio frame (chunk)
    # initial audio sample
    first_audio_sample = np.int((np.round(initial_time * fps) / fps) * conf.input['sr'])

    audio = []
    for id in range(len(mic_ids)):
        seq = sorted(glob.glob(sequence_path + mic_ids[id] + '*.wav'))[0]
        aud, sr = sf.read(seq)
        # Extract only the chunk
        aud = aud[first_audio_sample:first_audio_sample + num_samples]
        aud = utils.pad_audio_clip(aud, num_samples) # pad in the case the extracted segment is too short
        audio.append(aud)

    audio = np.transpose(np.array(audio))
    return audio, sr


def generate_audio_tensor(audio, sr):
    ## ======================= compute log mel features ===================

    winlen = conf.spectrog_param['winlen']
    hoplen = conf.spectrog_param['hoplen']
    numcep = conf.spectrog_param['numcep']
    n_fft = conf.spectrog_param['n_fft']
    fmin = conf.spectrog_param['fmin']
    fmax = conf.spectrog_param['fmax']

    ## ----------- LOG MEL SPECTROGRAMS TENSOR -------------
    if conf.input['features'] == 'Log-mel':
        # This generates a stack of log mel spectrograms
        tensor = [] # log mel tensor
        channel_num = audio.shape[1]
        for idx in range(channel_num):
            logmel_sp = utils.generate_mel_spectrograms(audio[:, idx], sr, winlen, hoplen, numcep, n_fft, fmin, fmax)
            tensor.append(logmel_sp)
        tensor = np.concatenate(tensor, axis=0) # (n_channels, timebins, freqbins)

    ## ----------- GCC SPECTROGRAMS TENSOR -------------
    elif conf.input['features'] == 'GCC-PHAT':
        tensor = [] # gcc_tensor
        channel_num = audio.shape[1]
        # ---> use all possible microphone pairs
        #for n in range(channel_num):
        #    for m in range(n + 1, channel_num):
        #        tensor.append(utils.generate_gcc_spectrograms(audio[:,m], audio[:,n], winlen, hoplen, numcep, n_fft))

        # ---> use reference mic (default 0, i.e. first mic)
        ref_mic_id = 0 #np.int(np.floor(channel_num/2))
        for n in range(channel_num):
            if not n == ref_mic_id:
                tensor.append(utils.generate_gcc_spectrograms(audio[:, n], audio[:, ref_mic_id], winlen, hoplen, numcep, n_fft))

        ## ---------- ADD mono log mel spect (1st channel only) ------------------------
        logmel = utils.generate_mel_spectrograms(audio[:, 0], sr, winlen, hoplen, numcep, n_fft, fmin, fmax)
        tensor.append(logmel)

        tensor = np.concatenate(tensor, axis=0) # (n_channels, timebins, freqbins)

    ## -------------- SALSA FEATURE EXTRACTION --------------
    elif conf.input['features'] == 'SALSA-Lite':
        tensor = salsa_extraction.extract_features(audio, conf=conf, ref_mic=1) # salsa-lite

    else:
        raise ValueError("""Input feature non supported: is '%s' a typo?""" %conf.input['features'])

    return tensor



class dataset_from_scratch(Dataset):
    def __init__(self, csv_file_path, dev_or_test, normalize=False, mean=None, std=None):
        self.csv_list = utils.csv_to_list(csv_file_path)[1:]  # [1:] is to remove first row ('name','time',etc)
        self.dev_or_test = dev_or_test
        self.normalize = normalize
        self.mean = mean
        self.std = std
        if dev_or_test == 'development': # 1 sec overlap for training
            self.frame_idx_list = utils.find_audio_frame_idx(self.csv_list, conf.input['frame_step_train'])
        else: # no overlap for testing
            self.frame_idx_list = utils.find_audio_frame_idx(self.csv_list, conf.input['frame_step_test'])

    def __len__(self):
        return int(len(self.frame_idx_list) / 1)

    def __getitem__(self, audio_seg):
        full_name = self.csv_list[self.frame_idx_list[audio_seg]][0]
        sequence = full_name[:-6] # remove '-cam01'
        cam = np.int(full_name[-2:])
        initial_time = np.float(self.csv_list[self.frame_idx_list[audio_seg]][1])
        target_coords = []
        pseudo_labels = []
        sp_activity = []

        for idx in range(self.frame_idx_list[audio_seg],
                         self.frame_idx_list[audio_seg] + (fps * conf.input['frame_len_sec'])):
            if idx >= len(self.csv_list):  # out of range i.e. the very end of malemonologue2_t2-cam22
                target_coords.append(1) # 1 default
                pseudo_labels.append(0)  # NOT_SPEAKING default
                sp_activity.append(0)  # NOT_SPEAKING default
            elif self.csv_list[idx][
                0] != full_name:  # end of a seq e.g. conv1_t1-cam22 finished and conv1_t2-cam01 starts
                target_coords.append(1)
                pseudo_labels.append(0)  # NOT_SPEAKING
                sp_activity.append(0)  # NOT_SPEAKING
            else:
                target_coords.append(self.csv_list[idx][2])
                if self.csv_list[idx][3] == 'SPEAKING':
                    pseudo_labels.append(1)
                else:  # NOT_SPEAKING
                    pseudo_labels.append(0)
                if self.csv_list[idx][4] == 'SPEAKING':
                    sp_activity.append(1)
                else:  # NOT_SPEAKING
                    sp_activity.append(0)

        if cam < 12:
            rig = '01'
        else:
            rig = '02'
            cam = cam - 11 # cam12 is cam01 on AVA rig2; cam13 is cam02, etc.
        cam = utils.cam_one_hot(cam)
        cam = np.expand_dims(cam, axis=0)

        # read audio files
        audio, sr = read_audio_file(sequence, self.dev_or_test, rig, initial_time)
        # compute features (e.g. GCC-PHAT or SALSA-Lite) and generate 16x960x64 input tensor
        tensor = generate_audio_tensor(audio, sr)

        if self.normalize:
            # Normalize feature
            n_scaler_chan = self.mean.shape[0]
            tensor = (tensor - self.mean) / self.std

        tensor = tensor.astype('float32')
        input_features = torch.from_numpy(tensor)
        target_coords = np.asarray(target_coords).astype('float32')
        pseudo_labels = np.asarray(pseudo_labels).astype('float32')
        sp_activity = np.asarray(sp_activity).astype('float32')

        if self.dev_or_test == 'development':
            return input_features, cam, target_coords, pseudo_labels, sp_activity, sequence
        else:  # == 'test'
            return input_features, cam, full_name, initial_time


class dataset_from_hdf5(Dataset):
    def __init__(self, h5py_dir, normalize=False, mean=None, std=None):

        self.h5_file = h5py.File(h5py_dir, 'r')
        self.normalize = normalize
        self.mean = mean
        self.std = std

    def __len__(self):
        return int((self.h5_file['features'].shape[0]) / 1)

    def __getitem__(self, audio_seg):

        features = self.h5_file['features'][audio_seg]
        cams = self.h5_file['cams'][audio_seg]
        target_coords = self.h5_file['target_coords'][audio_seg]
        pseudo_labels = self.h5_file['pseudo_labels'][audio_seg]
        speech_activity = self.h5_file['speech_activity'][audio_seg]
        sequence = self.h5_file['sequence'][audio_seg]


        if self.normalize:
            # Normalize feature
            n_scaler_chan = self.mean.shape[0]
            features = (features - self.mean) / self.std

        features = features.astype('float32')
        input_features = torch.from_numpy(features)

        return input_features, cams, target_coords, pseudo_labels, speech_activity, sequence
