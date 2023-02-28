# HORIZONTAL ACTIVE SPEAKER DETECTION AND LOCALIZATION ON THE TRAGIC TALKERS DATASET

This repo contains the code and models for the audio CRNN "student" network used in the paper "Leveraging Visual Supervision for Array-based Active Speaker Detection and Localization" (Under review).

## Dataset

- Download TragicTalkers from https://cvssp.org/data/TragicTalkers/ (username and password are required, check the license information on the website and contact d.berghi@surrey.ac.uk or davide.berghi@gmail.com to request the credentials)

- Unzip the audio data in the `data/TragicTalkers/` folder to get the following directory structure:
<pre>
	.
	└── data
	    ├── csv
	    ├── GT
	    └── TragicTalkers
		└── audio
		    ├── development
		    ├── README.txt
		    └── test
</pre>

## Dependencies

Install dependencies by running `pip install -r utils/requirements.txt`. Or manually install the modules in `utils/requirements.txt`.


## Get started

Open `core/config.py` and edit 'project_path' (line 8) to point to your working directory, and 'h5py_path' to point where you want the audio features (~44GB) to be stored.
Choose the desired supervisory condition (line 11). Default is 'GT_GT' (fully supervised).

There are 4 bash files (`0_make_hdf5.sh`, `1_train.sh`, `2_forward.sh`, `3_evaluation.sh`). Each allows setting the input argument "INFO". Choose a consistent string argument across the 4 scripts. It is used for the naming convention of the outputs and can be useful to personalize your experiments.

NOTE: You might need to make the bash files executable by running `chmod +x <file_name>.sh`


## 0 - EXTRACT INPUT FEATURES 

The network takes in input pre-extracted features from the multichannel audio. The code extracts the features from the development set and stores them in HDF5 binary format (`development_dataset.h5`) using the h5py module. 
Then it computes the mean and standard deviation vectors that will be used to normalize the input features before training (`feature_scaler.h5`).

Default feature is GCC-PHAT. To use SALSA-Lite: 
	open core/dataset.py 
	comment the "GCC SPECTROGRAM TENSOR" section in the generate_audio_tensor function (from line 66 to 86) 
	uncomment line 89 in the "SALSA FEATURE EXTRACTION" section 

Create h5py dataset (~44GB) by running:
 
	./0_make_hdf5.sh

This will create the `development_dataset.h5` and `feature_scaler.h5` files and store them in `[h5py_path]/h5py_[INFO]/`


## 1 - TRAINING

Start training by running:

	./1_train.sh

The model's weights will be saved in the checkpoint folder `ckpt/[INFO]/[LR]/`.
By default the boolean argument `TRAINWITHFOLDS` in `1_train.sh` is set to false. This training uses the entire development set to train the CRNN. 
Setting `TRAINWITHFOLDS=true` will start a 5-fold cross-validation training and save the training and validation loss vectors in `output/training_plots/[INFO]/`
Then the losses are averaged across the 5 folds. A log file and a plot with the results will be saved in `output/training_plots/[INFO]/`.
This procedure is only used to find suitable training meta-parameters. The default settings should work fine.


## 2 - FORWARD TEST SET

Forward pass the test set using the trained model. Run:

	./2_forward.sh

This will create a `test_forward.csv` file in `output/forward/[INFO]/[LR]/`


## 3 - EVALUATION

By default the argument `TOLERANCE` in `3_evaluation.sh` is set to 89 pixels, i.e. 2 degrees along the azimuth. Use 222 pixels for 5 degrees tolerance.
You can set `PLOTBOOL=true` to plot the precision-recall curve.
Run:
	
	./3_evaluation.sh

This will print the average distance (aD), the detection error (det err), the average precision (AP), and the F1 score (F1). 
A precision-recall "matrix" with the values of precision and recall achieved for each Sigmoid-sampled confidence threshold will be saved in a `precision_recall_[TOLERANCE]_sigmoid.csv` file.

(Optional) Uncommenting from line 170 to line 220 in `evaluation.py` will reproduce (and plot if `PLOTBOOL=true`) the results reported in the paper. 


## (Optional) Make qualitative video demo

The script `utils/make_video.py` allows generating a video of one of the test sequences to qualitatively check your model's results. 
Modify the sequence name (line 11), 'info' (line 12), and learning rate 'lr' (line 13) as desired. The output will be an mp4 video of the selected sequence and camera view with a vertical line indicating the horizontal speaker position predicted by the model specified in 'info' and 'lr'.

The script works if you have the video sequences of TragicTalkers in `data/TragicTalkers/` 


## Pretrained models

In `ckpt/` are available the fully supervised (GT-GT) pretrained models with the GCC-PHAT input features and SALSA-Lite.
