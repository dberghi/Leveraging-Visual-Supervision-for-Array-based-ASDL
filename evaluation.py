#!/usr/bin/python

import numpy as np
import glob,  csv, os, argparse, json, pickle
from tqdm import tqdm
from matplotlib.ticker import MultipleLocator
import matplotlib.pyplot as plt
from utils.speech_activity import eval_speech_activity, speech_activity_union
from utils.precision_recall_utils import pred_gt_comparison, compute_prec_rec, Prec_rec_seq, precision_recall_postprocessing
import utils.utils as utils
import core.config as conf


base_path = conf.input['project_path']
threshold = 0.5
fps = conf.input['fps']
gt_path = base_path + 'data/GT/'
json_path = base_path + 'data/csv/seq_lengths.json'


evalSet_dict = {
    "conversation1_t3": 0,
    "femalemonologue2_t3": 1,
    "interactive1_t2": 2,
    "interactive4_t3": 3,
    "malemonologue2_t3": 4
}



def main():
    fwd_file = base_path + 'output/forward/%s/%f/test_forward.csv' % (args.info, args.lr)
    # path to save pickle file with sequence-wise precision-recall matrices (used to compute statistics)
    savePcklPath = base_path + 'output/forward/%s/%f/seq_wise_prec_rec_%d_sigmoid.pckl' % (
    args.info, args.lr, args.tolerance)
    # path to save csv file with overall precision-recall matrix
    saveCsvPath = base_path + 'output/forward/%s/%f/precision_recall_%d_sigmoid.csv' % (
    args.info, args.lr, args.tolerance)

    ## ----------- load json file with sequences' lengths
    f = open(json_path)
    lengths = json.load(f)

    # Initialize precision-recall.csv file
    with open(saveCsvPath, 'w') as file:
        writer = csv.writer(file)
        writer.writerow(['confidence', 'true pos', 'GT pos', 'predict pos', 'precision', 'recall'])
        # generate precision-recall mat of zeros
        prec_rec_init = Prec_rec_seq(sequence=None, rig=None)
        for row in prec_rec_init.precision_recall_matrix:
            writer.writerow(row)

    gt_seq_list = sorted(glob.glob(gt_path + '/*'))
    fwd_list = utils.csv_to_list(fwd_file)[1:] # [1:] is to ignore first row ['name', 'time', etc]


    # ------------ initialize error matrix (gt_seq x rig) i.e. (5 seqs, 2 rigs)
    correct_mat = np.zeros((len(evalSet_dict), 2), dtype=int)
    wrong_mat = np.zeros((len(evalSet_dict), 2), dtype=int)
    VA_TP_mat = np.zeros((len(evalSet_dict), 2), dtype=int)
    error_mat = np.zeros((len(evalSet_dict), 2), dtype=float)
    # -------------

    with open(saveCsvPath, 'r') as file:
        precision_recall_reader = csv.reader(file)
        overall_precision_recall_mat = np.array([row for row in precision_recall_reader])
        overall_precision_recall_mat = overall_precision_recall_mat[1:] # ignore first row

    list_of_prec_rec_mats = []

    for sequence_path in gt_seq_list:
        sequence = os.path.basename(sequence_path)
        print('Sequence: %s' % (sequence))
        rigs_paths = sorted(glob.glob(sequence_path + '/*'))
        for rig_path in rigs_paths:
            rig = int(os.path.basename(rig_path))
            print('Rig: %d' %int(rig))

            # create new precision-recall matrix (there will be one for each test sequence and each rig)
            current_prec_rec_obj = Prec_rec_seq(sequence, rig) # init with zeros
            currect_prec_rec_mat = current_prec_rec_obj.precision_recall_matrix
            GT_files_path = sorted(glob.glob(rig_path + '/*.csv'))

            for count, GT_file_path in enumerate(tqdm(GT_files_path)):
                GT_file = os.path.basename(GT_file_path)
                seq_indices = utils.find(fwd_list, GT_file[:-7])
                fwd_sequence = list(np.array(fwd_list)[seq_indices])

                sequence_length = round(float(lengths[sequence]), 2)
                num_frames = int(round(sequence_length * fps))
                gt_list = utils.csv_to_list(GT_file_path)

                if sequence[:-4] == 'conversation' or sequence[:-4] == 'interactive':
                    gt_union = speech_activity_union(gt_list, num_frames) # unify
                else:
                    gt_union = []
                    for i in range(num_frames):
                        gt_union.append([gt_list[i][4], int(gt_list[i][1])])

                # not well optimized...
                overall_precision_recall_mat = pred_gt_comparison(gt_union, fwd_sequence, overall_precision_recall_mat, args.tolerance)
                currect_prec_rec_mat = pred_gt_comparison(gt_union, fwd_sequence, currect_prec_rec_mat, args.tolerance)

                num_cor, num_wr, tp, err = eval_speech_activity(gt_union, fwd_sequence, threshold)

                # update sequence-wise count
                correct_mat[evalSet_dict[sequence], rig-1] = correct_mat[evalSet_dict[sequence], rig-1] + num_cor
                wrong_mat[evalSet_dict[sequence], rig-1] = wrong_mat[evalSet_dict[sequence], rig-1] + num_wr
                VA_TP_mat[evalSet_dict[sequence], rig-1] = VA_TP_mat[evalSet_dict[sequence], rig-1] + tp
                error_mat[evalSet_dict[sequence], rig-1] = error_mat[evalSet_dict[sequence], rig-1] + err

            currect_prec_rec_mat = compute_prec_rec(currect_prec_rec_mat)
            current_prec_rec_obj.update(currect_prec_rec_mat)
            list_of_prec_rec_mats.append(current_prec_rec_obj)

    overall_precision_recall_mat = compute_prec_rec(overall_precision_recall_mat)

    # update overall csv file
    with open(saveCsvPath, 'w') as file:
        writer = csv.writer(file)
        writer.writerow(['confidence', 'true pos', 'GT pos', 'predict pos', 'precision', 'recall'])
        for prec_rec_rows in overall_precision_recall_mat:
            writer.writerow(prec_rec_rows)

    # save pickle file
    with open(savePcklPath, 'wb') as fil:
        pickle.dump(list_of_prec_rec_mats, fil)

    aD_mat = np.reshape((error_mat / VA_TP_mat), len(evalSet_dict)*2)
    aD_mean = np.mean(aD_mat)
    aD_std_error = np.std(aD_mat, ddof=1) / np.sqrt(np.size(aD_mat))

    detection_mat = np.reshape((correct_mat/(correct_mat+wrong_mat)), len(evalSet_dict)*2)
    detection_mean = np.mean(detection_mat)
    detection_std_error = np.std(detection_mat, ddof=1) / np.sqrt(np.size(detection_mat))

    print('')
    print('%s - aD mean: %f pixels' %(args.info, aD_mean))
    print('%s - aD standard error: %f pixels' %(args.info, aD_std_error))
    print('%s - Detection error mean: %f %%' % (args.info, ((1-detection_mean)*100)))
    print('%s - Detection standard error: %f' % (args.info, detection_std_error))



    ## ------------------------------------------------------------------------------------------------------
    ## ---- From the precision-recall values we can now compute (and plot) the AP and the F1 score
    ## ------------------------------------------------------------------------------------------------------

    """
        USAGE:
            precision_recall_postprocessing(csv_file_path, info='baseline name', plot_bool=False, seq_wise_F1s=False,
                            tolerance=89, label='', symbol='', color='b', linewidth=1.5)
        INPUTS:
            csv_file_path: path to csv file with overall precision-recall matrix.
            info: name of the baseline for print(). It is convenient when printing multiple baselines.
            plot_bool: set True to plot precision-recall curve
            seq_wise_F1s: set True to save pickle file with sequence-wise F1 scores (useful for statistical tests)
            tolerance: use 89px for 2 deg tolerance or 222px for 5 deg
            label: name used in the plot legend 
            symbol: color and symbol used to plot best F1 score value
            color: color and style for precision-recall curve
            linewidth: curve line width
    """
    precision_recall_postprocessing(saveCsvPath, info=args.info, seq_wise_F1s=False, plot_bool=args.plot_bool,
                                    symbol='ob', label=args.info)


    ## ---------- Uncomment to print and plot AP and F1 scores reported in the paper -------------------------

    '''
    csv_mono = base_path + 'output/forward/mono/%f/precision_recall_89_sigmoid.csv' % (0.0001)
    csv_2 = base_path + 'output/forward/stereo/%f/precision_recall_89_sigmoid.csv' % (0.0001)
    csv_16 = base_path + 'output/forward/16mics/%f/precision_recall_89_sigmoid.csv' % (0.0001)
    csv_asc = base_path + 'output/forward/ASC/%f/precision_recall_89_sigmoid.csv' % (999)
    csv_asc_s = base_path + 'output/forward/ASC(s)/%f/precision_recall_89_sigmoid.csv' % (999)
    csv_talkNet = base_path + 'output/forward/TalkNet/%f/precision_recall_89_sigmoid.csv' % (999)

    csv_salsa_GT_GT = base_path + 'output/forward/salsa-lite_GT_GT/%f/precision_recall_89_sigmoid.csv' % (0.00007)
    csv_gcc_GT_GT = base_path + 'output/forward/gcc_GT_GT/%f/precision_recall_89_sigmoid.csv' % (0.0005)
    csv_gcc_GT_VAD = base_path + 'output/forward/gcc_GT_VAD/%f/precision_recall_89_sigmoid.csv' % (0.00009)

    csv_gcc_ASC_s_GT = base_path + 'output/forward/gcc_ASC(s)_GT/%f/precision_recall_89_sigmoid.csv' % (0.00009)
    csv_gcc_ASC_s_VAD = base_path + 'output/forward/gcc_ASC(s)_VAD/%f/precision_recall_89_sigmoid.csv' % (0.00007)

    csv_gcc_ASC_GT = base_path + 'output/forward/gcc_ASC_GT/%f/precision_recall_89_sigmoid.csv' % (0.00009)
    csv_gcc_ASC_VAD = base_path + 'output/forward/gcc_ASC_VAD/%f/precision_recall_89_sigmoid.csv' % (0.0001)

    csv_gcc_TalkNet_GT = base_path + 'output/forward/gcc_TalkNet_GT/%f/precision_recall_89_sigmoid.csv' % (0.00009)
    csv_gcc_TalkNet_VAD = base_path + 'output/forward/gcc_TalkNet_VAD/%f/precision_recall_89_sigmoid.csv' % (0.0001)

    mono = precision_recall_postprocessing(csv_mono, info='Mono')
    stereo = precision_recall_postprocessing(csv_2, info='Stereo', plot_bool=False, symbol='C0v', color='C0-.',
                                             label='Stereo', linewidth=2.5)
    mics_16 = precision_recall_postprocessing(csv_16, info='16 Mics', plot_bool=False, symbol='C1v', color='C1-.',
                                              label='16 Mics', linewidth=2.5)
    asc = precision_recall_postprocessing(csv_asc, info='ASC', plot_bool=False, color='--b', symbol='ob', label='ASC',
                                          linewidth=2.5)
    asc_s = precision_recall_postprocessing(csv_asc_s, info='ASC(s)', plot_bool=False, color='b', symbol='ob',
                                            label='ASC(s)', linewidth=2.5)
    talkNet = precision_recall_postprocessing(csv_talkNet, info='TalkNet', plot_bool=True, color='--m', symbol='om',
                                              label='TalkNet', linewidth=3)
    salsa_GT_GT = precision_recall_postprocessing(csv_salsa_GT_GT, info='GT-GT (sal-lit)', plot_bool=True, symbol='g^',
                                                  color='g', label='GT-GT (sal-lit)')
    gcc_GT_GT = precision_recall_postprocessing(csv_gcc_GT_GT, info='GT-GT (gcc)', plot_bool=True, symbol='r^',
                                                color='r', label='GT-GT (gcc)', linewidth=2)
    gcc_GT_VAD = precision_recall_postprocessing(csv_gcc_GT_VAD, info='GT-VAD (gcc)', plot_bool=True, symbol='r^',
                                                 color=':r', label='GT-VAD (gcc)', linewidth=2)
    gcc_ASC_s_GT = precision_recall_postprocessing(csv_gcc_ASC_s_GT, info='ASC(s)-GT (gcc)', plot_bool=True,
                                                   symbol='rs', color='-r', label='ASC(s)-GT (gcc)')
    gcc_ASC_s_VAD = precision_recall_postprocessing(csv_gcc_ASC_s_VAD, info='ASC(s)-VAD (gcc)', plot_bool=True,
                                                    symbol='cs', color=':c', label='ASC(s)-VAD (gcc)')
    gcc_ASC_GT = precision_recall_postprocessing(csv_gcc_ASC_GT, info='ASC-GT (gcc)', plot_bool=True, symbol='gx',
                                                 color='g', label='ASC-GT (gcc)', linewidth=2)
    gcc_ASC_VAD = precision_recall_postprocessing(csv_gcc_ASC_VAD, info='ASC-VAD (gcc)', plot_bool=True, symbol='gx',
                                                  color=':g', label='ASC-VAD (gcc)', linewidth=2)
    gcc_TalkNet_GT = precision_recall_postprocessing(csv_gcc_TalkNet_GT, info='TalkNet-GT (gcc)', plot_bool=True,
                                                     symbol='bD', color='b', label='TalkNet-GT (gcc)', linewidth=2)
    gcc_TalkNet_VAD = precision_recall_postprocessing(csv_gcc_TalkNet_VAD, info='TalkNet-VAD (gcc)', plot_bool=True,
                                                      symbol='bD', color=':b', label='TalkNet-VAD (gcc)', linewidth=2)
    '''




    # plot formatting
    if args.plot_bool:
        plt.xlabel('Recall', fontsize=20)
        plt.ylabel('Precision', fontsize=20)
        # plt.title(title, fontsize=18)
        spacing = 0.1  # This can be your user specified spacing.
        minorLocator = MultipleLocator(spacing)
        ticks = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
        plt.xticks(ticks)
        plt.grid()
        plt.xlim(0.5, 1)
        plt.ylim(0.5, 1.01)
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        plt.legend(fontsize=18)  # , loc='lower right')
        plt.show()



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Specify arguments for evaluation')
    parser.add_argument('--lr', type=float, default=0.0005, metavar='LR',
                        help='learning rate (default: %f)' % conf.training_param['learning_rate'])
    parser.add_argument('--info', type=str, default='default', metavar='S',
                        help='Add additional info for storing (default: ours)')
    parser.add_argument('--tolerance', type=int, default=89, metavar='TOL',
                        help='tolerance (default: 89)')
    parser.add_argument('--plot-bool', default=False, action='store_true',
                        help='set True to plot precision-recall curve')
    args = parser.parse_args()

    main()
