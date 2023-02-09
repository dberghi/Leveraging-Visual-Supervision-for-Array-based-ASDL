#!/usr/bin/python

import csv, pickle, os
import matplotlib.pyplot as plt
import numpy as np


def pred_gt_comparison(gt_union, fwd_sequence, precision_recall_matrix, tolerance):

    for prec_rec_row in range(len(precision_recall_matrix)):
        confidence = float(precision_recall_matrix[prec_rec_row,0])
        tp = float(precision_recall_matrix[prec_rec_row,1])
        gt_pos = float(precision_recall_matrix[prec_rec_row, 2])
        pred_pos = float(precision_recall_matrix[prec_rec_row, 3])

        for frame in range(len(gt_union)):
            if gt_union[frame][0] == 'SPEAKING':
                gt_pos += 1
                if (float(fwd_sequence[frame][3])*100 >= confidence) and ((
                        abs(float(fwd_sequence[frame][2]) * 2448 - float(gt_union[frame][1]))) <= tolerance):
                    tp += 1
            if float(fwd_sequence[frame][3])*100 >= confidence:
                pred_pos += 1

        # update
        precision_recall_matrix[prec_rec_row] = [confidence, tp, gt_pos, pred_pos, 0, 0]

    return precision_recall_matrix

def compute_prec_rec(precision_recall_matrix):
    for prec_rec_row in range(len(precision_recall_matrix)):
        if float(precision_recall_matrix[prec_rec_row,3]) == 0.0:
            precision = 0
        else:
            precision = round( # round to 12 sig digits
                float(precision_recall_matrix[prec_rec_row, 1]) / float(precision_recall_matrix[prec_rec_row, 3]), 12)
        recall = round( # round to 12 sig digits
            float(precision_recall_matrix[prec_rec_row, 1]) / float(precision_recall_matrix[prec_rec_row, 2]), 12)

        precision_recall_matrix[prec_rec_row,4] = precision
        precision_recall_matrix[prec_rec_row,5] = recall

    return precision_recall_matrix


class Prec_rec_seq:
    def __init__(self, sequence, rig):
        precision_recall_matrix = []
        for conf in range(50, -50, -1):
            # INITIALISE CSV: [confidence, true_positives, false_positives, false negatives, precision, recall]
            # sig_conf = 1/(1 + np.exp(-(conf)/5))
            sig_conf = 1 / (1 + np.exp(-(conf) / 4))
            precision_recall_matrix.append([sig_conf * 100, 0, 0, 0, 0, 0])
        # add final zero confidence row
        precision_recall_matrix.append([0.0, 0, 0, 0, 0, 0])
        self.precision_recall_matrix = np.array(precision_recall_matrix)
        self.sequence = sequence
        self.rig = rig
    def update(self, new_mat):
        self.precision_recall_matrix = new_mat


def save_sequence_wise_F1s(path, idx, tolerance):
    pickle_path = os.path.dirname(path) + '/seq_wise_prec_rec_%d_sigmoid.pckl' %(tolerance)
    with (open(pickle_path, "rb")) as pickleFile:
        prec_recs = pickle.load(pickleFile)
    F1s = []
    for prec_rec_mat in prec_recs:
        mat = prec_rec_mat.precision_recall_matrix
        prec = mat[idx,4]
        rec = mat[idx,5]
        F1s.append(2*prec*rec / (prec+rec))

    with open(os.path.dirname(path) + '/F1_scores_%d.pckl' %(tolerance), 'wb') as fil:
        pickle.dump(F1s, fil)

    # print standard error
    F1_mean = np.mean(F1s)
    F1_std_error = np.std(F1s, ddof=1) / np.sqrt(np.size(F1s))
    print('------------Mean of the F1s: %f' %F1_mean)
    print('------------Standard error of the F1s: %f' %F1_std_error)




def max_F1_score(precision_array, recall_array):
    precision_array = np.array(precision_array)
    recall_array = np.array(recall_array)
    prod_array = precision_array * recall_array
    sum_array = precision_array + recall_array
    F1_max = 0
    idx_max = 0 # index representing the precision and recall that gives the highest F1 score
    F1_array = []
    confidence_array = []
    for i in range(len(prod_array)):
        F1 = 2 * prod_array[i] / sum_array[i]
        F1_array.append(F1)
        confidence_array.append(99-i)
        if F1 > F1_max:
            F1_max = F1
            idx_max = i
    return F1_max, idx_max, F1_array, confidence_array # idx is useful to recover the correct threshold value

def curve_smoother(precision_array): # as explained in the evaluation guidelines of the Pascal VOC2012 challenge
    for i in range(len(precision_array)):
        precision_array[i] = np.max(precision_array[i:])
    return precision_array

def AUC(precision_array, recall_array):
    tot_area = 0
    current_prec = precision_array[0]
    init_rec = 0
    fin_rec = recall_array[0]
    flag = 0
    for i in range(1,len(recall_array)):
        if precision_array[i] == current_prec:
            fin_rec = recall_array[i]
            if i == len(recall_array)-1: # check if this is the last loop before it ends
                area = current_prec * (fin_rec - init_rec)
                tot_area = tot_area + area
        else:
            area = current_prec * (fin_rec - init_rec)
            tot_area = tot_area + area
            ix = i-1
            if ix < 0:
                ix = 0
            init_rec = recall_array[ix]
            fin_rec = recall_array[i]
            current_prec = precision_array[i]

    return tot_area


class precision_recall_postprocessing:
    def __init__(self, csv_file_path, info='Unnamed baseline', plot_bool=False, seq_wise_F1s=False,
                 tolerance=89, label='', symbol='', color='b', linewidth=1.5):
        self.info = info
        self.seq_wise_F1s = seq_wise_F1s # Used for statistical tests
        self.tolerance = tolerance
        self.label = label
        precision_array = []
        self.recall_array = []
        self.linewidth = linewidth
        with open(csv_file_path, "r") as file:
            next(file) # skip header line
            reader = csv.reader(file)
            csv_data = list(reader)
            self.matrix = np.array([row for row in csv_data])
        for row in csv_data:
            precision_array.append(float(row[4]))
            self.recall_array.append(float(row[5]))

        # make precision monotonically decreasing as in Pascal VOC2012
        self.precision_array = curve_smoother(precision_array)
        # Compute area under the curve
        self.auc = AUC(self.precision_array, self.recall_array)
        # Compute F1 score, F1_array and idx_array are used to plot F1 against confidence
        [self.F1, self.F1_idx, self.F1_array, self.conf_array] = max_F1_score(self.precision_array, self.recall_array)

        print('%s - AP = %f' % (self.info, self.auc))
        print('%s - F1 score: %f' % (self.info, self.F1)) #at confidence: %s' % (self.info, self.F1, self.matrix[self.F1_idx, 0]))
        if self.seq_wise_F1s:
            save_sequence_wise_F1s(csv_file_path, self.F1_idx, self.tolerance)
        print()

        # plot curve
        if plot_bool:
            plt.plot(self.recall_array, self.precision_array, color, label=self.label,linewidth=self.linewidth)
            b = plt.plot(self.recall_array[self.F1_idx], self.precision_array[self.F1_idx], symbol)
            plt.setp(b,'markersize', 8)
            plt.setp(b,'markerfacecolor', "None")