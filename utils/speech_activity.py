#!/usr/bin/python




def speech_activity_union(gt_list, num_frames):
    union = []
    for idx in range(num_frames):
        if gt_list[idx][4] == 'SPEAKING' or gt_list[idx + num_frames][4] == 'SPEAKING':
            if gt_list[idx][4] == 'SPEAKING':
                union.append(['SPEAKING', int(gt_list[idx][1])])
            else:
                union.append(['SPEAKING', int(gt_list[idx + num_frames][1])])
        else:
            union.append(['NOT_SPEAKING', 0])
    return union


def eval_speech_activity(gt_union, fwd_sequence, threshold):
    num_correct = 0
    num_wrong = 0
    distance_err = 0
    true_pos = 0

    for frame in range(len(gt_union)):

        if (float(fwd_sequence[frame][3]) >= threshold and gt_union[frame][0] == 'SPEAKING') or (
                        float(fwd_sequence[frame][3]) < threshold and gt_union[frame][0] == 'NOT_SPEAKING'):
            if (float(fwd_sequence[frame][3]) >= threshold and gt_union[frame][0] == 'SPEAKING'):
                #distance_err = distance_err + (abs(union[frame][1] - (float(fwd_list[frame][2])))) # if forward file is expressed in frames
                distance_err = distance_err + (abs(gt_union[frame][1]-(float(fwd_sequence[frame][2])*2448)))
                true_pos += 1
            num_correct += 1
        else:
            num_wrong += 1

    return num_correct, num_wrong, true_pos, distance_err

