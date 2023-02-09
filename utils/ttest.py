#!/usr/bin/python

import pickle
import scipy.stats as stats
import core.config as conf

base_path = conf.input['project_path']
tolerance = 89

file_GT_GT = base_path + 'output/forward/gcc_GT_GT/%f/F1_scores_%d.pckl' %(0.0005, tolerance)
file_GT_VAD = base_path + 'output/forward/gcc_GT_VAD/%f/F1_scores_%d.pckl' %(0.00009, tolerance)


def main():
    # Load pickle file with greater F1 scores (on average)
    with (open(file_GT_GT, "rb")) as pickleFile:
        greater = pickle.load(pickleFile)
    # Load pickle file with smaller F1 scores (on average)
    with (open(file_GT_VAD, "rb")) as pickleFile:
        smaller = pickle.load(pickleFile)

    # Perfor paired sample t-test to check statistical significance of "grater" being greater of "smaller"
    results = stats.ttest_rel(greater, smaller, alternative='greater')

    print(results) # check whether pvalue < 5%

if __name__ == "__main__":
    main()