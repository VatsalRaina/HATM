#! /usr/bin/env python

import argparse
import os
import sys

import numpy as np

import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot as plt


from sklearn.metrics import precision_recall_curve
from sklearn.metrics.pairwise import cosine_similarity

commandLineParser = argparse.ArgumentParser(description='Compute features from labels.')
#commandLineParser.add_argument('prompt_path', type=str,
 #                              help='which should be loaded')
#commandLineParser.add_argument('save_path', type=str,
 #                              help='which should be loaded')
commandLineParser.add_argument('--strip_start_end', action='store_true', help='whether to strip the <s> </s> marks at the beginning and end of prompts in sorted_topics.txt file (used for legacy sorted_topics.txt formatting')


def main(argv=None):
    args = commandLineParser.parse_args()
    if not os.path.isdir('CMDs'):
        os.mkdir('CMDs')
    with open('CMDs/get_bert_dists.cmd', 'a') as f:
        f.write(' '.join(sys.argv) + '\n')

   
    hatm_dists = np.loadtxt('/home/alta/relevance/vr311/data_vatsal/LINSK/Andrey/LINSKuns03evl03_ALL_naive/info/hatm_dists.txt', dtype=np.float32)
    pr_ids = np.loadtxt('/home/alta/relevance/vr311/data_vatsal/LINSK/Andrey/LINSKuns03evl03_ALL_naive/pr_ids.txt', dtype=np.float32)
    
    os.chdir("/home/alta/relevance/vr311/models_min_data/baseline/simGrid/ens")


    tl = np.loadtxt(fname = "./rep_and_com/eval_unseen/labels.txt")
    tp = np.loadtxt(fname = "./rep_and_com/eval_unseen/predictions.txt")

    thresholds = np.linspace(0.1,0.33, 100)
    f_close = []
    f_far = []

    for th in thresholds:
        print(th)
        close_idxs = []
        far_idxs = []

        for i, val in enumerate(hatm_dists):
            if val > th:
                far_idxs.append(i)
            else:
                close_idxs.append(i)

        #print(close_idxs)
        #print(far_idxs)

        close_prs = []
        far_prs = []

        for j, val in enumerate(pr_ids):
            if val in close_idxs:
                close_prs.append(j)
            else:
                far_prs.append(j)    

        tl_close = tl[close_prs]
        tp_close = tp[close_prs]
    
        tl_far = tl[far_prs]
        tp_far = tp[far_prs]

        pos=-1
        pl1, rl1, tv1 = precision_recall_curve(1-tl_close, 1.-tp_close)
        for i, val in enumerate(tv1):
            if val > 0.563:
                pos = i
                break
        t = tv1[pos]
        p = pl1[pos]
        r = rl1[pos]
        fs = (1.+0.5**2) * ( (p * r) / (0.5**2 * p + r) )
        f_close.append(fs) 

        pl1, rl1, tv1 = precision_recall_curve(1-tl_far, 1.-tp_far)
        for i, val in enumerate(tv1):
            if val > 0.563:
                pos = i
                break
        t = tv1[pos]
        p = pl1[pos]
        r = rl1[pos]
        fs = (1.+0.5**2) * ( (p * r) / (0.5**2 * p + r) )
        f_far.append(fs)


    plt.plot(thresholds, f_close, label='CLOSE')
    plt.plot(thresholds, f_far, label='FAR')
    plt.xlabel('HATM distance threshold')
    plt.ylabel('$F_{0.5}$')
    plt.legend()
    plt.savefig('./thresh.png')

 
    """
    save_path = '/home/alta/relevance/vr311/models_min_data/baseline/simGrid/ens/rep_and_com/uns_CLOSE'
    path1 = os.path.join(save_path, 'labels.txt')
    np.savetxt(path1, tl_close)
    path2 = os.path.join(save_path, 'predictions.txt')
    np.savetxt(path2, tp_close)
   
    save_path = '/home/alta/relevance/vr311/models_min_data/baseline/simGrid/ens/rep_and_com/uns_FAR'
    path1 = os.path.join(save_path, 'labels.txt')
    np.savetxt(path1, tl_far)
    path2 = os.path.join(save_path, 'predictions.txt')
    np.savetxt(path2, tp_far)
    """

if __name__ == '__main__':
    main()
