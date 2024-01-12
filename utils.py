from os import times
from re import M
import numpy as np
import torch
import torch.nn as nn
from collections import defaultdict
import wandb

class permute_change(nn.Module):
    def __init__(self, n1, n2, n3):
        super(permute_change, self).__init__()
        self.n1 = n1
        self.n2 = n2
        self.n3 = n3
        
    def forward(self, x):
        x = x.permute(self.n1, self.n2, self.n3)
        return x

def register(args, counter, seed_count):
    config = vars(args)
    logger = wandb.init(project="v2.20.1", name='test-{}-{}'.format(counter, seed_count), config=config)
    return logger

def find_epoch(acc_epoch_seeds_s1, acc_epoch_seeds_s2, counter):
    res_max = 0
    if acc_epoch_seeds_s2[0] != None:
        for cur in acc_epoch_seeds_s2:
            res_max += cur["max_acc_s2"]
        res_max = res_max / len(acc_epoch_seeds_s2)
        for cur in acc_epoch_seeds_s2:
            cur["max_acc_s2"] = -1

    def find_one(acc_epoch_seeds):
        res = defaultdict(list)
        for i in range(len(acc_epoch_seeds)):
            cur = acc_epoch_seeds[i]
            for (k, v) in cur.items():
                res[k].append(v)

        max_k, max_v = -1, -1
        for (k, v) in res.items():
            mv = np.mean(v)
            # res[k] = mv
            if mv > max_v:
                max_k = k
                max_v = mv
        return max_v, max_k

    if None not in acc_epoch_seeds_s1:
        max_v1, max_k1 = find_one(acc_epoch_seeds_s1)
    else:
        max_v1, max_k1 = 0, 0
    if acc_epoch_seeds_s2[0] != None:
        max_v2, max_k2 = find_one(acc_epoch_seeds_s2)
    else:
        max_v2, max_k2 = 0, 0

    # res_max = []
    # for i in range(len(acc_epoch_seeds)):
    #     cur = acc_epoch_seeds[i]
    #     res_max.append(cur[max(cur)])

    # my-test-project
    # test-citeseer
    config = {"max_acc_avg_s1":max_v1, "max_acc_avg_s1_from":max_k1, "max_acc_avg_s2":max_v2, "max_acc_avg_s2_from":max_k2, \
              "max_s2": res_max}
    temp = wandb.init(project="v2.20.1", name='test-{}-sum'.format(counter), config=config)
    temp.finish()
    print("**test-{}-sum** max_acc_avg1:{} max_acc_avg1_from:{} max_acc_avg2:{} max_acc_avg2_from:{} max_s2:{}" \
            .format(counter, max_v1, max_k1, max_v2, max_k2, res_max))


def repeat(args0):

    # cora
    # lr1_set = [0.0001, 0.0005, 0.001, 0.003]                             # [0.0005]  not ok:[0.003]
    # weight_decay1_set = [0.0001, 0.0005, 0.001, 0.003, 0.005]     # not ok: [0.005] 
    # tau1_set = [0.8]                       # [1.2] [0.8] 
    # batch_size_set = [0]
    # dropout_set = [0.3]                           # [0.1]
    # dim_hid_set = [512]                      # [1024]

    # citeseer
    # lr1_set = [0.0005] # [0.0005, 0.001]      # not ok: [0.0001, 0.003]
    # weight_decay1_set = [0.1] # [0.005, 0.01, 0.05, 0.1] # [0.03, 0.08] [0.3, 0.5]     # not ok: [0.0001] must try: [0.008, 0.01]
    # tau1_set = [1.3]                             # [1.2] [0.8] 
    # batch_size_set = [256]  # [128, 256, 512, 0]        # [64]
    # dropout_set = [0.3]                       
    # dim_hid_set = [512]     # [256, 512]                 # [1024]
    # dim_out_set = [16]
    # times_aug_set = [16]

    # cora_ml
    # lr1_set = [0.0005, 0.001]  
    # weight_decay1_set = [0.01, 0.1] 
    # tau1_set = [1.1, 1.2, 1.3]              
    # batch_size_set = [256, 512]  # [128, 256, 512, 0]  not ok: [0, 128]
    # dropout_set = [0.3]                       
    # dim_hid_set = [512]  # not ok : [256]  
    # dim_out_set = [16]
    # times_aug_set = [16]
    # drop_edge_rate_set = [[0.2, 0.3], [0.3, 0.4]]
    # drop_feature_rate_set = [[0.2, 0.3], [0.1, 0.3]]

    # for lr1 in lr1_set:
    #     for weight_decay1 in weight_decay1_set:
    #         for tau1 in tau1_set:
    #             for batch_size in batch_size_set:
    #                 for dropout in dropout_set:
    #                     for dim_hid in dim_hid_set:
    #                         for dim_out in dim_out_set:
    #                             for times_aug in times_aug_set:
    #                                 for de in drop_edge_rate_set:
    #                                     for df in drop_feature_rate_set:
    #                                         args0.lr1 = lr1
    #                                         args0.weight_decay1 = weight_decay1
    #                                         args0.tau1 = tau1
    #                                         args0.batch_size = batch_size
    #                                         args0.dropout = dropout
    #                                         args0.dim_hid = dim_hid
    #                                         args0.dim_out = dim_out
    #                                         args0.times_aug = times_aug
    #                                         args0.drop_edge_rate = de
    #                                         args0.drop_feature_rate = df
    #                                         yield args0
                                    
    args0.name_hid = "test-101"
    args0.READ = False
    args0.lr1 = 0.0005
    args0.weight_decay1 = 0.1
    args0.tau1 = 1.1
    args0.batch_size = 256
    args0.dim_hid = 512
    args0.epochs1 = 150
    args0.drop_edge_rate = [0.2, 0.3]
    args0.drop_feature_rate = [0.2, 0.3]

    # lr2_set = [0.0005, 0.001, 0.003, 0.005, 0.01]  
    # weight_decay2_set = [0.00005, 0.0001, 0.0003, 0.0005, 0.001] # cora_ml not ok: 0.03# citeseer not ok: 0.003, 0.001, 0.0005
    # expand_set = [2]    # [2]
    # shrink_set = [1, 2, 3]
    # for lr2 in lr2_set:
    #     for weight_decay2 in weight_decay2_set:
    #         for expand in expand_set:
    #             for shrink in shrink_set:
    #                 args0.lr2 = lr2
    #                 args0.weight_decay2 = weight_decay2
    #                 args0.expand = expand
    #                 args0.shrink = shrink
    #                 yield args0  
    args0.lr2 = 0.005
    args0.weight_decay2 = 0.0001
    args0.expand = 1
    args0.shrink = 1
    yield args0

    # dim_out_set = [16, 24, 32]
    # times_aug_set = [16, 24, 32]
    # for dim_out in dim_out_set:
    #     for times_aug in times_aug_set:
    #         args0.dim_out = dim_out
    #         args0.times_aug = times_aug
    #         yield args0
    # args0.dim_out = 16
    # args0.times_aug = 16

    # drop_edge_rate_set = [[0.2, 0.3], [0.1, 0.3], [0.1, 0.2]]
    # drop_feature_rate_set = [[0.2, 0.3], [0.2, 0.4], [0.1, 0.2]]
    # for drop_edge_rate in drop_edge_rate_set:
    #     for drop_feature_rate in drop_feature_rate_set:
    #         args0.drop_edge_rate = drop_edge_rate
    #         args0.drop_feature_rate = drop_feature_rate
    #         yield args0
    # args0.drop_edge_rate = [0.2, 0.3]
    # args0.drop_feature_rate = [0.2, 0.3]

    # tau_set = [0.7, 0.8, 0.9]
    # batch_size_set = [64, 128, 256, 0]
    # dropout_set = [0.3]
    # dim_hid_set = [256, 512, 1024]
    # for tau in tau_set:
    #     for batch_size in batch_size_set:
    #         for dropout in dropout_set:
    #             for dim_hid in dim_hid_set:
    #                 args0.tau1, args0.tau2 = tau, tau
    #                 args0.batch_size = batch_size
    #                 args0.dropout = dropout
    #                 args0.dim_hid = dim_hid
    #                 yield args0
    # args0.tau = 0.8
    # args0.batch_size = 0
    # args0.dropout = 0.3
    # args0.dim_hid = 512

    # dim_out_set = [16, 24, 32]
    # times_aug_set = [16, 24, 32]
    # for dim_out in dim_out_set:
    #     for times_aug in times_aug_set:
    #         args0.dim_out = dim_out
    #         args0.times_aug = times_aug
    #         yield args0

    # for citeseer
    # args0.dataset = 'citeseer'
    # dim_out_set = [12, 16, 18]
    # times_aug_set = [12, 16, 18]
    # dim_r_set = [12, 16]
    # for dim_out in dim_out_set:
    #     for times_aug in times_aug_set:
    #         for dim_r in dim_r_set:
    #             if dim_r >= dim_out or dim_r >= times_aug:
    #                 continue
    #             args0.dim_out = dim_out
    #             args0.times_aug = times_aug
    #             args0.dim_r = dim_r
    #             yield args0

    # drop_edge_rate_set = [[0.1, 0.2, 0.3], [0.1, 0.2, 0.1], [0.2, 0.3, 0.4]]
    # drop_feature_rate_set = [[0.1, 0.2, 0.3], [0.1, 0.2, 0.1], [0.2, 0.3, 0.4]]
    # for drop_edge_rate in drop_edge_rate_set:
    #     for drop_feature_rate in drop_feature_rate_set:
    #         args0.drop_edge_rate = drop_edge_rate
    #         args0.drop_feature_rate = drop_feature_rate
    #         yield args0

    # lr_set = [0.0001, 0.0005, 0.001, 0.003]
    # tau_set = [0.6, 0.7, 0.8, 0.9]
    # batch_size_set = [64, 128, 256]
    # for lr in lr_set:
    #     for tau in tau_set:
    #         for batch_size in batch_size_set:
    #             args0.lr = lr
    #             args0.tau1, args0.tau2 = tau, tau
    #             args0.batch_size = batch_size
    #             yield args0
                
