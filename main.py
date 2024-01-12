# from unicodedata import name
import numpy as np
import torch

import time
import argparse
import random
import os
from torch.utils.tensorboard import SummaryWriter

from sklearn.preprocessing import normalize, OneHotEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import f1_score

from preprocess import *
from aug import *
from models import *
from eval import *
from utils import *

LOG_FILE = "param-v2.30.4"
DETAIL_FILE = 'log/cora_ml/details-v2.30.4.txt'

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str,  default='cora_ml', help='Dataset:cora/citeseer/WikiCS')
    parser.add_argument('--sparse', type=bool,  default=True, help='')
    parser.add_argument('--seed', type=int,  default=1232, help='1232')

    parser.add_argument('--times_aug', type=int,  default=16, help='16')
    parser.add_argument('--drop_edge_rate', nargs="+", type=float, default=[0.2, 0.3], help='')
    parser.add_argument('--drop_feature_rate', nargs="+", type=float, default=[0.2, 0.3], help='')

    # train
    parser.add_argument('--epochs1', type=int,  default=400, help='')
    parser.add_argument('--epochs2', type=int,  default=3000, help='')
    parser.add_argument('--batch_size', type=int,  default=0, help='')
    parser.add_argument('--lr1', type=float,  default=0.0005, help='')    
    parser.add_argument('--lr2', type=float,  default=0.01, help='')
    parser.add_argument('--weight_decay1', type=float,  default=0.001, help='0.0005') 
    parser.add_argument('--weight_decay2', type=float,  default=0.0001, help='0.0005') 

    # tencoder
    parser.add_argument('--dim_hid', type=int,  default=256, help='256')  
    parser.add_argument('--dim_out', type=int,  default=16, help='16')   
    parser.add_argument('--dropout', type=float,  default=0.3, help='0.3')

    # contrastive
    parser.add_argument('--tau1', type=float,  default=0.8, help='0.8')
    parser.add_argument('--tau2', type=float,  default=0.8, help='0.8')

    # tdecompose
    # parser.add_argument('--dim_r', type=int,  default=12, help='18')
    parser.add_argument('--expand', type=int,  default=2, help='1')
    parser.add_argument('--shrink', type=int,  default=1, help='2')

    # model-setting
    parser.add_argument('--NUM_MODEL', type=str,  default='two-model', help='one-model, two-model')
    parser.add_argument('--NUM_LR', type=str,  default='two-lr', help='one-lr, two-lr')

    # perturb
    parser.add_argument('--attack', type=str, default="no", help="no, random, meta, nettack")  
    parser.add_argument('--ptb_rate', type=float, default=0.05, help="noise ptb_rate")

    args = parser.parse_args()
    return args


def run(args, counter, seed_count):
    dataset, sparse = args.dataset, args.sparse
    times_aug, drop_edge_rate, drop_feature_rate = args.times_aug, args.drop_edge_rate, args.drop_feature_rate
    epochs1, epochs2 = args.epochs1, args.epochs2
    batch_size, lr1, lr2, weight_decay1, weight_decay2 = args.batch_size, args.lr1, args.lr2, args.weight_decay1, args.weight_decay2
    dim_hid, dim_out, dropout = args.dim_hid, args.dim_out, args.dropout
    tau1, tau2 = args.tau1, args.tau2
    expand, shrink = args.expand, args.shrink
    NUM_MODEL, NUM_LR = args.NUM_MODEL, args.NUM_LR
    name_hid, READ = args.name_hid, args.READ

    seed = args.seed          # 1234
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    print("counter", counter, seed_count)
    time_start = time.time()


    if args.ptb_rate == 0:   
        adj, features, labels, y, idx_train, idx_val, idx_test = load_npz_lcc(args.dataset)
        if args.attack == "nettack":
            from deeprobust.graph.data import Dataset, PrePtbDataset
            perturbed_data = PrePtbDataset(root='../data/perturb/nettack/', name=dataset, attack_method="nettack",
                                        ptb_rate=0.05)

            idx_test = perturbed_data.get_target_nodes()
    else:                    
        adj, features, labels, y, idx_train, idx_val, idx_test = load_data(dataset)

    print("num_nodes", features.shape[0])
    print("num_features", features.shape[1])
    print("num_edges", adj.sum())

    ''' perturb '''
    
    from deeprobust.graph.global_attack import Random
    from deeprobust.graph.data import Dataset, PrePtbDataset

    attack = args.attack       # "random"
    ptb_rate = args.ptb_rate
    if attack == "no" or ptb_rate == 0:
        perturbed_adj = adj
    elif attack == "random":
        attacker = Random()
        n_perturbations = int(ptb_rate * (adj.sum()//2))
        attacker.attack(adj, n_perturbations, type='add')    # 只能add或者remove，flip是针对有向图的翻转  5278*2+527*2=11610 加也是对称加
        perturbed_adj = attacker.modified_adj
        perturbed_adj = perturbed_adj.tocsr()
    elif attack == 'meta' or attack == 'nettack':
        data = Dataset(root='../data/raw/', name=dataset, setting='prognn')
        adj, features, labels = data.adj, data.features, data.labels
        idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test
        
        y = torch.LongTensor(labels)
        classes = max(labels) + 1 
        one_hot_label = np.zeros(shape=(labels.shape[0], classes))
        one_hot_label[np.arange(0,labels.shape[0]), labels] = 1
        labels = torch.Tensor(one_hot_label)

        perturbed_data = PrePtbDataset(root='../data/perturb/{}/'.format(attack), name=dataset, attack_method=attack,
                                        ptb_rate=ptb_rate if ptb_rate > 0 else 1.0)
        perturbed_adj = perturbed_data.adj if ptb_rate > 0 else adj    
        if attack == 'nettack':
            idx_test = perturbed_data.get_target_nodes()

    adj = perturbed_adj
    num_nodes = features.shape[0]
    print("num_nodes", num_nodes)

    ''' preprocess '''

    features, _ = preprocess_features(features)         
    adj = normalize_adj(adj + sp.eye(adj.shape[0]))    
    features = torch.FloatTensor(features).cuda()              

    sparse = True
    if sparse:
        adj = sparse_mx_to_torch_sparse_tensor(adj).cuda()   
    else:                                      
        adj = torch.Tensor(adj.todense()).cuda()

    ''' augmentation '''
    
    def get_aug():
        x_aug1 = torch.FloatTensor(features.shape[0], features.shape[1], times_aug).cuda()
        x_aug2 = torch.FloatTensor(features.shape[0], features.shape[1], times_aug).cuda()
        adj_aug1 = None
        adj_aug2 = None

        for i in range(times_aug):
            # x_aug1[:,:,i] = drop_feature(features, drop_feature_rate[i%3])
            # x_aug2[:,:,i] = drop_feature(features, drop_feature_rate[i%3])
            x_aug1[:,:,i] = drop_feature(features, drop_feature_rate[0])
            x_aug2[:,:,i] = drop_feature(features, drop_feature_rate[1])
            if i == 0:
                adj_aug1 = drop_adj(adj, drop_edge_rate[0]).unsqueeze(2)
                adj_aug2 = drop_adj(adj, drop_edge_rate[1]).unsqueeze(2)
            else:
                adj_aug1 = torch.cat((adj_aug1, drop_adj(adj, drop_edge_rate[0]).unsqueeze(2)), dim=2)
                adj_aug2 = torch.cat((adj_aug2, drop_adj(adj, drop_edge_rate[1]).unsqueeze(2)), dim=2)
        
        return x_aug1, x_aug2, adj_aug1, adj_aug2


    modelS1 = TCLS1(dim_in=features.shape[1], dim_hid=dim_hid, dim_out=dim_out, num_channels=times_aug, num_nodes=num_nodes, tau1=tau1, dropout=dropout).cuda()
    modelS2 = TCLS2(dim_out=dim_out, num_channels=times_aug, num_nodes=num_nodes, dropout=dropout, expand=expand, shrink=shrink).cuda()
    optimizerS1 = torch.optim.Adam(modelS1.parameters(), lr=lr1, weight_decay=weight_decay1)
    optimizerS2 = torch.optim.Adam(modelS2.parameters(), lr=lr2, weight_decay=weight_decay2)

    aug_store = None
    def train_s1(epoch):
        modelS1.train()
        optimizerS1.zero_grad()

        x_aug1, x_aug2, adj_aug1, adj_aug2 = get_aug()

        loss_cl_s1 = modelS1.forward(x_aug1, x_aug2, adj_aug1, adj_aug2, batch_size=batch_size)
        loss = loss_cl_s1
        loss.backward()
        optimizerS1.step()
        return loss_cl_s1

    def train_s2(hidden_ori):
        modelS2.train()
        optimizerS2.zero_grad()

        loss_res2 = modelS2.forward(hidden_ori)
        loss = loss_res2
        loss.backward()
        optimizerS2.step()
        return loss_res2

        loss_res, loss_cl_s2 = model.forward_s2(batch_size=batch_size)
        loss = loss_res + loss_cl_s2
        loss.backward()
        optimizer.step()
        return loss_res, loss_cl_s2

    def test_temp():
        modelS1.eval()
        Z = modelS1.embed_temp(features, adj)     
        if attack in {"no", "random", "meta"}:
            print(1)
            res = classification(Z.detach(), y, ratio=0.1)    
            return res["F1Mi"]["mean"]                          # for no, meta
        elif attack == "nettack":
            print(2)
            res = label_classification_2(Z.detach(), labels, idx_train, idx_val, idx_test)    # for nettack
            return res

    def test(hidden_ori):
        modelS2.eval()
        Z = modelS2.embed(hidden_ori)   
        if attack in {"no", "random", "meta"}:
            print(1)
            res = classification(Z.detach(), y, ratio=0.1)    
            return res["F1Mi"]["mean"]                          # for no, meta
        elif attack == "nettack":
            print(2)
            res = label_classification_2(Z.detach(), labels, idx_train, idx_val, idx_test)    # for nettack
            return res

    READ = False
    if READ == False or os.path.exists("./hid/{}/v1.30.2/s1-{}-seed{}-e{}.pt".format(dataset, name_hid, str(seed), str(epochs1)) ) == False:
        # Stage 1
        max_acc_temp, epoch_temp = 0, 0
        loss_temp = 9999999
        acc_loss, epoch_loss = None, None
        acc_epoch_all_s1 = {}
        for epoch in range(epochs1+1):
            losses = []
            
            loss = train_s1(epoch)
            losses.append(loss.detach().cpu())

            if epoch % 20 == 0:
                print("Epoch:{:4d} Loss:{:4f}".format(epoch, np.array(losses).mean(axis=0) ))
            if epoch % 50 == 0:
                acc_temp = test_temp()
                
                # max acc
                if acc_temp > max_acc_temp:
                    max_acc_temp = acc_temp
                    epoch_temp = epoch
                # min loss 
                loss_e = np.array(losses).mean(axis=0)
                if loss_e < loss_temp:
                    loss_temp = loss_e
                    acc_loss = acc_temp
                    epoch_loss = epoch
                # logger
                acc_epoch_all_s1.update({"Epoch-{}".format(epoch): acc_temp})

        print("[s1 max acc]  Epoch: {} Acc_temp:{:4f}".format(epoch_temp, max_acc_temp))
        print("[s1 min loss] Epoch: {} Loss: {:4f} Acc_loss:{:4f}".format(epoch_loss, loss_temp, acc_loss))
        
        with open('./log/{}/{}.txt'.format(dataset, LOG_FILE), 'a') as f:
            f.write(str(args))
            f.write('\n')
            f.write("[s1 max acc]  Epoch: {} Acc_temp:{:4f}\n".format(epoch_temp, max_acc_temp))
            f.write("[s1 min loss] Epoch: {} Loss: {:4f} Acc_loss:{:4f}\n".format(epoch_loss, loss_temp, acc_loss))

        modelS1.eval()
        hidden_ori = modelS1.get_hidden_ori(features, adj)

    elif READ == True:
        hidden_ori = torch.load("./hid/{}/v1.30.2/s1-{}-seed{}-e{}.pt".format(dataset, name_hid, str(seed), str(epochs1)) )
    
        Z_ori = hidden_ori.reshape(num_nodes, -1)
        res = classification(Z_ori, y, ratio=0.1)
        acc_epoch_all_s1 = {"Epoch-{}".format(epochs1): res["F1Mi"]["mean"]}

    # Stage 2
    acc_max = 0
    epoch_acc_max = 0
    max_acc_temp, epoch_temp = 0, 0
    loss_temp = 9999999
    acc_loss, epoch_loss = None, None
    acc_epoch_all_s2 = {}
    for epoch in range(epochs2+1):
        losses = []

        loss_res = train_s2(hidden_ori)
        losses.append(loss_res.detach().cpu())

        if epoch % 50 == 0:
            acc = test(hidden_ori)
            print("Epoch:{:5d} Loss:{:4f}".format(epoch, np.array(losses).mean(axis=0)) )
            if acc > acc_max:
                acc_max = acc
                epoch_acc_max = epoch
            loss_e = np.array(losses).mean(axis=0)
            if loss_e < loss_temp:
                loss_temp = loss_e
                acc_loss = acc
                epoch_loss = epoch
            acc_epoch_all_s2.update({"s2-Epoch-{}".format(epoch): acc})

    
    # --- test ---

    print("[s2 max acc] Max acc: {:4f} from Epoch {:5d}".format(acc_max, epoch_acc_max))
    print("[s2 min loss] Epoch: {} Loss: {:4f} Acc_loss:{:4f}".format(epoch_loss, loss_temp, acc_loss))
    acc_epoch_all_s2.update({"max_acc_s2": acc_max})
    return acc_epoch_all_s1, acc_epoch_all_s2



if __name__ == "__main__":

    args0 = parse_args()

    counter = 0
    
    seed_set =  [15610, 77401, 29614, 78084, 1517, 25616]
    
    for args in iter(repeat(args0)): 
        acc_epoch_seeds_s1, acc_epoch_seeds_s2 = [], []
        counter += 1
        seed_count = 0
        for seed in seed_set:
            args.seed = seed
            seed_count += 1
            acc_epoch_all_s1, acc_epoch_all_s2 = run(args, counter, seed_count)
            acc_epoch_seeds_s1.append(acc_epoch_all_s1)
            acc_epoch_seeds_s2.append(acc_epoch_all_s2)
        find_epoch(acc_epoch_seeds_s1, acc_epoch_seeds_s2, counter)

        




