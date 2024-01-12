import numpy as np
import torch
import sys
import pickle
import scipy
import scipy.sparse as sp
import networkx as nx

import os.path as osp
from torch_geometric.datasets import Planetoid, CitationFull, WikiCS, Coauthor, Amazon
import torch_geometric.transforms as T
# from ogb.nodeproppred import PygNodePropPredDataset

'''
---------------------------- load data ----------------------------
'''

def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):           
        index.append(int(line.strip()))
    return index                           

def sample_mask(idx, l):
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)


def load_data_all(dataset_str):
    if dataset_str in ['cora', 'citeseer']:
        adj, features, labels, idx_train, idx_val, idx_test = load_data(dataset_str)

        y = torch.argmax(torch.Tensor(labels), dim=1)           

        print("features size:", features.shape, "  y size:", y.shape)

        features, _ = preprocess_features(features)        
        adj = normalize_adj(adj + sp.eye(adj.shape[0]))    

    
        features = torch.FloatTensor(features).cuda()             
        labels = torch.FloatTensor(labels).long().cuda()
        idx_train = torch.LongTensor(idx_train).cuda()
        idx_val = torch.LongTensor(idx_val).cuda()
        idx_test = torch.LongTensor(idx_test).cuda()

        sparse = True
        if sparse:
            adj = sparse_mx_to_torch_sparse_tensor(adj).cuda()
        else:                                      
            adj = torch.Tensor(adj.todense()).cuda()

        return features, adj, labels, y


    elif dataset_str in ['WikiCS']:
        dataset = load_data_other(dataset_str)
        data = dataset[0]
        edge_index = data.edge_index
        eles = torch.ones(edge_index.size(1)).long()
        adj = scipy.sparse.csr.csr_matrix((eles, (edge_index[0], edge_index[1])), shape=(data.x.size(0), data.x.size(0)))
        adj = normalize_adj(adj + sp.eye(adj.shape[0]))     # scipy.sparse.coo.coo_matrix
        sparse = True
        if sparse:
            adj = sparse_mx_to_torch_sparse_tensor(adj).cuda()    
        else:                                      
            adj = torch.Tensor(adj.todense()).cuda()

        features = torch.FloatTensor(data.x).cuda()
        y = torch.LongTensor(data.y).cuda()

        return features, adj, None, y

from deeprobust.graph.data import Dataset

def load_npz_lcc(name):
    data = Dataset(root='./data/{}/'.format(name), name=name, setting='prognn')
    adj, features, labels = data.adj, data.features, data.labels
    idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test
    labels = torch.LongTensor(labels)

    y = torch.LongTensor(labels)
    classes = max(labels) + 1 
    one_hot_label = np.zeros(shape=(labels.shape[0], classes))
    one_hot_label[np.arange(0,labels.shape[0]), labels] = 1
    labels = torch.Tensor(one_hot_label)

    return adj, features, labels, y, idx_train, idx_val, idx_test

def load_data(dataset_str, npz=True): 
    if npz == False and dataset_str in ["cora", "citeseer"]:
    
        """Load data."""
        names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
        objects = []
        for i in range(len(names)):
            with open("./data/{}/ind.{}.{}".format(dataset_str, dataset_str, names[i]), 'rb') as f:
                if sys.version_info > (3, 0):
                    objects.append(pickle.load(f, encoding='latin1'))
                else:
                    objects.append(pickle.load(f))

        x, y, tx, ty, allx, ally, graph = tuple(objects)

        test_idx_reorder = parse_index_file("./data/{}/ind.{}.test.index".format(dataset_str, dataset_str))   
        test_idx_range = np.sort(test_idx_reorder)                                                       

        if dataset_str == 'citeseer':                                                   
            test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
            tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))       
            tx_extended[test_idx_range-min(test_idx_range), :] = tx                   
            tx = tx_extended                                                     
            ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))                 
            ty_extended[test_idx_range-min(test_idx_range), :] = ty
            ty = ty_extended                                                              

        features = sp.vstack((allx, tx)).tolil()                                        
        features[test_idx_reorder, :] = features[test_idx_range, :]                      
        adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))                           

        labels = np.vstack((ally, ty))
        labels[test_idx_reorder, :] = labels[test_idx_range, :]

        idx_test = test_idx_range.tolist()              
        idx_train = range(len(y))                       
        idx_val = range(len(y), len(y)+500)              

        y = torch.argmax(torch.Tensor(labels), dim=1)
        print("features size:", features.shape, "  y size:", y.shape)

        return adj, features, labels, y, idx_train, idx_val, idx_test   

    elif npz == True:
        
        adj_matrix, attr_matrix, labels, y, idx_train, idx_val, idx_test = load_npz_sym(dataset_str)


        return adj_matrix, attr_matrix, labels, y, idx_train, idx_val, idx_test


def load_npz_sym(file_name):
    from deeprobust.graph.data import Dataset

    data = Dataset(root='./data/{}/'.format(file_name), name=file_name, setting='prognn')
    adj, features, labels = data.adj, data.features, data.labels
    idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test

    y = torch.LongTensor(labels)
    classes = max(labels) + 1 
    one_hot_label = np.zeros(shape=(labels.shape[0], classes))
    one_hot_label[np.arange(0,labels.shape[0]), labels] = 1
    labels = torch.Tensor(one_hot_label)

    return adj, features, labels, y, idx_train, idx_val, idx_test


def load_npz(file_name, to_sym=True):

    if not file_name.endswith('.npz'):
        file_name += '.npz'
    with np.load("./data/cora_ml/"+file_name, allow_pickle=True) as loader:
        loader = dict(loader)
        adj_matrix = sp.csr_matrix((loader['adj_data'], loader['adj_indices'],
                                              loader['adj_indptr']), shape=loader['adj_shape'])

        if to_sym == True:
            temp = adj_matrix+adj_matrix.T
            temp.data = np.ones(len(temp.data))
            adj_matrix = temp

        if 'attr_data' in loader:
            attr_matrix = sp.csr_matrix((loader['attr_data'], loader['attr_indices'],
                                                   loader['attr_indptr']), shape=loader['attr_shape'])
        else:
            attr_matrix = None

        labels = loader.get('labels')
        y = torch.LongTensor(labels)

        classes = max(labels) + 1 
        one_hot_label = np.zeros(shape=(labels.shape[0], classes))
        one_hot_label[np.arange(0,labels.shape[0]), labels] = 1

        idx_test = range(1)
        idx_train = range(1)
        idx_val = range(1)

    return adj_matrix, attr_matrix, labels, y, idx_train, idx_val, idx_test


def load_data_other(dataset_str):
    path = osp.expanduser('./data')
    path = osp.join(path, dataset_str)

    root_path = osp.expanduser('./datasets')

    if dataset_str == 'WikiCS':
        return WikiCS(root=path, transform=T.NormalizeFeatures())
    
    elif dataset_str == 'Coauthor-CS':
        return Coauthor(root=path, name='cs', transform=T.NormalizeFeatures())

    elif dataset_str == 'Coauthor-Phy':
        return Coauthor(root=path, name='physics', transform=T.NormalizeFeatures())

    elif dataset_str == 'Amazon-Computers':
        return Amazon(root=path, name='computers', transform=T.NormalizeFeatures())

    elif dataset_str == 'Amazon-Photo':
        return Amazon(root=path, name='photo', transform=T.NormalizeFeatures())

    return (CitationFull if dataset_str == 'dblp' else Planetoid)(osp.join(root_path, 'Citation'), dataset_str, transform=T.NormalizeFeatures())
    


def sparse_to_tuple(sparse_mx, insert_batch=False):
    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()                            
        if insert_batch:
            coords = np.vstack((np.zeros(mx.row.shape[0]), mx.row, mx.col)).transpose()
            values = mx.data
            shape = (1,) + mx.shape
        else:
            coords = np.vstack((mx.row, mx.col)).transpose()  
            values = mx.data                          
            shape = mx.shape                             
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])   
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx


def preprocess_features(features):         
    rowsum = np.array(features.sum(1))      
    r_inv = np.power(rowsum, -1).flatten() 
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)            
    features = r_mat_inv.dot(features)    
    return features.todense(), sparse_to_tuple(features)


def normalize_adj(adj):
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))                   
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()  
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)          
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo() 

def preprocess_adj(adj):
    adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))   
    return sparse_to_tuple(adj_normalized)


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)   


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

