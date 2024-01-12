import numpy as np
import torch


def drop_feature(x, drop_prob):
    drop_mask = torch.empty((x.size(1), ), dtype=torch.float32, device=x.device).uniform_(0, 1) < drop_prob
    x = x.clone()
    x[:, drop_mask] = 0

    return x 


def drop_adj(adj, drop_prob):


    indices = adj.coalesce().indices()       
    values = adj.coalesce().values()        
    drop_mask = torch.empty((indices.size(1), ), dtype=torch.float32, device=adj.device).uniform_(0, 1) > drop_prob
    indices_n = indices[:, drop_mask]
    values_n = values[drop_mask]
    
    return torch.sparse.FloatTensor(indices_n, values_n, adj.coalesce().size())


def add_adj(adj, add_prob):
    
    indices = adj.coalesce().indices() 
    values = adj.coalesce().values() 
    num_nodes = adj.size(0)
    num_add_edge = int(values.size(0) * add_prob)

    edge_add = np.random.choice(num_nodes, (num_add_edge, 2)).tolist()
    edge_ori = adj.coalesce().indices().t().cpu().numpy().tolist()
    edge_add_f = []
    for e in edge_add:
        if e not in edge_ori:
            edge_add_f.append(e)
    edge_add_f = torch.LongTensor(edge_add_f).t().cuda()
    edge_all = torch.cat((indices, edge_add_f), dim=1)

    values_add = torch.FloatTensor(np.random.choice(values.cpu().numpy(), edge_add_f.size(1))).cuda()
    values_all = torch.cat((values, values_add))

    return torch.sparse.FloatTensor(edge_all, values_all, adj.coalesce().size())


def disturb_adj(adj, prob):
    drop_prob = prob / 2
    add_prob = prob / 2

    indices = adj.coalesce().indices() 
    values = adj.coalesce().values()

    drop_mask = torch.empty((indices.size(1), ), dtype=torch.float32, device=adj.device).uniform_(0, 1) > drop_prob
    indices_n = indices[:, drop_mask]
    values_n = values[drop_mask]

    num_nodes = adj.size(0)
    num_add_edge = int(values.size(0) * add_prob)
    edge_add = np.random.choice(num_nodes, (num_add_edge, 2)).tolist()
    edge_ori = adj.coalesce().indices().t().cpu().numpy().tolist()
    edge_add_f = []
    for e in edge_add:
        if e not in edge_ori:
            edge_add_f.append(e)
    edge_add_f = torch.LongTensor(edge_add_f).t().cuda()
    values_add = torch.FloatTensor(np.random.choice(values.cpu().numpy(), edge_add_f.size(1))).cuda()
    
    edge_all = torch.cat((indices_n, edge_add_f), dim=1)
    values_all = torch.cat((values_n, values_add))

    return torch.sparse.FloatTensor(edge_all, values_all, adj.coalesce().size())


