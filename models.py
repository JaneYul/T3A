import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from layers import *
from utils import *

class TEncoder(nn.Module):
    def __init__(self, dim_in, dim_hid, dim_out, num_channels, num_nodes, dropout):
        super(TEncoder, self).__init__()

        self.gc1 = TEncoderlayer(dim_in, dim_hid, num_channels, num_nodes)
        self.gc15 = TEncoderlayer(dim_hid, dim_hid, num_channels, num_nodes)
        self.gc2 = TEncoderlayer(dim_hid, dim_out, num_channels, num_nodes)
        self.gc3 = TEncoderlayer(dim_in, dim_out, num_channels, num_nodes)
        self.dropout = dropout
        self.dp = nn.Dropout(self.dropout)
        self.nn3 = TensorLinear(num_channels, num_channels, num_nodes, dim_out)   # 第三个维度(num_channels)上的投影

        self.gcn1 = GraphConvolution(dim_in, dim_hid)
        self.gcn2 = GraphConvolution(dim_hid, dim_out)
        self.act = F.relu

    def forward(self, x, adj):

        out = F.relu(self.gc1(x, adj))
        out = F.dropout(out, self.dropout, training=self.training)
        out = self.gc2(out, adj)
        out = self.nn3(out)
        return out



class TDecompose(nn.Module):
    def __init__(self, num_nodes, dim_d, num_channels, dropout, expand, shrink):    # num_nodes dim_d num_channels
        super(TDecompose, self).__init__()
        self.num_nodes = num_nodes
        self.dim_d = dim_d
        self.num_channels = num_channels
        self.dropout = dropout
            
        self.fc1 = nn.Linear(num_channels, num_channels)
        self.fc2 = nn.Linear(dim_d, dim_d)
        self.fc3 = nn.Linear(num_channels, num_channels)
        self.fc4 = nn.Linear(dim_d, dim_d)

        self.act = nn.PReLU()

        n_1, n_2, n_3 = num_nodes, dim_d, num_channels
        self.A1_hat = nn.Parameter(torch.Tensor(n_3*expand, n_1, n_2//shrink))
        self.B1_hat = nn.Parameter(torch.Tensor(n_3*expand, n_2//shrink, n_2))
        self.net1 = nn.Sequential(permute_change(1, 2, 0),
                                  nn.Linear(int(n_3*expand), int(n_3*expand), bias=False),
                                  nn.LeakyReLU(),
                                  nn.Linear(int(n_3*expand), n_3, bias=False))
        self.A2_hat = nn.Parameter(torch.Tensor(n_3*expand, n_1, n_2//shrink))
        self.B2_hat = nn.Parameter(torch.Tensor(n_3*expand, n_2//shrink, n_2))
        self.net2 = nn.Sequential(permute_change(1, 2, 0),
                                  nn.Linear(int(n_3*expand), int(n_3*expand), bias=False),
                                  nn.LeakyReLU(),
                                  nn.Linear(int(n_3*expand), n_3, bias=False))
        self.criterion = nn.MSELoss()

        self.reset_parameters()

    def reset_parameters(self):

        stdv = 1. / math.sqrt(self.A1_hat.size(0))
        self.A1_hat.data.uniform_(-stdv, stdv)
        self.A2_hat.data.uniform_(-stdv, stdv)
        stdv = 1. / math.sqrt(self.B1_hat.size(0))
        self.B1_hat.data.uniform_(-stdv, stdv)
        self.B2_hat.data.uniform_(-stdv, stdv)
         
    def reg_term(t):
        return torch.norm(t) / t.numel()

    def reg_term_one(self):
        return self.reg_term(self.A1_hat) + self.reg_term(self.B1_hat) + self.reg_term(self.A2_hat) + self.reg_term(self.B2_hat)

    def reg_term_all(self):
        reg = 0
        gamma = 0.00001
        for name, p in self.named_parameters():
            if name in ["A1_hat", "A2_hat"]:
                reg += gamma*torch.norm(p[:,1:,:]-p[:,:-1,:],1)
            elif name in ["B1_hat", "B2_hat"]:
                reg += gamma*torch.norm(p[:,:,1:]-p[:,:,:-1],1)
            elif name in ["net1.3.weight", "net2.3.weight"]:
                reg += gamma*torch.norm(p[1:,:]-p[:-1,:],1)
        return reg        

    
    def forward(self, input1, input2):

        res1 = torch.bmm(self.A1, self.B1)          # [n, d, c]  
        res1 = self.fc1(res1)
        res1 = self.act(res1)
        res1 = res1.permute(0, 2, 1)
        res1 = self.fc2(res1)
        res1 = res1.permute(0, 2, 1)
        loss_res1 = torch.norm(res1 - input1)

        res2 = torch.bmm(self.A2, self.B2)            
        res2 = self.fc3(res2)
        res2 = self.act(res2)
        res2 = res2.permute(0, 2, 1)
        res2 = self.fc4(res2)
        res2 = res2.permute(0, 2, 1)
        loss_res2 = torch.norm(res2 - input2)

        return self.A1, self.B1, res1, res2, loss_res1+loss_res2

    def forward_2(self, input1, input2):

        res1 = torch.matmul(self.A1_hat, self.B1_hat)
        res1 = self.net1(res1)
        loss_res1 = self.criterion(res1, input1)

        res2 = torch.matmul(self.A2_hat, self.B2_hat)
        res2 = self.net1(res2) # self.net2(res2)
        loss_res2 = self.criterion(res2, input2)

        return None, None, res1, res2, loss_res1+loss_res2 # +loss_reg

    def forward_ori(self, input):
        res1 = torch.matmul(self.A1_hat, self.B1_hat)
        res1 = self.net1(res1)
        loss_res1 = self.criterion(res1, input) # torch.norm

        return res1, loss_res1 # +loss_reg


class TContrastive(nn.Module):
    def __init__(self, num_nodes, dim_a, dim_b, tau, dropout):
        super(TContrastive, self).__init__()
        self.num_nodes = num_nodes
        self.tau = tau

        self.fc1 = torch.nn.Linear(dim_a*dim_b, dim_a*dim_b)
        self.fc2 = torch.nn.Linear(dim_a*dim_b, dim_a*dim_b)
        self.fc3 = torch.nn.Linear(dim_a*dim_b, dim_a*dim_b)
        self.fc4 = torch.nn.Linear(dim_a*dim_b, dim_a*dim_b)

        self.fc5 = torch.nn.Linear(dim_b, dim_b)
        self.fc6 = torch.nn.Linear(dim_b, dim_b)
        self.fc7 = torch.nn.Linear(dim_a, dim_a)
        self.fc8 = torch.nn.Linear(dim_a, dim_a)

        self.w_r1 = nn.Parameter(torch.FloatTensor([0.5]))
        self.w_r2 = nn.Parameter(torch.FloatTensor([0.5]))

    def projection_r1(self, z):
        z = F.elu(self.fc1(z))
        return self.fc2(z)

    def projection_c1(self, z):
        z = F.elu(self.fc3(z))
        return self.fc4(z)

    def forward_v1(self, h1, h2, batch_size=0):  
        if batch_size == 0:
            l1 = self.semi_loss(h1, h2)    
            l2 = self.semi_loss(h2, h1)
        else:
            l1 = self.batched_semi_loss(h1, h2, batch_size)
            l2 = self.batched_semi_loss(h2, h1, batch_size)

        ret = (l1 + l2) * 0.5
        ret = ret.mean() 

        return ret

    def forward_m1(self, ta, tb, batch_size=0):   
        z1 = ta.reshape(ta.shape[0], -1)
        z2 = tb.reshape(ta.shape[0], -1)
        h1 = self.projection_r1(z1)           
        h2 = self.projection_r1(z2)
        loss_row = self.forward_v1(h1, h2, batch_size)

        z1 = ta.permute(0, 2, 1).reshape(ta.shape[0], -1)
        z2 = tb.permute(0, 2, 1).reshape(ta.shape[0], -1)
        h1 = self.projection_c1(z1)           
        h2 = self.projection_c1(z2)
        loss_col = self.forward_v1(h1, h2)

        w_row = max(self.w_r1[0], 0)
        w_row = min(w_row, 1)
        return w_row*loss_row + (1-w_row)*loss_col



class TCL(nn.Module):
    def __init__(self, dim_in, dim_hid, dim_out, num_channels, num_nodes, tau1, tau2, dim_r, dropout, MC_WAY, expand, shrink):
        super(TCL, self).__init__()
        self.num_channels = num_channels

        self.tencoder = TEncoder(dim_in, dim_hid, dim_out, num_channels, num_nodes, dropout)
        self.tdecompose = TDecompose(num_nodes, dim_out, num_channels, dim_r, dropout, expand, shrink)

        self.contrastive1 = TContrastive(num_nodes, dim_out, num_channels, tau1, dropout)  
        self.MC_WAY = MC_WAY

        self.hidden1 = None
        self.hidden2 = None
        self.hidden_ori = None


    def forward_s1(self, x1, x2, adj1, adj2, batch_size):
        hidden1 = self.tencoder(x1, adj1)   # [2708, 128, 3]  [2708 32, 32]
        hidden2 = self.tencoder(x2, adj2)

        loss_cl_s1 = self.contrastive1.forward_m1(hidden1, hidden2, batch_size=batch_size)

        self.hidden1 = hidden1
        self.hidden2 = hidden2

        return loss_cl_s1

    def get_hidden_ori(self, x, adj):
        xt = x.repeat(self.num_channels, 1, 1).permute(1, 2, 0)
        adjt = adj.to_dense().repeat(self.num_channels, 1, 1).permute(1, 2, 0)
        Z = self.tencoder(xt, adjt)
        self.hidden_ori = Z


    def to_detach(self):
        self.hidden1 = self.hidden1.detach()
        self.hidden2 = self.hidden2.detach()
        self.hidden_ori = self.hidden_ori.detach()


    def forward_s2(self, batch_size=0):
        _, loss_s2 = self.tdecompose.forward_ori(self.hidden_ori)
        return loss_s2

        _, _, rhidden1, rhidden2, loss_res = self.tdecompose.forward_2(self.hidden1, self.hidden2)   # H': [2708, 16, 16], [2708, 16, 16]

        loss_cl_s2 = self.contrastive2.forward_m1(rhidden1, rhidden2, batch_size=batch_size)

        return loss_res, loss_cl_s2

    def embed(self):   
        Zo = self.hidden_ori.reshape(self.hidden_ori.shape[0], -1)

        rhidden_ori, _ = self.tdecompose.forward_ori(self.hidden_ori)
        Zr = rhidden_ori.reshape(rhidden_ori.shape[0], -1)
        return Zr



    def embed_temp(self, x, adj):

        xt = x.repeat(self.num_channels, 1, 1).permute(1, 2, 0)
        adjt = adj.to_dense().repeat(self.num_channels, 1, 1).permute(1, 2, 0)
        Z = self.tencoder(xt, adjt)

        Z = Z.detach().reshape(Z.shape[0], -1)
        return Z

        z01 = self.hidden1.detach().reshape(self.hidden1.shape[0], -1)
        z02 = self.hidden2.detach().reshape(self.hidden2.shape[0], -1)

        Z = torch.cat((z01, z02), dim=1)

        return Z




class TCLS1(nn.Module):
    def __init__(self, dim_in, dim_hid, dim_out, num_channels, num_nodes, tau1, dropout):
        super(TCLS1, self).__init__()
        self.num_channels = num_channels
        self.tencoder = TEncoder(dim_in, dim_hid, dim_out, num_channels, num_nodes, dropout)
        self.contrastive1 = TContrastive(num_nodes, dim_out, num_channels, tau1, dropout)   # 两个对比模块 对比矩阵的大小

    def forward(self, x1, x2, adj1, adj2, batch_size):
        hidden1 = self.tencoder(x1, adj1)   # [2708, 128, 3]  [2708 32, 32]
        hidden2 = self.tencoder(x2, adj2)

        
        loss_cl_s1 = self.contrastive1.forward_m1(hidden1, hidden2, batch_size=batch_size)

        self.hidden1 = hidden1
        self.hidden2 = hidden2
        return loss_cl_s1

    def get_hidden_ori(self, x, adj):
        xt = x.repeat(self.num_channels, 1, 1).permute(1, 2, 0)
        adjt = adj.to_dense().repeat(self.num_channels, 1, 1).permute(1, 2, 0)
        Z = self.tencoder(xt, adjt)
        return Z.detach()

    def embed_temp(self, x, adj):

        xt = x.repeat(self.num_channels, 1, 1).permute(1, 2, 0)
        adjt = adj.to_dense().repeat(self.num_channels, 1, 1).permute(1, 2, 0)
        Z = self.tencoder(xt, adjt)

        Z = Z.detach().reshape(Z.shape[0], -1)
        return Z

        z01 = self.hidden1.detach().reshape(self.hidden1.shape[0], -1)
        z02 = self.hidden2.detach().reshape(self.hidden2.shape[0], -1)

        Z = torch.cat((z01, z02), dim=1)

        return Z


class TCLS2(nn.Module):
    def __init__(self, dim_out, num_channels, num_nodes, dropout, expand, shrink, MC_WAY):
        super(TCLS2, self).__init__()
        self.tdecompose = TDecompose(num_nodes, dim_out, num_channels, dropout, expand, shrink)
        self.MC_WAY = MC_WAY

    def forward(self, hidden_ori, batch_size=0):
        _, loss_s2 = self.tdecompose.forward_ori(hidden_ori)
        return loss_s2

        _, _, rhidden1, rhidden2, loss_res = self.tdecompose.forward_2(self.hidden1, self.hidden2)   # H': [2708, 16, 16], [2708, 16, 16]

        loss_cl_s2 = self.contrastive2.forward_m1(rhidden1, rhidden2, batch_size=batch_size)

        return loss_res, loss_cl_s2

    def embed(self, hidden_ori):     

        rhidden_ori, _ = self.tdecompose.forward_ori(hidden_ori)
        Zr = rhidden_ori.reshape(rhidden_ori.shape[0], -1)
        return Zr







class GCN(nn.Module):
    def __init__(self, dim_in, dim_hid, dim_out, dropout):
        super(GCN, self).__init__()
        self.gc1 = GraphConvolution(dim_in, dim_hid)
        self.gc2 = GraphConvolution(dim_hid, dim_out)
        self.dropout = dropout
        self.act = F.relu

    def forward(self, features, adj):
        out = self.gc1(features, adj)
        out = self.act(out)
        out = F.dropout(out, self.dropout, training=self.training)

        out = self.gc2(out, adj)
        out = self.act(out)
        return out
       
