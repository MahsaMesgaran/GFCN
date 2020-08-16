import torch.nn as nn
import torch.nn.functional as F
from layers import GraphConvolution


class GCN(nn.Module):
    def __init__(self, nfeat, nhid , nclass, dropout):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nfeat, nhid)
        self.gc3 = GraphConvolution(nhid, nclass)
        self.gc4 = GraphConvolution(nfeat, nclass)
        self.dropout = dropout
        
    
    def forward(self, x, adj):
        
        a_skip0 = self.gc2(x)
        a_skip0 = F.dropout(a_skip0, self.dropout, training=self.training)        
        a_skip1 = self.gc4(x)
        a_skip1 = F.dropout(a_skip1, self.dropout, training=self.training)
        
        m = nn.RReLU()
        x = m(self.gc1(x, adj) + a_skip0)
        x = self.gc3(x, adj) + a_skip1

        self.w1 = self.gc1.weight
        self.w2 = self.gc2.weight
        self.w3 = self.gc3.weight
        self.w4 = self.gc4.weight
        
        return x, self.w1, self.w2, self.w3, self.w4

    