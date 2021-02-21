from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class SimpleNet(nn.Module):

    def __init__(self,
                in_features,
                out_features):
        super(SimpleNet, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        self.GCNconv1 = GCNConv(self.in_features, 64)
        self.GCNconv2 = GCNConv(64, self.out_features)
 
    def forward(self, x, edge_index):
        
        x = self.GCNconv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.GCNconv2(x, edge_index)

        return x



class FullyConnectNet(nn.Module):
    def __init__(self,
                in_features,
                out_features):
        super(FullyConnectNet,self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.l1 = nn.Linear(self.in_features,128)
        self.l2 = nn.Linear(128,256)
        self.l3  = nn.Linear(256,128)
        self.lout = nn.Linear(128,self.out_features)

    def forward(self,x, *args):
        x = F.elu(self.l1(x))
        x = F.elu(self.l2(x))
        x = F.dropout(x,0.3)
        x = F.elu(self.l3(x))
        x = F.elu(self.lout(x))

        return x