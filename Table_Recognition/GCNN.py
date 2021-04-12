import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GravNetConv


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



class SimpleNetDeep(nn.Module):

    def __init__(self,
                in_features,
                out_features):
        super(SimpleNetDeep, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        self.GCNconv1 = GCNConv(self.in_features, 64)

        self.GCNconv2 = GCNConv(64, 64)
        self.GCNconv3 =  GCNConv(64, 64)
        self.GCNconv4 =  GCNConv(64, self.out_features)
 
    def forward(self, x, edge_index):
        
        x = self.GCNconv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.GCNconv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.GCNconv3(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)

        x = self.GCNconv4(x, edge_index)
        return x


class SimpleGravNet(nn.Module):

    def __init__(self,
                in_features,
                out_features):
        super(SimpleGravNet, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        self.GravNetconv1 = GravNetConv(self.in_features, 36, 4, 22, 40)
        self.GravNetconv2 = GravNetConv(36, 36, 4, 22, 40)
        self.GravNetconv3 = GravNetConv(36, 48, 4, 22, 40)
        self.GravNetconv4 = GravNetConv(48, 48, 4, 22, 40)
        self.Lout = nn.Linear(self.GravNetconv1.out_channels + self.GravNetconv2.out_channels
                              + self.GravNetconv3.out_channels + self.GravNetconv4.out_channels, self.out_features )
        
 
    def forward(self, x, __):
        
        x1 = self.GravNetconv1(x)
        #x = F.relu(x)
        #x = F.dropout(x, training=self.training)
        x2 = self.GravNetconv2(x1)
        
        x3 = self.GravNetconv3(x2)
        #x = F.relu(x)
        #x = F.dropout(x, training=self.training)
        x4 = self.GravNetconv4(x3)
                
        return F.relu(self.Lout(torch.cat([x1,x2,x3,x4], dim=1)))



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
        




        

        return x