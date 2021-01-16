import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class SimpleNet(torch.nn.Module):

    def __init__(self,
                in_features,
                out_features):
        super(SimpleNet, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        self.GCNconv1 = GCNConv(self.in_features, 16)
        self.GCNconv2 = GCNConv(16, self.out_features)
 
    def forward(self, x, edge_index):
        
        x = self.GCNconv1conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.GCNconv2conv2(x, edge_index)

        return F.log_softmax(x, dim=1)