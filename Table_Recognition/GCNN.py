import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class SimpleNet(torch.nn.Module):

    def __init__(self,
                out_features,
                num_features=None):
        super(SimpleNet, self).__init__()
        #self.num_features = num_features
        self.out_features = out_features
        self.conv1 = None
        self.conv2 = None

    def define_layers(self,num_features):
        #Created as the "num_features" parameter is not defined yet, when network is initialized

        self.num_features = num_features
        self.conv1 = GCNConv(self.num_features, 16)
        self.conv2 = GCNConv(16, self.out_features)

 
    def forward(self, x, edge_index):
        assert self.conv1 is not None, "Remeber to run 'define_layers' method before forward pass!!"
        
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1)