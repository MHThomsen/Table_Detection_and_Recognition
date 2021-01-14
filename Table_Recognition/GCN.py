import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class SimpleNet(torch.nn.Module):

    def __init__(self, num_features):
        super(Net, self).__init__()
        self.num_features = num_features

        self.conv1 = GCNConv(self.num_features, 16)
        self.conv2 = GCNConv(16, 32)

    def forward(self, x, edge_index):
        
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1)