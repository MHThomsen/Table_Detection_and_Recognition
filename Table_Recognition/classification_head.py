from torch import nn
from torch.nn import functional as F


def head_loss(predicted_logits,targets):
    loss = nn.BCEWithLogitsLoss()
    
    return loss(predicted_logits,targets)



class head_v1(nn.Module):
    def __init__(self, 
                input_shape,
                hidden_units = 512):
        super(head_v1,self).__init__()
        self.input_shape = input_shape
        self.hidden_units = hidden_units


        self.fc1 = nn.Linear(self.input_shape,self.hidden_units)
        self.fc2 = nn.Linear(self.hidden_units,self.hidden_units)

        self.out == nn.Linear(self.hidden_units,1)


    def forward(self,x):
        x = x.flatten(start_dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        return self.out(x)



            