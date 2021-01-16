import torch
from torch.autograd import Variable
from torch.nn.parameter import Parameter
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.init as init

from torch.nn import Linear, Conv2d, BatchNorm2d, MaxPool2d, Dropout2d, AvgPool2d
from torch.nn.functional import relu, elu, relu6, sigmoid, tanh, softmax




def compute_conv_dim(dim_size,kernel_size,padding,stride):
    return int((dim_size - (kernel_size-1) + 2 * padding -1) / stride + 1)


# define network
class FeatureNet_v1(nn.Module):
    def __init__(self,input_channels=1
                    ,img_h=766
                    ,img_w=1366
                    ,num_classes=4):
        super(FeatureNet_v1, self).__init__()
        self.input_channels = input_channels
        
        self.img_h = img_h
        self.img_w = img_w
        

        self.num_classes = num_classes
        
        
        self.conv_1 = Conv2d(in_channels=self.input_channels,
                             out_channels=8,
                             kernel_size=3,
                             stride=1,
                             padding=0)
        
        dim_c1_w = compute_conv_dim(self.img_w,3,0,1)
        dim_c1_h = compute_conv_dim(self.img_h,3,0,1)
        
        self.conv_2 = Conv2d(in_channels=8,
                            out_channels = 16,
                            kernel_size = 3,
                            stride=1,
                            padding=0)
        
        dim_c2_w = compute_conv_dim(dim_c1_w,3,0,1)
        dim_c2_h = compute_conv_dim(dim_c1_h,3,0,1)
        
        self.pool_1 = MaxPool2d(5,stride=2)
        
        dim_p1_w = compute_conv_dim(dim_c2_w,5,0,2)
        dim_p1_h = compute_conv_dim(dim_c2_h,5,0,2)
        
        self.conv_3 = Conv2d(in_channels=16,
                            out_channels=32,
                            kernel_size=5,
                            stride=1,
                            padding=0)
        
        dim_c3_w = compute_conv_dim(dim_p1_w,5,0,1)
        dim_c3_h = compute_conv_dim(dim_p1_h,5,0,1)
        
        self.pool_2 = AvgPool2d(5,stride=3)
        
        dim_p2_w = compute_conv_dim(dim_c3_w,5,0,3)  
        dim_p2_h = compute_conv_dim(dim_c3_h,5,0,3)  
        
      
        self.dropout = Dropout2d(p=0.5)
       
        
        
        #parameters for output layer = dim_p2*dim_p2*128
        self.l1_in_features = dim_p2_w*dim_p2_h*32
        
        self.l_1 = Linear(in_features=self.l1_in_features,
                        out_features=128)
        
        
        self.l_out = Linear(in_features=128, 
                            out_features=num_classes,
                            bias=False)
    
    def forward(self, x): # x.size() = [batch, channel, height, width]
        x = self.conv_1(x)
        x = elu(self.conv_2(x))
        x = self.pool_1(x)
        x = elu(self.conv_3(x))
        x = self.pool_2(x)

        x = self.dropout(x)
        
        # torch.Tensor.view: http://pytorch.org/docs/master/tensors.html?highlight=view#torch.Tensor.view
        #   Returns a new tensor with the same data as the self tensor,
        #   but of a different size.
        # the size -1 is inferred from other dimensions 
        
        x = x.view(-1,self.l1_in_features)
        x = relu(self.l_1(x))
        return softmax(self.l_out(x),dim=1)

    def feature_forward(self,x): # x.size() = [batch, channel, height, width]
        #After training network, then use this to output feature map
        with torch.no_grad():
            if self.training:
                print("Model in Training mode - exiting forward pass.")
                return None
            else:
                x = elu(self.conv_1(x))
                x = elu(self.conv_2(x))
                x = self.pool_1(x)
                x = elu(self.conv_3(x))
                x = self.pool_2(x)
        return x
        

