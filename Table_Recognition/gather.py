import numpy as np


#Collection of different "gather" functions, for preparing word coordinate + feature map data before
#inputting in GCNN

#ALL functions should follow these params:

#input: 
    # vertex_features: batch_size * max_number_of_vertices (250) * 5
    # feature map: batch_size * c * h * w

#return:
    # features: tensor of size batch_size * max_number_of_vetices * num_features, where num_features can vary between functions

class simplest_gather():
    
    def __init__(self):
        self.out_dim = 5
    
    def gather(self,vertex_features,feature_map):
        return vertex_features