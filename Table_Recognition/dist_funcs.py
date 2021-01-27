import torch

def abs_dist(features1,features2):
    '''
    Takes as input two tensors and calculates difference funcion. 
    '''
    
    return torch.abs(features1-features2)