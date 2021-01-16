import numpy as np

import torch 
import torch.nn as nn
import torch.nn.functional as F

#import simplest network modules
from feature_CNN import FeatureNet_v1 
from GCNN import SimpleNet
from classification_head import head_v1
from gather import simplest_gather
from dist_funcs import abs_dist
#from dist_funcs import simple_dist



class VexMoutNet(nn.Module):
    #Final network, encapsulating all modules
    def __init__(self, 
                feature_net = None,
                gcnn = None,
                classification_head = None,
                gather_func = None,
                distance_func = None,
                img_h = 768,
                img_w = 1366,
                gcn_out_dim=100,
                max_sampling_size=5):
        super(VexMoutNet,self).__init__()
        self.feature_net = feature_net
        self.gcnn = gcnn
        
        
        self.classification_head_cells = classification_head
        self.classification_head_rows = classification_head
        self.classification_head_cols = classification_head
        
        self.gather_func = gather_func
        self.distance_func = distance_func
        self.max_sampling_size = max_sampling_size


        if self.feature_net is None:
            self.feature_net = FeatureNet_v1()

        if self.gather_func is None:
            self.gather_func = simplest_gather()

        if self.gcnn is None:
            self.gcnn = SimpleNet(in_features = self.gather_func.out_dim,out_features=gcn_out_dim)

        if self.distance_func is None:
            self.distance_func = abs_dist

        if self.classification_head_cells is None:
            self.classification_head_cells = self.classification_head_rows = self.classification_head_cols = head_v1(input_shape=gcn_out_dim)

        #Freeze feature net parameters
        for param in self.feature_net.parameters():
            param.requires_grad=False


    def head_loss(self,predicted_logits,targets):
        loss = nn.BCEWithLogitsLoss()
        return loss(predicted_logits,targets)





    def forward(self,data_dict):
        #input is dictionary of all data

        #after running through feature net, gcn then sampling happens. Thus after sampling we reconstruct the ground truth data, based on the sample and three adjacency matrices
        #output should be (predictions, targets)_rows/cols/cells

        #data_dict:
        #'imgs': images in a batch
        #'num_words': number of words on each image 
        #'vertex_features': (x1,y1,x2,y2,word_lenth) of all words on each image
        #'edge_index' : created from visibility graph, adjacency input to GCNN
        #'adjacency_matrix_cells' 
        #'adjacency_matrix_cols'
        #'adjacency_matrix_rows'

        
        
        #Do not include feature generation or gather function in gradients
        with torch.no_grad:
            #Create feature map
            features = self.feature_net.feature_forward(data_dict['imgs'])

            #Gather function:
            gcn_features = self.gather_func.gather(data_dict['vertex_features'],features)

        
        #Initialize GCNN layers (they depend on the number of features from the gather function)
        self.gcn.define_layers(num_features=gcn_features.shape(2))


        graph_features = self.gcn(gcn_features,data_dict['edge_index'])


        #Now ready to input into 3 separate classification heads

        #if training, take sample from adjacency matrix
        

        #TODO find ud hvordan faster rcnn bruger loss dict
        loss_dict = {}
        
        if self.training:
            #Cells

            #extract targets from adjacency matrix based on indexes

            #extract features and calculate distance function on all pairs

            #put features through classification head


            #calculate loss based 

            loss_dict['cells'] = None






        else:
            predictions = self.classification_head(graph_features)











def simple_dist(vertex_features):
    return None


def get_sample_features(sample,vertex_features,dist_func):
    temp = vertex_features.numpy()
    feats = []
    for p in sample:
        feats.append(dist_func(temp[p[0]],temp[p[1]]))
    
    return torch.tensor(feats)

def get_sample_targets(sample,matrix):
    targets = []
    for p in sample:
        targets.append(matrix[p])
    return torch.stack(targets)

def even_sampling(num_words, matrix, max_samps = 5):
    #removed "mans" from inputs
    #num_features = mans.shape[1]
    matrix = matrix[:num_words,:num_words].numpy()

    pairs = set()
    neg_pairs = set()



    #training: monte carlo sampling
    for idx, row in enumerate(matrix):
        neighbours = np.where(row == 1)[0]
        neighbours = neighbours[neighbours != idx]
        non_neighbours = np.where(row == 0)[0]
        num_neighbours = len(neighbours)
        num_non_neighbours = len(non_neighbours)


        if num_neighbours!=0:
            #we take up to max_samps positive matches
            pos_idx = np.random.choice(neighbours, size=num_neighbours, replace=False)
            neg_idx = np.random.choice(non_neighbours, size=num_non_neighbours, replace=False)


        if num_non_neighbours == 0:
            max_samps = 1

        if num_neighbours == 0:
            cnt_p = 1
        else:
            cnt_p = 0
            for i in pos_idx:
                pair = (min([idx, i]),max([idx, i]))

                #Avoid duplicate pairs
                if pair not in pairs:
                    cnt_p += 1

                    #INSERT get_distance(pair, method) function that takes in a pair of indexes and returns
                    pairs.add(pair)

                if cnt_p == max_samps:
                    break


        if cnt_p > 0:
            cnt_n = 0
            for j in neg_idx:
                pair = (min([idx, j]),max([idx, j]))

                #Avoid duplicate pairs
                if pair not in neg_pairs:
                    cnt_n+=1

                    #INSERT get_distance(pair, method) function that takes in a pair of indexes and returns
                    neg_pairs.add(pair)

                if cnt_n >= cnt_p:
                    break

    #Concatenate the two sets
    #First some sanity check
    l1 = len(pairs)
    l2 = len(neg_pairs)
    
    pairs.update(neg_pairs)
    
    assert len(pairs) == l1+l2, "Merging pairs and neg_pairs resulted in overlap. It should not!"
    
    
    return list(pairs)
    

