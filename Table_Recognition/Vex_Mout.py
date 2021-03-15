import numpy as np

import torch 
import torch.nn as nn
import torch.nn.functional as F

#import simplest network modules
from feature_CNN import FeatureNet_v1 
from GCNN import SimpleNet
from classification_head import head_v1
from gather import simplest_gather,slice_gather
import collapser_funcs
from dist_funcs import abs_dist
from utils import get_stats



from time import process_time



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
                gcnn_out_dim=16,
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

        if self.gather_func is None:

            #TODO uncomment
            #self.gather_func = simplest_gather()

            #TODO define slice size in config
            #self.collapser_func = collapser_funcs.mean_channel_collapser(32,43,16)
            #self.collapser_func = collapser_funcs.max_collapser(32,43,16)
            self.collapser_func = collapser_funcs.mean_2d_collapser(32,43,16)

            self.gather_func = slice_gather(self.collapser_func)




        if self.feature_net is None:
            self.gather_func = simplest_gather()
            print("No featureNet defined in VexMout - using simplestgather func!")
        else:
            #Freeze feature net parameters
            for param in self.feature_net.parameters():
                param.requires_grad=False

                    

        if self.gcnn is None:
            self.gcnn = SimpleNetDeep(in_features = self.gather_func.out_dim,out_features=gcnn_out_dim)

        if self.distance_func is None:
            self.distance_func = abs_dist

        if self.classification_head_cells is None:
            self.classification_head_cells = head_v1(input_shape=gcnn_out_dim)
            self.classification_head_rows = head_v1(input_shape=gcnn_out_dim)
            self.classification_head_cols = head_v1(input_shape=gcnn_out_dim)



    def head_loss(self,predicted_logits,targets,pos_weight):
        loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        return loss(predicted_logits,targets)

    def get_sample_targets(self,sample,matrix):
        
        #TODO Kan nok optimeres
        targets = []
        for p in sample:
            pair = (p[0],p[1])
            targets.append(matrix[pair])
        return torch.stack(targets)

    def get_sample_features(self,sample,vertex_features,dist_func):
        return dist_func(vertex_features[sample[:,0]],vertex_features[sample[:,1]])

    def even_sampling(self,num_words, matrix, max_samps = 5):
        #removed "mans" from inputs
        #num_features = mans.shape[1]
        matrix = matrix[:num_words,:num_words].cpu().numpy()

        pairs = set()
        neg_pairs = set()
        
        #training: monte carlo sampling
        for idx, row in enumerate(matrix):
            neighbours = np.where(row == 1)[0]
            neighbours = neighbours[neighbours != idx]
            non_neighbours = np.where(row == 0)[0]
            num_neighbours = len(neighbours)
            num_non_neighbours = len(non_neighbours)
            pos_idx = None
            neg_idx = None


            if num_neighbours!=0:
                #Randomly permute index arrays
                pos_idx = np.random.choice(neighbours, size=num_neighbours, replace=False)
                neg_idx = np.random.choice(non_neighbours, size=num_non_neighbours, replace=False)

                

            if num_non_neighbours == 0:
                max_samps = 1

            if num_neighbours == 0:
                neg_idx = np.random.choice(non_neighbours,size=1,replace=False)
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
        npos = len(pairs)
        nneg = len(neg_pairs)
        
        pairs.update(neg_pairs)
        
        assert len(pairs) == npos+nneg, "Merging pairs and neg_pairs resulted in overlap. It should not!"
        
        
        return torch.tensor(list(pairs)),npos,nneg




    def forward(self,
                data_dict, device,
                pred_thresh=0.5):
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


        batch_size = len(data_dict['vertex_features'])


        #Do not include feature generation or gather function in gradients
        
        with torch.no_grad():
            #Create feature map
            
            if self.feature_net is not None:
                feature_map = self.feature_net.feature_forward(data_dict['imgs'])
            else:
                feature_map = [None]*batch_size    
            #Gather function:
            gcnn_input_features = []
            for i in range(batch_size):
                gcnn_input_features.append(self.gather_func.gather(data_dict['vertex_features'][i],feature_map[i]))

            #gcnn_input_features = self.gather_func.gather(data_dict['vertex_features'],feature_map)

       
        '''
        ################################ ACTIVATE GRADIENTS ######################################
        '''
        
        graph_features = []
        for i in range(batch_size):
            graph_features.append(self.gcnn(gcnn_input_features[i],data_dict['edge_indexes'][i]))

        
        #Now ready to input into 3 separate classification heads

        #if training, take sample from adjacency matrices
        if self.training:
            cells_tgt = []
            cols_tgt = []
            rows_tgt = []

            cells_feat = []
            cols_feat = []
            rows_feat = []  

            cells_p = 0
            cells_n = 0
            cols_p = 0
            cols_n = 0
            rows_p = 0
            rows_n = 0 

            for idx,nw in enumerate(data_dict['num_words']):
                
                cells_sample,cells_pos,cells_neg = self.even_sampling(nw,data_dict['adjacency_matrix_cells'][idx])
                cells_tgt.append(self.get_sample_targets(cells_sample,data_dict['adjacency_matrix_cells'][idx]))   
                cells_feat.append(self.get_sample_features(cells_sample,graph_features[idx],self.distance_func))
                cells_p+=cells_pos
                cells_n+=cells_neg
                
                cols_sample,cols_pos,cols_neg = self.even_sampling(nw,data_dict['adjacency_matrix_cols'][idx])
                cols_tgt.append(self.get_sample_targets(cols_sample,data_dict['adjacency_matrix_cols'][idx]))
                cols_feat.append(self.get_sample_features(cols_sample,graph_features[idx],self.distance_func))
                cols_p+=cols_pos
                cols_n+=cols_neg
                
                rows_sample,rows_pos,rows_neg = self.even_sampling(nw,data_dict['adjacency_matrix_rows'][idx])    
                rows_tgt.append(self.get_sample_targets(rows_sample,data_dict['adjacency_matrix_rows'][idx]))
                rows_feat.append(self.get_sample_features(rows_sample,graph_features[idx],self.distance_func))
                rows_p+=rows_pos
                rows_n+=rows_neg
            
            
            #Collect everything in batch to single tensor, to pass throgh classification head    
            cells_targets = torch.cat(cells_tgt,dim=0).float().to(device)
            cells_features = torch.cat(cells_feat,dim=0)
            
            cols_targets = torch.cat(cols_tgt,dim=0).float().to(device)
            cols_features = torch.cat(cols_feat,dim=0)
            
            rows_targets = torch.cat(rows_tgt,dim=0).float().to(device)
            rows_features = torch.cat(rows_feat,dim=0)
            
            #Get predictions
            cells = self.classification_head_cells(cells_features).reshape(-1).to(device)
            cols = self.classification_head_cols(cols_features).reshape(-1).to(device)
            rows = self.classification_head_rows(rows_features).reshape(-1).to(device)

          
            loss_cells = self.head_loss(cells,cells_targets,torch.tensor([cells_n/cells_p]).to(device))
            loss_cols = self.head_loss(cols,cols_targets,torch.tensor([cols_n/cols_p]).to(device))
            loss_rows = self.head_loss(rows,rows_targets,torch.tensor([rows_n/rows_p]).to(device))
            
            #Calculate statistics 
            stat_dict = get_stats(cells,cols,rows,cells_targets,cols_targets,rows_targets,pred_thresh)


            return loss_cells, loss_cols, loss_rows, stat_dict

        else:
            with torch.no_grad():
                
                #TODO Delete variables before this step to clear memory

                #################### FEATURES ####################
                #Get features as tensor of size sum(num_words)_over_batch x num_features
                f = []
                for idx,nw in enumerate(data_dict['num_words']):
                    
                    
                    #Create two different tensor repeats. Gives all combinations of words
                    #with 3 repeats:
                    #t1: [1,2,3] -> [1,1,1,2,2,2,3,3,3]
                    #t2: [1,2,3] -> [1,2,3,1,2,3,1,2,3]
                    t1 = torch.repeat_interleave(graph_features[idx], repeats=nw.item(), dim=0)
                    t2 = graph_features[idx].repeat(nw.item(),1)

                    f.append(torch.stack([t1,t2],dim=1))
                tmp = torch.cat(f,dim=0)

                

                #calculate distance function on all pairs:
                classification_features = self.distance_func(tmp[:,0,:],tmp[:,1,:])
                del tmp 
                
                pred_dict = {}
                #Get predictions on
                pred_dict['cells'] = self.classification_head_cells(classification_features)
                pred_dict['cols'] = self.classification_head_cols(classification_features)
                pred_dict['rows'] = self.classification_head_rows(classification_features)


                return pred_dict





                









