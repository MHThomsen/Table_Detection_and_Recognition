import numpy as np
import torch

#Collection of different "gather" functions, for preparing word coordinate + feature map data before
#inputting in GCNN

#ALL functions should follow these params:

#input: 
    # vertex_features: batch_size * max_number_of_vertices (250) * 5
    # feature map: batch_size * c * h * w

#return:
    # features: tensor of size batch_size * max_number_of_vetices * num_features, where num_features can vary between functions


class simplest_gather():
    
    def __init__(self, collapser_function = None):
        self.out_dim = 4
    
    def gather(self,vertex_features,feature_map):
        return vertex_features[:,:4]


class slice_gather():

    def __init__(self,collapser_function):
        
        self.collapser_func = collapser_function
        self.out_dim = 4 + self.collapser_func.out_dim

    def gather(self,vertex_features,feature_map):
    
        img_l, img_h = (768, 1366)

        #x, y -dimensions of feature map
        max_x, max_y = feature_map.shape[1], feature_map.shape[2]

        l,h = 43.0, 16.0

        #the height of a single pixel
        #pxl_l, pxl_h = 1/feature_map.shape[1], 1/feature_map.shape[2]
        
        image_feats = []
        for word in vertex_features:
            #get x,y coordinates of word
            x1, y1, x2, y2, _ = word.cpu().numpy()
            #for this, we denormalize in order to utilize np.floor and np.ceil
            x1, y1, x2, y2 = x1*img_l, y1*img_h, x2*img_l, y2*img_h 
            #calculate centre of word
            c = np.floor(((x1+(x2-x1)/2),(y1+(y2-y1)/2)))

            #define x-coordinates of slice
            x_slice = int(c[0]-np.floor(l/2)), int(c[0]+np.ceil(l/2))
            if min(x_slice) < 0:
                x_slice = (x_slice[0]+abs(min(x_slice)),x_slice[1]+abs(min(x_slice)))
            if max(x_slice) > max_x:
                x_slice = (x_slice[0]-(max(x_slice)-max_x),x_slice[1]-(max(x_slice)-max_x))

            #define y-coordinates of slice
            y_slice = int(c[1]-np.floor(h/2)),int(c[1]+np.ceil(h/2))
            if min(y_slice) < 0:
                y_slice = (y_slice[0]+abs(min(y_slice)),y_slice[1]+abs(min(y_slice)))
            if max(y_slice) > max_y:
                y_slice = (y_slice[0]-(max(y_slice)-max_y),y_slice[1]-(max(y_slice)-max_y))

            #get 3D slice of word, flatten and concatenate with vertex features
            slice_3d = feature_map[:,x_slice[0]:x_slice[1],y_slice[0]:y_slice[1]]

            slice_collapsed = self.collapser_func.collapse(slice_3d)
            image_feats.append(torch.cat((word[:4], slice_collapsed)))
        return torch.stack(image_feats,dim=0)


    

'''        collapsus = collapser_function(channels,width,height) '''
