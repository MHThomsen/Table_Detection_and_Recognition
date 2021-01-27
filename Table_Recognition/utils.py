import numpy as np
import torch
import tfrecord
import cv2

from tfrecord.torch.dataset import TFRecordDataset


def visibility_matrix(torch_df,num_words):
    '''indentify neighbours to the right and down and generate visibility matrix / neighbourhood graph.
        for each node, we indentify it's closest neighbour to the right and the closest neighbour below.
    input: numpy array of shape (words, [x1, x2, y1, y2])
    output: visibility matrix of shape (words, words)'''
    
    #remove last column (word_length)
    npdf = torch_df.numpy()
    
    #Only create matrix of size matching number of words
    matrix = np.zeros((num_words, num_words))

    for i,row1 in enumerate(npdf):
        if i == num_words:
            break

        #xmin = 0
        #ymin = 1
        #xmax = 2
        #ymax = 3 

        min_down = 10**6
        min_right = 10**6
        min_down_idx = None
        min_right_idx = None

        for j,row2 in enumerate(npdf):
            if j == num_words:
                break
            if i != j:
                #Right neighbour
                if row1[1] <= row2[1] <= row1[3] or row1[1] <= row2[3] <= row1[3] or row2[1] <= row1[1] <= row2[3] or row2[1] <= row1[3] <= row2[3]:
                    if  0 <= row2[0]-row1[2] <= min_right:
                        min_right_idx, min_right = j, row2[0]-row1[2]

                #Down neighbour
                if row1[0] <= row2[0] <= row1[2] or row1[0] <= row2[2] <= row1[2] or row2[0] <= row1[0] <= row2[2] or row2[0] <= row1[2] <= row2[2]:
                    if 0 <= row2[1]-row1[3] <= min_down:
                        min_down_idx, min_down = j, row2[1]-row1[3]

        if min_right_idx != None:
            matrix[i,min_right_idx] = 1
            matrix[min_right_idx, i] = 1    
        if min_down_idx != None:
            matrix[i,min_down_idx] = 1
            matrix[min_down_idx, i] = 1
            
    source = []
    target = []

    for i, row in enumerate(matrix):
        for j, edge in enumerate(row):
            if edge == 1:
                source.append(i)
                target.append(j)

    edge_index = torch.tensor([source, target], dtype=torch.long)

    return edge_index


def tfrecord_transforms(elem,
                   max_height = 768,
                   max_width = 1366,
                   num_of_max_vertices = 250,
                   max_length_of_word = 30,
                   batch_size = 8):
    """
    Function used to transform the data loaded by the TFRecord dataloader.
    Parameters are defind in TIES datageneration, defines the size and complexity of the generated tables. DO NOT CHANGE  
    """
    with torch.no_grad():
    
        #Everything is flattened in tfrecord, so needs to be reshaped. 

        #Images are in range [0,255], need to be in [0,1]
        #If image max is over 1 , then normalize: 
        data_dict =  {}
        
        #Torch dimensions: B x C x H x W
        #inputting grayscale, so only 1 dimension
        
        if torch.max(elem['image']) > 1:
            data_dict['imgs'] = (elem['image']/255).reshape(batch_size,1,max_height,max_width)
        else:
            data_dict['imgs'] = elem['image'].reshape(batch_size,1,max_height,max_width)
         
        #Extract number of words for each image:
        num_words = elem['global_features'][:,2]
        data_dict['num_words'] = num_words

        
        
        v = elem['vertex_features'].reshape(batch_size,num_of_max_vertices,5).float()
        #normalizaing words coordinates to be invariant to image size 
        v[:,:,0] = v[:,:,0]/max_width
        v[:,:,1] = v[:,:,1]/max_height
        v[:,:,2] = v[:,:,2]/max_width
        v[:,:,3] = v[:,:,3]/max_height

        #data_dict['vertex_features'] = v

        vertex_feats = []
        for idx,vf in enumerate(v):
            tmp = vf[0:num_words[idx]]
            #tmp.requires_grad=True
            vertex_feats.append(tmp)

        data_dict['vertex_features'] = vertex_feats  
                
        #Calculate visibility matrix for each batch element
        edge_index = []
        for idx,vex in enumerate(v):
            edge_index.append(visibility_matrix(vex,num_words[idx]))
         
        data_dict['edge_index'] = edge_index

        #TODO maybe "shave" matrices down to num_words*num_words in size. 
        #Reshape adjacency matrices
        data_dict['adjacency_matrix_cells'] = elem['adjacency_matrix_cells'].reshape(batch_size,
                                                                                     num_of_max_vertices,
                                                                                     num_of_max_vertices)
        
        data_dict['adjacency_matrix_cols'] = elem['adjacency_matrix_cols'].reshape(batch_size,
                                                                                     num_of_max_vertices,
                                                                                     num_of_max_vertices)
        
        data_dict['adjacency_matrix_rows'] = elem['adjacency_matrix_rows'].reshape(batch_size,
                                                                                     num_of_max_vertices,
                                                                                     num_of_max_vertices)
        
        return data_dict

def rescale_img_quad(image,
                    output_size=600,
                    crop_first = True):
    
    """function to rescale images to fit to input dimension of feature CNN (default 600). 
    Quadratic image, so largest dimension is fitted, and smallest dimension is padded with 0

    output_size: output height and width of image
    crop_first: images received imght have white borders, which messes up rscaling. Ability to first crop images 
    # TODO Find a way to make cropping and retain corresponding image values between feature map and image data. 
    """
    image = image[0,:,:].numpy()
    h, w = image.shape[:2]    
    H = False
    if h > w:
        H = True
        scaling_factor = h/output_size
        new_h = output_size
        new_w = int(w/scaling_factor)
    else:
        scaling_factor = w/output_size
        new_h = int(h/scaling_factor)
        new_w = output_size
    
    dim = (new_w,new_h)
    img = cv2.resize(image,dim)
    
    
    #Now for padding
    #Center padding
    if H:
        #Height is largest dimension, so pad width
        pad_l = int((output_size-new_w)/2)
        
        #Test if dimensions fit: 
        if 2*pad_l + new_w != output_size:
            #correct the number by adding a small value to right padding
            pad_r = (output_size - (pad_l*2 + new_w)) + pad_l 
        else:
            pad_r = pad_l
            
        img = cv2.copyMakeBorder(img,0,0,pad_l,pad_r,cv2.BORDER_CONSTANT,value=1)
    else:
        #Width is largest dimension
        
        pad_t = int((output_size-new_h)/2)
        
        if 2*pad_t + new_h != output_size:
            pad_b = (output_size - (2*pad_t + new_h)) + pad_t
        else:
            pad_b = pad_t
        
        img = cv2.copyMakeBorder(img,pad_t,pad_b,0,0,cv2.BORDER_CONSTANT,value=1)

    
    out = torch.tensor(img).reshape(1,output_size,output_size)
    return out