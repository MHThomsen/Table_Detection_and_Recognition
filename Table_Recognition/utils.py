import numpy as np



def visbility_matrix(npdf):
    '''indentify neighbours to the right and down and generate visibility matrix / neighbourhood graph.
        for each node, we indentify it's closest neighbour to the right and the closest neighbour below.
    input: numpy array of shape (words, [x1, x2, y1, y2])
    output: visibility matrix of shape (words, words)'''

    
    matrix = np.zeros((npdf.shape[0], npdf.shape[0]))

    for i,row1 in enumerate(npdf):

        min_down = 10^6
        min_right = 10^6
        min_down_idx = None
        min_right_idx = None

        for j,row2 in enumerate(npdf):
            if i != j:
                #Right neighbour
                if row1[2] <= row2[2] <= row1[3] or row1[2] <= row2[3] <= row1[3]:
                    if  0 < row2[0]-row1[1] <= min_right:
                        min_right_idx, min_right = j, row2[0]-row1[1]

                #Down neighbour
                if row1[0] <= row2[0] <= row1[1] or row1[0] <= row2[1] <= row1[1]:
                    if 0 < row2[2]-row1[3] <= min_down:
                        min_down_idx, min_down = j, row2[2]-row1[3]

        if min_right_idx != None:
            matrix[i,min_right_idx] = 1
            matrix[min_right_idx, i] = 1    
        if min_down_idx != None:
            matrix[i,min_down_idx] = 1
            matrix[min_down_idx, i] = 1
            
    return matrix


