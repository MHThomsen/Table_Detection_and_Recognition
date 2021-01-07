import pandas as pd
import numpy as np
from collections import deque




def preprocess_text(text):
    punc = ',.:/"'
    clean_text = ''
    
    for c in text:
        if c not in punc:
            clean_text+=c.lower().strip()
    return clean_text


def give_graph(adjacency_matrix):
    graph = {}
    for i in range(adjacency_matrix.shape[0]):
        row_i = adjacency_matrix[i]
        temp = deque()

        for idx,val in zip(np.argsort(row_i),np.sort(row_i)):
            if val == 0:
                break
            else:
                temp.appendleft(idx)

        #if len(temp) != 0:
        graph[i] = temp

    return graph



def adjacency_graphs(df,Column_THRES=10):

    Row_THRESH = int(np.median(df['height'])*0.5)
    #height_THRES = int(image.shape[0]/3)

    right_matrix = np.zeros(shape=[df.shape[0],df.shape[0]],dtype=int)
    r_down_matrix = np.zeros(shape=[df.shape[0],df.shape[0]],dtype=int)
    left_matrix = np.zeros(shape=[df.shape[0],df.shape[0]],dtype=int)
    l_down_matrix = np.zeros(shape=[df.shape[0],df.shape[0]],dtype=int)
    
    #Extract from pandas into numpy array for much faster processing
    npdf = df[['xmin','xmax','ymin','ymax','page_num']].to_numpy(dtype=int)

    #row indexes: 
    #xmin = 0
    #xmax = 1
    #ymin = 2
    #ymax = 3
    #page_num=4


    for i, row1 in enumerate(npdf):
        for j,row2 in enumerate(npdf):
            #Check if on same page
            if row1[4] == row2[4]:
                #Check if on same line
                if (abs(row1[3] - row2[3]) <= Row_THRESH or abs(row1[2] - row2[2]) <= Row_THRESH):
                    #Word to the right 
                    if row1[1] < row2[0]:
                        right_matrix[i][j] = -abs(row1[1] - row2[0])
                    #Word to the left
                    elif row2[1] < row1[0]:
                        left_matrix[i][j] = -abs(row1[0] - row2[1])
                #Check if words are left aligned:
                elif abs(row1[0] - row2[0]) <= Column_THRES  and row1[3] < row2[2]:
                    l_down_matrix[i][j] = -abs(row1[3] - row2[2])
                #Check if words are right aligned: 
                elif abs(row1[1] - row2[1]) <= Column_THRES and row1[3] < row2[2]:
                    r_down_matrix[i][j] = -abs(row1[3] - row2[2])
                #Check if words are right aligned:
    return give_graph(right_matrix), give_graph(left_matrix), give_graph(r_down_matrix),give_graph(l_down_matrix)




def median_space_distance(df,r):
    dists = []
    xmins = df['xmin']
    xmaxs = df['xmax']
    for idx,line in r.items():
        try:
            idx2 = line[0]
            d = xmins[idx2]-xmaxs[idx]
            dists.append(d)
        except:
            continue
        ll = np.array(dists,dtype=int)   
    return np.median(ll)





def x_dist(df,i,j):
    return df.iloc[j]['xmin'] - df.iloc[i]['xmax']