'''
graph2head.py'''

import numpy as np



indsæt = 4

matrix = data_dict['adjacency_matrix_rows'][indsæt]
num_words = data_dict['num_words'][indsæt]




def generate_distances(mans, num_words, matrix, max_samps = 5, training=True):
    num_features = mans.shape[1]
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




    
    #test: 
    