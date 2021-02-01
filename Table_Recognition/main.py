import torch
import tfrecord
import numpy as np
import os
import pickle

from tfrecord.torch.dataset import TFRecordDataset

from utils import get_all_targets, get_stats, tfrecord_preparer
from feature_CNN import FeatureNet_v1
from GCNN import SimpleNet
from Vex_Mout import VexMoutNet
import config
from tqdm import tqdm


from time import process_time

"Very early version of main loop. Testing imports and pipeline atm."


#Load list of tfRecords from folder: 
folder_path = os.getcwd()+r'\tfrecords'
#folder_path = "C:\Users\Jesper\Desktop\DataGeneration\Data_Outputs"

#load filenames of folder: 
tfrecord_files = os.listdir(folder_path)



#######################################################################################################
########################################### Params ####################################################
#######################################################################################################
batch_size = 8

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

num_epochs = 5

prediction_thres = 0.5



#######################################################################################################
########################################### Load models ###############################################
#######################################################################################################

#load Feature CNN model
featurenet_path = os.getcwd()+r"\models\FeatureNet_v1.pt"
featurenet = FeatureNet_v1()
featurenet.load_state_dict(torch.load(featurenet_path,map_location=torch.device('cpu')))
featurenet.eval()


model = VexMoutNet()


# move model to the right device
model.to(device)

#stats of previous training. loss decay + f1 development
Stats = {}

# construct an optimizer
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005,
                            momentum=0.9, weight_decay=0.0005)
# and a learning rate scheduler
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                step_size=3,
                                                gamma=0.1)




#######################################################################################################
########################################### Training Loop #############################################
#######################################################################################################

p = process_time()




for epoch in range(num_epochs):
    data_time = 0
    trans_time = 0
    forward_time = 0
    
    
    
    
    model.train()
    loop = tqdm(enumerate(tfrecord_files), total=len(tfrecord_files))
    for idx, record in loop:

        t = process_time()
        tfrecord_path = os.path.join(folder_path,record)
        #Maybe tfrecords need to be generated with more files, to make loading more effective? 
        dataset = TFRecordDataset(tfrecord_path, config.index_path, config.tfrecord_description)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
        
        data_time+=process_time()-t
        for batch in loader:
            t = process_time()
            data_dict = tfrecord_preparer(batch,device=device)
            trans_time+=process_time()-t

            optimizer.zero_grad()

            t = process_time()
            loss_cells, loss_cols, loss_rows, stat_dict = model(data_dict,prediction_thres)
            forward_time+= process_time()-t
            
            total_loss = loss_cells + loss_cols + loss_rows 
            
            total_loss.backward()
            optimizer.step()
            lr_scheduler.step()

            #Iterate over, put tensors to device
            loop.set_description(f'Train Epoch [{epoch}/{num_epochs-1}]')
            loop.set_postfix_str(s=f"Total_loss = {round(total_loss.item(),4)}, Cells = {round(loss_cells.item(),4)}, Cols = {round(loss_cols.item(),4)}, Rows = {round(loss_rows.item(),4)}, F1_Cells = {round(stat_dict['cells']['f1'],4)}, F1_Cols = {round(stat_dict['cols']['f1'],4)}, F1_Rows = {round(stat_dict['rows']['f1'],4)}")

    torch.cuda.empty_cache()
    model.eval()
    ################
    #TODO indsæt VALIDATION DATA i stedet for tfrecordfiles 
    #####################
    validation_batch_size = 2
    loop = tqdm(enumerate(tfrecord_files), total=len(tfrecord_files))
    

    valid_forward_time = 0
    valid_targets_time = 0
    valid_head = 0
    valid_stat = 0
    valid_trans = 0

    for idx, record in loop:
        tfrecord_path = os.path.join(folder_path,record)
        #Maybe tfrecords need to be generated with more files, to make loading more effective? 
        dataset = TFRecordDataset(tfrecord_path, config.index_path, config.tfrecord_description)
        loader = torch.utils.data.DataLoader(dataset, batch_size=validation_batch_size)
        for batch in loader:
            t = process_time()
            data_dict = tfrecord_preparer(batch,device=device,batch_size=validation_batch_size)
            valid_trans += process_time()-t


            t = process_time()
            preds_dict = model(data_dict,prediction_thres)
            valid_forward_time+=process_time()-t


            t = process_time()
            targets_cells, targets_cols, targets_rows = get_all_targets(data_dict)
            valid_targets_time+= process_time()-t


            t = process_time()
            loss_cells = model.head_loss(preds_dict['cells'].reshape(-1),targets_cells)
            loss_cols =  model.head_loss(preds_dict['cols'].reshape(-1),targets_cols)
            loss_rows = model.head_loss(preds_dict['rows'].reshape(-1),targets_rows)
            total_loss = loss_cells+loss_cols+loss_rows
            valid_head+=process_time()-t

            t = process_time()
            stat_dict = get_stats(preds_dict['cells'],preds_dict['cols'],preds_dict['rows'],targets_cells,targets_cols,targets_rows,prediction_thres)
            valid_stat+=process_time()-t

            #Iterate over, put tensors to device
            loop.set_description(f'Validation Epoch [{epoch}/{num_epochs}]')
            loop.set_postfix_str(s=f"Total_loss = {round(total_loss.item(),4)}, Cells = {round(loss_cells.item(),4)}, Cols = {round(loss_cols.item(),4)}, Rows = {round(loss_rows.item(),4)}, F1_Cells = {round(stat_dict['cells']['f1'],4)}, F1_Cols = {round(stat_dict['cols']['f1'],4)}, F1_Rows = {round(stat_dict['rows']['f1'],4)}")
    
    
    print(f'¤¤¤¤¤TRAINING: Data: {data_time}, transforms: {trans_time}, forward: {forward_time}')
    print(f'#####VALIDATION: Forward: {valid_forward_time}, Targets: {valid_targets_time}, Head: {valid_head}, Stats: {valid_stat}, valid_trans{valid_trans}')

    if idx % 100 == 0:
        Stats['total_loss'].append(total_loss)
        Stats['loss_cells'].append(loss_cells)
        Stats['loss_cols'].append(loss_cols)   
        Stats['loss_rows'].append(loss_rows)
        Stats['f1_cells'].append(stat_dict['cells']['f1'])
        Stats['f1_cols'].append(stat_dict['cols']['f1'])
        Stats['f1_rows'].append(stat_dict['rows']['f1'])


#GEM MODEL OGSÅ!!!

with open('Stats.pickle', 'wb') as handle:
    pickle.dump(Stats, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
 
