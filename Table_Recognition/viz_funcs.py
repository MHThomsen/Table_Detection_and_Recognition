import torch
from Vex_Mout import VexMoutNet

import tfrecord
import tensorflow as tf
from tfrecord.torch.dataset import TFRecordDataset

from utils import get_all_targets, get_stats, tfrecord_preparer

import matplotlib.pyplot as plt

from PIL import Image, ImageDraw
import math

def visualize(idx, data_dict, preds_dict, pred_thresh, category):
    
    if category=='cells':
        outline = (0, 255, 0, 0)
    elif category == 'rows':
        outline = (254, 62, 255, 0)
    else:
        outline = (255,0,0,0)
    
    preds = {}

    for k,v in preds_dict.items():
        preds[k] = v >= pred_thresh
    
    img = data_dict['imgs'][0][0,0,:,:].numpy()*255
    shape = img.shape
    img = Image.fromarray(img).convert("RGB")

    for i in range(preds[category].shape[0]):
        if i==idx:
            x1, y1, x2, y2 = data_dict['vertex_features'][0][i,:4]
            x1, y1, x2, y2 = int(x1*shape[1]), int(y1*shape[0]), int(x2*shape[1]), int(y2*shape[0])
            draw = ImageDraw.Draw(img)
            draw.rectangle([x1, y1, x2, y2], outline=(255, 159, 0, 0), width=2)

        elif preds[category][idx,i] == True:
            x1, y1, x2, y2 = data_dict['vertex_features'][0][i,:4]
            x1, y1, x2, y2 = int(x1*shape[1]), int(y1*shape[0]), int(x2*shape[1]), int(y2*shape[0])
            draw = ImageDraw.Draw(img)
            draw.rectangle([x1, y1, x2, y2], outline=outline, width=2)
        
    return img



def viz(model_name, tf_records_path, record_num, word_num, table_num, pred_thresh):

    model_path = "C:/Users/Jesper/Desktop/TableRecognition/Table_Detection_and_Recognition/Table_Recognition/models/{}/model.pt".format(model_name)
    model = VexMoutNet()
    model.load_state_dict(torch.load(model_path))
    model.eval()

    path = "C:/Users/Jesper/Desktop/TableRecognition/Table_Detection_and_Recognition/Table_Recognition/Data/{}".format(tf_records_path)
    files = os.listdir(path)
    record = files[record_num]
    device=torch.device("cpu")

    batch_size = 1
    #variables for tfrecord loader
    index_path = None
    tfrecord_description = {"imgs": "float", 
                   "num_words": "int",
                   "vertex_features": "float",
                   "adjacency_matrix_cells":"int",
                   "adjacency_matrix_cols":"int",
                   "adjacency_matrix_rows":"int",
                   "num_edges":'int',
                   "edge_indexes" : 'int'}

    tfrecord_path = os.path.join(path,record)
    dataset = TFRecordDataset(tfrecord_path, index_path, tfrecord_description)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
    
    for idx,tmp in enumerate(loader):
        batch = tmp
        if idx == table_num:
            break
    
    data_dict = tfrecord_preparer(batch,device=device, batch_size=batch_size)
    
    preds_dict = model(data_dict,device, 0.5)
    for k,v in preds_dict.items():
        reshap = int(math.sqrt(v.shape[0]))
        preds_dict[k] = torch.sigmoid(v.reshape(reshap,reshap)) 
        
    img_cells = visualize(word_num, data_dict, preds_dict, pred_thresh, 'cells')
    img_rows = visualize(word_num, data_dict, preds_dict, pred_thresh, 'rows')
    img_cols = visualize(word_num, data_dict, preds_dict, pred_thresh, 'cols')
    return img_cells, img_rows, img_cols