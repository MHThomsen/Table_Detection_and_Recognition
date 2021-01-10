import torch
import tfrecord
import numpy as np
import os

from tfrecord.torch.dataset import TFRecordDataset

from utils import tfrecord_transforms, rescale_img_quad
from feature_CNN import FeatureNet_v1


import config



"Very early version of main loop. Testing imports and pipeline atm."





#Load list of tfRecords from folder: 

folder_path = "/Users/Morten/Desktop/DTU/DL_SpecialKursus/TIES_DataGeneration/Data_Outputs"

#load filenames of folder: 
tfrecord_files = os.listdir(folder_path)


#load Feature CNN model
model_path = "./models/FeatureNet_v1.pt"

model = FeatureNet_v1()
model.load_state_dict(torch.load(model_path,map_location=torch.device('cpu')))
model.eval()



#####################################################
#EARLY TRAINING LOOP:
#####################################################
#Iterate all tfrecord files, load each as a batch of batch_size = 8 (default)
tfrecord_path = "/Users/Morten/Desktop/DTU/DL_SpecialKursus/TIES_DataGeneration/Data_Outputs/P3VVFC13XZDAYL3UOJH9.tfrecord"

#Maybe tfrecords need to be generated with more files, to make loading more effective? 
dataset = TFRecordDataset(tfrecord_path, config.index_path, config.tfrecord_description)
loader = torch.utils.data.DataLoader(dataset, batch_size=8)


elem = next(iter(loader))
data_dict = tfrecord_transforms(elem)

#print(data_dict.keys())

#print(data_dict['imgs'].shape)


##########################################################################################################
#####################################################
images = [rescale_img_quad(img) for img in data_dict['imgs']]
images = torch.stack(images, dim=0)
#print(d.shape) 

# TODO Find a way to make cropping and retain corresponding image values between feature map and image data. 
#####################################################
##########################################################################################################
"""
import matplotlib.pyplot as plt

img = d[1,0,:,:]

plt.figure(figsize=(20,10))
plt.tight_layout()
plt.imshow(img,cmap='gray')
plt.show()
"""


#Run images through the features network
features = model.feature_forward(images)
print(features.shape)


#image features are a collection of {data_dict,features}




