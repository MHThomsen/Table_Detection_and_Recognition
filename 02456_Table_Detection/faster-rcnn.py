import pandas as pd
import sys
import numpy as np
import os
import json
import re
import torch
import ast
import cv2
import io
import pickle
from PIL import Image
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.transforms import functional as F
import engine, transforms, utils, coco_eval, coco_utils
from engine import train_one_epoch, evaluate
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from RPN_custom import RPN_custom




def collate_fn(batch):
    return tuple(zip(*batch))

def resize_bbox(image_shape, newsize, bbox):
    x1, y1, w, h = bbox
    w_conv = newsize[0]/image_shape[0]
    h_conv = newsize[1]/image_shape[1]
    (x1, y1, w, h) = (x1*w_conv, y1*h_conv, w*w_conv, h*h_conv)
    return x1, y1, w, h

images_path = 'Detection/images/'

class TableBank():
    def __init__(self,images_path, Train, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        if Train:
            self.annotations = Annotations_train
        else:
            self.annotations = Annotations_test
        self.images_path = images_path
        #self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = os.path.join(self.images_path,
                                self.annotations.iloc[idx]['filename'])
        
        image = Image.open(img_path).convert("RGB")
        image_shape = image.size
        newsize = (550,700)
        image = image.resize(newsize,resample=Image.BILINEAR)
        bboxes = ast.literal_eval(self.annotations.iloc[idx]['BoundingBoxes'])

        #boxes = np.zeros((1,4), dtype=np.float32)
        areas = []
        boxes = []
        for i in range(len(bboxes)):
            (x1, y1, w, h) = resize_bbox(image_shape, newsize, bboxes[i])
            x2 = x1+w
            y2 = y1+h
            boxes.append([x1,y1,x2,y2])
            areas.append((h*w))
            #if area < h*w:
            #    area = h*w
            #    boxes[0,:] = [x1,y1,x2,y2]

        # convert everything into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # there is only one class
        labels = torch.ones((len(boxes)), dtype=torch.int64)
        #labels = torch.ones(1, dtype = torch.int64)
        areas = torch.as_tensor(areas, dtype=torch.int64)
        iscrowd = torch.zeros(len(boxes), dtype=torch.int64)
        
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = torch.tensor([idx])
        target["area"] = areas
        target["iscrowd"] = iscrowd        

        image = F.to_tensor(image)

        return image, target

Annotations_train = pd.read_csv('Detection/annotations/Annotations_train_sample.csv')
Annotations_test = pd.read_csv('Detection/annotations/Annotations_test_sample.csv')

#Annotations_train = pd.read_csv('Detection/annotations/ant_mini_train.csv')
#Annotations_test = pd.read_csv('Detection/annotations/ant_mini_test.csv')


TBdata = TableBank(images_path, Train=True)
TBdata_test = TableBank(images_path,Train=False)



def record_losses(model, data_loader, device, Ls,model_name):
    with torch.no_grad():
        #cpu_device = torch.device("cpu")
        model.train()
        #metric_logger = utils.MetricLogger(delimiter="  ")

        if model_name == 'RPNresnet50':
            record_losses = {'loss_objectness':0, 'loss_rpn_box_reg':0}
        else:
            record_losses = {'loss_classifier':0, 'loss_box_reg':0, 'loss_objectness':0, 'loss_rpn_box_reg':0}
        counter = 0
        for images, targets in data_loader:
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            loss_dict = model(images, targets)
            #print(loss_dict)
            del images, targets
            for k, v in loss_dict.items():
                record_losses[k] += v
            counter += 1

        for k,v in record_losses.items():
            record_losses[k] = v/counter
        

        losses = sum(loss for loss in record_losses.values())

        Ls['total loss'].append(losses)
        
        Ls['loss_objectness'].append(record_losses['loss_objectness'])
        Ls['loss_rpn_box_reg'].append(record_losses['loss_rpn_box_reg'])

        if model_name !='RPNresnet50':
            Ls['loss_classifier'].append(record_losses['loss_classifier'])
            Ls['loss_box_reg'].append(record_losses['loss_box_reg'])
    return Ls


def main(network):
    # train on the GPU or on the CPU, if a GPU is not available
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # our dataset has two classes only - background and person
    num_classes = 2

    #dataset = torch.utils.data.Subset(TBdata,[range(len(TBdata))])
    indices = torch.randperm(len(TBdata)).tolist()
    dataset = torch.utils.data.Subset(TBdata, indices[:])
    indices_ = torch.randperm(len(TBdata_test)).tolist()
    dataset_val = torch.utils.data.Subset(TBdata_test, indices_[:])

    # get the model using our helper function
        #model = get_model_instance_segmentation(num_classes)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=8,
                                sampler=None, num_workers=0, collate_fn=collate_fn)
    dataloader_val = torch.utils.data.DataLoader(dataset_val, batch_size=8,
                                sampler=None, num_workers=0, collate_fn=collate_fn)
    
    #Calculated statistics on training data:
    #Transform parameters
    min_size = 550  
    max_size = 700  
    image_means = [0.9492,0.9492,0.9492]
    image_stds = [0.1158,0.1158,0.1158]

    if network == 'resnet50':
        backbone = resnet_fpn_backbone('resnet50', True)
        model = FasterRCNN(backbone, num_classes, min_size=min_size, max_size=max_size, image_mean = image_means, image_std = image_stds)

    elif network =='resnet18':
        backbone = resnet_fpn_backbone('resnet18', True)
        model = FasterRCNN(backbone, num_classes, min_size=min_size, max_size=max_size, image_mean = image_means, image_std = image_stds)
        
    elif network == 'resnet152':
        backbone = resnet_fpn_backbone('resnet152', True)
        model = FasterRCNN(backbone, num_classes, min_size=min_size, max_size=max_size, image_mean = image_means, image_std = image_stds)
    
    elif network == 'RPNresnet50':
        backbone = resnet_fpn_backbone('resnet50', True)
        model = RPN_custom(backbone, num_classes, min_size=min_size, max_size=max_size, image_mean = image_means , image_std = image_stds)
    
    elif network == 'RPNresnet152':
        backbone = resnet_fpn_backbone('resnet152', True)
        model = RPN_custom(backbone, num_classes, min_size=min_size, max_size=max_size, image_mean = image_means , image_std = image_stds)
        

    # move model to the right device
    model.to(device)

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005,
                                momentum=0.9, weight_decay=0.0005)
    # and a learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=3,
                                                   gamma=0.1)

    
    num_epochs = 10
    Ls = {'total loss':[],'loss_classifier':[], 'loss_box_reg':[], 'loss_objectness':[], 'loss_rpn_box_reg':[]}
    Ls_val = {'total loss':[],'loss_classifier':[], 'loss_box_reg':[], 'loss_objectness':[], 'loss_rpn_box_reg':[]}
    
    for epoch in range(num_epochs):
        # train for one epoch, printing every 10 iterations
        train_one_epoch(model, optimizer, dataloader, device, epoch, print_freq=10)
        # update the learning rate
        lr_scheduler.step()
        # evaluate on the test dataset
        #evaluate(model, dataloader_test, device=device)
        Ls_val = record_losses(model, dataloader_val, device, Ls_val,network)

        #record losses
        Ls = record_losses(model, dataloader, device, Ls,network)
    

    #If folder does not exist already, create it
    output_loc = f'./{network}/'

    if not os.path.exists(output_loc):
        os.makedirs(output_loc)
    

    
    torch.save(model.state_dict(), output_loc+'model.pt')
    

    print("That's it!")
    return Ls, Ls_val, num_epochs

model_name = 'RPNresnet152'

output_loc = f'./{model_name}/'

Ls, Ls_val, num_epochs = main(model_name)


for k, v in Ls.items():
    for i in range(len(v)):
        Ls[k][i] = v[i].cpu().numpy()

with open(output_loc+"losses train dict {}.pickle".format(model_name), "wb") as handle:
    pickle.dump(Ls, handle)


for k, v in Ls_val.items():
    for i in range(len(v)):
        Ls_val[k][i] = v[i].cpu().numpy()

with open(output_loc+"losses val dict {}.pickle".format(model_name), "wb") as handle:
    pickle.dump(Ls_val, handle)
