import torchvision
import torch
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
import torchvision.transforms as T
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt

def get_model(num_classes=2,weights=torchvision.models.detection.MaskRCNN_ResNet50_FPN_Weights.DEFAULT, device=None):
    #num_classes = 2
    # load an instance segmentation model pre-trained pre-trained on COCO
    
    '''
    if weights=='imagenet_v2':
        weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2
    else:
        weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1
    '''    
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights=weights)
    
    #model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # and replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                        hidden_layer,
                                                        num_classes)
    
    device=get_device(device)
    
    model=model.to(device)
    model=model.double()
    
    return model
    
    
def get_device(device='default'):
    if not(device):
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    if device == 'cpu':
        device = torch.device('cpu')
    elif device == 'cuda':
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
        
    return device