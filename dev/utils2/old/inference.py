import torchvision
import torch
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
import torchvision.transforms as T
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os


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
    
    
def get_batch(data_loader_train,batch_num=5):
    images=None
    for k in range(batch_num):
        images, targets=next(iter(data_loader_train))
        #images=images.detach().cpu()
        #images=images.permute(1,2,0)
    return images, targets
    
    
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
    

def get_trained_model(PATH=r'D:\All_files\pys\AI_algos\Liv\TestMaskRCNN\Well96\weights',
                      fname='weight_epoch_39_iter_45.pth',
                      device='gpu'):
    # get model
    model=get_model(num_classes=2,device=device)
    model.load_state_dict(torch.load(os.path.join(PATH,fname)))
    return model
    
    
def get_inference(model=None,
                  device='gpu',
                  data_loader=None,
                  batch_num=115):
    
    # get batch numbefr
    images,targets=get_batch(data_loader_train=data_loader,batch_num=batch_num)
    
    if model : model.eval();
    
    # predict
    images = list(image.to(device) for image in images)    
    targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

    with torch.no_grad():
        out = model(images)
    
    return images, out , targets#.cpu()
    
        
def get_inference_test(model,
                       img_path=r'C:\Users\bdutta\work\pys\Cam\data\Test\well96_test_1018_1100.png',
                       resize=(256,256)):

    img_test = Image.open(img_path).convert("RGB")

    transforms_image=T.Compose([T.PILToTensor(),T.Resize(resize,T.InterpolationMode.NEAREST)])
    img1=transforms_image(img_test)
    img1=img1.double().unsqueeze(0)/255.0

    model.eval()
    with torch.no_grad():
        out=model(img1)
    
    return img_test, out
    

def sort_list(out):
    #break;   
    # sort list to 4*6 grid
    lst=np.zeros((24,4))
    for idx1 in range(1,len(lst)): 
        lst[idx1,:]=out[0]['boxes'][idx1-1].detach().numpy()


    xval=lst[:,[0,2]].mean(1)
    yval=lst[:,[1,3]].mean(1)

    # sort y
    ym=np.argsort(yval)
    ymr=ym.reshape(4,6)
    #print(ymr)

    pos=np.zeros((4,6)).astype('int')
    # sort x
    for i in range(4):
        pos[i,:]=np.argsort(xval[ymr[i,:]])

    #print(pos)

    for i in range(4): 
        dummy=ymr[i,:]
        ymr[i,:]=dummy[pos[i,:]]
    return ymr


def plots(img_test,out,ymr):
    # plotting
    fig,ax=plt.subplots(4,6)#,figsize=(10,9))
    col_list=np.zeros((4,6,3))
    for idx1 in range(4):
        for idx2 in range(6):
            if 6*idx1+idx2>0:
                test_image=cv2.resize(np.array(img_test),(256,256))
                sort_mask=ymr[idx1,idx2]-1
                #print(sort_mask)
                dummy=out[0]['masks'][sort_mask].detach().squeeze().numpy()
                _,thresh1=cv2.threshold(dummy, 0, 1, cv2.THRESH_BINARY)
                test_image[thresh1 == 0] = 0

                ax[idx1,idx2].imshow(test_image)
                ax[idx1,idx2].axis('off')
                #print(out[0]['boxes'][sort_mask].detach().numpy())

                #col_list[6*idx1+idx2,:]=test_image.sum(axis=0).sum(axis=0)/np.sum(test_image[:,:,0]>0)
                col_list[idx1,idx2,:]=test_image.sum(axis=0).sum(axis=0)/np.sum(test_image[:,:,0]>0)
            else:
                #print(8*idx1+idx2)
                ax[idx1,idx2].imshow(np.zeros((256,256)),cmap='gray')
                ax[idx1,idx2].axis('off')
                #col_list[0,:]=[0,0,0]
                
    return col_list


def get_rgb_patch(img, out, verbose=False):
    #break;   
    ymr=sort_list(out)
    #col_list=plots(img,out,ymr)
    if verbose:
        col_list=plots_new(img,out,ymr)
    else:
        col_list=get_colors(img, out, ymr)
        
    return col_list
    
       
def plots_new(img_test, out, ymr):
    
    fig,ax=plt.subplots(4,6)#,figsize=(10,9))
    
    col_list_2=np.zeros((4,6,3))
    bbox_list=np.zeros((24,4)).astype('int')
    
    # get centroids, height and width of mask
    for idx1 in range(4):
        for idx2 in range(6):
            sort_mask=ymr[idx1,idx2]-1
            bbox_list[6*idx1+idx2,:]=out[0]['boxes'][sort_mask].detach().squeeze().numpy().astype('int')
    #print(bbox_list)
    xmean=bbox_list[:,[0,2]].mean(axis=1).astype('int').reshape(-1,1)
    ymean=bbox_list[:,[1,3]].mean(axis=1).astype('int').reshape(-1,1)
    w=np.tile(int((bbox_list[:,2]-bbox_list[:,0])[1:].min()*0.2),[bbox_list.shape[0],1])
    h=np.tile(int((bbox_list[:,3]-bbox_list[:,1])[1:].min()*0.2),[bbox_list.shape[0],1])
    bbox_list=np.hstack((xmean,ymean,w,h))
    
    ymr=ymr-1
    
    #print(bbox_list)
    
    # get color values and plots
    for idx1 in range(4):
        for idx2 in range(6):
            if 6*idx1+idx2>0:
                test_image=cv2.resize(np.array(img_test),(256,256))

                #sort_mask=ymr[idx1,idx2]-1
                dummy=out[0]['masks'][ymr[idx1,idx2]].detach().squeeze().numpy()
                _,thresh1=cv2.threshold(dummy, 0, 1, cv2.THRESH_BINARY)
                test_image[thresh1 == 0] = 0
                
                pos = bbox_list[6*idx1+idx2,:]
                #pos= out[0]['boxes'][sort_mask].detach().squeeze().numpy().astype('int')
                #w=int((pos[2]-pos[0])*0.1)
                #h=int((pos[3]-pos[1])*0.1)
                #w=size[0]
                #h=size[1]
                #dummy=test_image[pos[1]+w:pos[3]-w,pos[0]+h:pos[2]-h,:]
                #dummy=test_image[pos[1]+w:pos[1]+5*w,pos[0]+h:pos[0]+5*h,:]
                
                dummy=test_image[pos[1]-pos[3]:pos[1]+pos[3],pos[0]-pos[2]:pos[0]+pos[2],:]

                ax[idx1,idx2].imshow(dummy)
                ax[idx1,idx2].axis('off')
                #col_list[6*idx1+idx2,:]=test_image.sum(axis=0).sum(axis=0)/np.sum(test_image[:,:,0]>0)
                #col_list_2[6*idx1+idx2,:]=dummy.sum(axis=0).sum(axis=0)/(dummy.shape[0]*dummy.shape[1])
                col_list_2[idx1,idx2,:]=dummy.sum(axis=0).sum(axis=0)/(dummy.shape[0]*dummy.shape[1])
            else:
                ax[idx1,idx2].imshow(np.zeros((bbox_list[1,3],bbox_list[1,2])),cmap='gray')
                ax[idx1,idx2].axis('off')
                #col_list[0,:]=[0,0,0]
                
    #fig.subplots_adjust(wspace=0.02,hspace=0.02)
    #path=r'C:\Users\bdutta\work\pys\Cam\plots'
    #plt.savefig(os.path.join(path,'segments.png'))
    
    return col_list_2


def get_colors(img_test, out, ymr):
    
    col_list_2=np.zeros((4,6,3))
    bbox_list=np.zeros((24,4)).astype('int')
    
    # get centroids, height and width of mask
    for idx1 in range(4):
        for idx2 in range(6):
            sort_mask=ymr[idx1,idx2]-1
            bbox_list[6*idx1+idx2,:]=out[0]['boxes'][sort_mask].detach().squeeze().numpy().astype('int')
    #print(bbox_list)
    xmean=bbox_list[:,[0,2]].mean(axis=1).astype('int').reshape(-1,1)
    ymean=bbox_list[:,[1,3]].mean(axis=1).astype('int').reshape(-1,1)
    w=np.tile(int((bbox_list[:,2]-bbox_list[:,0])[1:].min()*0.2),[bbox_list.shape[0],1])
    h=np.tile(int((bbox_list[:,3]-bbox_list[:,1])[1:].min()*0.2),[bbox_list.shape[0],1])
    bbox_list=np.hstack((xmean,ymean,w,h))
    
    ymr=ymr-1
    
    #print(bbox_list)
    
    # get color values and plots
    for idx1 in range(4):
        for idx2 in range(6):
            if 6*idx1+idx2>0:
                test_image=cv2.resize(np.array(img_test),(256,256))
                dummy=out[0]['masks'][ymr[idx1,idx2]].detach().squeeze().numpy()
                _,thresh1=cv2.threshold(dummy, 0, 1, cv2.THRESH_BINARY)
                test_image[thresh1 == 0] = 0
                
                pos = bbox_list[6*idx1+idx2,:]
 
                dummy=test_image[pos[1]-pos[3]:pos[1]+pos[3],pos[0]-pos[2]:pos[0]+pos[2],:]
                #col_list_2[6*idx1+idx2,:]=dummy.sum(axis=0).sum(axis=0)/(dummy.shape[0]*dummy.shape[1])
                col_list_2[idx1, idx2,:]=dummy.sum(axis=0).sum(axis=0)/(dummy.shape[0]*dummy.shape[1])
    
    return col_list_2
