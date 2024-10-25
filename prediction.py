import torch
import os
os.environ['CUDA_VISIBLE_DEVICES']="1"
torch.set_printoptions(profile='full')
import numpy as np
from PIL import Image
from UNet import UNet
from pre_processed import img_Normalize, hwc_to_chw, fore_Mask
from data_loader import get_ids
import cv2

def mask_to_image(image):
    return image * 255

def predict_img(net, origin_img, gpu=True):
    net.eval()
    img = np.array(origin_img,dtype=np.float32)
    img = img_Normalize(img)

    transimg = hwc_to_chw(img)
    transimg = torch.from_numpy(transimg).unsqueeze(0)

    if(gpu):
        transimg = transimg.cuda()

    with torch.no_grad():
        SI = net(transimg)

    return SI

if __name__=='__main__':
    net = UNet(in_chan=1)
    height = 256
    width = 256
    dir_checkpoint = '/home/liang/Data/Test/checkpoint/L0SMNet.pth'
    dir_img = '/home/liang/Data/L0Smooth/img/'
    dir_save = '/home/liang/Data/L0Smooth/result/'
    imgsuffix = '.png'

    net.cuda()
    net.load_state_dict(torch.load(dir_checkpoint))

    ids = get_ids(dir_img)
    count = 1
    for id in ids:
        imgorigin = Image.open(dir_img + id + imgsuffix)
        imggray = imgorigin.convert('L')
        imgcorrect = predict_img(net,imggray)
        imgcorrect = mask_to_image(imgcorrect)
        img = torch.squeeze(imgcorrect)
        img = img.cpu().numpy()
        cv2.imwrite(dir_save+id+imgsuffix,img)
        count = count + 1