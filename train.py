import sys
import os
os.environ['CUDA_VISIBLE_DEVICES']="1"
import numpy as np
import pandas as pd
import random
import torch
torch.set_printoptions(profile="full")
import torch.nn as nn
from torch import optim
import matplotlib.pyplot as plt
from L0Smooth_loss import L0SmoothLoss
from torch.optim import lr_scheduler
from UNet import UNet
from pre_processed import split_dataset, batch
from data_loader import get_ids,get_images_ids

def weights_init(m):
    if isinstance(m,nn.Conv2d):
        nn.init.xavier_normal_(m.weight.data)
        nn.init.constant_(m.bias,0.1)
    elif isinstance(m,nn.BatchNorm2d):
        nn.init.constant_(m.weight,1)
        nn.init.constant_(m.bias,0)

def train_net(net,epochs=100,batch_size=1,lr=0.0001,gpu=True):

    dir_img = '/home/liang/Data/L0Smooth/img/'
    dir_checkpoint = '/home/liang/Data/Test/checkpoint/'#'/home/liang/Data/BiasData/BrainWeb/checkpoint/'#
    dir_excel = '/home/liang/Data/BiasData/excel/'

    ids = get_ids(dir_img)
    id_dataset = split_dataset(ids)
    train_data = []
    for t in id_dataset['train']:
        train_data.append(t)

    print('Training parameters(cos):Epochs:{} Batch size:{} Learning rate:{} Training size:{} '.format(epochs,batch_size,lr,len(id_dataset['train'])))

    train_num = len(id_dataset['train'])
    optimizer = optim.Adam(net.parameters(),lr=lr,betas=(0.9,0.999))
    scheduler_corr = lr_scheduler.ExponentialLR(optimizer,gamma=0.9999,last_epoch=-1)
    criterion_l0 = L0SmoothLoss()
    kappa = 2
    lamb = 0.005
    beta = 2 * lamb
    epochs_x = []
    train_loss_y = []

    for epoch in range(epochs):
        print('Epoch {}/{}'.format(epoch+1,epochs))

        net.train()
        random.shuffle(train_data)
        train_imgs = get_images_ids(train_data,dir_img)
        sum_loss = 0

        for j,b in enumerate(batch(train_imgs,batch_size)):
            imgs = np.array([i[1] for i in b]).astype(np.float32)
            imgs = torch.from_numpy(imgs)
            if gpu:
                imgs = imgs.cuda()

                pred = net(imgs)
                loss = criterion_l0(imgs, pred, lamb, beta)
                sum_loss = sum_loss + loss.item()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        epochs_x.append((epoch + 1))
        if(beta > 1e5):
            beta = 1e5
        else:
            beta = beta * kappa
        train_loss_y.append((sum_loss / train_num))
        print ('Epoch {} finished---loss:{}'.format((epoch+1),sum_loss / train_num))
        print('Learning rate of L0SmoothNet is {}'.format(optimizer.state_dict()['param_groups'][0]['lr']))
        scheduler_corr.step()
        torch.save(net.state_dict(), dir_checkpoint + 'L0SMNet.pth')
        print('Checkpoint{} with the latest version is saved'.format(epoch + 1))

    # train_lossT = [train_loss_y]
    # train_lossT = np.array(train_lossT)
    # train_loss = train_lossT.T
    # data_loss = pd.DataFrame(train_loss)
    # data_loss.columns = ['Loss']
    # writexcel = pd.ExcelWriter(dir_excel + 'HCPB.xlsx')
    # data_loss.to_excel(writexcel,'page_1',float_format='%.15f')
    # writexcel.save()

    line_loss, = plt.plot(epochs_x,train_loss_y,color='red',linewidth = 1.0, linestyle = '--')
    plt.title("The training curves")
    plt.legend(handles = [line_loss], labels = ['loss of L0 smooth'])
    plt.show()

if __name__=='__main__':
    net=UNet(in_chan=1)
    net.apply(weights_init)

    if torch.cuda.is_available():
        net.cuda()

    epochs=20
    batch_size = 1
    lr = 0.001
    gpu = True
    sigma = 5

    try:
        train_net(net=net,epochs=epochs,batch_size=batch_size,lr=lr,gpu=True)
    except KeyboardInterrupt:
        torch.save(net.state_dict(),'Interrupt.pth')
        print('Interruption occurred')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)