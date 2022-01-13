import torch
from torch.utils.data import DataLoader
from torchvision import transforms as tf
from torch.utils.data.sampler import SubsetRandomSampler
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
import matplotlib.pyplot as plt
from skimage.io import imread
import os
import numpy as np

from Unet import Unet as net
from LIP_dataset import LIP_image
from pairloss2 import PairContrastiveLoss

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

student = net().to(device)
teacher = net().to(device)
head1 = nn.Conv2d(32, 12, 1, 1, 0, bias=True).to(device)
head2 = nn.Conv2d(32, 12, 1, 1, 0, bias=True).to(device)
#teacher.load_state_dict(student.state_dict())
criterion1 = PairContrastiveLoss(12, 32, 16).to(device)
#criterion2 = cfg.DiceLoss()

param = list(student.parameters()) + list(teacher.parameters()) + list(criterion1.parameters()) + list(head1.parameters()) + list(head2.parameters())
optimizer1 = optim.Adam(param, lr=5e-4)
#optimizer2 = optim.Adam(head.parameters(), lr=5e-4)

transform_train = tf.Compose([
        tf.ToTensor(),
        tf.ToPILImage(),
        tf.RandomCrop((256,256), pad_if_needed=True),
        tf.RandomHorizontalFlip(),
        tf.ToTensor()
     ])


path = "/home/DATA/ymh/data/TrainVal_images/train_images"
dataset_unlabaled = LIP_image(path, transform_train)
unlabeled_batch = 16
unlabeled_loader = DataLoader(dataset=dataset_unlabaled, batch_size=unlabeled_batch, shuffle=True)


for epoch in range(300):
    running_loss = 0
    for x1 in unlabeled_loader:
        ## ---- self supervised ---- ##
        x1 = x1.float().to(device)
                
        student.train()
        teacher.train()
        optimizer1.zero_grad()
        
        rep1_s = student(x1)
        rep1_t = teacher(x1)
        
        output1_s = head1(rep1_s)
        output1_t = head2(rep1_t)

        loss_self = criterion1(output1_s, output1_t)#.detach()) + criterion1(output1_t, output1_s.detach())
        
        loss_self.backward()
        optimizer1.step()
        '''
        ## ---- supervised ---- ##
        x0, y0 = labeled_loader.__iter__().next()
        x0 = x0.float().to(device)
        y0 = y0.float().to(device)
        
        student.eval()
        teacher.eval()
        optimizer2.zero_grad()
        
        with torch.no_grad():
            rep0_s = student(x0)
            rep0_t = teacher(x0)
        
        output0_s = head(rep0_s)
        output0_t = head(rep0_t)
        loss_sup = criterion2(output0_s, y0) + criterion2(output0_t, y0)
        
        loss_sup.backward()
        optimizer2.step()
        '''
        #running_loss1 += loss_sup.item()
        running_loss += loss_self.item()
            
    running_loss /= len(unlabeled_loader)
    print("="*100)
    print("[Epoch:%d] [Loss:%f]" % ((epoch+1), running_loss))

    if (epoch+1) % 1 == 0:
        visual = output1_t[0]
        visual = visual.argmax(dim=0, keepdim=True).squeeze()
        print(visual.shape)
        for i in range(12):
            single_map = visual == i
            print(single_map.shape)
            plt.imshow(single_map.detach().cpu().numpy(), cmap='gray')
            plt.title('Epoch %03d Class %02d' % ((epoch+1), i))
            plt.savefig("/home/DATA/ymh/ultra/save/img2/epoch%03d_%02d.png" % ((epoch+1), i))