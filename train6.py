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

#from self_sup_net import Our_Unet_singlegpu as net
from network_C_sev2 import Our_Unet_singlegpu as net
from Dataset import USDataset as newset
import config as cfg
from utils import rand_translation as T
#from loss1 import AgoodLoss
from pairloss2 import PairContrastiveLoss

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

student = net().to(device)
teacher = net().to(device)
#head1 = nn.Conv2d(32, 6, 1, 1, 0, bias=True).to(device)
#head2 = nn.Conv2d(32, 6, 1, 1, 0, bias=True).to(device)
#teacher.load_state_dict(student.state_dict())
criterion1 = PairContrastiveLoss(32, 32, 16).to(device)
#criterion2 = cfg.DiceLoss()

param = list(student.parameters()) + list(teacher.parameters()) + list(criterion1.parameters())# + list(head1.parameters()) + list(head2.parameters())
optimizer1 = optim.Adam(param, lr=5e-4)
#optimizer2 = optim.Adam(head.parameters(), lr=5e-4)

transform_train = tf.Compose([
        tf.ToTensor(),
        tf.ToPILImage(),
        #tf.Resize((442,565)),
        tf.Resize((256,256)),
        tf.RandomAffine(0, shear=[-15, 15, -15, 15]),
        tf.ToTensor()
     ])

transform_valid = tf.Compose([
        tf.ToTensor(),
        tf.ToPILImage(),
        #tf.Resize((442,565)),
        tf.Resize((256,256)),        
        tf.ToTensor(),
     ])

path1 = "/home/DATA/ymh/ultra/newset/wrist_train/wrist_HM70A"
path2 = "/home/DATA/ymh/ultra/newset/wrist_target/wrist_HM70A"
dataset_labeled = newset('wrist_HM70A', path1, path2, transform_train, transform_train)
dataset_unlabaled = newset('wrist_HM70A', path1, path2, transform_train, transform_train)
dataset_val = newset('wrist_HM70A', path1, path2, transform_valid, transform_valid)

labeled_idx = cfg.wrist_HM70A_labeled_idx
unlabeled_idx = cfg.wrist_HM70A_unlabeled_idx
valid_idx = cfg.wrist_HM70A_valid_idx
n_val_sample = len(valid_idx)
labeled_sampler = SubsetRandomSampler(labeled_idx)
unlabeled_sampler = SubsetRandomSampler(unlabeled_idx)
valid_sampler = SubsetRandomSampler(valid_idx)

labeled_batch = 4
unlabeled_batch = 12
labeled_loader = DataLoader(dataset=dataset_labeled, batch_size=labeled_batch, sampler=labeled_sampler)
unlabeled_loader = DataLoader(dataset=dataset_unlabaled, batch_size=unlabeled_batch, sampler=unlabeled_sampler)
valid_loader = DataLoader(dataset=dataset_val, batch_size=1, sampler=valid_sampler)

sample_img, _ = valid_loader.__iter__().next()

#tau = 0.1
for epoch in range(300):
    running_loss = 0
    for x1, _ in unlabeled_loader:
        ## ---- self supervised ---- ##
        x1 = x1.float().to(device)
                
        student.train()
        teacher.train()
        optimizer1.zero_grad()
        
        rep1_s = student(x1)
        rep1_t = teacher(x1)
        
        #output1_s = head1(rep1_s)
        #output1_t = head2(rep1_t)

        loss_self = criterion1(rep1_s, rep1_t)#.detach()) + criterion1(output1_t, output1_s.detach())
        
        loss_self.backward()
        optimizer1.step()
        
        #for target_param, param in zip(teacher.parameters(), student.parameters()):
        #    target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)
        
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
        teacher.eval()
        visual = teacher(sample_img.float().to(device))
        visual = visual.argmax(dim=1, keepdim=True).squeeze()
        print(visual.shape)
        for i in range(32):
            single_map = visual == i
            print(single_map.shape)
            plt.imshow(single_map.detach().cpu().numpy(), cmap='gray')
            plt.title('Epoch %03d Class %02d' % ((epoch+1), i))
            plt.savefig("/home/DATA/ymh/ultra/save/img1/epoch%03d_%02d.png" % ((epoch+1), i))