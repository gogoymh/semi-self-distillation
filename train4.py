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
from network_C_sev import Our_Unet_singlegpu as net
from Dataset import USDataset as newset
import config as cfg
from utils import rand_translation as T
#from loss1 import AgoodLoss
from pairloss2 import PairContrastiveLoss

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

student = net().to(device)
teacher = net().to(device)
#teacher.load_state_dict(student.state_dict())
criterion = PairContrastiveLoss(6, 32, 16).to(device)

param = list(student.parameters()) + list(teacher.parameters()) + list(criterion.parameters())
optimizer = optim.Adam(param, lr=5e-4)

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

labeled_batch = 3
unlabeled_batch = 12
labeled_loader = DataLoader(dataset=dataset_labeled, batch_size=labeled_batch, sampler=labeled_sampler)
unlabeled_loader = DataLoader(dataset=dataset_unlabaled, batch_size=unlabeled_batch, sampler=unlabeled_sampler)
valid_loader = DataLoader(dataset=dataset_val, batch_size=1, sampler=valid_sampler)


#criterion = nn.CrossEntropyLoss()
#alpha = 1
#beta = torch.linspace(0.1, 6, steps=300)
#beta = 6
#ema_tau = 0.01
#ema_tau = torch.linspace(0, 0.01, steps=300)

logsoftmax = nn.LogSoftmax(dim=1)

sample_img, _ = valid_loader.__iter__().next()

#C = 0
#t = 0.05
#tau = 0.1

#criterion = AgoodLoss()


for epoch in range(300):
    running_loss = 0
    student.train()
    teacher.train() # batch norm layer가 있는 network면 차이가 난다.
    for x1, _ in unlabeled_loader:
        ## ---- unlabeled data ---- ##
        x1 = x1.float().to(device)
        
        ## ---- unsupervised ---- ##
        output1_s = student(x1)
        output1_t = teacher(x1)

        loss = criterion(output1_s, output1_t.detach()) + criterion(output1_t, output1_s.detach())
        
        '''
        logprobs_s = logsoftmax(output1_s/t)
        logprobs_t = logsoftmax(output1_t/t)
        
        loss_1 = - (((output1_t-C)/t).softmax(dim=1).detach() * logprobs_s).sum(dim=1).mean()
        loss_2 = - (((output1_s-C)/t).softmax(dim=1).detach() * logprobs_t).sum(dim=1).mean()
        
        loss = (loss_1 + loss_2)/2
        '''

        loss.backward()
        optimizer.step()
        
        #C = (1.0 - tau) * C + tau * torch.cat((output1_s.detach(), output1_t.detach()), dim=0).mean(dim=0)
        
        running_loss += loss.item()
        #print(loss.item())
            
    running_loss /= len(unlabeled_loader)
    print("="*100)
    print("[Epoch:%d] [Loss:%f]" % ((epoch+1), running_loss))

    if (epoch+1) % 1 == 0:
        teacher.eval()
        visual = teacher(sample_img.float().to(device))
        visual = visual.argmax(dim=1, keepdim=True).squeeze()
        print(visual.shape)
        for i in range(6):
            single_map = visual == i
            print(single_map.shape)
            plt.imshow(single_map.detach().cpu().numpy(), cmap='gray')
            plt.title('Epoch %03d Class %02d' % ((epoch+1), i))
            plt.savefig("/home/DATA/ymh/ultra/save/img1/epoch%03d_%02d.png" % ((epoch+1), i))


