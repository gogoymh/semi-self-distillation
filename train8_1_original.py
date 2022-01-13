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
#from utils import rand_translation as T
#from loss1 import AgoodLoss
#from ViT_patchloss6 import PairContrastiveLoss

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

student = net().to(device)
#teacher = net().to(device)
#head1 = nn.Conv2d(32, 8, 1, 1, 0, bias=True).to(device)
#head2 = nn.Conv2d(32, 3, 1, 1, 0, bias=True).to(device)
output1 = nn.Sequential(
    nn.Conv2d(32, 1, 1, 1, 0, bias=True),
    nn.Sigmoid()).to(device)
#output2 = nn.Sequential(
#    nn.Conv2d(3, 1, 1, 1, 0, bias=True),
#    nn.Sigmoid()).to(device)
#teacher.load_state_dict(student.state_dict())
#criterion1 = PairContrastiveLoss(8, 8, 1, latent=512)
#loss_checkpoint = torch.load("/home/DATA/ymh/ultra/save/model/pairloss.pth")
#loss_checkpoint = torch.load("/data/ymh/US_segmentation/save/model/pairloss.pth")
#criterion1.load_state_dict(loss_checkpoint["model_state_dict"])
#criterion1.to(device)
#criterion1.eval()
criterion2 = cfg.DiceLoss()

param = list(student.parameters()) + list(output1.parameters())# + list(criterion1.parameters())  + list(head1.parameters())# + list(head2.parameters())+ list(teacher.parameters())+ list(output2.parameters())
optimizer1 = optim.Adam(param, lr=5e-4)

#param_output = list(output1.parameters()) + list(output2.parameters())
#optimizer2_1 = optim.Adam(param_output, lr=5e-4)
#optimizer2_2 = optim.Adam(output2.parameters(), lr=5e-4)

transform_train = tf.Compose([
        tf.ToTensor(),
        tf.ToPILImage(),
        #tf.Resize((442,565)),
        tf.Resize((224,224)),
        tf.RandomAffine(0, shear=[-15, 15, -15, 15]),
        tf.ToTensor()
     ])

transform_valid = tf.Compose([
        tf.ToTensor(),
        tf.ToPILImage(),
        #tf.Resize((442,565)),
        tf.Resize((224,224)),        
        tf.ToTensor(),
     ])

path1 = "/home/DATA/ymh/ultra/newset/wrist_train/wrist_HM70A"
#path1 = "/data/ymh/US_segmentation/newset/wrist_train/wrist_HM70A"
path2 = "/home/DATA/ymh/ultra/newset/wrist_target/wrist_HM70A"
#path2 = "/data/ymh/US_segmentation/newset/wrist_target/wrist_HM70A"
dataset_train = newset('wrist_HM70A', path1, path2, transform_train, transform_train)
dataset_val = newset('wrist_HM70A', path1, path2, transform_valid, transform_valid)

train_idx = cfg.wrist_HM70A_train_idx
valid_idx = cfg.wrist_HM70A_valid_idx
n_val_sample = len(valid_idx)
train_sampler = SubsetRandomSampler(train_idx)
valid_sampler = SubsetRandomSampler(valid_idx)

train_loader = DataLoader(dataset=dataset_train, batch_size=4, sampler=train_sampler)
valid_loader = DataLoader(dataset=dataset_val, batch_size=1, sampler=valid_sampler)

for epoch in range(300):
    print("="*100)
    running_loss1 = 0
    for x0, y0 in train_loader:
        student.train()
        #teacher.train()
        output1.train()
        optimizer1.zero_grad()
        
        ## ---- supervised ---- ##
        x0 = x0.float().to(device)
        y0 = y0.float().to(device)
        
        rep0_s = student(x0)
        output0_s = output1(rep0_s)
        
        loss = criterion2(output0_s, y0)
        
        loss.backward()
        optimizer1.step()
        
        running_loss1 += loss.item()
            
    running_loss1 /= len(train_loader)
    print("[Epoch:%d] [Loss sup:%f]" % ((epoch+1), running_loss1))

    if (epoch+1) % 1 == 0:
        ## ---- for student ---- ##
        student.eval()
        output1.eval()
        
        mean_accuracy = 0
        mean_precision = 0
        mean_recall = 0
        mean_dice = 0

        for x0, y0 in valid_loader:
            x0 = x0.float().to(device)
            y0 = y0 >= 0.5
            y0 = y0.to(device)
    
            with torch.no_grad():
                output0 = output1(student(x0))
    
            pred = output0 >= 0.5        
            pred = pred.view(-1)
                    
            Trues = pred[pred == y0.view(-1)]
            Falses = pred[pred != y0.view(-1)]
                
            TP = (Trues == 1).sum().item()
            TN = (Trues == 0).sum().item()
            FP = (Falses == 1).sum().item()
            FN = (Falses == 0).sum().item()
        
            accuracy = (TP + TN)/(TP + TN + FP + FN)
            if TP == 0:
                precision = 0
                recall = 0
                dice = 0
            else:
                precision = TP / (TP + FP)
                recall = TP / (TP + FN)
                dice = (2 * TP) / ((2 * TP) + FP + FN)

            mean_accuracy += accuracy
            mean_precision += precision
            mean_recall += recall
            mean_dice += dice
    
        mean_accuracy /= n_val_sample
        mean_precision /= n_val_sample
        mean_recall /= n_val_sample
        mean_dice /= n_val_sample
                
        print("[Student]", end=" ")
        print("[Accuracy:%f]" % mean_accuracy, end=" ")
        print("[Precision:%f]" % mean_precision, end=" ")
        print("[Recall:%f]" % mean_recall, end=" ")
        print("[F1 score:%f]" % mean_dice)
        '''
        ## ---- for teacher ---- ##
        teacher.eval()
        mean_accuracy = 0
        mean_precision = 0
        mean_recall = 0
        mean_dice = 0

        for x0, y0 in valid_loader:
            x0 = x0.float().to(device)
            y0 = y0 >= 0.5
            y0 = y0.to(device)
    
            with torch.no_grad():
                output0 = output2(head2(teacher(x0)))
    
            pred = output0 >= 0.5        
            pred = pred.view(-1)
                    
            Trues = pred[pred == y0.view(-1)]
            Falses = pred[pred != y0.view(-1)]
                
            TP = (Trues == 1).sum().item()
            TN = (Trues == 0).sum().item()
            FP = (Falses == 1).sum().item()
            FN = (Falses == 0).sum().item()
        
            accuracy = (TP + TN)/(TP + TN + FP + FN)
            if TP == 0:
                precision = 0
                recall = 0
                dice = 0
            else:
                precision = TP / (TP + FP)
                recall = TP / (TP + FN)
                dice = (2 * TP) / ((2 * TP) + FP + FN)

            mean_accuracy += accuracy
            mean_precision += precision
            mean_recall += recall
            mean_dice += dice
    
        mean_accuracy /= n_val_sample
        mean_precision /= n_val_sample
        mean_recall /= n_val_sample
        mean_dice /= n_val_sample
                
        print("[Teacher]", end=" ")
        print("[Accuracy:%f]" % mean_accuracy, end=" ")
        print("[Precision:%f]" % mean_precision, end=" ")
        print("[Recall:%f]" % mean_recall, end=" ")
        print("[F1 score:%f]" % mean_dice)
        '''

