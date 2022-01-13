import torch
from torch.utils.data import DataLoader
from torchvision import transforms as tf
from torch.utils.data.sampler import SubsetRandomSampler
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
#import matplotlib.pyplot as plt
#from skimage.io import imread
import os
import numpy as np

#from self_sup_net import Our_Unet_singlegpu as net
#from network_C import Our_Unet_singlegpu as net
from network_C_sev2 import Our_Unet_singlegpu as net
from Dataset import USDataset as newset
from Dataset import Numberset
import config as cfg

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model1 = net().to(device)
model2 = net().to(device)

last1 = nn.Sequential(
    nn.Conv2d(32, 1, 1, 1, 0, bias=False),
    nn.Sigmoid()
    ).to(device)
last2 = nn.Sequential(
    nn.Conv2d(32, 1, 3, 2, 1, bias=False),
    nn.Upsample(scale_factor=2, mode='nearest'),
    nn.Sigmoid()
    ).to(device)

param1 = list(model1.parameters()) + list(last1.parameters())
param2 = list(model2.parameters()) + list(last2.parameters())

param = param1 + param2
optimizer = optim.Adam(param, lr=5e-4)

transform_train = tf.Compose([
        tf.ToTensor(),
        tf.ToPILImage(),
        tf.Resize((224,224)),
        tf.RandomAffine(0, shear=[-15, 15, -15, 15]),
        tf.ToTensor()
     ])

transform_valid = tf.Compose([
        tf.ToTensor(),
        tf.ToPILImage(),
        tf.Resize((224,224)),
        tf.ToTensor(),
     ])

#path1 = "/home/DATA/ymh/ultra/newset/wrist_train/wrist_HM70A"
#path1 = "/data/ymh/US_segmentation/newset/wrist_train/wrist_HM70A"
#path2 = "/home/DATA/ymh/ultra/newset/wrist_target/wrist_HM70A"
#path2 = "/data/ymh/US_segmentation/newset/wrist_target/wrist_HM70A"
#path1 = "/DATA/ymh/semisup/newset/wrist_train/wrist_HM70A"
#path2 = "/DATA/ymh/semisup/newset/wrist_target/wrist_HM70A"
path1 = "/DATA/ymh_ksw/dataset/US_image/newset/wrist_train/wrist_HM70A"
path2 = "/DATA/ymh_ksw/dataset/US_image/newset/wrist_target/wrist_HM70A"
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
unlabeled_batch = 4
labeled_loader = DataLoader(dataset=dataset_labeled, batch_size=labeled_batch, sampler=labeled_sampler)
unlabeled_loader = DataLoader(dataset=dataset_unlabaled, batch_size=unlabeled_batch, sampler=unlabeled_sampler)
valid_loader = DataLoader(dataset=dataset_val, batch_size=1, sampler=valid_sampler)

#numset = Numberset(num_models)
#num_loader = DataLoader(dataset=numset, batch_size=2, shuffle=True)
criterion = cfg.DiceLoss()

for epoch in range(300):
    running_loss1 = 0
    running_loss2 = 0    
    
    model1.train()
    model2.train()
    
    for x1, _ in unlabeled_loader:
        ## ---- unlabeled data ---- ##
        x1 = x1.float().to(device)
        
        ## ---- labeled data ---- ##
        x0, y0 = labeled_loader.__iter__().next()
        x0 = x0.float().to(device)
        y0 = y0.float().to(device)
        
        ## ---- update student ---- ##
        optimizer.zero_grad()
        
        output1_0 = last1(model1(x0))
        output2_0 = last2(model2(x0))
        
        loss_sup = criterion(output1_0, y0) + criterion(output2_0, y0)
        
        output1_1 = last1(model1(x1))
        output2_1 = last2(model2(x1))
        
        loss_self1 = criterion(output1_1, (output2_1 >= 0.5).detach()) + criterion(output1_0, (output2_0 >= 0.5).detach())
        loss_self2 = criterion(output2_1, (output1_1 >= 0.5).detach()) + criterion(output2_0, (output1_0 >= 0.5).detach())
        loss_self = loss_self1 + loss_self2
        
        loss = loss_sup + loss_self
        loss.backward()
        optimizer.step()
        
        running_loss1 += loss_sup.item()
        running_loss2 += loss_self.item()
        #print(loss.item())
    
    running_loss1 /= len(unlabeled_loader)
    running_loss2 /= len(unlabeled_loader)
    print("="*100)
    print("[Epoch:%d] [Loss sup:%f] [Loss self:%f]" % ((epoch+1), running_loss1, running_loss2))
    #print(prob)
    

    '''
    for teacher in models:
        student = models[np.random.choice(num_models, size=1, p=prob)[0]]
        for target_param, param in zip(teacher.parameters(), student.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - ema) + param.data * ema)
    '''
    if (epoch+1) % 1 == 0:
        ## ---- for student ---- ##
        model1.eval()
        mean_accuracy = 0
        mean_precision = 0
        mean_recall = 0
        mean_dice = 0

        for x0, y0 in valid_loader:
            x0 = x0.float().to(device)
            y0 = y0 >= 0.5
            y0 = y0.to(device)
                
            with torch.no_grad():
                output0 = last1(model1(x0))
    
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
                
        print("[Model1]", end=" ")
        print("[Accuracy:%f]" % mean_accuracy, end=" ")
        print("[Precision:%f]" % mean_precision, end=" ")
        print("[Recall:%f]" % mean_recall, end=" ")
        print("[F1 score:%f]" % mean_dice)
        
        ## ---- ---- #3
        model2.eval()
        mean_accuracy = 0
        mean_precision = 0
        mean_recall = 0
        mean_dice = 0

        for x0, y0 in valid_loader:
            x0 = x0.float().to(device)
            y0 = y0 >= 0.5
            y0 = y0.to(device)
                
            with torch.no_grad():
                output0 = last2(model2(x0))
    
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
                
        print("[Model2]", end=" ")
        print("[Accuracy:%f]" % mean_accuracy, end=" ")
        print("[Precision:%f]" % mean_precision, end=" ")
        print("[Recall:%f]" % mean_recall, end=" ")
        print("[F1 score:%f]" % mean_dice)