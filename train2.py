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
from network_C import Our_Unet_singlegpu as net
from Dataset import USDataset as newset
import config as cfg
from utils import rand_translation as T

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

student = net().to(device)
teacher = net().to(device)
#teacher.load_state_dict(student.state_dict())

param = list(student.parameters()) + list(teacher.parameters())
optimizer = optim.Adam(param, lr=5e-4)

transform_train = tf.Compose([
        tf.ToTensor(),
        tf.ToPILImage(),
        tf.Resize((442,565)),
        tf.RandomAffine(0, shear=[-15, 15, -15, 15]),
        tf.ToTensor()
     ])

transform_valid = tf.Compose([
        tf.ToTensor(),
        tf.ToPILImage(),
        tf.Resize((442,565)),
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
unlabeled_batch = 3
labeled_loader = DataLoader(dataset=dataset_labeled, batch_size=labeled_batch, sampler=labeled_sampler)
unlabeled_loader = DataLoader(dataset=dataset_unlabaled, batch_size=unlabeled_batch, sampler=unlabeled_sampler)
valid_loader = DataLoader(dataset=dataset_val, batch_size=1, sampler=valid_sampler)


criterion = criterion = cfg.DiceLoss()
alpha = 1
beta = torch.linspace(0.1, 6, steps=300)
#beta = 6
#ema_tau = 0.01
ema_tau = torch.linspace(0, 0.01, steps=300)

for epoch in range(300):
    running_loss1 = 0
    running_loss2 = 0    
    student.train()
    teacher.train() # batch norm layer가 있는 network면 차이가 난다.
    for x1, _ in unlabeled_loader:
        ## ---- unlabeled data ---- ##
        x1 = x1.float().to(device)
        
        ## ---- labeled data ---- ##
        x0, y0 = labeled_loader.__iter__().next()
        x0 = x0.float().to(device)
        y0 = y0.float().to(device)

        ## ---- supervised ---- ##
        optimizer.zero_grad()
        
        output0_s = student(x0)
        output0_t = teacher(x0)
        loss_sup = criterion(output0_s, y0) + criterion(output0_t, y0)
        
        ## ---- unsupervised ---- ##
        output1_s = student(x1)
        output1_t = teacher(x1)

        loss_self = criterion(output1_s, output1_t.detach()) + criterion(output1_t, output1_s.detach())
        
        loss = alpha * loss_sup + beta[epoch] * loss_self
        loss.backward()
        optimizer.step()
        
        ## ---- Cross mean ---- ##
        for target_param, param in zip(teacher.parameters(), student.parameters()):
            #target_param.data.copy_(target_param.data * (1.0 - ema_tau[epoch]) + param.data * ema_tau[epoch])
            for_teacher = target_param.data * (1.0 - ema_tau[epoch]) + param.data * ema_tau[epoch]
            for_student = target_param.data * ema_tau[epoch] + param.data * (1.0 - ema_tau[epoch])
            target_param.data.copy_(for_teacher)
            param.data.copy_(for_student)
        
        running_loss1 += loss_sup.item()
        running_loss2 += loss_self.item()
        #print(loss.item())
            
    running_loss1 /= len(unlabeled_loader)
    running_loss2 /= len(unlabeled_loader)
    print("="*100)
    print("[Epoch:%d] [Loss sup:%f] [Loss self:%f]" % ((epoch+1), running_loss1, running_loss2))

    if (epoch+1) % 1 == 0:
        ## ---- for student ---- ##
        student.eval()
        mean_accuracy = 0
        mean_precision = 0
        mean_recall = 0
        mean_dice = 0

        for x0, y0 in valid_loader:
            x0 = x0.float().to(device)
            y0 = y0 >= 0.5
            y0 = y0.to(device)
    
            with torch.no_grad():
                output0 = student(x0)
    
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
                output0 = teacher(x0)
    
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


