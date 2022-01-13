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
from network_C import Our_Unet_singlegpu as net
from Dataset import USDataset as newset
import config as cfg

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


model0 = net().to(device)
model1 = net().to(device)
model2 = net().to(device)

params1 = list(model0.parameters()) + list(model1.parameters())
params2 = list(model0.parameters()) + list(model2.parameters())

optimizer = optim.Adam(params1, lr=5e-4)
optimizer = optim.Adam(params2, lr=5e-4)

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

path1 = "/home/DATA/ymh/ultra/newset/wrist_train/wrist_HM70A"
#path1 = "/data/ymh/US_segmentation/newset/wrist_train/wrist_HM70A"
path2 = "/home/DATA/ymh/ultra/newset/wrist_target/wrist_HM70A"
#path2 = "/data/ymh/US_segmentation/newset/wrist_target/wrist_HM70A"
#path1 = "/DATA/ymh/semisup/newset/wrist_train/wrist_HM70A"
#path2 = "/DATA/ymh/semisup/newset/wrist_target/wrist_HM70A"
#path1 = "/DATA/ymh_ksw/dataset/US_image/newset/wrist_train/wrist_HM70A"
#path2 = "/DATA/ymh_ksw/dataset/US_image/newset/wrist_target/wrist_HM70A"
dataset_labeled = newset('wrist_HM70A', path1, path2, transform_train, transform_train)
dataset_unlabaled = newset('wrist_HM70A', path1, path2, transform_train, transform_train)
dataset_val = newset('wrist_HM70A', path1, path2, transform_valid, transform_valid)

labeled_idx1 = cfg.wrist_HM70A_labeled_half1_idx
labeled_idx2 = cfg.wrist_HM70A_labeled_half2_idx
unlabeled_idx1 = cfg.wrist_HM70A_unlabeled_half1_idx
unlabeled_idx2 = cfg.wrist_HM70A_unlabeled_half2_idx
valid_idx = cfg.wrist_HM70A_valid_idx

n_val_sample = len(valid_idx)
labeled_sampler1 = SubsetRandomSampler(labeled_idx1)
labeled_sampler2 = SubsetRandomSampler(labeled_idx2)
unlabeled_sampler1 = SubsetRandomSampler(unlabeled_idx1)
unlabeled_sampler2 = SubsetRandomSampler(unlabeled_idx2)
valid_sampler = SubsetRandomSampler(valid_idx)

labeled_batch = 4
unlabeled_batch = 4
labeled_loader1 = DataLoader(dataset=dataset_labeled, batch_size=labeled_batch, sampler=labeled_sampler1)
labeled_loader2 = DataLoader(dataset=dataset_labeled, batch_size=labeled_batch, sampler=labeled_sampler2)
unlabeled_loader1 = DataLoader(dataset=dataset_unlabaled, batch_size=unlabeled_batch, sampler=unlabeled_sampler1)
unlabeled_loader2 = DataLoader(dataset=dataset_unlabaled, batch_size=unlabeled_batch, sampler=unlabeled_sampler2)
valid_loader = DataLoader(dataset=dataset_val, batch_size=1, sampler=valid_sampler)

criterion = cfg.DiceLoss()

for epoch in range(300):
    running_loss1 = 0
    running_loss2 = 0    
    
    #model1.train()
    #model2.train()
    
    for _ in range(len(unlabeled_loader1)):
        optimizer.zero_grad()
        
        ## ---- model1 ---- ##
        model1.train()
        model2.eval()
        
        x0_1, y0_1 = labeled_loader1.__iter__().next()
        x0_1 = x0_1.float().to(device)
        y0_1 = y0_1.float().to(device)
        
        x1_1, _ = unlabeled_loader1.__iter__().next()
        x1_1 = x1_1.float().to(device)
        
        output1_0_1 = model1(x0_1)
        #output2_0_1 = model2(x0_1)
        loss_sup1 = criterion(output1_0_1, y0_1)
        
        output1_1_1 = model1(x1_1)
        output2_1_1 = model2(x1_1)
        loss_self1 = criterion(output1_1_1, (output2_1_1 >= 0.5).detach())# + criterion(output1_0_1, (output2_0_1 >= 0.5).detach())
        '''
        loss1 = loss_sup1 + loss_self1
        loss1.backward()
        optimizer1.step()
        '''
        ## ---- model2 ---- ##
        model1.eval()
        model2.train()
        
        x0_2, y0_2 = labeled_loader2.__iter__().next()
        x0_2 = x0_2.float().to(device)
        y0_2 = y0_2.float().to(device)
        
        x1_2, _ = unlabeled_loader2.__iter__().next()
        x1_2 = x1_2.float().to(device)
        
        #output1_0_2 = model1(x0_2)
        output2_0_2 = model2(x0_2)
        loss_sup2 = criterion(output2_0_2, y0_2)
        
        output1_1_2 = model1(x1_2)
        output2_1_2 = model2(x1_2)        
        loss_self2 = criterion(output2_1_2, (output1_1_2 >= 0.5).detach())# + criterion(output2_0_2, (output1_0_2 >= 0.5).detach())
        '''
        loss2 = loss_sup2 + loss_self2
        loss2.backward()
        optimizer2.step()
        
        ## ---- backward ---- ##
        '''
        loss1 = loss_sup1 + loss_self1
        loss2 = loss_sup2 + loss_self2
        loss = loss1 + loss2
        loss.backward()
        optimizer.step()
        
        running_loss1 += (loss_sup1.item() + loss_sup2.item())
        running_loss2 += (loss_self1.item() + loss_self2.item())
        #print(loss.item())
    
    running_loss1 /= len(unlabeled_loader1)
    running_loss2 /= len(unlabeled_loader2)
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
                output0 = model1(x0)
    
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
                output0 = model2(x0)
    
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