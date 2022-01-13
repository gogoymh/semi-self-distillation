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
from network_C import Our_Unet_singlegpu as net
from Dataset import USDataset as newset
from Dataset import Numberset
import config as cfg

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

num_models = 8

models = nn.ModuleList()
for i in range(num_models):
    models.append(net().to(device))

optimizers = []
for i in range(num_models):
    optimizers.append(optim.Adam(models[i].parameters(), lr=5e-4))


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
path1 = "/DATA/ymh/semisup/newset/wrist_train/wrist_HM70A"
path2 = "/DATA/ymh/semisup/newset/wrist_target/wrist_HM70A"
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
prob = torch.ones(num_models) / num_models
loss = torch.ones(num_models)

ema = 0.1

for epoch in range(300):
    running_loss1 = 0
    running_loss2 = 0    
    
    #loss = torch.zeros(num_models)
    #cnt = torch.zeros(num_models)

    for x1, _ in unlabeled_loader:
        ## ---- unlabeled data ---- ##
        x1 = x1.float().to(device)
        
        ## ---- labeled data ---- ##
        x0, y0 = labeled_loader.__iter__().next()
        x0 = x0.float().to(device)
        y0 = y0.float().to(device)
        
        #chosen = num_loader.__iter__().next()
        chosen = np.random.choice(num_models, size=2, replace=False, p=prob)
        models[chosen[0]].train()
        models[chosen[1]].train()
        
        ## ---- update student ---- ##
        optimizers[chosen[0]].zero_grad()
        optimizers[chosen[1]].zero_grad()
        
        output1_0 = models[chosen[0]](x0)
        output2_0 = models[chosen[1]](x0)
        
        loss_sup1 = criterion(output1_0, y0) 
        loss_sup2 = criterion(output2_0, y0)
        
        output1_1 = models[chosen[0]](x1)
        output2_1 = models[chosen[1]](x1)
        
        loss_self1 = criterion(output1_1, output2_1.detach())
        loss_self2 = criterion(output2_1, output1_1.detach())
        
        loss1 = loss_sup1 + loss_self1
        loss2 = loss_sup2 + loss_self2
        loss1.backward()
        loss2.backward()
        optimizers[chosen[0]].step()
        optimizers[chosen[1]].step()
        
        loss[chosen[0]] = (1-ema) * loss[chosen[0]].item() + ema * loss_sup1.item()
        loss[chosen[1]] = (1-ema) * loss[chosen[1]].item() + ema * loss_sup2.item()
        #cnt[chosen[0]] += 1
        #cnt[chosen[1]] += 1
        
        prob = 1 - loss
        prob = prob.numpy()
        prob = prob / np.sum([prob])
        
        running_loss1 += (loss_sup1.item() + loss_sup2.item())
        running_loss2 += (loss_self1.item() + loss_self2.item())
        #print(loss.item())
    
    running_loss1 /= len(unlabeled_loader)
    running_loss2 /= len(unlabeled_loader)
    print("="*100)
    print("[Epoch:%d] [Loss sup:%f] [Loss self:%f]" % ((epoch+1), running_loss1, running_loss2))
    #print(prob)
    
    #print(loss / cnt)
    #prob = 2 - (loss / cnt)
    #prob = prob.numpy()
    #prob = prob / np.sum([prob])
    #prob = prob / prob.sum()
    #prob = prob.softmax(dim=0)
    #print(prob)
    
    if (epoch+1) % 1 == 0:
        for i in range(num_models):
            ## ---- for student ---- ##
            models[i].eval()
            mean_accuracy = 0
            mean_precision = 0
            mean_recall = 0
            mean_dice = 0

            for x0, y0 in valid_loader:
                x0 = x0.float().to(device)
                y0 = y0 >= 0.5
                y0 = y0.to(device)
                
                with torch.no_grad():
                    output0 = models[i](x0)
    
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
                
            print("[Model%d]" % i, end=" ")
            print("[Accuracy:%f]" % mean_accuracy, end=" ")
            print("[Precision:%f]" % mean_precision, end=" ")
            print("[Recall:%f]" % mean_recall, end=" ")
            print("[F1 score:%f]" % mean_dice)
