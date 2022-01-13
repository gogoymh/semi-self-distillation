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
import config as cfg

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model1 = net().to(device)
model2 = net().to(device)
model3 = net().to(device)
model4 = net().to(device)
model5 = net().to(device)
model6 = net().to(device)
model7 = net().to(device)
model8 = net().to(device)
model9 = net().to(device)
model10 = net().to(device)
model11 = net().to(device)
model12 = net().to(device)

params = list(model1.parameters()) + list(model2.parameters()) + list(model3.parameters()) + list(model4.parameters())
params += list(model5.parameters()) + list(model6.parameters()) + list(model7.parameters()) + list(model8.parameters())
params += list(model9.parameters()) + list(model10.parameters()) + list(model11.parameters()) + list(model12.parameters())
optimizer = optim.Adam(params, lr=5e-4)
#ema_tau = 0.01
#ema_tau = torch.linspace(0.99, 0.01, steps=300)

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

path1 = "/DATA/ymh/newset/wrist_train/wrist_HM70A"
path2 = "/DATA/ymh/newset/wrist_target/wrist_HM70A"
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


criterion = cfg.DiceLoss()

for epoch in range(300):
    running_loss1 = 0
    running_loss2 = 0    
    model1.train()
    model2.train()
    model3.train()
    model4.train()
    model5.train()    
    model6.train()
    model7.train()
    model8.train()   
    model9.train()    
    model10.train()
    model11.train()
    model12.train() 
    for x1, _ in unlabeled_loader:
        ## ---- unlabeled data ---- ##
        x1 = x1.float().to(device)
        
        ## ---- labeled data ---- ##
        x0, y0 = labeled_loader.__iter__().next()
        x0 = x0.float().to(device)
        y0 = y0.float().to(device)
        
        ## ---- update student ---- ##
        optimizer.zero_grad()
        
        output1_0 = model1(x0)
        output2_0 = model2(x0)
        output3_0 = model3(x0)
        output4_0 = model4(x0)
        output5_0 = model5(x0)
        output6_0 = model6(x0)
        output7_0 = model7(x0)
        output8_0 = model8(x0)
        output9_0 = model9(x0)
        output10_0 = model10(x0)
        output11_0 = model11(x0)
        output12_0 = model12(x0)
        
        loss_sup = criterion(output1_0, y0) + criterion(output2_0, y0) + criterion(output3_0, y0) + criterion(output4_0, y0)
        loss_sup += criterion(output5_0, y0) + criterion(output6_0, y0) + criterion(output7_0, y0) + criterion(output8_0, y0)
        loss_sup += criterion(output9_0, y0) + criterion(output10_0, y0) + criterion(output11_0, y0) + criterion(output12_0, y0)
        
        output1_1 = model1(x1)
        output2_1 = model2(x1)
        output3_1 = model3(x1)
        output4_1 = model4(x1)
        output5_1 = model5(x1)
        output6_1 = model6(x1)
        output7_1 = model7(x1)
        output8_1 = model8(x1)
        output9_1 = model9(x1)
        output10_1 = model10(x1)
        output11_1 = model11(x1)
        output12_1 = model12(x1)

        y1 = output1_1 + output2_1 + output3_1 + output4_1 
        y1 += output5_1 + output6_1 + output7_1 + output8_1
        y1 += output9_1 + output10_1 + output11_1 + output12_1
        y1 = (y1/12).detach()
        
        loss_self = criterion(output1_1, y1) + criterion(output2_1, y1) + criterion(output3_1, y1) + criterion(output4_1, y1)
        loss_self += criterion(output5_1, y1) + criterion(output6_1, y1) + criterion(output7_1, y1) + criterion(output8_1, y1)
        loss_self += criterion(output9_1, y1) + criterion(output10_1, y1) + criterion(output11_1, y1) + criterion(output12_1, y1)
        
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
        '''

