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
#from ViT_patchloss6_4 import PairContrastiveLoss

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

student = net().to(device)
#teacher = net().to(device)
#head1 = nn.Conv2d(32, 2, 1, 1, 0, bias=True).to(device)
#head2 = nn.Conv2d(32, 3, 1, 1, 0, bias=True).to(device)
output1 = nn.Sequential(
    nn.Conv2d(32, 1, 1, 1, 0, bias=True),
    nn.Sigmoid()).to(device)
output2 = nn.Sequential(
    nn.Conv2d(32, 1, 1, 1, 0, bias=True),
    nn.Sigmoid()).to(device)
#teacher.load_state_dict(student.state_dict())
#criterion1 = PairContrastiveLoss(2, 8, 1, latent=512)
#loss_checkpoint = torch.load("/home/DATA/ymh/ultra/save/model/pairloss.pth")
#loss_checkpoint = torch.load("/data/ymh/US_segmentation/save/model/pairloss.pth")
#criterion1.load_state_dict(loss_checkpoint["model_state_dict"])
#criterion1.to(device)
#criterion1.eval()
criterion2 = cfg.DiceLoss()

param = list(student.parameters()) + list(output1.parameters()) + list(output2.parameters())# + list(criterion1.parameters())  + list(head1.parameters())# + list(head2.parameters())+ list(teacher.parameters())+ list(output2.parameters())
optimizer1 = optim.Adam(param, lr=5e-4)

#param_output = list(output1.parameters()) + list(output2.parameters())
#optimizer2 = optim.Adam(student.parameters(), lr=5e-4)

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

#sample_img, _ = valid_loader.__iter__().next()
#eps = torch.linspace(0., 0., 300)

for epoch in range(300):
    print("="*100)
    running_loss1 = 0
    running_loss2 = 0
    for x1, _ in unlabeled_loader:
        student.train()
        output1.train()
        output2.train()
        #head1.train()
        #criterion1.train()
        optimizer1.zero_grad()
        
        x0, y0 = labeled_loader.__iter__().next()
        ## ---- supervised ---- ##
        x0 = x0.float().to(device)
        y0 = y0.float().to(device)
        
        rep0 = student(x0)
        output0_s = output1(rep0)
        output0_t = output2(rep0)
        
        loss_sup = criterion2(output0_s, y0) + criterion2(output0_t, y0)
        
        x1 = x1.float().to(device)        
        
        rep1 = student(x1)
        output1_s = output1(rep1)
        output1_t = output2(rep1)

        loss_self = criterion2(output1_s, output1_t.detach()) + criterion2(output1_t, output1_s.detach())
        loss = loss_sup + loss_self
        loss.backward()
        optimizer1.step()
        
        running_loss1 += loss_sup.item()
        running_loss2 += loss_self.item()
            
    running_loss1 /= len(labeled_loader)
    running_loss2 /= len(unlabeled_loader)
    print("[Epoch:%d] [Loss sup:%f] [Loss self:%f]" % ((epoch+1), running_loss1, running_loss2))

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

