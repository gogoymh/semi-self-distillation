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
#from network_C_sev2 import Our_Unet_singlegpu as net
from network_C import Our_Unet_singlegpu as net
from Dataset import USDataset as newset
import config as cfg
#from utils import rand_translation as T
#from loss1 import AgoodLoss
#from ViT_patchloss7_2_2 import PairContrastiveLoss

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

student = net().to(device)
teacher = net().to(device)
#teacher.load_state_dict(student.state_dict())
#output2 = nn.Sequential(
#    nn.Conv2d(3, 1, 1, 1, 0, bias=True),
#    nn.Sigmoid()).to(device)
#teacher.load_state_dict(student.state_dict())
#criterion1 = PairContrastiveLoss(2, 256, 32, 8, 1)
#loss_checkpoint = torch.load("/home/DATA/ymh/ultra/save/model/pairloss.pth")
#loss_checkpoint = torch.load("/data/ymh/US_segmentation/save/model/pairloss.pth")
#criterion1.load_state_dict(loss_checkpoint["model_state_dict"])
#criterion1.to(device)
#criterion1.eval()
criterion2 = cfg.DiceLoss()

#param1 = list(student.parameters()) + list(criterion1.parameters()) + list(output1.parameters())
#optimizer1 = optim.Adam(param1, lr=5e-4)

param2 = list(student.parameters()) + list(teacher.parameters())
optimizer2 = optim.Adam(param2, lr=5e-4)

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

#path1 = "/DATA/ymh_ksw/dataset/US_image/newset/wrist_train/wrist_HM70A"
path1 = "/home/DATA/ymh/ultra/newset/wrist_train/wrist_HM70A"
#path1 = "/data/ymh/US_segmentation/newset/wrist_train/wrist_HM70A"
#path2 = "/DATA/ymh_ksw/dataset/US_image/newset/wrist_target/wrist_HM70A"
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


beta = 1.0
ema_tau = 0.1

def rand_bbox(W, H, lam):
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int64(W * cut_rat)
    cut_h = np.int64(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2

for epoch in range(300):
    print("="*100)
    running_loss1 = 0
    running_loss2 = 0
    for x1, _ in unlabeled_loader:
        #student.eval()
        #teacher.eval()
        student.train()
        teacher.train()
        
        x0, y0 = labeled_loader.__iter__().next()
        ## ---- supervised ---- ##
        x0 = x0.float().to(device)
        y0 = y0.float().to(device)
        
        x1 = x1.float().to(device)
        y1_s = student(x1).detach()
        y1_t = teacher(x1).detach()
        
        input = torch.cat((x0, x1), dim=0)
        target_s = torch.cat((y0, y1_t), dim=0)
        target_t = torch.cat((y0, y1_s), dim=0)
        
        #student.train()
        #teacher.train()
        
        optimizer2.zero_grad()
        
        lam = np.random.beta(beta, beta)
        rand_index = torch.randperm(input.size()[0]).to(device)
        
        bbx1, bby1, bbx2, bby2 = rand_bbox(224, 224, lam)
        input[:, :, bbx1:bbx2, bby1:bby2] = input[rand_index, :, bbx1:bbx2, bby1:bby2]
        #target_a = target
        #target_b = target[rand_index,:,:,:]
        target_s[:, :, bbx1:bbx2, bby1:bby2] = target_s[rand_index, :, bbx1:bbx2, bby1:bby2]
        target_t[:, :, bbx1:bbx2, bby1:bby2] = target_t[rand_index, :, bbx1:bbx2, bby1:bby2]
        
        #lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (input.size()[-1] * input.size()[-2]))

        output_s = student(input)
        output_t = teacher(input)
        loss = criterion2(output_s, target_s) + criterion2(output_t, target_t)
        #loss = criterion2(output, target_a) * lam + criterion2(output, target_b) * (1. - lam)
        
        loss.backward()
        optimizer2.step()
        
        #for target_param, param in zip(teacher.parameters(), student.parameters()):
        #    #target_param.data.copy_(target_param.data * (1.0 - ema_tau[epoch]) + param.data * ema_tau[epoch])
        #    target_param.data.copy_(target_param.data * (1.0 - ema_tau) + param.data * ema_tau)
        
        running_loss1 += loss.item()
            
    running_loss1 /= len(labeled_loader)
    running_loss2 /= len(unlabeled_loader)
    print("[Epoch:%d] [Loss sup:%f] [Loss self:%f]" % ((epoch+1), running_loss1, running_loss2))

    if (epoch+1) % 1 == 0:
        ## ---- for student ---- ##
        student.eval()
        #output1.eval()
        #head1.eval()
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
        

