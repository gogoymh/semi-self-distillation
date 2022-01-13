import numpy as np
from models.modeling import VisionTransformer, CONFIGS

import torch
import torch.nn as nn

class PairContrastiveLoss(nn.Module):
    def __init__(self, channel=3, hidden=768, size=16):
        super().__init__()
        
        self.patch_embeddings = nn.Conv2d(channel, hidden, kernel_size=size, stride=size)
        self.sim = nn.CosineSimilarity(dim=-1)
        n_patches = (224 // size) * (224 // size)

        self.position_embeddings = nn.Parameter(torch.zeros(1, n_patches+1, hidden))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden))
        
        self.hidden = hidden
        
    def forward(self, x, y):
        x = self.patch_embeddings(x)
        y = self.patch_embeddings(y)
        B, D, H, W = x.shape
        
        cls_tokens = self.cls_token.expand(B, -1, -1)
        
        x = x.flatten(2)
        x = x.transpose(-1, -2)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.position_embeddings

        y = y.flatten(2)
        y = y.transpose(-1, -2)
        y = torch.cat((cls_tokens, y), dim=1)
        y = y + self.position_embeddings
        
        x = x.reshape(-1, self.hidden).unsqueeze(1)
        y = y.reshape(-1, self.hidden).unsqueeze(0)
        

        SIM = self.sim(x,y)
        
        eye = torch.eye(x.shape[0]).to(x.device)
        
        close = SIM[eye.long().type(torch.bool)]
        far = SIM[(1-eye).long().type(torch.bool)]

        loss = - torch.log( (close/0.5).exp().sum() / (far/0.5).exp().sum() )

        return loss


if __name__ == "__main__":
    model_type = "ViT-B_16"
    config = CONFIGS[model_type]

    num_classes = 10 #if dataset == "cifar10" else 100

    img_size = 224
    model = VisionTransformer(config, img_size, zero_head=True, num_classes=num_classes)

    pretrained_dir = "/home/DATA/ymh/ViT-pytorch/ViT-B_16.npz"
    model.load_from(np.load(pretrained_dir))
    
    print(model.transformer.embeddings.state_dict())
    
    loss = PairContrastiveLoss(3, 768, 16)
    
    loss.load_state_dict(model.transformer.embeddings.state_dict(), strict=False)
    torch.save({'model_state_dict': loss.state_dict()}, "/home/DATA/ymh/ultra/save/model/pairloss.pth")
    
    print("saved")