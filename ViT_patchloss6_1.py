import torch
import torch.nn as nn

class PairContrastiveLoss(nn.Module):
    def __init__(self, channel=3, hidden=768, size=16, padding=0, latent=1024):
        super().__init__()
        
        self.patch_embeddings = nn.Conv2d(channel, hidden, kernel_size=size, stride=size, padding=padding)
        self.sim = nn.CosineSimilarity(dim=-1)
        #n_patches = (image // size) * (image // size)

        self.latent1 = nn.Parameter(torch.randn(1, latent, hidden))
        self.latent2 = nn.Parameter(torch.randn(1, latent, hidden))
        
    def forward(self, x):
        patch = self.patch_embeddings(x)
        B, D, H, W = x.shape
        
        latent1 = self.latent1.expand(B, -1, -1)
        latent2 = self.latent2.expand(B, -1, -1)
        
        patch = patch.flatten(2)
        patch = patch.transpose(-1, -2)
        
        patch = patch.unsqueeze(2)
        latent1 = latent1.unsqueeze(1)
        latent2 = latent2.unsqueeze(1)        
        
        far = self.sim(patch,latent1)
        far = far.reshape(B, -1)
        
        close = self.sim(patch,latent2)
        close = close.reshape(B, -1)        
        
        loss = - torch.log( (close/0.5).exp().sum(dim=1) / (far/0.5).exp().sum(dim=1) )

        return loss.mean()

if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    a = torch.randn((6,3,224,224)).to(device)
    #b = torch.randn((6,3,224,224)).to(device)
    criterion = PairContrastiveLoss().to(device)
    
    print(criterion(a))