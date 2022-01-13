import torch
import torch.nn as nn

class PairContrastiveLoss(nn.Module):
    def __init__(self, channel=3, hidden=768, size=16, padding=0, latent=1024):
        super().__init__()
        
        self.patch_embeddings = nn.Conv2d(channel, hidden, kernel_size=size, stride=size, padding=padding)
        self.sim = nn.CosineSimilarity(dim=-1)
        
        self.latent = nn.Parameter(torch.randn(1, latent, hidden))
        
    def forward(self, x):
        patch = self.patch_embeddings(x)
        B, D, H, W = x.shape
        
        patch = patch.flatten(2)
        patch = patch.transpose(-1, -2)
        
        one = patch.unsqueeze(2)
        two = patch.unsqueeze(1)
        
        SIM = self.sim(one,two)
        eye = torch.eye(SIM.shape[1]).to(x.device)
        
        far = SIM[:, (1-eye).long().type(torch.bool)]
        
        
        latent = self.latent.expand(B, -1, -1)
        latent = latent.unsqueeze(1)
        
        close = self.sim(patch.unsqueeze(2),latent).reshape(B, -1)
        
        loss = - torch.log( (close/0.5).exp().sum(dim=1) / (far/0.5).exp().sum(dim=1) )

        return loss.mean()

if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    a = torch.randn((3,2,224,224)).to(device)
    #b = torch.randn((6,3,224,224)).to(device)
    criterion = PairContrastiveLoss(2, 8, 4, latent=256).to(device)
    
    print(criterion(a))
