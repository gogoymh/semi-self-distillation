import torch
import torch.nn as nn

class PairContrastiveLoss(nn.Module):
    def __init__(self, num_class, latent=128, channel=32, hidden=768, size=16, padding=0):
        super().__init__()
        
        self.patch_embeddings = nn.Conv2d(channel, hidden, kernel_size=size, stride=size, padding=padding)
        self.sim = nn.CosineSimilarity(dim=-1)

        self.latent = nn.Parameter(torch.randn(1, latent * num_class, hidden))
        self.num_class = num_class
        self.num_latent = latent
        
    def forward(self, x, label):
        patch = self.patch_embeddings(x)
        B, D, H, W = x.shape
        
        latent = self.latent.expand(B, -1, -1)
        
        patch = patch.flatten(2)
        patch = patch.transpose(-1, -2)
        
        patch = patch.unsqueeze(1)
        latent = latent.unsqueeze(2)
        
        SIM = self.sim(latent, patch)
        SIM = SIM.reshape(B, self.num_latent, self.num_class, H, W)
        SIM = SIM.mean(dim=1)
        
        mask = torch.cat(((label < 0.5), (label >= 0.5)), dim=1)
        
        close = SIM[mask]
        far = SIM[~mask]
        
        #loss = - torch.log(  (close/0.5).exp().sum() / ((close/0.5).exp().sum() + (far/0.5).exp().sum() ) )
        loss = - torch.log(  (close/0.5).exp().sum() / (far/0.5).exp().sum() ) 

        return loss.mean()

if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    a = torch.randn((1,32,14,14)).to(device)
    b = torch.rand((1,1,14,14)).to(device)
    criterion = PairContrastiveLoss(2, 128, 32, 8, 1).to(device)

    criterion.eval()
    print(criterion(a, b))








