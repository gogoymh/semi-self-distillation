import torch
import torch.nn as nn

class PairContrastiveLoss(nn.Module):
    def __init__(self, channel=3, hidden=768, size=16, padding=0, latent=1024):
        super().__init__()
        
        self.pixel_embeddings = nn.Conv2d(channel, hidden, kernel_size=1, stride=1, padding=padding)
        self.patch_embeddings = nn.Conv2d(channel, hidden, kernel_size=size, stride=size, padding=padding)
        self.sim = nn.CosineSimilarity(dim=-1)

        self.patch_size = size*size
    
        #self.latent = nn.Parameter(torch.randn(1, latent, hidden))
        
    def forward(self, x):
        pixel = self.pixel_embeddings(x)
        patch = self.patch_embeddings(x)
        B, D, H, W = x.shape
        
        pixel = pixel.flatten(2)
        pixel = pixel.transpose(-1, -2)
        
        patch = patch.flatten(2)
        patch = patch.transpose(-1, -2)
        
        pixel = pixel.unsqueeze(2)
        patch = patch.unsqueeze(1)
        
        SIM = self.sim(pixel,patch)
        SIM = SIM.reshape(B, self.patch_size, SIM.shape[2], SIM.shape[2])
        
        eye = torch.eye(SIM.shape[2]).to(x.device)
        #close = SIM[:,:,eye.long().type(torch.bool)]
        far = SIM[:,:,(1-eye).long().type(torch.bool)]
        
        #loss = - torch.log( (close/0.5).exp().sum() / (far/0.5).exp().sum() )
        loss = - torch.log( 1 / (far/0.5).exp().sum() )

        return loss.mean()
        

if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    a = torch.randn((3,32,224,224)).to(device)
    #b = torch.randn((6,3,224,224)).to(device)
    criterion = PairContrastiveLoss(channel=32, hidden=8, size=8, padding=0, latent=128).to(device)
    
    print(criterion(a))