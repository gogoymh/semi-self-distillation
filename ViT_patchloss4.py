import torch
import torch.nn as nn

class PairContrastiveLoss(nn.Module):
    def __init__(self, channel=3, hidden=768, size=16):
        super().__init__()
        
        self.patch_embeddings = nn.Conv2d(channel, hidden, kernel_size=size, stride=size)
        self.sim = nn.CosineSimilarity(dim=-1)
        n_patches = (224 // size) * (224 // size)

        #self.position_embeddings = nn.Parameter(torch.zeros(1, n_patches+1, hidden))
        #self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden))
        
        #self.hidden = hidden
        
    def forward(self, x, y):
        x = self.patch_embeddings(x)
        y = self.patch_embeddings(y)
        B, D, H, W = x.shape
        
        #cls_tokens = self.cls_token.expand(B, -1, -1)
        
        x = x.flatten(2)
        x = x.transpose(-1, -2)
        #x = torch.cat((cls_tokens, x), dim=1)
        #x = x + self.position_embeddings

        y = y.flatten(2)
        y = y.transpose(-1, -2)
        #y = torch.cat((cls_tokens, y), dim=1)
        #y = y + self.position_embeddings
        
        x = x.unsqueeze(2)
        y = y.unsqueeze(1)

        SIM = self.sim(x,y)
        
        eye = torch.eye(SIM.shape[1]).to(x.device)
        
        #close = SIM[:, eye.long().type(torch.bool)]
        far = SIM[:, (1-eye).long().type(torch.bool)]
        
        #loss = - torch.log( (close/0.5).exp().sum() / (far/0.5).exp().sum(dim=1) )
        loss = - torch.log( 1 / (far/0.5).exp().sum(dim=1) )

        return loss.mean()

if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    a = torch.randn((6,3,224,224)).to(device)
    b = torch.randn((6,3,224,224)).to(device)
    criterion = PairContrastiveLoss(3,768,16).to(device)
    
    print(criterion(a,b))