import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.modules.utils import _pair


logsoftmax = nn.LogSoftmax(dim=-1)
def crossentropy(x, y):
    B = x.shape[0]
    logprobs = logsoftmax(x)
    loss = - (y.softmax(dim=-1).detach() * logprobs).sum(dim=-1)
    loss = loss.reshape(B,B,-1).mean(dim=-1)
    return loss

class PairContrastiveLoss(nn.Module):
    def __init__(self, channel, hidden, size):
        super().__init__()
        
        self.patch = nn.Conv2d(channel, hidden, kernel_size=size, stride=size)
        self.sim = nn.CosineSimilarity(dim=-1)
        n_patches = (224 // size) * (224 // size)
        #print(n_patches)
        self.position_embeddings = nn.Parameter(torch.zeros(1, n_patches+1, hidden))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden))
        
        self.hidden = hidden
        
    def forward(self, x, y):
        x = self.patch(x)
        y = self.patch(y)
        B, D, H, W = x.shape
        
        #x = x.reshape(-1,H,W).reshape(B*D,H*W).unsqueeze(1)
        #y = y.reshape(-1,H,W).reshape(B*D,H*W).unsqueeze(0)
        
        #x = x.permute(0,2,3,1).unsqueeze(1)
        #y = y.permute(0,2,3,1).unsqueeze(0)
        
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
        
        #CE = crossentropy(x, y)
        SIM = self.sim(x,y)
        
        eye = torch.eye(x.shape[0]).to(x.device)
        
        close = SIM[eye.long().type(torch.bool)]
        far = SIM[(1-eye).long().type(torch.bool)]

        
        #close = CE[eye.long().type(torch.bool)]
        
        #loss = - torch.log (close.sum() / far.exp().sum())
        #loss = close.mean() / far.mean()
        loss = - torch.log( (close/0.5).exp().sum() / (far/0.5).exp().sum() )
        #loss = - torch.log( (close*2).exp().sum() / (far*2).exp().sum() )

        return loss

if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    a = torch.randn((6,3,224,224)).to(device)
    b = torch.randn((6,3,224,224)).to(device)
    criterion = PairContrastiveLoss(3,768,16).to(device)
    
    print(criterion(a,b))
