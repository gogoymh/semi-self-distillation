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
        
    def forward(self, x, y):
        B = x.shape[0]
        x = self.patch(x)
        y = self.patch(y)
        x = x.permute(0,2,3,1).unsqueeze(1)
        y = y.permute(0,2,3,1).unsqueeze(0)
        
        CE = crossentropy(x, y)
        
        eye = torch.eye(B).to(x.device)
        
        #close = CE * eye
        #far = CE * (1-eye)
        
        close = CE[eye.long().type(torch.bool)]
        far = CE[(1-eye).long().type(torch.bool)]

        loss = close.mean() / far.mean()
        #loss = close.sum() / far.sum()

        return loss
    
if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    a = torch.randn((4,6,224,224)).to(device)
    b = torch.randn((4,6,224,224)).to(device)
    criterion = PairContrastiveLoss(6,128,16).to(device)
    
    print(criterion(a,b))


