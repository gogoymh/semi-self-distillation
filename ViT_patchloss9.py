import torch
import torch.nn as nn

class PairContrastiveLoss(nn.Module):
    def __init__(self, size=2):
        super().__init__()
        
        self.pool = nn.MaxPool2d(size)
        self.upsample = nn.Upsample(scale_factor=size, mode='nearest')
        
    def forward(self, x):
        x = (x >= 0.5).float()
        x = self.pool(x)
        x = self.upsample(x)
        
        return x

if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    a = torch.rand((1,1,8,8)).to(device)
    criterion = PairContrastiveLoss(2).to(device)

    print(a)
    print(criterion(a))








