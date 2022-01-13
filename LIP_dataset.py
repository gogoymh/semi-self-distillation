import torch
from torch.utils.data import Dataset
from skimage.io import imread
import os
import numpy as np
#import random
from torchvision import transforms

class LIP_image(Dataset):
    def __init__(self, path, input_transform=None):
        super().__init__()

        self.path = path
        self.file = os.listdir(path)
        
        self.input_transform = input_transform
        
        self.normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        
        self.len = len(self.file)
        
    def __getitem__(self, index):
        
        input_img = imread(os.path.join(self.path, self.file[index]))
        if len(input_img.shape) == 2:
            input_img = np.stack((input_img, input_img, input_img), axis=2)

        input_img = self.input_transform(input_img)
        input_img = self.normalize(input_img)
        
        return input_img
    
    def __len__(self):
        return self.len
    
if __name__ == "__main__":
    path = "C://Users//Minhyeong//Downloads//TrainVal_images//TrainVal_images//train_images//"
    
    from torchvision import transforms as tf
    
    transform_valid = tf.Compose([
        tf.ToTensor(),
        tf.ToPILImage(),
        tf.RandomCrop((256,256), pad_if_needed=True),
        tf.RandomHorizontalFlip(),
        tf.ToTensor()
     ])
    
    a = LIP_image(path, transform_valid)
    
    import matplotlib.pyplot as plt
    
    index = np.random.choice(a.len, 1)[0]
    b = a.__getitem__(index)

    b = b.permute(1,2,0).numpy() * 0.5 + 0.5
    
    plt.imshow(b)
    #plt.savefig("/DATA/ymh_ksw/script/input2.png")
    plt.show()
    plt.close()