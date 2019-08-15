#%%
import matplotlib.pyplot as plt
import matplotlib.cm as CM
import os
from PIL import Image
import pandas as pd
import numpy as np
import torchvision.transforms as standard_transforms
import torch

from model import CSRNet
from utils import loadModel


if __name__ == '__main__':
    img_path = './datasets/gcc2shhb/shhb/img/1.jpg'
    den_path = './datasets/gcc2shhb/shhb/den/1.csv'
    pretrained_model_path = './pretrained/39.pth'

    img = Image.open(img_path)
    if img.mode == 'L':
        img = img.convert('RGB')
    den = pd.read_csv(den_path).values
    den = den.astype(np.float32, copy=False)

    plt.imshow(img)
    plt.figure()
    plt.imshow(den,cmap=CM.jet)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    mean_std = ([0.5, 0.5, 0.5],[0.25, 0.25, 0.25])
    img_transform = standard_transforms.Compose([
        standard_transforms.ToTensor(),
        standard_transforms.Normalize(*mean_std)
    ])
    img = img_transform(img)
    img = img.unsqueeze(dim=0)
    img = img.to(device)

    net = CSRNet().to(device)
    net = loadModel(net, pretrained_model_path)

    predDmap = net(img).detach().squeeze()
    plt.figure()
    plt.imshow(predDmap,cmap=CM.jet)
    

#%%
