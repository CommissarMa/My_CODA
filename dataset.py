#%%
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import os
from PIL import Image
import pandas as pd
import numpy as np
import torchvision.transforms as standard_transforms

import utils.transforms as own_transforms


class HeadCountDataset(Dataset):
    def __init__(self, root, mode, main_transform=None, img_transform=None, gt_transform=None):
        self.img_path = os.path.join(root, 'img')
        self.gt_path = os.path.join(root, 'den')
        self.data_files = [filename for filename in os.listdir(self.img_path) \
                           if os.path.isfile(os.path.join(self.img_path,filename))]
        self.num_samples = len(self.data_files) 
        self.main_transform=main_transform  
        self.img_transform = img_transform
        self.gt_transform = gt_transform

    def __getitem__(self, index):
        fname = self.data_files[index]
        img, den = self.read_image_and_gt(fname)      
        if self.main_transform is not None:
            img, den = self.main_transform(img,den) 
        if self.img_transform is not None:
            img = self.img_transform(img)         
        if self.gt_transform is not None:
            den = self.gt_transform(den)               
        return img, den

    def __len__(self):
        return self.num_samples

    def read_image_and_gt(self,fname):
        img = Image.open(os.path.join(self.img_path,fname))
        if img.mode == 'L':
            img = img.convert('RGB')

        # den = sio.loadmat(os.path.join(self.gt_path,os.path.splitext(fname)[0] + '.mat'))
        # den = den['map']
        den = pd.read_csv(os.path.join(self.gt_path,os.path.splitext(fname)[0] + '.csv'), sep=',',header=None).values
        
        den = den.astype(np.float32, copy=False)    
        den = Image.fromarray(den)  
        return img, den


def loading_data(root, mode, batch_size=1):
    mean_std = ([0.5, 0.5, 0.5],[0.25, 0.25, 0.25])
    log_para = 1
    if mode == 'train':
        main_transform = own_transforms.Compose([
            own_transforms.RandomHorizontallyFlip()
        ])
    else:
        main_transform = None
    
    img_transform = standard_transforms.Compose([
        standard_transforms.ToTensor(),
        standard_transforms.Normalize(*mean_std)
    ])

    gt_transform = standard_transforms.Compose([
        own_transforms.LabelNormalize(log_para)
    ])

    restore_transform = standard_transforms.Compose([
        own_transforms.DeNormalize(*mean_std),
        standard_transforms.ToPILImage()
    ])

    dataset = HeadCountDataset(root, mode, main_transform, img_transform, gt_transform)

    if mode == 'train':
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    else:
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    return dataloader, restore_transform


if __name__ == '__main__':
    dataloader, re_trans = loading_data('./datasets/gcc2shhb/gcc','train',1)
    import matplotlib.pyplot as plt
    for i, (img,den) in enumerate(dataloader):
        print(img.shape)
        print(den.shape)
        # plt.imshow(img)
        # plt.figure()
        # den = np.array(den)
        # plt.imshow(den)
        # print(den.mode)
        break


#%%
