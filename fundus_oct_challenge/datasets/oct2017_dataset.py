import os
import torch
from torch.utils.data import Dataset
from torchvision.io import read_image, ImageReadMode
import torchvision.transforms.functional as fn

class OCT2017(Dataset):
    def __init__(self, root_dir='OCT2017', mode='train', transforms=None):
        assert mode in ['train', 'val', 'test'], "Mode should be 'train', 'val', or 'test'"
        
        self.root_dir = root_dir
        self.mode = mode
        self.transforms = transforms

        self.class_names = os.listdir(root_dir)
        self.n_classes = len(self.class_names)

        self.image_dir = os.path.join(root_dir, mode)

        self.target_image_size = (800, 1100) # how to expand image to target size? reduce target size?

        if mode == 'test':
            self.image_list = []
            for class_name in self.class_names:
                self.image_list.extend([os.path.join(folder_path, class_name, img) for img in os.listdir(os.path.join(root_dir, class_name))])
        elif mode == 'train':
            self.image_list = pd.read_csv(os.path.join(root_dir, 'train.csv'), index_col=False)['image_name']
        elif mode == 'val':
            self.image_list = pd.read_csv(os.path.join(root_dir, 'val.csv'), index_col=False)['image_name']

    def __getitem__(self):
        img_path = os.path.join(self.image_dir, self.image_list[idx])
        img = read_image(img_path, mode=ImageReadMode.GRAY).float() / 255.0
        
        if self.target_image_size[0] <= img.shape[0] and self.target_image_size[1] <= img.shape[1]:
            crop = fn.center_crop(img, output_size=self.target_image_size)
        else:
            resize = fn.resize(img, size=self.target_image_size)

        if self.transforms:
            img, _ = self.transforms(img, img)

        return img


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import numpy as np

    mode = 'test'
    root_dir = './data/OCT2017/'
    folder_path = os.path.join(root_dir, mode)
    img_path = './data/OCT2017/train/CNV/CNV-163081-232.jpeg'

    img = read_image(img_path, mode=ImageReadMode.GRAY)
    print(img.shape)
    img = fn.resize(img, size=[800, 1100])
    print(img.shape)

    plt.figure(figsize=(15, 10))
    plt.imshow(img.moveaxis(0, -1))
    plt.axis("off")
    plt.show()
