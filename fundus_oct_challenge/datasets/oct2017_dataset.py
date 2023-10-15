import os
import torch
from torch.utils.data import Dataset
from torchvision.io import read_image, ImageReadMode
import torchvision.transforms.functional as fn

import pandas as pd

from fundus_oct_challenge.utils.model_utils import normalize


class OCT2017(Dataset):
    def __init__(self, root_dir='OCT2017', mode='train', transforms=None, task="classification", task_frequency=[1]):
        assert mode in ['train', 'val', 'test'], "Mode should be 'train', 'val', or 'test'"
        
        self.root_dir = root_dir
        self.mode = mode
        self.transforms = transforms

        self.task = task
        
        self.target_image_size = (800, 1100) # how to expand image to target size? reduce target size?

        if mode == 'test':
            self.image_list = []
            self.image_dir = os.path.join(root_dir, 'test')
            self.class_names = os.listdir(os.path.join(root_dir, 'test'))

            for class_name in self.class_names:
                self.image_list.extend([os.path.join(folder_path, class_name, img) for img in os.listdir(os.path.join(root_dir, class_name))])
        elif mode == 'train':
            self.image_dir = os.path.join(root_dir, 'train')
            self.class_names = os.listdir(os.path.join(root_dir, 'train'))
            self.image_list = pd.read_csv(os.path.join(root_dir, 'train.csv'), index_col=False)['image_name']

        elif mode == 'val':
            self.image_dir = os.path.join(root_dir, 'train')
            self.class_names = os.listdir(os.path.join(root_dir, 'train'))
            self.image_list = pd.read_csv(os.path.join(root_dir, 'val.csv'), index_col=False)['image_name']
        
        self.n_classes = len(self.class_names)
        self.class_mapping = dict(zip(self.class_names, range(self.n_classes)))

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        label = self.image_list[idx].split("-")[0]
        img_path = os.path.join(self.image_dir, label, self.image_list[idx])

        img = read_image(img_path, mode=ImageReadMode.GRAY).float() / 255.0
        
        if self.target_image_size[0] <= img.shape[0] and self.target_image_size[1] <= img.shape[1]:
            img = fn.center_crop(img, output_size=self.target_image_size, antialias=False)
        else:
            img = fn.resize(img, size=self.target_image_size, antialias=False)

        if self.transforms:
            img, _ = self.transforms(img, img)

        return img, self.class_mapping[label], self.task
        
    def get_label_name(self, label_id):
        return list(self.class_mapping.keys())[label_id]


if __name__ == '__main__':
    pass
