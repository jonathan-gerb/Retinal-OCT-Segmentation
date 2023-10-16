import os
import glob
import torch
import pandas as pd
import numpy as np
import re
import cv2
from scipy import ndimage
from pathlib import Path
from torch.utils.data import Dataset
import torchvision.transforms.functional as fn
from torchvision.io import read_image, ImageReadMode

class CombinedOCTDataset(Dataset):
    def __init__(self, root_dir='retinal_oct_dataset_collection', mode='train', transforms=None, task="reconstruction", datasets=["kermany2018"], img_size=(800, 1104), max_ds_size=-1):
        assert mode in ['train', 'val', 'test'], "Mode should be 'train', 'val', or 'test'"
        self.root_dir = root_dir
        self.mode = mode
        assert task == "reconstruction" or task == "classification", "CombinedOCTDataset only supports reconstruction or classification"
        self.task = task
        self.img_size = img_size
        self.max_ds_size = max_ds_size
        self.datasets = datasets
        self.available_datasets = {}
        self.available_datasets['classification'] = ["GOALS", "kermany2018", "neh_ut_2021", "OCTID"]
        self.available_datasets['reconstruction'] = ["GOALS", "kermany2018", "neh_ut_2021", "2015_BOE_CHIU", "OCTID"]

        for dataset in self.datasets:
            assert dataset in self.available_datasets[task], f"{dataset} not available for task: {task}"

        self.label2classidx = {
            "NORMAL": 0,
            "GLAUCOMA": 1,
            'CNV': 2, 
            'DME': 3, 
            "DRUSEN": 4 
        }

        self.transforms = transforms
        
        self.dataset_paths = {}
        # fills dataset_paths
        self.load_datasets()
        self.build_index()


    def build_index(self):
        self.all_images = []
        self.all_labels = []
        for key in self.dataset_paths.keys():
            for img_path, label in zip(self.dataset_paths[key]['images'], self.dataset_paths[key]['labels']):
                self.all_images.append(img_path)
                self.all_labels.append(label)


    def load_datasets(self):
        for dataset in self.datasets:
            if dataset.lower() == "goals":
                self.load_goals()
            if dataset.lower() == "kermany2018":
                self.load_kermany2018()
            if dataset.lower() == "neh_ut_2021":
                self.load_neh_ut2021()
            if dataset.lower() == "2015_boe_chiu":
                self.load_chiu2015()
            if dataset.lower() == "octid":
                self.load_octid()
        
    def load_goals(self):
            # Determine directories based on mode
        if self.mode == 'train':
            image_dir = os.path.join(self.root_dir, 'GOALS', 'Train', 'Image')
            # mask_dir = os.path.join(self.root_dir, 'GOALS', 'Train', 'Layer_Masks')
            excel_path = os.path.join(self.root_dir, 'GOALS', 'Train', 'Train_GC_GT.xlsx')
        elif self.mode == 'val':
            image_dir = os.path.join(self.root_dir, 'GOALS', 'Validation', 'Image')
            # mask_dir = os.path.join(self.root_dir, 'GOALS',' Validation', 'Layer_Masks')
            excel_path = os.path.join(self.root_dir, 'GOALS', 'Validation', 'Val_GC_GT.xlsx')
        elif self.mode == 'test':
            image_dir = os.path.join(self.root_dir, 'GOALS', 'Test', 'Image')
            # mask_dir = os.path.join(self.root_dir, 'GOALS', 'Test', 'Layer_Masks')
            excel_path = os.path.join(self.root_dir, 'GOALS', 'Test', 'Test_GC_GT.xlsx')
        else:
            raise NotImplementedError(f"No mode: {self.mode} implemented")
        
        image_list = glob.glob(f"{image_dir}/*.png")
        # mask_list = glob.glob(f"{mask_dir}/*.png")
        
        # Sort the lists
        image_list.sort()
        # mask_list.sort()

        df = pd.read_excel(excel_path)
        image_label_dict = dict(zip(df['ImgName'], df['GC_Label']))
        ids = [int(Path(filename).stem) for filename in image_list]
        labels = [image_label_dict.get(id, None) for id in ids]

        self.dataset_paths["goals"] = {}
        self.dataset_paths["goals"]['images'] = image_list
        # self.dataset_paths["goals"]['masks'] = mask_list
        self.dataset_paths["goals"]['labels'] = labels
        self.dataset_paths["goals"]['labelsnames'] = {"NORMAL": 0, "GLAUCOMA": 1}
        assert len(image_list) == len(labels)

    def load_kermany2018(self):
        if self.mode == 'train':
            image_dir = os.path.join(self.root_dir, 'KERMANY2018', 'training', 'train')
        elif self.mode == 'val':
            image_dir = os.path.join(self.root_dir, 'KERMANY2018', 'training', 'val')
        elif self.mode == 'test':
            image_dir = os.path.join(self.root_dir, 'KERMANY2018', 'test')

        image_list = []
        labels = []
        label2classidx = {
            "NORMAL": 0,
            'CNV': 2, 
            'DME': 3, 
            "DRUSEN": 4 
        }

        for classname in ['NORMAL', 'CNV', 'DME', "DRUSEN"]:
            class_images = list((Path(image_dir) / classname).glob("*.jpeg"))
            class_images = [str(pathname) for pathname in class_images]
            labels += [label2classidx[classname]] * len(class_images)
            image_list += class_images

        self.dataset_paths["kermany2018"] = {}
        self.dataset_paths["kermany2018"]['images'] = image_list
        self.dataset_paths["kermany2018"]['labels'] = labels
        self.dataset_paths["kermany2018"]['labelsnames'] = label2classidx
        assert len(image_list) == len(labels), f"{len(image_list)}, {len(labels)}"

    def load_neh_ut2021(self):
        image_dir = os.path.join(self.root_dir, 'NEH_UT_2021', 'NEH_UT_2021RetinalOCTDataset')
        csv_path = os.path.join(self.root_dir, 'NEH_UT_2021', 'data_information.csv')
        
        labels = []
        label2classidx = {
            "NORMAL": 0,
            'CNV': 2, 
            "DRUSEN": 4 
        }

        # i'll just manually split the data into train test and val based on a ratio of
        # 0.5, 0.25, 0.25, this isnt entirely correct as normal has fewer patients but it's ok
        indices_to_keep = []
        if self.mode == 'train':
            indices_to_keep = list(range(1, 81))
        elif self.mode == 'val':
            indices_to_keep = list(range(81, 121))
        elif self.mode == 'test':
            indices_to_keep = list(range(121, 160))

        df = pd.read_csv(csv_path)
        
        image_list = []
        labels = []
        
        for patient_class in np.unique(df['Class']):
            df_classwise = df[df['Class'] == patient_class]
            
            for patient_index in np.unique(df_classwise['Patient ID']):
                if patient_index not in indices_to_keep:
                    continue
                df_patientwise = df_classwise[df_classwise['Patient ID'] == patient_index]
                
                for i in range(len(df_patientwise)):
                    patient_path = Path(df_patientwise.iloc[i]['Directory'])
                    image_filepath = str(image_dir / patient_path)
                    if df_patientwise.iloc[i]['Label'].lower() == 'normal':
                        labels.append(0)
                    elif df_patientwise.iloc[i]['Label'].lower() == 'drusen':
                        labels.append(4)
                    elif df_patientwise.iloc[i]['Label'].lower() == 'cnv':
                        labels.append(2)
                        
                    image_list.append(image_filepath)

        self.dataset_paths["neh_ut2021"] = {}
        self.dataset_paths["neh_ut2021"]['images'] = image_list
        self.dataset_paths["neh_ut2021"]['labels'] = labels
        self.dataset_paths["neh_ut2021"]['labelsnames'] = label2classidx
        assert len(image_list) == len(labels)


    def load_chiu2015(self):
        image_dir = os.path.join(self.root_dir, '2015_BOE_CHIU')

        patient_to_keep = []
        if self.mode == 'train':
            patient_to_keep = list(range(1, 6))
        elif self.mode == 'val':
            patient_to_keep = list(range(6, 9))
        elif self.mode == 'test':
            patient_to_keep = list(range(9, 11))

        image_list = []
        for patient in patient_to_keep:
            patient_images = list((Path(image_dir) / f"Subject_{patient:02}").glob("*.png"))
            patient_images = [str(pathname) for pathname in patient_images]
            image_list += patient_images

        image_list.sort()
        self.dataset_paths["chiu2015"] = {}
        self.dataset_paths["chiu2015"]['images'] = image_list

        labels = [0] * len(image_list)
        self.dataset_paths["chiu2015"]['labels'] = labels
        assert len(image_list) == len(labels)

    def load_octid(self):
        image_dir = os.path.join(self.root_dir, 'OCTID')

        index_to_keep = []
        if self.mode == 'train':
            index_to_keep = list(range(0, 101))
        elif self.mode == 'val':
            index_to_keep = list(range(101, 151))
        elif self.mode == 'test':
            index_to_keep = list(range(151, 207))

        image_list = list(Path(image_dir).glob("*.jpeg"))
        image_list = [str(pathname) for pathname in image_list]
        # only keep the right range of images
        image_list = [filepath for filepath in image_list if int(re.sub(r'[^0-9]', '', filepath)) in index_to_keep]
        image_list.sort()

        self.dataset_paths["octid"] = {}
        self.dataset_paths["octid"]['images'] = image_list
        labels = [0] * len(image_list)
        self.dataset_paths["octid"]['labels'] = labels
        assert len(image_list) == len(labels)

    def replace_white_sides(self, img):
        # replace white sides of rotated images
        threshold = 253
        binary_mask = (img >= threshold).astype(np.int32)
        labeled, num_features = ndimage.label(binary_mask)
        for i in range(1, num_features + 1):
            blob_size = (labeled == i).sum()
            if blob_size > 10:  # adjust as needed
                img[labeled == i] = 0  # replacing with black for this example
        return img

    def __len__(self):
        # shorten dataset artifically to get faster epochs
        if self.max_ds_size == -1:
            return len(self.all_images)
        else:
            return self.max_ds_size

    def __getitem__(self, idx):
        img_path = self.all_images[idx]
        label = self.all_labels[idx]
        if ".tif" in img_path:
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img.ndim == 2:
                img = img[np.newaxis, :, :]
        else:
            img = read_image(img_path, mode=ImageReadMode.GRAY).numpy()

        # replace white sides of rotated images, takes numpy input
        img = self.replace_white_sides(img)

        # Convert back to tensor and normalize
        img = torch.from_numpy(img).float() / 255.0

        if img is None or img.nelement() == 0:
            print('img was none or empty, sampling index 0 instead')
            return self[0]

        img = fn.resize(img, size=self.img_size, antialias=True)
        
        if self.transforms:
            try:
                img, _ = self.transforms(img, img)
            except:
                print(img_path, label, img.shape)
                return

        if self.task == 'reconstruction':
            return img, img, self.task
        if self.task == 'classification':
            return img, label, self.task

    def get_label_name(self, label_id):
        return list(self.label2classidx.keys())[label_id]

