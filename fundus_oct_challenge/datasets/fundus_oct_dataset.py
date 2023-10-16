import os
import torch
from torch.utils.data import Dataset
from torchvision.io import read_image, ImageReadMode

class FundusOctDataset(Dataset):
    def __init__(self, root_dir='GOALS', mode='train', transforms=None, task="segmentation", separate_bottom_bg=True):
        assert mode in ['train', 'val', 'test'], "Mode should be 'train', 'val', or 'test'"
        self.root_dir = root_dir
        self.mode = mode
        self.task = task
        self.transforms = transforms
        self.separate_bottom_bg = separate_bottom_bg
        self.mapping_dict = {
            0: 1,
            80: 2,
            160: 3,
            255: 0,
        }
        
        self.target_image_size = (800, 1100)
        self.label2id = {
            "background_above": 0,
            "retinal nerve fiber layer": 1,
            "ganglion cell-inner plexiform layer": 2,
            "choroidal layer": 3,
            "background_between": 4,
            "background_below": 5,
        }

        # Create a lookup table, used to replace the png uint8 values with class label values
        self.lookup_table = torch.arange(256)
        for k, v in self.mapping_dict.items():
            self.lookup_table[k] = v
        
        # Determine directories based on mode
        if mode == 'train':
            self.image_dir = os.path.join(root_dir, 'Train', 'Image')
            self.mask_dir = os.path.join(root_dir, 'Train', 'Layer_Masks')
        elif mode == 'val':
            self.image_dir = os.path.join(root_dir, 'Validation', 'Image')
            self.mask_dir = os.path.join(root_dir, 'Validation', 'Layer_Masks')
        elif mode == 'test':
            self.image_dir = os.path.join(root_dir, 'Test', 'Image')
            self.mask_dir = os.path.join(root_dir, 'Test', 'Layer_Masks')
        else:
            raise NotImplementedError(f"No mode: {mode} implemented")
        
        self.image_list = os.listdir(self.image_dir)
        
    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_list[idx])
        mask_path = os.path.join(self.mask_dir, self.image_list[idx])

        img = read_image(img_path, mode=ImageReadMode.GRAY).float() / 255.0
        mask = read_image(mask_path, mode=ImageReadMode.GRAY).squeeze().long()

        mask = self.preprocess_png_mask(mask, seperate_bottom_bg=self.separate_bottom_bg).unsqueeze(dim=0)
        assert torch.max(mask) < 20, f"found class with number higher than 20, something is probably wrong. {torch.max(mask)}"

        if self.transforms:
            img, mask = self.transforms(img, mask)
        return img, mask, self.task, img_path

    def get_label_name(self, label_id):
        return list(self.label2id.keys())[label_id]

    def preprocess_png_mask(self, mask, seperate_bottom_bg=True):
        """
        add additional mask labels based from background class
        """
        # replace values
        mask = self.map_values(mask)

        # change class of background below to a specific second background class
        if seperate_bottom_bg:
            first_line_value = self.label2id["retinal nerve fiber layer"]
            second_line_value = self.label2id["ganglion cell-inner plexiform layer"]
            third_line_value = self.label2id["choroidal layer"]
            above_value = self.label2id["background_above"]
            between_value = self.label2id["background_between"]
            below_value = self.label2id["background_below"]

            # Iterate over columns
            for j in range(mask.shape[1]):
                first_line_indices = torch.where(mask[:, j] == first_line_value)[0]
                second_line_indices = torch.where(mask[:, j] == second_line_value)[0]
                third_line_indices = torch.where(mask[:, j] == third_line_value)[0]
                
                if first_line_indices.numel() > 0 and second_line_indices.numel() > 0 and third_line_indices.numel() > 0:
                    topmost_first_line_index = first_line_indices[0]
                    topmost_second_line_index = second_line_indices[0]
                    bottommost_second_line_index = second_line_indices[-1]
                    topmost_third_line_index = third_line_indices[0]
                    bottommost_third_line_index = third_line_indices[-1]
                    
                    # Set values above the first line to above_value
                    mask[:topmost_first_line_index, j] = above_value
                    
                    # Set values between the second and third line to between_value
                    mask[bottommost_second_line_index + 1:topmost_third_line_index, j] = between_value
                    
                    # Set values below the third line to below_value
                    mask[bottommost_third_line_index + 1:, j] = below_value
        else:
            return mask
        
        return mask
    
    def map_values(self, tensor):
        # Use the tensor values as indices to get the mapped values
        return torch.index_select(self.lookup_table, 0, tensor.reshape(-1)).reshape(tensor.shape)
