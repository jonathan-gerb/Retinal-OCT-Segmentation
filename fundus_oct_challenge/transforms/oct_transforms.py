from fnmatch import translate
import torchvision
# v2 transforms are faster and in beta (but they're fine), this removes the warning for using them.
torchvision.disable_beta_transforms_warning()
import torchvision.transforms.v2 as T
from torchvision.transforms.v2 import functional as F
import torch


def get_transforms(mode='train'):
    """
    Function to get transformation for image instance.

    Usage:
        train_transforms = get_transforms(mode='train')
        val_transforms = get_transforms(mode='val')
        test_transforms = get_transforms(mode='test')
    """

    class ComposeJoint(object):
        def __init__(self, transforms=None):
            self.transforms = transforms or []

        def __call__(self, img, mask):
            for transform in self.transforms:
                if isinstance(transform, tuple):
                    # Apply transformations only to image
                    img = transform[0](img)
                else:
                    # Joint transformations
                    img, mask = transform(img, mask)
            return img, mask

    class RandomHorizontalFlipJoint(object):
        def __init__(self, p=0.5):
            self.p = p

        def __call__(self, img, mask):
            if torch.rand(1) < self.p:
                return F.hflip(img), F.hflip(mask)
            return img, mask

        def __str__(self):
            return "RandomHorizontalFlip"

    class RandomRotationJoint(object):
        def __init__(self, degrees=10):
            self.degrees = (-degrees, degrees)

        def __call__(self, img, mask):
            angle = self.degrees[0] + (self.degrees[1] - self.degrees[0]) * torch.rand(1)
            return F.rotate(img, angle, fill=float(img.min())), F.rotate(mask, angle, fill=0)
        
        def __str__(self):
            return "RandomRotation"
   
    class VerticalShiftJoint(object):
        def __init__(self, translate=(0, 60)):
            self.translate = translate

        def __call__(self, img, mask):
         # Idea for vertical dynamic shift based on the mask: This should replace the default (0,60) translate. code is a bit long though and I need to refer to mask
         #   mask_upper_limit =  torch.nonzero(torch.tensor(mask[:, 0]) == 1)[0].item()
         #   mask_lower_limit = torch.nonzero(torch.tensor(mask[:, 0])) == 3).squeeze(-1)[-1]).item()
         #   for column in range(mask.shape[1])[1:]:
        #        upper_mask_edge = torch.nonzero(torch.tensor(mask[:, column])) == 1)[0].item()
        #        lower_mask_edge = torch.nonzero(torch.tensor(mask[:, column])) == 3).squeeze(-1)[-1]).item()
        #        if upper_mask_edge < mask_upper_limit: mask_upper_limit = upper_mask_edge
        #        if lower_mask_edge >  mask_lower_limit: mask_lower_limit = lower_mask_edge
        #    shift = random.randrange(-upper_mask_edge,(mask.shape[1][-1] - mask_lower_limit)) 
        #    transformation_matrix = torch.tensor([[1, 0, 0],
         #                             [0, 1, shift]])
        #   return (F.affine(img, matrix=transformation_matrix, fill=0), 
        #          F.affine(mask, matrix=transformation_matrix,  fill=255)))

            shift = self.translate[0] + (self.translate[1] - self.translate[0]) * torch.rand(1)
            return (
                F.affine(img, angle=0, translate=(0, -shift), scale=1., shear=0, fill=0), 
                F.affine(mask, angle=0, translate=(0, -shift), scale=1., shear=0, fill=255)
            )

        def __str__(self):
            return "VerticalShift"

    class RandomScaleJoint(object):
        def __init__(self, min_scale=1.0, max_scale=1.2):
            self.min_scale = min_scale
            self.max_scale = max_scale

        def __call__(self, img, mask):
            scale = self.min_scale + (self.max_scale - self.min_scale) * torch.rand(1)
            return (
                F.affine(img, angle=0, translate=(0, 0), scale=scale, shear=0), 
                F.affine(mask, angle=0, translate=(0, 0), scale=scale, shear=0)
            )
        
        def __str__(self):
            return "RandomScale"

    # Joint augmentations
    joint_transforms_train = [
        RandomHorizontalFlipJoint(p=0.5),
        RandomRotationJoint(), 
        VerticalShiftJoint(),
        RandomScaleJoint()
        # ... Add any other joint transformations for training mode
    ]

    joint_transforms_val = [
        # ... Add any other joint transformations for validation/test mode (usually less augmentations)
    ]

    # Image-only augmentations
    image_transforms_train = [
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        T.RandomApply([T.GaussianBlur(3, sigma=(0.1, 2.0))], p=0.5),
        # ... Add any other image-only transformations for training mode
        (T.Normalize(mean=[0.485], std=[0.229]),)
    ]

    image_transforms_val = [
        # ... Add any other image-only transformations for validation/test mode
        (T.Normalize(mean=[0.485], std=[0.229]),)
    ]

    # Compose based on mode
    if mode == 'train':
        all_transforms = joint_transforms_train + image_transforms_train
    else:
        all_transforms = joint_transforms_val + image_transforms_val

    composed_transforms = ComposeJoint(all_transforms)

    return composed_transforms

# Code for visual testing of the transformations
#
# if __name__ == "__main__":
#     from torchvision.io import read_image
#     from pathlib import Path
#     import numpy as np
#     import os
#     import matplotlib.pyplot as plt

#     image_path_train = Path("data/GOALS/Train/Image")
#     mask_path_train = Path("data/GOALS/Train/Layer_Masks")

#     train_imgs = np.random.choice(list(image_path_train.iterdir()), size=1)
#     img = read_image(str(train_imgs[0]))
#     mask = read_image(str(mask_path_train / Path(train_imgs[0].name)))

#     transformations = [
#             RandomHorizontalFlipJoint(p=1),
#             RandomRotationJoint(), 
#             VerticalShiftJoint(),
#             RandomScaleJoint()
#         ]

#     n_columns = 5
#     n_rows = int((len(transformations) + 1) / n_columns)
#     height = max(3, n_rows * 3)
#     width = 15
    
#     plt.figure(figsize=(width, height))
#     plt.subplot(n_rows, n_columns, 1)
#     plt.imshow(img.permute(1, 2, 0).numpy())
#     plt.title("Original image")

#     for i, transform in enumerate(transformations):
#         transformed_img, _ =  transform(img, mask)

#         plt.subplot(n_rows, n_columns, i + 2)
#         plt.imshow(transformed_img.permute(1, 2, 0).numpy())
#         plt.title(str(transform))

#     plt.show()
