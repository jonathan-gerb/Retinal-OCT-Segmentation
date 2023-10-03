from fnmatch import translate
import torchvision
# v2 transforms are faster and in beta (but they're fine), this removes the warning for using them.
torchvision.disable_beta_transforms_warning()
import torchvision.transforms.v2 as T
from torchvision.transforms.v2 import functional as F
import torch

def get_transforms(mode='train'):
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

    class RandomRotationJoint(object):
        def __init__(self, degrees=10):
            self.degrees = (-degrees, degrees)

        def __call__(self, img, mask):
            angle = self.degrees[0] + (self.degrees[1] - self.degrees[0]) * torch.rand(1)
            return F.rotate(img, angle, fill=float(img.min())), F.rotate(mask, angle, fill=0)

    class VerticalShiftJoint(object):
        def __init__(self, translate=(0, 60)):
            self.translate = translate

        def __call__(self, img, mask):
            shift = self.translate[0] + (self.translate[1] - self.translate[0]) * torch.rand(1)
            return (
                F.affine(img, angle=0, translate=(0, -shift), scale=1., shear=0, fill=0), 
                F.affine(mask, angle=0, translate=(0, -shift), scale=1., shear=0, fill=0)
            )

        def __str__(self):
            return "VerticalShift"
    # Joint augmentations
    joint_transforms_train = [
        RandomHorizontalFlipJoint(p=0.5),
        RandomRotationJoint(), 
        VerticalShift(),
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

# Usage
train_transforms = get_transforms(mode='train')
val_transforms = get_transforms(mode='val')
test_transforms = get_transforms(mode='test')
