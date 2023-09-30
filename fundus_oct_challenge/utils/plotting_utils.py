import matplotlib.pyplot as plt
import torch

colormap_6class = torch.Tensor([
    [47,79,79],
    [0,250,154],
    [255,127,80],
    [255,0,0],
    [34,139,34],
    [0,191,255],
])


def apply_colormap(segmentation, swap_to_cf=True):
    # move segmentation to device of colormap (should be to cpu)
    segmentation = segmentation.to(colormap_6class.device)
    colored_seg = colormap_6class[segmentation]
    
    # swap to channel first
    if swap_to_cf:
        colored_seg = torch.permute(colored_seg, (2,0,1))

    return colored_seg.byte()