from .fundus_oct_dataset import FundusOctDataset
from .oct2017_dataset import OCT2017


def get_dataset(
        dataset='fundus',
        root_dir='GOALS', 
        mode='train', 
        transforms=None, 
        task="segmentation",
        batch_size=None
    ):
    """
    Choosing dataset based on dataset name.

    Args:
        dataset (str, optional): dataset name. Defaults to 'fundus'.
        root_dir (str, optional): root directory for the dataset. Defaults to 'GOALS'.
        mode (str, optional): in which mode dataset will be used. Defaults to 'train'.
        transforms (_type_, optional): transformation to apply. Defaults to None.
        tasks (list, optional): tasks dataset will be used for. Defaults to ["segmentation","reconstruction"].
        task_frequency (list, optional): frequency of each ask. Defaults to [0.5, 0.5].
        batch_size (_type_, optional): . Defaults to None.

    Returns:
        Required dataset instance.
    """

    if dataset == 'fundus':
        return FundusOctDataset(root_dir, mode, transforms, task)
    elif dataset == 'oct2017':
        return OCT2017(root_dir, mode, transforms, task)
    else:
        raise NotImplementedError(f"Dataset '{dataset}' not yet implemented")
