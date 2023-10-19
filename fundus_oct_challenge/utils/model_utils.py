from torch.utils.data import Subset

def normalize(weights):
    total = sum(weights)
    if total == 0:  # avoid division by zero
        return [0] * len(weights)
    return [w / total for w in weights]


def get_limited_dataset(original_dataset, k):
    # Take the first k samples from the original dataset
    indices = list(range(k))
    limited_dataset = Subset(original_dataset, indices)
    return limited_dataset