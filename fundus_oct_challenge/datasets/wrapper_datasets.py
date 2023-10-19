from torch.utils.data import Dataset

class OversampledDataset(Dataset):
    def __init__(self, base_dataset, oversampling_factor=1):
        self.base_dataset = base_dataset
        self.oversampling_factor = oversampling_factor

    def __len__(self):
        return len(self.base_dataset) * self.oversampling_factor

    def __getitem__(self, idx):
        # Wrap around the base dataset to get items
        return self.base_dataset[idx % len(self.base_dataset)]
