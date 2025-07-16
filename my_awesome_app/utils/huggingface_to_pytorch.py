import numpy as np
from PIL import Image
from torch.utils.data import Dataset

import numpy as np
from PIL import Image
from torch.utils.data import Dataset


class HFDatasetToTorch(Dataset):
    def __init__(self, hf_dataset, transform=None):
        self.hf_dataset = hf_dataset
        self.transform = transform

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, idx):
        example = self.hf_dataset[idx]
        image = example.get("image") or example.get("img")
        if image is None:
            raise KeyError("Neither 'image' nor 'img' found in dataset sample.")
        label = example["label"]
        if isinstance(image, Image.Image):
            image = image.convert("RGB")
        image = np.array(image)
        if self.transform:
            image = self.transform(Image.fromarray(image))  # transform expects PIL
        return image, label
