import torch
from mlbnb.types import Split
from torch.utils.data import Dataset, random_split
from torchvision import datasets, transforms

from config.config import Paths


class MnistDataset(Dataset):
    def __init__(
        self,
        paths: Paths,
        val_fraction: float,
        split: Split,
        generator: torch.Generator,
    ):
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )

        if split == Split.TEST:
            self.dataset = datasets.MNIST(
                root=paths.data,
                train=False,
                download=True,
                transform=transform,
            )
        else:
            dataset = datasets.MNIST(
                root=paths.data,
                train=True,
                download=True,
                transform=transform,
            )

            n_val = int(val_fraction * len(dataset))
            train, val = random_split(
                dataset, [len(dataset) - n_val, n_val], generator=generator
            )
            self.dataset = train if split == Split.TRAIN else val

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]
