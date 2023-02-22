import logging

import pytorch_lightning as pl
from numpy import load
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms

log = logging.getLogger(__name__)


class MnistDataset(Dataset):
    """
    MNIST fashion dataset based on numpy arrays.

    Attributes
    ----------
    images: np.ndarray
        the dataset images
    transform: torchvision.transforms
        the image transform
    labels: np.ndarray
        the dataset labels
    ineference : bool
        indicates if there is are labels for the current dataset.
    """

    def __init__(self, images, labels=None, transform=None, stage=None):
        """Initializes images, labels and inference flag."""
        self.images = images
        self.transform = transform
        self.stage = stage
        if self.stage != "predict":
            self.labels = labels

    def __len__(self):
        """Gets the size of the dataset."""
        return len(self.labels)

    def __getitem__(self, idx):
        """Gets the image with the specified id."""
        images = self.images[idx]
        if self.transform:
            images = self.transform(images).float()
        if self.stage != "predict":
            labels = self.labels[idx]
            return images, labels
        else:
            return images


class MnistDataModule(pl.LightningDataModule):
    """
    Loads the saved tensors in data/processed, normalizes them and returns the DataLoaders
    through its methods.
    """

    def __init__(self, data_dir: str = "data/processed", batch_size: int = 32):
        super().__init__()
        log.info(f"Loading data from: {data_dir}")
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(mean=[0.485], std=[0.229])]
        )

    def prepare_data(self):
        return

    def setup(self, stage: str):
        if stage == "fit":
            train_images = load(self.data_dir + "/train_images.npy")
            train_labels = load(self.data_dir + "/train_labels.npy")
            dataset_full = MnistDataset(
                train_images, train_labels, transform=self.transform
            )
            train_size = int(len(dataset_full) * 0.9)
            val_size = len(dataset_full) - train_size
            self.train_dataset, self.val_dataset = random_split(
                dataset_full, [train_size, val_size]
            )

        if stage == "test":
            test_images = load(self.data_dir + "/test_images.npy")
            test_labels = load(self.data_dir + "/test_labels.npy")
            self.test_dataset = MnistDataset(
                test_images, test_labels, transform=self.transform
            )

        if stage == "predict":
            self.predict_dataset = MnistDataset(
                test_images, test_labels, transform=self.transform, stage=stage
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=16
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=16
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=16
        )

    def predict_dataloader(self):
        return DataLoader(
            self.predict_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=16,
        )

    def teardown(self, stage: str):
        # Used to clean-up when the run is finished
        return
