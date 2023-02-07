import os
from typing import Tuple

import torch
from torch.utils.data import DataLoader, Dataset


class MnistDataset(Dataset):
    """
    Dataloader for the MNIST dataset based on numpy arrays.
    """

    def __init__(self, images, labels=None, inference=False):
        self.inference = inference
        self.images = torch.from_numpy(images).float()
        if not inference:
            self.labels = torch.from_numpy(labels).long()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if not self.inference:
            return self.images[idx], self.labels[idx]
        else:
            return self.images[idx]


def get_latest_version(model_dir: str = "models") -> int:
    """
    Gets the latest version of the model.
    If no previous version exists, it returns -1.

    Parameters
    ----------
    model_dir : str, optional
        directory with the pretrained, saved models

    Returns
    -------
    version : int
        version of the trained model (-1 means that no pretrained model exists)
    """

    version = -1
    for folder in os.listdir(model_dir):
        if folder.startswith("v"):
            curr_version = int(folder[1:])
            version = max(version, curr_version)

    return version


def save_latest_model(model: torch.nn.Module, model_dir="models") -> None:
    """
    Saves the latest model in the "latest" fubfolder in the "models" folder

    Parameters
    ----------
    model : torch.nn.Module
        the pretrained network
    model_dir : str, optional
        directory with the pretrained, saved models

    Returns
    -------
    None
    """
    version = get_latest_version()
    if version > -1:
        latest_model_path = os.path.join(model_dir, "latest")
        if not os.path.exists(latest_model_path):
            os.makedirs(latest_model_path)
        print(f"Saving LATEST model in directory: {latest_model_path}/model.pth")
        torch.save(model.state_dict(), latest_model_path + "/model.pth")


def load_tensors(
    data_dir="data/processed",
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """
    Loads the saved tensors in data/processed.

    Parameters
    ----------
    data_dir : str, optional
        directory with the processed (.py) MNIST data

    Returns
    -------
    (trainloader, validloader) : tuple
        dataloaders for the training and validation MNIST dataset
    """

    # alternative to move to the preset directory
    print(f"Loading data from: {data_dir}")

    train_images = torch.load(data_dir + "/train_images.pt")
    train_labels = torch.load(data_dir + "/train_labels.pt")
    valid_images = torch.load(data_dir + "/valid_images.pt")
    valid_labels = torch.load(data_dir + "/valid_labels.pt")

    # Create the custom dataset object
    train_dataset = MnistDataset(train_images, train_labels)
    valid_dataset = MnistDataset(valid_images, valid_labels)

    # Create the DataLoader
    trainloader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
    validloader = DataLoader(valid_dataset, batch_size=32, shuffle=False, num_workers=2)

    return trainloader, validloader


def save_model(model: torch.nn.Module, version: int, model_dir="models") -> None:
    """
    Saves the model in a subfolder in the "models" folder that is named after the version
    of the model (i.e. v1, v2, v3 etc...)

    Parameters
    ----------
    model : torch.nn.Module
        the pretrained network
    version : int
        the version of the pretrained model
    model_dir : str, optional
        directory with the pretrained, saved models

    Returns
    -------
    None
    """
    version_dir = os.path.join(model_dir, "v{}".format(version))
    if not os.path.exists(version_dir):
        os.makedirs(version_dir)
    model_path = os.path.join(version_dir, "model.pth")
    print(f"Saving model in directory: {model_path}")
    torch.save(model.state_dict(), model_path)


def load_model(model, model_dir="models"):
    """
    Loads the last saved version of the model, or if there is none, does nothing.

    Parameters
    ----------
    model : torch.nn.Module
       an instance of the pretrained model
    model_dir : str, optional
        directory with the pretrained, saved models

    Returns
    -------
    model : torch.nn.Module
        the latest version of the pretrained network
    """
    print(f"Loading model from directory: {model_dir}")

    version = get_latest_version(model_dir)
    latest_dir = model_dir + "/latest"
    if os.path.exists(latest_dir):
        print(f"Loading the latest model. This is likely version: v{version}")
        model_path = os.path.join(model_dir, "latest", "model.pth")
        model.load_state_dict(torch.load(model_path))
    else:
        print("Loading the newest version, the <latest> subfolder does not exist!")
        if version > -1:
            model_path = os.path.join(model_dir, "v{}".format(version), "model.pth")
            if os.path.exists(model_path):
                model.load_state_dict(torch.load(model_path))
            else:
                print(
                    "Warning: model version v{} does not exist. Unable to load model.".format(
                        version
                    )
                )
        else:
            print(
                "Warning: no previous version of the model exists. Creating an empty (v0) model..."
            )

    return model
