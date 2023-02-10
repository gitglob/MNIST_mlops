# -*- coding: utf-8 -*-
import argparse
from pathlib import Path
from typing import List

import numpy as np
import torch
from model import MyModel
from torch import nn
from torch.utils.data import DataLoader
from utils.model_utils import ModelUtils, MnistDataset
import hydra
import logging

log = logging.getLogger(__name__)


def preprocess(data: List[np.ndarray]) -> List[torch.Tensor]:
    """
    Convert numpy data (images) into pytorch tensors and normalizes them with a mean of 0
    and std. deviation of 1.

    The formula used is taken from: https://stats.stackexchange.com/questions/46429/transform-data-to-desired-mean-and-standard-deviation

    Parameters
    ----------
    data : list [np.array, np.array]
        the test data [images, labels]

    Returns
    -------
    data_norm : list [torch.Tensor, torch.Tensor]
        the normalized test data [images, labels]
    """

    log.info(
        "\nPreprocessing data: Converting to tensor and normalizing with μ=0 and σ=1"
    )

    images, labels = data

    # normalize features
    log.info("Preprocessing features")
    # current mean and std dev
    m1 = np.mean(images, axis=(1, 2)).reshape(-1, 1)
    s1 = np.std(images, axis=(1, 2)).reshape(-1, 1)
    # log.info(f"Shape of μ, σ vector (should be 5000x1): {np.shape(m1), np.shape(s1)}")
    # desired mean and std dev
    m2 = 0
    s2 = 1

    def normalizer(xi, mi, si):
        return m2 + ((xi - mi) * (s2 / si))

    images_norm = np.array(
        [normalizer(x_i, m_i, s_i) for (x_i, m_i, s_i) in zip(images, m1, s1)]
    )
    m2 = np.mean(images_norm, axis=(1, 2)).reshape(-1, 1)
    s2 = np.std(images_norm, axis=(1, 2)).reshape(-1, 1)
    # log.info(f"Shape of μ, σ vector (should be 5000x1): {np.shape(m2), np.shape(s2)}")

    # normalize labels
    log.info("Preprocessing labels")
    # current mean and std dev
    m1 = np.mean(labels).reshape(-1, 1)
    s1 = np.std(labels).reshape(-1, 1)
    # log.info(f"Shape of μ, σ vector (should be 1x1): {np.shape(m1), np.shape(s1)}")
    # desired mean and std dev
    m2 = 0
    s2 = 1

    labels_norm = m2 + ((labels - m1) * (s2 / s1))
    labels_norm = labels.reshape(-1, 1)
    m2 = np.mean(labels_norm).reshape(-1, 1)
    s2 = np.std(labels_norm).reshape(-1, 1)
    # log.info(f"Shape of μ, σ vector (should be 1x1): {np.shape(m2), np.shape(s2)}")

    data_norm = [images_norm, labels_norm]

    return data_norm


def load_data(input_filepath: str) -> List[np.ndarray]:
    """
    Loads the train and test MNIST data from the data/raw folder

    Parameters
    ----------
    input_filepath : torch.nn.Module
        the model to predict with

    Returns
    -------
    data : list [np.array, np.array]
        the test data [images, labels]
    """

    log.info(f"\nLoading data from: {input_filepath}")
    data = np.load(input_filepath)
    images = data["images"]
    log.info(f"Shape of images: {images.shape}")
    labels = data["labels"]
    _ = data["allow_pickle"]

    data = [images, labels]

    return data


def predict(model: torch.nn.Module, dataloader: torch.utils.data.DataLoader) -> None:
    """
    Create predictions based on the latest version of the trained model.

    Parameters
    ----------
    model : torch.nn.Module
        the model to predict with
    dataloader: torch.utils.data.DataLoader
        the dataloader with the test data

    Returns
    -------
    None
    """

    correct = 0
    loss = 0
    criterion = nn.CrossEntropyLoss()
    model.eval()
    for images, labels in dataloader:
        images = images.resize_(images.size()[0], 784)
        labels = labels.flatten()

        features, output = model.forward(images)
        ps = torch.exp(output)
        _, predicted = torch.max(ps, dim=1)

        loss += criterion(output, labels).item()
        correct += (predicted == labels).sum().item()

        # log.info(f"labels: {labels.T}")
        # log.info(f"predictions: {predicted}")

    log.info(f"Mean Loss: {loss/len(dataloader)} , Accuracy: {correct/len(dataloader)}")


def data2dataloader(data: List[np.ndarray]) -> torch.utils.data.DataLoader:
    """
    Converts the numpy array data to torch tensors and then to a dataloader.

    Parameters
    ----------
    data : list, [np.array, np.array]
        the images and labels of the test data

    Returns
    -------
    dataloader: torch.utils.data.DataLoader
        the dataloader with the test data
    """

    images, labels = data
    images = np.array(images)
    labels = np.array(labels)

    # Create the custom dataset object
    dataset = MnistDataset(images, labels)

    # Create the DataLoader
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=2)

    return dataloader


def main(model_dir: str, data_fpath: str) -> None:
    """
    Runs prediction scripts to test the pretrained NN on new, raw, test data (.npz).

    Parameters
    ----------
    model_dir : str
        the directory of the trained models
    data_fpath : str
        the path to the data to test upon

    Returns
    -------
    None
    """
    # initialize Hydra with the path to the config.yaml file
    hydra.initialize(version_base=None, config_path="../../conf")
    cfg = hydra.compose(config_name="config.yaml")

    # initialize torch seed
    torch.manual_seed(cfg._general_.random_seed)

    # load model
    model = MyModel(cfg._model_.input_dim, cfg._model_.latent_dim, cfg._model_.output_dim)
    
    # initialize model helper
    util = ModelUtils(log, model)
    # load model
    util.load_model()

    # load data
    data = load_data(data_fpath)
    data = preprocess(data)
    dataloader = data2dataloader(data)

    # predict the labels of the data
    predict(util.model, dataloader)

    return


if __name__ == "__main__":
    # setup logging format
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    
    # parse arguments
    parser = argparse.ArgumentParser(
        description="Predict new data with the trained model."
    )
    parser.add_argument(
        "model_dir",
        nargs="?",
        type=str,
        default="models",
        help="the path to the trained model <model.pth>",
    )
    parser.add_argument(
        "data_fpath",
        nargs="?",
        type=str,
        default="data/raw/train_2.npz",
        help="the path to the .npz data",
    )

    args = parser.parse_args()
    model_dir = args.model_dir
    data_fpath = args.data_fpath

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    log.info("Running inference on new, untrained upon data...")
    main(model_dir, data_fpath)
