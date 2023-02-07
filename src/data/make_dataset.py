# -*- coding: utf-8 -*-
import logging
import os
from pathlib import Path
from typing import List

import click
import numpy as np
import torch
from dotenv import find_dotenv, load_dotenv


def load_mnist(mnist_dir: str) -> List[np.ndarray]:
    """
    Loads the train and validation MNIST data from the data/raw folder

    Parameters
    ----------
    mnist_dir : str, optional
        the directory where the raw MNIST (.npz) data are

    Returns
    -------
    data_out : list
        4d list with the numpy arrays containing the training and validation
        data [train_images, train_labels, validation_images, validation_labels].
    """

    print(f"\nLoading MNIST data from: {mnist_dir}")
    data = np.load(mnist_dir + "/train_1.npz")
    train_images = data["images"]
    print(f"Shape of training images: {train_images.shape}")
    train_labels = data["labels"]
    _ = data["allow_pickle"]

    data = np.load(mnist_dir + "/test.npz")
    validation_images = data["images"]
    print(f"Shape of validation images: {validation_images.shape}")
    validation_labels = data["labels"]
    _ = data["allow_pickle"]

    data_out = [
        train_images,
        train_labels,
        validation_images,
        validation_labels,
    ]

    return data_out


def normalize(image: np.ndarray, m2: int = 0, s2: int = 1) -> np.ndarray:
    """
    Function to normalize a given image with a specific mean and std. deviation

    Parameters
    ----------
    image : numpy.array or PIL.image, optional
        the image to normalize
    m2 : float
        desired new mean
    s2 : float
        desired new standard deviation

    Returns
    -------
    image_norm : numpy.array or PIL.image, optional
        the input image, normalized
    """
    # current mean and std dev
    m1 = np.mean(image)
    s1 = np.std(image)

    image_norm = m2 + ((image - m1) * (s2 / s1))

    return image_norm


def preprocess(data: List[np.ndarray]) -> List[torch.Tensor]:
    """
    Convert numpy data (images) into pytorch tensors and normalizes them with a
    mean of 0 and std. deviation of 1.

    Source: https://stats.stackexchange.com/questions/46429/transform-data-to-desired-mean-and-standard-deviation

    Parameters
    ----------
    data : list
        the MNIST data in the form of a 4d list with numpy arrays
        [train_images, train_labels, validation_images, validation_labels]

    Returns
    -------
    data_out : list
        the MNIST data in the same form of a 4d list, but with normalized
        tensors with mean=0 and std_dev=1
    """

    print("\nPreprocessing data: Converting to tensor and normalizing with μ=0 and σ=1")

    # transforms = T.Compose([
    #                   T.ToTensor(),
    #                   T.Normalize(
    #                     mean=[0.485, 0.456, 0.406],
    #                     std=[0.229, 0.224, 0.225])
    #             ])

    train_images, train_labels, valid_images, valid_labels = data

    # normalize features
    print("Preprocessing features")
    out_images = []
    for x in [train_images, valid_images]:
        # current mean and std dev
        m1 = np.mean(x, axis=(1, 2)).reshape(-1, 1)
        s1 = np.std(x, axis=(1, 2)).reshape(-1, 1)
        # print(f"Shape of μ, σ vector (should be 5000x1): {np.shape(m1), np.shape(s1)}")
        # desired mean and std dev
        m2 = 0
        s2 = 1

        normalizer = lambda xi, mi, si: m2 + ((xi - mi) * (s2 / si))
        x_norm = np.array(
            [normalizer(x_i, m_i, s_i) for (x_i, m_i, s_i) in zip(x, m1, s1)]
        )
        m2 = np.mean(x_norm, axis=(1, 2)).reshape(-1, 1)
        s2 = np.std(x_norm, axis=(1, 2)).reshape(-1, 1)
        # print(f"Shape of μ, σ vector (should be 5000x1): {np.shape(m2), np.shape(s2)}")

        out_images.append(x)

    # normalize labels
    print("Preprocessing labels")
    out_labels = []
    for x in [train_labels, valid_labels]:
        # current mean and std dev
        m1 = np.mean(x).reshape(-1, 1)
        s1 = np.std(x).reshape(-1, 1)
        # print(f"Shape of μ, σ vector (should be 1x1): {np.shape(m1), np.shape(s1)}")
        # desired mean and std dev
        m2 = 0
        s2 = 1

        x_norm = m2 + ((x - m1) * (s2 / s1))
        x_norm = x.reshape(-1, 1)
        m2 = np.mean(x_norm).reshape(-1, 1)
        s2 = np.std(x_norm).reshape(-1, 1)
        # print(f"Shape of μ, σ vector (should be 1x1): {np.shape(m2), np.shape(s2)}")

        out_labels.append(x)

    data_out = [out_images[0], out_labels[0], out_images[1], out_labels[1]]

    return data_out


def save_data(data: List[np.ndarray], save_dir: str) -> None:
    """
    Saves the incoming normalized pytorch tensors in the specified filepath.

    Parameters
    ----------
    data : list
        the MNIST data in the form of a 4d list with normalized tensors (m=0,
        s=1) [train_images, train_labels, validation_images, validation_labels]
    save_dir : str
        the directory where the normalized tensors should be saved in as .pt files

    Returns
    -------
    None
    """

    print(f"\nSaving data at: {save_dir}")

    # if the save_dir doesn't exist, create and save it, along with an empty .gitkeep file
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        open(save_dir + "/.gitkeep", "w").close()

    train_images, train_labels, valid_images, valid_labels = data
    torch.save(train_images, save_dir + "/train_images.pt")
    torch.save(train_labels, save_dir + "/train_labels.pt")
    torch.save(valid_images, save_dir + "/valid_images.pt")
    torch.save(valid_labels, save_dir + "/valid_labels.pt")


@click.command()
@click.argument(
    "input_datadir",
    default="/home/glob/Documents/github/dtu_mlops/mlops_cookiecutter/data/raw",
    type=click.Path(exists=True),
)
@click.argument(
    "output_datadir",
    default="/home/glob/Documents/github/dtu_mlops/mlops_cookiecutter/data/processed",
    type=click.Path(),
)
def main(input_datadir: str, output_datadir: str) -> None:
    """
    Runs data processing scripts to turn raw data from (../raw) into cleaned data ready
    to be analyzed (saved in ../processed).

    Parameters
    ----------
    input_datadir : str, argument
        the MNIST data in the form of a 4d list with normalized tensors (m=0, s=1)
        [train_images, train_labels, validation_images, validation_labels]
    output_datadir : str, argument
        the directory where the normalized tensors should be saved in as .pt files

    Returns
    -------
    None
    """
    logger = logging.getLogger(__name__)
    logger.info("making final data set from raw data")

    data = load_mnist(input_datadir)
    data = preprocess(data)
    save_data(data, output_datadir)

    return


if __name__ == "__main__":
    # setup logging format
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
