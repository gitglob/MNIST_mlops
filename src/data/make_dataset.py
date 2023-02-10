# -*- coding: utf-8 -*-
import logging
import os
from pathlib import Path
from typing import List, Union

import click
import numpy as np
import torch
from dotenv import find_dotenv, load_dotenv
import hydra
from omegaconf import OmegaConf
import logging

log = logging.getLogger(__name__)


def load_mnist(raw_dir: str, train_version: Union[str, int] = "all") -> List[np.ndarray]:
    """
    Loads the train and validation MNIST data from the data/raw folder

    Parameters
    ----------
    raw_dir : str
        the directory where the raw MNIST (.npz) data are
    train_version : Union[str, int], optional
        the specific training dataset that the model
            - "first" means the first one
            - an integer specifies the version x, meaning the "train_x.npz" dataset
            - "all" means to load the data from every single file

    Returns
    -------
    data_out : List[np.ndarray]
        4d list with the numpy arrays containing the training and validation
        data [train_images, train_labels, validation_images, validation_labels].
    """

    log.info(f"\nLoading MNIST data from: {raw_dir}")

    # Get a list of all files in the directory
    files = os.listdir(raw_dir)

    # Filter the list to only include files that match the pattern "train_x.npz"
    train_files = [f for f in files if f.startswith("train_") and f.endswith(".npz")]

    # Sort the list of train files by the number x in ascending order
    train_files.sort(key=lambda x: int(x.split("_")[1].split(".")[0]))

    # check which version of data we want
    if train_version == "all":
        # Load each train file and split it into train_images and train_labels
        train_image_list = []
        train_label_list = []
        for train_file in train_files:
            file_path = os.path.join(raw_dir, train_file)
            data = np.load(file_path)
            train_image_list.append(data["images"])
            train_label_list.append(data["labels"])

            # Concatenate the train images and labels lists into a single numpy array
            train_images = np.concatenate(train_image_list)
            train_labels = np.concatenate(train_label_list)
    else:
        if train_version == "first":
            # Load the first (smallest x) train file with numpy.load
            train_file = os.path.join(raw_dir, train_files[0])
        else:
            # Train file name
            train_fname = "train_" + str(train_version) + ".npz"

            # Load the specific training dataset that was asked
            train_file = os.path.join(raw_dir, train_fname)

        # Load the specific training data file
        data = np.load(train_file)
        train_images = data["images"]
        train_labels = data["labels"]

    log.info(f"Shape of training images: {train_images.shape}")

    data = np.load(raw_dir + "/test.npz")
    validation_images = data["images"]
    log.info(f"Shape of validation images: {validation_images.shape}")
    validation_labels = data["labels"]

    data_out = [
        train_images,
        train_labels,
        validation_images,
        validation_labels,
    ]

    return data_out


def normalize(image: np.ndarray, m1: int, s1: int, m2: int = 0, s2: int = 1) -> np.ndarray:
    """
    Function to normalize a given image with a specific mean and std. deviation.
    Source: https://stats.stackexchange.com/questions/46429/transform-data-to-desired-mean-and-standard-deviation


    Parameters
    ----------
    image : numpy.array or PIL.image, optional
        the image to normalize
    m1 : float
        mean of the initial dataset pre-normalization
    s1 : float
        standard deviation of the initial dataset pre-normalization
    m2 : float
        desired new mean
    s2 : float
        desired new standard deviation

    Returns
    -------
    image_norm : numpy.array or PIL.image, optional
        the input image, normalized
    """

    image_norm = m2 + ((image - m1) * (s2 / s1))

    return image_norm

def preprocess(data: List[np.ndarray], m2: int = 0, s2: int = 1) -> List[torch.Tensor]:
    """
    Convert numpy data (images) into pytorch tensors and normalizes them with a
    mean of 0 and std. deviation of 1.

    Parameters
    ----------
    data : list
        the MNIST data in the form of a 4d list with numpy arrays
        [train_images, train_labels, validation_images, validation_labels]
    m2: int
        the new desired mean for the normalized images
    s2: int
        the new desired std. deviation for the normalized images

    Returns
    -------
    data_out : list
        the MNIST data in the same form of a 4d list, but with normalized
        tensors with mean=0 and std_dev=1
    """

    log.info(
        "\nPreprocessing data: Converting to tensor and normalizing with μ=0 and σ=1"
    )

    # If we were using the mean and std. dev. of Imagenet, it would look like this:
    # (however, this is only essential if we are using an Imagenet pretrained model)
    # transforms = T.Compose([
    #                   T.ToTensor(),
    #                   T.Normalize(
    #                     mean=[0.485, 0.456, 0.406],
    #                     std=[0.229, 0.224, 0.225])
    #             ])

    train_images, train_labels, valid_images, valid_labels = data
    log.info(
        f"Shape of [train_images, train_labels, valid_images, valid_labels]: {np.shape(train_images), np.shape(train_labels), np.shape(valid_images), np.shape(valid_labels)}"
    )

    # normalize images
    log.info("Normalizing images and labels")
    data_out = []
    for x in [train_images, train_labels, valid_images, valid_labels]:
        log.info(f"Shape of x vector (should be mx28x28): {np.shape(x)}")

        # extract initial mean and std. dev.
        m1 = np.mean(x)
        s1 = np.std(x)
        log.info(f"Mean, std. dev.: {m1, s1}")

        # normalizer = lambda xi, mi, si: m2 + ((xi - mi) * (s2 / si))
        x_norm = np.array(
            [normalize(x_i, m1, s1, m2, s2) for x_i in x]
        )

        log.info(f"Shape of x_norm vector (should be mx28x28): {np.shape(x_norm)}")
        # extract final mean and std. dev.
        m2 = np.mean(x_norm)
        s2 = np.std(x_norm)
        log.info(f"New mean, std. dev.: {m2, s2}")

        data_out.append(x_norm)

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

    log.info(f"\nSaving data at: {save_dir}")

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
    default="/home/glob/Documents/github/MNIST_mlops/data/raw",
    type=click.Path(exists=True),
)
@click.argument(
    "output_datadir",
    default="/home/glob/Documents/github/MNIST_mlops/data/processed",
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

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # get hydra diretories
    conf_dir = os.path.join(project_dir, "conf")
    conf_path = os.path.join(conf_dir, "config.yaml")

    # initialize Hydra with the path to the config.yaml file
    hydra.initialize(version_base=None, config_path="../../conf")
    cfg = hydra.compose(
        config_name="config.yaml"
    )  # <class 'omegaconf.dictconfig.DictConfig'>
    print(f"configuration: \n {OmegaConf.to_yaml(cfg)}")

    # initialize torch seed
    torch.manual_seed(cfg._general_.random_seed)

    log.info("making final data set from raw data")

    data = load_mnist(input_datadir)
    data = preprocess(data, cfg._data_.mean, cfg._data_.std_dev)
    save_data(data, output_datadir)

    return


if __name__ == "__main__":
    # setup logging format
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
