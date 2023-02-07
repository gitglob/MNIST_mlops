import argparse
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.manifold import TSNE
from typing import List


def visualize_metrics(
    e: int,
    version: int,
    train_steps: List[int],
    test_steps: List[int],
    train_losses: list,
    train_accuracies: list,
    test_losses: list,
    test_accuracies: list,
    model_dir: str = "models",
    vis_dir: str = "reports/figures/metrics",
) -> None:
    """
    Visualizes and saves the figure of the loss and accuracy over time of the trained
    model in reports/difures/metrics.

    Parameters
    ----------
    e : array_like
        the 2d features of the dataset
    version :
        the version of the model
    train_steps : int
        the number of batches that have been processed
    test_steps : int
        the number of times the model has been evaluated on the validation data with the
        testloader
    train_losses : numpy.array
        the loss over the train_steps
    train_accuracies : numpy.array
        the accuracy over the train_steps
    test_losses : numpy.array
        the loss over the test_steps
    test_accuracies : numpy.array
        the accuracy over the test_steps
    model_dir : str, optional
        the directory where the models are saved
    vis_dir : str, optional
        the directory where the metric plot should be saved

    Returns
    -------
    None
    """

    if version == "latest":
        vis_version_dir = os.path.join(vis_dir, version)
    else:
        vis_version_dir = os.path.join(vis_dir, "v{}".format(version))

    fig, (ax1, ax2) = plt.subplots(2)

    ax1.plot(train_steps, train_losses[0 : len(train_steps)], label="Train")
    ax1.plot(test_steps, test_losses[0 : len(test_steps)], "-*", label="Validation")
    ax1.set_ylabel("NLL Loss")
    ax1.set_title("Loss")
    ax1.legend()

    ax2.plot(train_steps, train_accuracies[0 : len(train_steps)], label="Train")
    ax2.plot(
        test_steps,
        test_accuracies[0 : len(test_steps)],
        "-*",
        label="Validation",
    )
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Accuracy")
    ax2.legend()

    fig.supxlabel("step")

    fig.suptitle(f"Model: v{version} , Epoch: {e} , Step: {train_steps[-1]}")

    if not os.path.exists(vis_version_dir):
        os.makedirs(vis_version_dir)
    figdir = vis_version_dir + "/metrics.png"
    if version != "latest":
        print(f"Saving figure in: {figdir}")
    else:
        print(f"Saving LATEST figure in: {figdir}")
    plt.savefig(figdir)
    plt.close()


def visualize_features(
    features: np.ndarray, vis_dir: str = "reports/figures/features"
) -> None:
    """
    Visualizes and saves the 2d features of the training data in reports/difures/features.

    Parameters
    ----------
    features : array_like
        the 2d features of the dataset
    vis_dir : vis_dir, optional
        the directory where the feature plot should be saved

    Returns
    -------
    None
    """

    vis_version_dir = os.path.join(vis_dir, "latest")

    fig, ax = plt.subplots()

    ax.scatter(features[:, 0], features[:, 1], s=1)
    ax.set_xlabel("Dimension #1")
    ax.set_ylabel("Dimension #2")
    ax.set_title("2D features from training data")

    if not os.path.exists(vis_version_dir):
        os.makedirs(vis_version_dir)
    figdir = vis_version_dir + "/features.png"
    print(f"Saving LATEST figure in: {figdir}")
    plt.savefig(figdir)
    plt.close()


def extract_features(
    model: torch.nn.Module, dataloader: torch.utils.data.DataLoader
) -> np.ndarray:
    """
    Extract the 2d features of the training data using sklearn's TSNE as a feature
    extractor.

    Parameters
    ----------
    model : nn.Module
        the trained neural network
    dataloader : torch.utils.data.Dataloader
        the dataloader that is used to load the training data

    Returns
    -------
    features_out : numpy.array
        the 2d features
    """

    feature_extractor = TSNE(
        n_components=2, learning_rate="auto", init="random", perplexity=3
    )

    features_2d_list = []
    model.eval()
    for images, labels in dataloader:
        images = images.resize_(images.size()[0], 784)
        labels = labels.flatten()

        features, _ = model.forward(images)
        features = features.detach().numpy()

        features_2d = feature_extractor.fit_transform(features)
        features_2d_list.extend(features_2d.tolist())

    features_out = np.array(features_2d_list)

    return features_out


def main(model_dir: str, data_fpath: str) -> None:
    """
    Calls all the necessary functions to visualize the 2d features of the training data.

    Parameters
    ----------
    model_dir : str
        the directory of the trained models
    data_fpath : str
        the path to the data the network was trained upon

    Returns
    -------
    None
    """
    # load model
    model = MyModel()
    model = load_model(model, model_dir)

    # load data
    data = load_data(data_fpath)
    data = preprocess(data)
    dataloader = data2dataloader(data)

    # extract features from the trainset
    print("Extracting features...")
    features = extract_features(model, dataloader)

    # visualize features
    print("Visualizing 2d features using TSNE...")
    visualize_features(features)

    return


if __name__ == "__main__":
    # parse arguments
    parser = argparse.ArgumentParser(
        description="Make visualization on the trained data with the trained model."
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
        default="data/raw/train_1.npz",
        help="the path to the .npz data",
    )

    args = parser.parse_args()
    model_dir = args.model_dir
    data_fpath = args.data_fpath

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # append sibling dir to system path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    import_dir = os.path.join(current_dir, "..", "models")
    sys.path.append(import_dir)

    # load functions from "models" sibling directory
    from model import MyModel
    from predict_model import data2dataloader, load_data, preprocess
    from utils.model_utils import load_model

    main(model_dir, data_fpath)
