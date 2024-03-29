import argparse
import io
import logging
import os
import sys
from pathlib import Path
from typing import List

import hydra
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from sklearn.manifold import TSNE
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

import wandb

log = logging.getLogger(__name__)

def report(model, train_dataloader):
    '''Produce classification report and confusion matrix.'''
    preds, target = [], []
    for batch in train_dataloader:
        x, y = batch
        probs = model(x)
        preds.append(probs.argmax(dim=-1))
        target.append(y.detach())

    target = torch.cat(target, dim=0)
    preds = torch.cat(preds, dim=0)

    report = classification_report(target, preds)
    with open("reports/classification_report.txt", 'w') as outfile:
        outfile.write(report)
    confmat = confusion_matrix(target, preds)
    disp = ConfusionMatrixDisplay(confusion_matrix = confmat, display_labels=np.arange(10).tolist())
    plt.figure(figsize=(8, 8), dpi=100)
    disp.plot(cmap=plt.cm.Blues)
    plt.savefig('reports/figures/confusion_matrix.png', dpi=100)

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

    ax1.plot(train_steps[0 : len(train_losses)], train_losses, label="Train")
    ax1.plot(test_steps[0 : len(test_losses)], test_losses, "-*", label="Validation")
    ax1.set_ylabel("NLL Loss")
    ax1.set_title("Loss")
    ax1.legend()

    ax2.plot(train_steps[0 : len(train_accuracies)], train_accuracies, label="Train")
    ax2.plot(
        test_steps[0 : len(test_accuracies)],
        test_accuracies,
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
        log.info(f"Saving figure in: {figdir}")
    else:
        log.info(f"Saving LATEST figure in: {figdir}")
    plt.savefig(figdir)

    # Save the figure to a buffer instead of a file
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)

    # Open the image from the buffer using PIL
    image = Image.open(buf)

    # Convert the image to an RGB numpy array
    image_array = np.array(image.convert("RGB"))

    # Log the image using wandb
    wandb.log({"metrics": wandb.Image(image_array)})

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
    log.info(f"Saving LATEST figure in: {figdir}")
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

    # extract features from the trainset
    log.info("Extracting features...")
    features = extract_features(util.model, dataloader)

    # visualize features
    log.info("Visualizing 2d features using TSNE...")
    visualize_features(features)

    return


if __name__ == "__main__":
    # setup logging format
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

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
    from utils.model_utils import ModelUtils

    main(model_dir, data_fpath)
