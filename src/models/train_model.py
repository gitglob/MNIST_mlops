import os
import sys
from typing import Callable, Optional, Tuple, Union

import torch
from model import MyModel
from torch import nn

import hydra
from omegaconf import OmegaConf
import logging

log = logging.getLogger(__name__)

from utils.model_utils import ModelUtils


def validation(util) -> Tuple[float, float]:
    """
    Validated the training process n the validation set.

    Parameters
    ----------
    model : list
        the NN model
    validloader : torch.utils.data.DataLoader
        the validation Dataloader
    criterion : Union[Callable, nn.Module]
        the MNIST data in the form of a 4d list with numpy arrays

    Returns
    -------
    (valid_loss, accuracy) : tuple
        the validation loss and the validation accuracy
    """

    accuracy = 0
    valid_loss = 0
    for images, labels in util.validloader:
        images = images.resize_(images.size()[0], 784)

        _, output = util.model.forward(images)
        valid_loss += util.criterion(output, labels).item()

        # Calculating the accuracy
        # Model's output is log-softmax, take exponential to get the probabilities
        ps = torch.exp(output)
        # Class with highest probability is our predicted class, compare with true label
        equality = labels.data == ps.max(1)[1]
        # Accuracy is number of correct predictions divided by all predictions,
        # just take the mean
        accuracy += equality.type_as(torch.FloatTensor()).mean().item()

    return valid_loss, accuracy


def train(util: ModelUtils, epochs: int = 5, print_every: int = 40) -> None:
    """
    Validated the training process n the validation set.

    Parameters
    ----------
    model : list
        the NN model
    validloader : torch.utils.data.DataLoader
        the validation Dataloader
    criterion : Union[Callable, nn.Module]
        the MNIST data in the form of a 4d list with numpy arrays

    Returns
    -------
    None
    """

    # train the network
    step = 0
    train_steps = []
    test_steps = []
    running_loss = 0
    running_correct = 0
    running_tot = 0
    train_losses = []
    train_accuracies = []
    test_losses = []
    test_accuracies = []
    try:
        for e in range(epochs):
            # util.Model in training mode, dropout is on
            util.model.train()
            for images, labels in util.trainloader:
                train_steps.append(step)

                # Flatten images into a 784 long vector
                images.resize_(images.size()[0], 784)

                util.optimizer.zero_grad()

                features, output = util.model.forward(images)
                ps = torch.nn.functional.softmax(output, dim=1)
                _, predicted = torch.max(ps, dim=1)

                # calculate loss
                loss = util.criterion(output, labels)
                loss.backward()
                util.optimizer.step()

                # calculate running loss
                running_loss += loss.item()

                # calculate running correct correct and total predictions
                running_correct += (predicted == labels).sum().item()
                running_tot += len(labels)

                # log current training loss and accuracy
                train_losses.append(loss.item())
                train_accuracies.append((predicted == labels).sum().item() / len(labels))

                if step % print_every == 0:
                    log.info("\n")
                    test_steps.append(step)

                    # util.Model in inference mode, dropout is off
                    util.model.eval()

                    # Turn off gradients for validation, will speed up inference
                    with torch.no_grad():
                        test_loss, accuracy = validation(util)

                    print(
                        "Epoch: {}/{}.. ".format(e + 1, epochs),
                        "Training Loss: {:.3f}.. ".format(running_loss / print_every),
                        "Training Accuracy: {:.3f}.. ".format(
                            running_correct / running_tot
                        ),
                        "Test Loss: {:.3f}.. ".format(test_loss / len(util.validloader)),
                        "Test Accuracy: {:.3f}".format(accuracy / len(util.validloader)),
                    )

                    # log current loss and accuracy
                    test_losses.append(test_loss / print_every)
                    test_accuracies.append(accuracy / len(util.validloader))

                    # update the latest version
                    util.update()
                    # save the model every logging period
                    util.save_model()

                    # visualize and save the loss and accuracy thus far
                    visualize_metrics(
                        e,
                        "latest",
                        train_steps,
                        test_steps,
                        train_losses,
                        train_accuracies,
                        test_losses,
                        test_accuracies,
                    )

                    # set running training loss and number of correct predictions to 0
                    running_loss = 0
                    running_correct = 0
                    running_tot = 0

                    # Make sure dropout and grads are on for training
                    util.model.train()

                # increase the step
                step += 1
    # if the training ends or is interrupted, we want the util.model and the last
    # visualizations to be saved in the "latest" folder
    except KeyboardInterrupt:
        util.update()
        util.save_latest_model(util.model)
        visualize_metrics(
            e,
            "latest",
            train_steps,
            test_steps,
            train_losses,
            train_accuracies,
            test_losses,
            test_accuracies,
        )

    util.save_latest_model(util.model)
    visualize_metrics(
        e,
        "latest",
        train_steps,
        test_steps,
        train_losses,
        train_accuracies,
        test_losses,
        test_accuracies,
    )


def main() -> None:
    """Runs train and validation scripts to train a NN based on the processed MNIST data
    in data/processed in the form of tensors."""

    # initialize Hydra with the path to the config.yaml file
    hydra.initialize(version_base=None, config_path="../../conf")
    cfg = hydra.compose(config_name="config.yaml")

    log.info(f"configuration: \n {OmegaConf.to_yaml(cfg)}")

    # initialize torch seed
    torch.manual_seed(cfg._general_.random_seed)

    # initialize
    model = MyModel(cfg._model_.input_dim, cfg._model_.latent_dim, cfg._model_.output_dim)

    # define optimizer and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg._train_.learning_rate)
    criterion = nn.CrossEntropyLoss()

    # initialize model helper
    util = ModelUtils(log, model, criterion, optimizer)
    
    # load model
    model = util.load_model()
    # load data
    util.load_tensors(batch_size=cfg._train_.batch_size)

    # define number of epochs and log frequency
    epochs = cfg._train_.epochs
    print_every = cfg._train_.print_every

    # train model
    train(util, epochs, print_every)


if __name__ == "__main__":
    # setup logging format
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # append sibling dir to system path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    import_dir = os.path.join(current_dir, "..", "visualization")
    sys.path.append(import_dir)

    from visualize import visualize_metrics

    main()
