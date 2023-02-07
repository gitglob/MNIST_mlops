import os
import sys
from typing import Callable, Optional, Tuple, Union

import torch
from model import MyModel
from torch import nn
from utils.model_utils import (
    get_latest_version,
    load_model,
    load_tensors,
    save_latest_model,
    save_model,
)


def validation(
    model: nn.Module,
    validloader: torch.utils.data.DataLoader,
    criterion: Union[Callable, nn.Module],
) -> Tuple[float, float]:
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
    for images, labels in validloader:
        images = images.resize_(images.size()[0], 784)

        _, output = model.forward(images)
        valid_loss += criterion(output, labels).item()

        # Calculating the accuracy
        # Model's output is log-softmax, take exponential to get the probabilities
        ps = torch.exp(output)
        # Class with highest probability is our predicted class, compare with true label
        equality = labels.data == ps.max(1)[1]
        # Accuracy is number of correct predictions divided by all predictions,
        # just take the mean
        accuracy += equality.type_as(torch.FloatTensor()).mean().item()

    return valid_loss, accuracy


def train(
    model: nn.Module,
    trainloader: torch.utils.data.DataLoader,
    testloader: torch.utils.data.DataLoader,
    criterion: Union[Callable, nn.Module],
    optimizer: Optional[torch.optim.Optimizer] = None,
    epochs: int = 5,
    print_every: int = 40,
) -> None:
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

    # check which version of the model we are running
    model_relative_dir = "models"
    version = get_latest_version(model_relative_dir) + 1

    # train the network
    if optimizer is None:
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
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
            # Model in training mode, dropout is on
            model.train()
            for images, labels in trainloader:
                train_steps.append(step)

                # Flatten images into a 784 long vector
                images.resize_(images.size()[0], 784)

                optimizer.zero_grad()

                features, output = model.forward(images)
                ps = torch.exp(output)
                _, predicted = torch.max(ps, dim=1)

                # calculate loss
                loss = criterion(output, labels)
                loss.backward()
                optimizer.step()

                # calculate running loss
                running_loss += loss.item()

                # calculate running correct correct and total predictions
                running_correct += (predicted == labels).sum().item()
                running_tot += len(labels)

                # log current training loss and accuracy
                train_losses.append(loss.item())
                train_accuracies.append((predicted == labels).sum().item() / len(labels))

                if step % print_every == 0:
                    test_steps.append(step)
                    # Model in inference mode, dropout is off
                    model.eval()

                    # Turn off gradients for validation, will speed up inference
                    with torch.no_grad():
                        test_loss, accuracy = validation(model, testloader, criterion)

                    print(
                        "Epoch: {}/{}.. ".format(e + 1, epochs),
                        "Training Loss: {:.3f}.. ".format(running_loss / print_every),
                        "Training Accuracy: {:.3f}.. ".format(
                            running_correct / running_tot
                        ),
                        "Test Loss: {:.3f}.. ".format(test_loss / len(testloader)),
                        "Test Accuracy: {:.3f}".format(accuracy / len(testloader)),
                    )

                    # log current loss and accuracy
                    test_losses.append(test_loss / print_every)
                    test_accuracies.append(accuracy / len(testloader))

                    # save the model every logging period
                    save_model(model, version)

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
                    model.train()

                # increase the step
                step += 1
    # if the training ends or is interrupted, we want the model and the last
    # visualizations to be saved in the "latest" folder
    except KeyboardInterrupt:
        save_latest_model(model)
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

    save_latest_model(model)
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
    # load model
    model = MyModel()
    model = load_model(model)

    # load data
    trainloader, testloader = load_tensors()

    # define optimizer and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    # define number of epochs and log frequency
    epochs = 10
    print_every = 500

    # train model
    train(
        model,
        trainloader,
        testloader,
        criterion,
        optimizer,
        epochs,
        print_every,
    )


if __name__ == "__main__":
    # append sibling dir to system path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    import_dir = os.path.join(current_dir, "..", "visualization")
    sys.path.append(import_dir)

    from visualize import visualize_metrics

    main()
