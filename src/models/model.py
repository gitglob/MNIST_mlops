from typing import Tuple

import numpy as np
from pytorch_lightning import LightningModule
from torch import Tensor, nn, optim

import wandb


class MyModel(LightningModule):
    """
    A class for a NN to train and make predictions on MNIST data.

    ...

    Attributes
    ----------
    name : str
        first name of the person
    surname : str
        family name of the person
    age : int
        age of the person

    Methods
    -------
    info(additional=""):
        log.infos the person's name and age.
    """

    def __init__(self, img_shape, latent_dim, output_dim):
        """
        Constructs all the layers of the NN.

        Parameters
        ----------
        img_shape : int
            shape of image (x, y, z)
        latent_dim : int
            size of latent dimension (latent features)
        output_dim : int
            size of output dimension
        """

        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(1, 64, 3),
            nn.LeakyReLU(),
            nn.Conv2d(64, 32, 3),
            nn.LeakyReLU(),
            nn.Conv2d(32, 16, 3),
            nn.LeakyReLU(),
            nn.Conv2d(16, 8, 3),
            nn.LeakyReLU(),
        )

        self.latent_extractor = nn.Sequential(
            nn.Flatten(), nn.Linear(8 * 20 * 20, latent_dim)
        )

        self.classifier = nn.Sequential(nn.Dropout(), nn.Linear(latent_dim, output_dim))

        self.criterium = nn.CrossEntropyLoss()

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Does a forward pass on the NN

        Parameters
        ----------
        x : torch.Tensor
            batch of data to perform a forward pass upon

        Returns
        -------
        features, x : torch.Tensor, torch.Tensor
            the 2d features before the output layer & the output of the NN
        """
        if x.ndim != 4:
            raise ValueError("Expected input to a 4D tensor")
        if x.shape[1] != 1 or x.shape[2] != 28 or x.shape[3] != 28:
            raise ValueError("Expected each sample to have shape [1, 28, 28]")

        # self.pshape(x)
        x = self.backbone(x)
        # self.pshape(x)
        latent_features = self.latent_extractor(x)
        # self.pshape(x)
        y = self.classifier(latent_features)
        # self.pshape(x)

        return y

    def pshape(self, x: Tensor) -> None:
        """
        prints the shape of the input tensor

        Parameters
        ----------
        x : torch.Tensor
            input tensor
        """
        print("Shape: ", x.shape)

    def training_step(self, batch, batch_idx):
        loss, acc = self._shared_eval_step(batch, batch_idx)

        # log with wandb
        self.log("train_loss", loss)
        self.log("train_acc", acc)

        # hist = np.histogram(preds.detach().numpy())
        # self.logger.experiment.log({'train_logits': wandb.Histogram(np_histogram=hist)})

        return loss

    def validation_step(self, batch, batch_idx):
        loss, acc = self._shared_eval_step(batch, batch_idx)
        metrics = {"val_acc": acc, "val_loss": loss}
        self.log_dict(metrics)
        return metrics

    def test_step(self, batch, batch_idx):
        loss, acc = self._shared_eval_step(batch, batch_idx)
        metrics = {"test_acc": acc, "test_loss": loss}
        self.log_dict(metrics)
        return metrics

    def _shared_eval_step(self, batch, batch_idx):
        # extract x,y and forward pass
        x, y = batch
        preds = self(x)
        # loss and accuracy
        loss = self.criterium(preds, y)
        acc = (y == preds.argmax(dim=-1)).float().mean()
        return loss, acc

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x, y = batch
        preds = self(x)
        return preds

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
