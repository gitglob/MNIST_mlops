import logging
import os
import sys
from pathlib import Path

import hydra
import pytorch_lightning as pl
import torch
from model import MyModel
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../../conf", config_name="config.yaml")
def main(cfg) -> None:
    """Runs train and validation scripts to train a NN based on the processed MNIST data
    in data/processed in the form of tensors."""
    # extract hydra configuration
    random_seed = cfg._general_.random_seed
    input_dim = cfg._model_.input_dim
    latent_dim = cfg._model_.latent_dim
    output_dim = cfg._model_.output_dim
    batch_size = cfg._train_.batch_size
    epochs = cfg._train_.epochs
    check_every = cfg._train_.check_every

    # set as working directory the MNIST_mlops folder
    project_dir = Path(__file__).resolve().parents[2]
    os.chdir(project_dir)

    # initialize torch seed
    torch.manual_seed(random_seed)

    # initialize
    model = MyModel(input_dim, latent_dim, output_dim)

    # initialize dataloaders
    data_module = MnistDataModule(batch_size=batch_size)

    # define trainer callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath="./models", monitor="val_loss", mode="min"
    )
    early_stopping_callback = EarlyStopping(
        monitor="val_loss", patience=3, verbose=True, mode="min"
    )

    # train model - pytorch lightning
    trainer = Trainer(
        precision="bf16",
        check_val_every_n_epoch=check_every,
        max_epochs=epochs,
        limit_train_batches=0.2,
        callbacks=[early_stopping_callback, checkpoint_callback],
        logger=pl.loggers.WandbLogger(project="MNIST_fashion"),
        profiler="simple",
    )
    trainer.fit(model, data_module)

    # test model
    trainer.test(model, data_module)


if __name__ == "__main__":
    # setup logging format
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # append sibling dir to system path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    import_dir_vis = os.path.join(current_dir, "..", "visualization")
    import_dir_data = os.path.join(current_dir, "..", "data")
    sys.path.append(import_dir_vis)
    sys.path.append(import_dir_data)

    from data_module import MnistDataModule

    main()
