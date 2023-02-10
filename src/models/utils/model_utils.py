import os
from typing import Tuple, List, Union

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


class ModelUtils:
    def __init__(self, log, model, criterion=None, optimizer=None, models_dir="models"):
        self.log = log

        self.version = None
        self.generation = None

        self.model = model
        self.models_dir = models_dir

        self.gen_dir = None
        self.version_dir = None

        self.criterion = criterion
        self.optimizer = optimizer

        self.gen_match_flag = (
            False  # does the current model match a previously trained one?
        )
        self.previous_version_flag = (
            False  # does the current model have a previous version?
        )

        self.init()

    def init(self):
        """
        Initializes the generation and version directories.
        """
        # "generation"
        gen_folders_flag, gen_model_summaries = self.parse_gen_models()
        self.get_maching_gen(gen_model_summaries)
        if not self.gen_match_flag:
            self.log.info(f"New model generation!")
        
        self.gen_dir = os.path.join(self.models_dir, "g{}".format(self.generation))
        self.log.info(f"Generation directory: {self.gen_dir}")

        # "latest"
        self.latest_dir = os.path.join(self.gen_dir, "latest")
        self.latest_model = os.path.join(self.gen_dir, "latest", "model.pth")

        # "version"
        self.get_latest_version()
        if not self.previous_version_flag:
            self.log.warn(f"New model version!")

        self.version_dir = os.path.join(self.gen_dir, "v{}".format(self.version))
        self.log.info(f"Version directory: {self.version_dir}")

        # get the model path
        self.model_path = os.path.join(self.version_dir, "model.pth")
        self.log.info(f"Model path: {self.model_path}")

    def update(self):
        """
        Updates model version.
        """

        self.version += 1
        self.version_dir = os.path.join(self.gen_dir, "v{}".format(self.version))
        self.latest_dir = os.path.join(self.gen_dir, "latest")
        self.model_path = os.path.join(self.version_dir, "model.pth")
        self.log.info(f"Model path: {self.model_path}")

    def read_summary(self, file_path):
        with open(file_path, "r") as file:
            summary = file.read()
        return summary

    def get_maching_gen(
        self, gen_model_summaries: List[List[Union[torch.nn.Module, int]]]
    ):
        """
        Gets the matching model generation.

        Parameters
        ----------
        models_dir : str, optional
            directory with the pretrained, saved models

        Returns
        -------
        gen_model_summaries : list
            a list with all the model generation architectures and their generation id
            i.e. [[model(), g1], [model(), g5]]
        """

        # extract gen names ids and summaries
        gen_names = [_[1] for _ in gen_model_summaries]
        gen_ids = [int(string[1:]) for string in gen_names]
        gen_summaries = [_[0] for _ in gen_model_summaries]

        # check if there are other generations
        if not gen_model_summaries:
            self.generation = 0
            self.log.warn(f"No other generations at all!! This becomes g0!")
        else:
            # iterate over the existing generations and check if one of the existing generations matches our current model
            for (i, gen_summary) in enumerate(gen_summaries):
                if gen_summary == str(self.model):
                    self.generation = gen_ids[i]
                    self.log.info(f"Matching generation: {self.generation}")
                    self.gen_match_flag = True
                    break

            if not self.gen_match_flag:
                self.generation = max(gen_ids) + 1
                self.log.warn(f"No matching generation! New generation: {self.generation}")

    def parse_gen_models(self):
        """
        Loads all the model generations.

        Parameters
        ----------
        models_dir : str, optional
            directory with the pretrained, saved models

        Returns
        -------
        gen_model_summaries : list
            a list with all the model generation architectures and their generation id
            i.e. [[model_summary, g1], [model_summary, g5]]
        """

        gen_folders_flag = True
        gen_model_summaries = []
        gs = [
            f
            for f in os.listdir(self.models_dir)
            if f.startswith("g") and os.path.isdir(os.path.join(self.models_dir, f))
        ]
        if not gs:
            gen_folders_flag = False
        for g in gs:
            subfolder_path = os.path.join(self.models_dir, g)
            gen_summary_path = os.path.join(subfolder_path, "model_summary.txt")
            if not os.path.exists(gen_summary_path):
                continue
            model_summary = self.read_summary(gen_summary_path)
            gen_model_summaries.append([model_summary, g])

        return gen_folders_flag, gen_model_summaries

    def get_latest_version(self) -> int:
        """
        Gets the latest version of the specific generation model.
        If no previous version exists, it returns -1.

        Parameters
        ----------
        gen_dir : str, optional
            directory with the pretrained, saved models of this generation

        Returns
        -------
        version : int
            version of the trained model (-1 means that no pretrained model exists)
        """

        versions = []
        if self.gen_match_flag:
            for folder in os.listdir(self.gen_dir):
                if folder.startswith("v"):
                    self.previous_version_flag = True
                    curr_version = int(folder[1:])
                    versions.append(curr_version)
                self.version = max(versions)

            if not self.previous_version_flag:
                self.log.warn(f"No previous versions. This becomes v0!")
                self.version = 0
        else:
            self.log.warn(f"No previous generations & versions. This becomes g0/v0!")
            self.version = 0


    def load_tensors(
        self,
        data_dir: str = "data/processed",
        batch_size: int = 32,
    ) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
        """
        Loads the saved tensors in data/processed.

        Parameters
        ----------
        data_dir : str, optional
            directory with the processed (.py) MNIST data
        batch_size: int, optional
            size of batches

        Returns
        -------
        (trainloader, validloader) : tuple
            dataloaders for the training and validation MNIST dataset
        """

        # alternative to move to the preset directory
        self.log.info(f"Loading data from: {data_dir}")

        train_images = torch.load(data_dir + "/train_images.pt")
        train_labels = torch.load(data_dir + "/train_labels.pt")
        valid_images = torch.load(data_dir + "/valid_images.pt")
        valid_labels = torch.load(data_dir + "/valid_labels.pt")

        # Create the custom dataset object
        train_dataset = MnistDataset(train_images, train_labels)
        valid_dataset = MnistDataset(valid_images, valid_labels)

        # Create the DataLoader
        self.trainloader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=2
        )
        self.validloader = DataLoader(
            valid_dataset, batch_size=batch_size, shuffle=False, num_workers=2
        )

    def load_model(self):
        """
        Loads the last saved version of the model, or if there is none, does nothing.

        Parameters
        ----------
        model : torch.nn.Module
            an instance of the pretrained model
        gen : int
            the generation of the model
        model_dir : str, optional
            directory with the pretrained, saved models

        Returns
        -------
        model : torch.nn.Module
            the latest version of the pretrained network
        """

        self.log.info(f"Loading model from directory: {self.model_path}")

        # get the latest verion and its dir
        if os.path.exists(self.latest_model):
            self.log.info(
                f"Loading the latest model -this is likely gen: g{self.generation} , version: v{self.version} -"
            )
            self.model.load_state_dict(torch.load(self.latest_model))
        elif self.previous_version_flag:
            self.log.info(
                "Loading the newest version, the <latest> subfolder does not exist!"
            )
            model_path = os.path.join(
                self.gen_dir, "v{}".format(self.version), "model.pth"
            )
            self.model.load_state_dict(torch.load(model_path))
        else:
            self.log.warn("No previous version of this model exists!")

    def save_model(self) -> None:
        """
        Saves the model in a subfolder in the "models" folder that is named after the
        generation and version of the model (i.e. g1/v1, g1/v2, g3/v6 etc...).
        It also saves a file with the information about the model archtecture.

        Parameters
        ----------
        model : torch.nn.Module
            the pretrained network
        version : int
            the version of the pretrained model
        self.models_dir : str, optional
            directory with the pretrained, saved models

        Returns
        -------
        None
        """

        # save model
        self.log.info(f"Saving model in: {self.model_path}")
        if not os.path.exists(self.version_dir):
            os.makedirs(self.version_dir)
        torch.save(self.model.state_dict(), self.model_path)

        # If this model didn't exist before, export to TorchScript and save
        if not self.gen_match_flag:
            model_scripted_path = os.path.join(self.gen_dir, "model_scripted.pt")
            self.log.info(f"Saving model Torchscript in: {model_scripted_path}")
            model_scripted = torch.jit.script(self.model)
            model_scripted.save(model_scripted_path)

            # Also save the model summary as txt
            model_summary_path = os.path.join(self.gen_dir, "model_summary.txt")
            self.log.info(f"Saving model summary in: {model_summary_path}")
            with open(model_summary_path, "w") as f:
                # Write model architecture
                f.write(str(self.model))

    def save_latest_model(self, model: torch.nn.Module, model_dir="models") -> None:
        """
        Saves the latest model in the "latest" fubfolder in the "models/gx/" folder

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

        if not os.path.exists(self.latest_dir):
            os.makedirs(self.latest_dir)
        self.log.info(
            f"Saving LATEST model in directory: {self.latest_dir}/model.pth"
        )
        torch.save(self.model.state_dict(), self.latest_dir + "/model.pth")
