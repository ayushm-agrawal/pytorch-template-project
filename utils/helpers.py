import os
from types import SimpleNamespace
from typing import Optional

import torch
from torch import nn, optim
from torchvision.datasets import CIFAR10

from models.alexnet import AlexNet
from preprocess.transforms import cifar10_transforms
from torch.utils.data import DataLoader


class Helpers:
    """
    Class contains Helper methods to load config objects
    required for model training

    Each method will update the specific configs attribute by loading the actual Object type

    Attributes:
        configs : SimpleNamespace
            a SimpleNamespace object containing model hyper_parameters and utilities

    """

    def __init__(self, configs: SimpleNamespace):
        """Inits Helpers class with configs"""
        self.configs = configs

    def load_optimizer(self) -> torch.optim.Optimizer:
        """
        Method loads respective Optimizer using the passed configs value

        Returns:
            (:class:~`torch.optim.optimizer`) Optimizer object

        Examples:
            Initiate the class and call the methods

            >>> helpers = Helpers(self.configs)
            >>> optimize =  helpers.load_optimizer()
        """

        # Load the model if it wasn't loaded
        if not isinstance(self.configs.model_name, nn.Module):
            self.configs.model_name = self.load_model()

        optimizer_name: str = str(self.configs.optimizer).lower()
        optimizer = None
        if optimizer_name == "adam":
            optimizer = optim.Adam(
                self.configs.model_name.parameters(),
                lr=self.configs.learning_rate,
                weight_decay=self.configs.weight_decay
            )
        elif optimizer_name == "sgd":
            optimizer = optim.SGD(
                self.configs.model_name.parameters(),
                lr=self.configs.learning_rate,
                weight_decay=self.configs.weight_decay,
                momentum=self.configs.momentum
            )
        # Add your required optimizer values in the elif statements

        return optimizer

    def load_criterion(self) -> Optional[nn.Module]:
        """
        Loads the criterion object from the `configs.criterion` attribute

        Returns:
            (:class:`~torch.nn.modules.loss`): Specific Loss object based on the Criterion name

        Examples:
            Initiate the class and call the methods

            >>> helpers = Helpers(self.configs)
            >>> _criterion = helpers.load_criterion()
        """

        criterion_name: str = str(self.configs.criterion).lower()
        criterion = None

        if criterion_name == "cross-entropy":
            criterion = nn.CrossEntropyLoss()

        # Add additional criterion in the elif statements

        return criterion

    def create_data_loaders(self) -> dict:
        """
        Creates :class:`~torch.utils.data.DataLoader` Object for Train, Validation and Test data

        Returns:
            (dict): Dictionary of :class:`~torch.utils.data.DataLoader` Objects for train, test and validation

        Examples:

            >>> helpers = Helpers()
            >>> data = helpers.create_data_loaders()
        """

        # update this line for your own transformation function
        train_transform, test_transform = cifar10_transforms()

        self.configs.data["train"] = "train" if self.configs.data["train"] is None else self.configs.data["train"]
        self.configs.data["test"] = "test" if self.configs.data["test"] is None else self.configs.data["test"]
        if self.configs.data["validation"] is None:
            self.configs.data["validation"] = "validation"

        train_path = os.path.join(self.configs.data["root"], self.configs.data["train"])
        test_path = os.path.join(self.configs.data["root"], self.configs.data["test"])

        # load datasets, downloading if needed
        train_set = CIFAR10(train_path, train=True, download=True,
                            transform=train_transform)
        test_set = CIFAR10(test_path, train=False, download=True,
                           transform=test_transform)

        train_loader = DataLoader(train_set,
                                  batch_size=self.configs.batch_size,
                                  num_workers=0)
        test_loader = DataLoader(test_set,
                                 batch_size=self.configs.batch_size,
                                 num_workers=0)

        return dict({"train_loader": train_loader, "test_loader": test_loader})

    def load_model(self) -> nn.Module:
        """
        Loads model object for the `configs.model_name` attribute

        Returns: None

        Examples:
            Initiate the class and call the methods

            >>> helpers = Helpers(self.configs)
            >>> helpers.load_model()
        """

        # check if model was previously loaded
        if not isinstance(self.configs.model_name, nn.Module):
            model_name: str = str(self.configs.model_name).lower()

            model = None
            if model_name == "alexnet":
                model = AlexNet(num_classes=self.configs.num_classes,
                                input_channels=self.configs.input_channels)

            # To add your own models, create a `<model_name>.py` file under `models` directory
            # import the model in this file and add your models in the elif statement
            # similar to the example model here

            return model
