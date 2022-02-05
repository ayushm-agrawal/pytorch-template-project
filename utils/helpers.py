import os
from types import SimpleNamespace
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

    def load_optimizer(self) -> None:
        """
        Method loads respective Optimizer using the passed configs value

        Returns: None

        Examples:
            Initiate the class and call the methods

            >>> helpers = Helpers(self.configs)
            >>> helpers.load_optimizer()
        """

        # Load the model if it wasn't loaded
        if not isinstance(self.configs.model_name, nn.Module):
            self.load_model()

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

        self.configs.optimizer = optimizer

    def load_criterion(self) -> None:
        """
        Loads the criterion object from the `configs.criterion` attribute

        Returns: None

        Examples:
            Initiate the class and call the methods

            >>> helpers = Helpers(self.configs)
            >>> helpers.load_criterion()
        """

        criterion_name: str = str(self.configs.criterion).lower()
        criterion = None

        if criterion_name == "cross-entropy":
            criterion = nn.CrossEntropyLoss()

        # Add additional criterion in the elif statements

        self.configs.criterion = criterion

    def create_data_loaders(self) -> None:
        """
        Creates :class:`~torch.utils.data.DataLoader` Object for Train, Validation and Test data

        Returns:
            :class:`~torch.utils.data.DataLoader` objects for train and test datasets

        Examples:

            >>> helpers = Helpers()
            >>> helpers.create_data_loaders()
        """

        # update this line for your own transformation function
        train_transform, test_transform = cifar10_transforms()

        if self.configs.data["train"] is None:
            self.configs.data["train"] = os.path.join(self.configs.data["root"], "train")

        if self.configs.data["validation"] is None:
            self.configs.data["validation"] = os.path.join(self.configs.data["root"], "validation")

        if self.configs.data["test"] is None:
            self.configs.data["test"] = os.path.join(self.configs.data["root"], "test")

        # load datasets, downloading if needed
        train_set = CIFAR10(self.configs.data["train"], train=True, download=True,
                            transform=train_transform)
        test_set = CIFAR10(self.configs.data["test"], train=False, download=True,
                           transform=test_transform)

        train_loader = DataLoader(train_set,
                                  batch_size=self.configs.batch_size,
                                  num_workers=0)
        test_loader = DataLoader(test_set,
                                 batch_size=self.configs.batch_size,
                                 num_workers=0)

        self.configs.data = dict({"train_loader": train_loader, "test_loader": test_loader})

    def load_model(self) -> None:
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

            self.configs.model_name = model
