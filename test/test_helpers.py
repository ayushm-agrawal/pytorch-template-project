import os.path

import pytest
import json
import shutil
from types import SimpleNamespace

import torch.optim
import torch.nn as nn

import models.alexnet
from utils.helpers import Helpers
from torch.utils.data import DataLoader


@pytest.fixture
def configs():
    configs_path = "configs.json"

    configs = SimpleNamespace(**json.load(open(os.path.abspath(configs_path))))
    return configs


def test_load_optimizer(configs):
    helpers = Helpers(configs)
    assert isinstance(configs.optimizer, str)
    assert isinstance(configs.model_name, str)
    # testing for adam
    helpers.load_optimizer()
    assert isinstance(configs.model_name, nn.Module)
    assert isinstance(configs.optimizer, torch.optim.Adam)

    # testing for sgd
    configs.optimizer = "sgd"
    helpers.load_optimizer()
    assert isinstance(configs.model_name, nn.Module)
    assert isinstance(configs.optimizer, torch.optim.SGD)


def test_load_model(configs):
    helpers = Helpers(configs)
    assert isinstance(configs.model_name, str)

    helpers.load_model()
    print(configs.model_name.__class__)
    assert isinstance(configs.model_name, models.alexnet.AlexNet)


def test_load_criterion(configs):
    helpers = Helpers(configs)

    assert isinstance(configs.criterion, str)

    helpers.load_criterion()

    assert isinstance(configs.criterion, nn.CrossEntropyLoss)


def test_create_data_loader(configs):
    helpers = Helpers(configs)

    assert configs.data['train'] is None
    assert configs.data['test'] is None
    assert configs.data['validation'] is None

    helpers.create_data_loaders()

    assert isinstance(configs.data, dict)
    assert isinstance(configs.data['train_loader'], DataLoader)
    assert isinstance(configs.data['test_loader'], DataLoader)

