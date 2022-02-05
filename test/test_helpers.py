import json
import os.path
from types import SimpleNamespace

import pytest
import torch.nn as nn
import torch.optim
from torch.utils.data import DataLoader

import models.alexnet
from utils.helpers import Helpers


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
    optimizer = helpers.load_optimizer()
    assert isinstance(optimizer, torch.optim.Adam)

    # testing for sgd
    configs.optimizer = "sgd"
    optimizer = helpers.load_optimizer()
    assert isinstance(optimizer, torch.optim.SGD)


def test_load_model(configs):
    helpers = Helpers(configs)
    assert isinstance(configs.model_name, str)

    model = helpers.load_model()
    assert isinstance(model, models.alexnet.AlexNet)


def test_load_criterion(configs):
    helpers = Helpers(configs)

    assert isinstance(configs.criterion, str)

    criterion = helpers.load_criterion()

    assert isinstance(criterion, nn.CrossEntropyLoss)


def test_create_data_loader(configs):
    helpers = Helpers(configs)

    assert configs.data['train'] is None
    assert configs.data['test'] is None
    assert configs.data['validation'] is None

    data = helpers.create_data_loaders()

    assert isinstance(data, dict)
    assert isinstance(data['train_loader'], DataLoader)
    assert isinstance(data['test_loader'], DataLoader)
