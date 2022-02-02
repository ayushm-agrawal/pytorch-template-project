import json
from types import SimpleNamespace
from utils.helpers import Helpers


def load_configs(path: str = "./configs.json") -> SimpleNamespace:
    """
    Function loads the configs.json file as a SimpleNamespace providing
    access to the attributes
    :param path: The relative/absolute path to the configs.json file
    :return: SimpleNamespace object of configs
    """
    configs = SimpleNamespace(**json.load(open(path)))

    # TODO: load appropriate configs objects from the dictionary
    helpers = Helpers(configs)
    for attribute in configs.__dict__.keys():
        if attribute == "optimizer":
            configs.optimizer = helpers.load_optimizer()
        if attribute == "criterion":
            configs.criterion = helpers.load_criterion()
        if attribute == "data_path":
            configs.data_path = helpers.load_data_loaders()
        if attribute == "model_name":
            configs.data_path = helpers.load_model()

    return configs
