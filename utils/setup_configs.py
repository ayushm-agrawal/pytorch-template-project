import json
from types import SimpleNamespace


def load_configs(path: str = "./configs.json") -> SimpleNamespace:
    """
    Function loads the configs.json file as a SimpleNamespace providing
    access to the attributes
    :param path: The relative/absolute path to the configs.json file
    :return: SimpleNamespace object of configs
    """
    configs = SimpleNamespace(**json.load(open(path)))

    # TODO: load appropriate configs objects from the dictionary

    return configs
