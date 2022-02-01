from types import SimpleNamespace


class Helpers:
    def __init__(self, configs: SimpleNamespace):
        self.configs = configs

    def load_optimizer(self):
        optimizer = self.configs.optimizer
        match optimizer:
            case "adam":
                pass
            case "sgd":
                pass

        raise NotImplementedError("Load Optimizer hasn't been implemented yet!")