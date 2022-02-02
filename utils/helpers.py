from types import SimpleNamespace


class Helpers:
    def __init__(self, configs: SimpleNamespace):
        self.configs = configs

    def load_optimizer(self):
        # TODO: check if the model was previously loaded

        optimizer = self.configs.optimizer
        match optimizer:
            case "adam":
                pass
            case "sgd":
                pass

        raise NotImplementedError("Load Optimizer hasn't been implemented yet!")

    def load_criterion(self):
        raise NotImplementedError("Load Criterion hasn't been implemented yet!")

    def load_data_loaders(self):
        raise NotImplementedError("Load DataLoader hasn't been implemented yet!")

    def load_model(self):
        raise NotImplementedError("Load DataLoader hasn't been implemented yet!")

