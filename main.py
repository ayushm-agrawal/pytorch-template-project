import utils.setup_configs as conf


def main():
    # load configs
    path_to_configs = "./configs.json"
    configs = conf.load_configs(path_to_configs)


if __name__ == '__main__':
    main()
