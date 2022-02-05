# Pytorch Template
[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://GitHub.com/Naereen/StrapDown.js/graphs/commit-activity)
[![PyPI pyversions](https://img.shields.io/badge/python-3.8-blue)](https://img.shields.io/badge/python-3.8-blue)
[![Torch version](https://img.shields.io/badge/torch-1.10.2-orange)](https://img.shields.io/badge/torch-1.10.2-orange)
[![Code Coverage](coverage.svg)](coverage.svg)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)


This project serves as a basic template for any Pytorch project. 

<b>The idea is to `standardize code` structures as much as possible for various projects</b>

___

## How to use

- Clone/Fork the repository and use the structure below as a reference to add/update/delete code.
- Use the `requirements.txt` file to add all the dependencies in your own virtual environment.

___

## Repository Structure
```
pytorch-template-project
├── models
│   └── alexnet.py
│ 
├── preprocess
│   └── transforms.py 
│ 
├── test
│   └── test_helpers.py
│ 
├── utils
│   ├── helpers.py
│   └── setup_configs.py
│ 
├── .coverage
│ 
├── .gitignore
│ 
├── configs.json
│ 
├── coverage.svg
│ 
├── main.py
│ 
├── README.md
│ 
└── requirements.txt
```

___

## Explanation of each file

[![Under Construction](https://img.shields.io/badge/Under-Construction-red)](https://img.shields.io/badge/Under-Construction-red)

<img src="https://freevector-images.s3.amazonaws.com/uploads/vector/preview/40797/FreeVectorConstructionIllustrationyc1121_generated.jpg" alt="drawing" style="width:200px;"/>

---
## Update the configs.json file to adjust the hyper parameters:

> This is just a base config file that I have provided. Please feel free to add additional configs attributes as needed.
> Although, I highly recommend not to change the names of the attributes unless necessary. 
> Certain attributes like `optimizer`, `criterion`, `data`, `model_name` are being used to load their specific Object types.
> If you must change the attribute names, then update the [setup_configs.py](utils/setup_configs.py) 
> and [helpers.py](utils/helpers.py) 
> to use the updated attribute names.

- `experiment_name` : The name of the experiment
- `model_name` : The architecture to use. Architectures included in the code are 
  - Options : `AlexNet`
  - Default : `alexnet`
- `optimizer` : The optimizer to use for the experiment
  - Options : `adam`, `sgd`
  - Default : `sgd`
- `criterion` : The Loss function to use for the experiment
  - Default : `cross-entropy`
- `num_epochs` : Number of epochs to train the model for
  - Default : `10` 
- `num_classes` : The number of classes for classification
  - Default : `10`
- `input_channels` : The input channels for the model
  - Default : `3`
- `learning_rate` : The learning rate to use for the experiments. We have not used Adaptive learning rate for the simplicity in interpreting the trends
  - Default : `0.001`
- `momentum` : Used for the optimizer
  - Default : `0.9`
- `weight_decay` : Used for the optimizer
  - Default : `1e-5`
- `batch_size` : The number of images per batch in the training dataset
  - Default : `32`
- `data` : (dict) The path for the dataset
  - `root` : Path for the root directory
  - `train`: **Relative Path** for the train folder from the root
  - `validation` : **Relative Path** for the validation folder from the root
  - `test` : **Relative Path** for the test folder from the root
- `seed` : This initializes different weights for each experiment. We average over the results to determine a general learning trend
  - Options : `Length of list defines the number of different experiments for a given arch and dataset`
  - Default : `[0, 123, 420, 69]`
- `model_save_path` : Path where the model will be saved after training
  - Default : `./trained_models/`

___

## Issues or Want to Contribute?

- If you have any issues, then create the issue with the fix you have one and create a pull request.
- If you have new features, submit a pull request, and I will review it.

[![For The Badge](https://forthebadge.com/images/badges/built-with-love.svg)](https://forthebadge.com)

