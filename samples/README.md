# Data format
## Introduction
> The Data is used for the temperature field reconstruction task of heat source systems. Default save path: `samples/data`

## Structures

* `data`: All the data for the training, validation, testing process
  * `train/`
    * `train/`: save path for training data
    * `train_val.txt`: training samples where the training process is using. 80% of the data is for training, 20% of the data is for validation.
  * `test/`
    * `test/`: save path for testing data
    * `test.txt`: testing samples where the testing process is using.

## Others

* Researchers can modify the `data_root` in `config/config.yml` to the data path where the data is saved.