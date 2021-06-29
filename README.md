# reconstruction
## Introduction
This project provides the implementation of the paper "Physics-Informed Deep Reversible Regression Model for Temperature Field Reconstruction of Heat-Source Systems". [[paper](https://arxiv.org/abs/2106.11929)]

## Requirements

    * Software
        * python
        * cuda
        * pytorch
    * Hardware
        * GPU with at least 4GB

## Environment construction

```python
pip install -r requirements.txt
```

## Running

The training, test and visualization can be accessed by running `main.py` file.

- The data root is put in `data_root` in configuration file `config/config.yml` .

- Training

  ```
  python main.py -m train
  ```

  or

  ```
  python main.py --mode=train
  ```

- Test

  ```
  python main.py -m test --test_check_num=21
  ```

  or

  ```
  python main.py --mode=test --test_check_num=21
  ```

  or

  ```
  python main.py -m=test -v=21
  ```

  where variable `test_check_num` is the number of the saved model for test.

- Prediction visualization

  ```
  python main.py -m plot --test_check_num=21
  ```

  or

  ```
  python main.py --mode=plot --test_check_num=21
  ```

  or

  ```
  python main.py -m=test -v=21
  ```

  where variable `test_check_num` is the number of the saved model for plotting.

## Project architecture

- `config`: the configuration file
  - `data.yml` describes the setups of the layout domain and heat sources
  - `config.yml` describes other configurations
- `notebook`: the test file for `notebook`
- `outputs`: the output results by `test` and `plot` module. The test results is saved at `outputs/*.csv` and the plotting figures is saved at `outputs/predict_plot/`.
- `src`: including surrogate model, training and testing files.
  - `test.py`: testing files.
  - `train.py`: training files.
  - `plot.py`: prediction visualization files.
  - `DeepRegression.py`: Model configurations.
  - `data`: data preprocessing and data loading files.
  - `models`: DNN surrogate models for the TFR-HSS task.
  - `loss`: physics-informed losses for training.
  - `utils`: useful tool function files.

* `docker`: start with docker.
* `lightning_logs`: saved models.

## One tiny example

One tiny example for training and testing can be accessed based on the following instruction.

- Some training and testing data are available at `samples/data`.
- Based on the original configuration file, run `python main.py` directly for a quick experience of this tiny example.

## Citing this work

If you find this work helpful for your research, please consider citing:

```
@article{gong2021,
    Author = {Zhiqiang Gong and Weien Zhou and Jun Zhang and Wei Peng and Wen Yao},
    Title = {Physics-Informed Deep Reversible Regression Model for Temperature Field Reconstruction of Heat-Source Systems},
    Booktitle = {arXiv preprint arXiv:2106.11929},
    Year = {2021}
}
```