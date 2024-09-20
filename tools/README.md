# Tools for Open Gaze Estimation

This folder contains the main scripts to train and test the models, as well as a set of tools to help with data preparation, configuration and visualization.

## Training and Testing

This repository follows the standard training and testing procedure for all models. More specifically, we follow the philosophy of [OpenMMLab](https://openmmlab.com/). If you have any questions about the training and testing procedure, please refer to the [mmengine documents](https://mmengine.readthedocs.io/zh-cn/latest/advanced_tutorials/config.html).

To train or test the model, please run `tools\train.py` or `tools\test.py`. Usually, your command should follow one of the patterns listed below:

```shell
python tools/train.py --work-dir <runs> <config>
python tools/test.py --work-dir <runs> <config> <checkpoint>
```

If new changes are made to the `template` package, you need to reinstall it (in development mode) for the modification to take effect. As a quick reference, run the following command (see also [setuptools user guide](https://setuptools.pypa.io/en/stable/userguide/quickstart.html)):

```shell
cd <folder-for-setup-py>
pip install --editable .
```

## Data Preparation

We provide a set of tools to help with data preparation. These tools are located in the `tools\dataset` folder. In general, these tools takes as input the root folder of the uncompressed dataset and outputs the processed data in the `data` folder (`data/mpiigaze` for instance).

| Dataset | Home | Download | Command |
| ------- | ---- | -------- | ------- |
| MPIIGaze | [MPIIGaze](https://www.mpi-inf.mpg.de/de/departments/computer-vision-and-machine-learning/research/gaze-based-human-computer-interaction/appearance-based-gaze-estimation-in-the-wild) | [Link](http://datasets.d2.mpi-inf.mpg.de/MPIIGaze/MPIIGaze.tar.gz) | `python tools/dataset/mpiigaze.py --dataset-path <path>` |
