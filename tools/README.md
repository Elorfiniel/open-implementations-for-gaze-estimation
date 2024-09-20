# Tools for Open Gaze Estimation

This folder contains the main scripts to train and test the models, as well as a set of tools to help with data preparation, configuration and visualization.

## Training and Testing

This repository follows the standard training and testing procedure for all models. More specifically, we follow the philosophy of [OpenMMLab](https://openmmlab.com/). If you have any questions about the training and testing procedure, please refer to the [mmengine documents](https://mmengine.readthedocs.io/zh-cn/latest/advanced_tutorials/config.html).

To train or test the model, please follow these steps:

1. Install the requirements for the project.
2. Set `PYTHONPATH` properly.
3. Run `tools\train.py` or `tools\test.py`.

Usually, your command should look like one of the following patterns:

```shell
python tools/train.py --work-dir <runs> <config>
python tools/test.py --work-dir <runs> <config> <checkpoint>
```

## Data Preparation

We provide a set of tools to help with data preparation. These tools are located in the `tools\dataset` folder. In general, these tools takes as input the root folder of the uncompressed dataset and outputs the processed data in the `data` folder (`data/mpiigaze` for instance).

| Dataset | Home | Download | Command |
| ------- | ---- | -------- | ------- |
| MPIIGaze | [MPIIGaze](https://www.mpi-inf.mpg.de/de/departments/computer-vision-and-machine-learning/research/gaze-based-human-computer-interaction/appearance-based-gaze-estimation-in-the-wild) | [Link](http://datasets.d2.mpi-inf.mpg.de/MPIIGaze/MPIIGaze.tar.gz) | `python tools/dataset/mpiigaze.py --dataset-path <path>` |

## FAQs

In case you encounter any problem, checkout the following sections first. In case of any other questions, please feel free to open an issue.

### Python Module Search Path

`PYTHONPATH` is an environment variable used by Python to specify additional directories where the Python interpreter should look for modules and packages. When you execute a Python script, paths specified are appended in `sys.path`, which usually contains, in order, the directory of the script, directories listed in the `PYTHONPATH` variable, and the default directories where Python's standard library modules and third-party library modules are installed.

Please make sure the Python interpreter knows where to find the `template` module (consists of files in the `template` folder). We've made our best effort to ensure a seamless experience with `mmengine`, which treats the `template` module as a package extending the OpenMMLab library. Running `tools/train.py` or `tools/test.py` should work out of the box (see the corresponding files for detail).

In case of any failure, please examine whether the `PYTHONPATH` environment variable has been correctly configured.
