# Open Implementations for Gaze Estimation

An unofficial implementation for several gaze estimation algorithms, trained on open gaze datasets including MPIIGaze, MPIIFaceGaze, etc. This repository invokes the [mmengine](https://github.com/open-mmlab/mmengine) library under the hood. The code is organized in a modular fashion, mimicking the style of other open-source projects in the OpenMMLab family.

## Quick Start

Follow these steps to get started:

1. Create a python virtual environment for dependencies (optional).

    ```shell
    python -m venv --upgrade-deps open-gaze

    open-gaze/Scripts/activate.bat # (windows: cmd)
    open-gaze/Scripts/Activate.ps1 # (windows: pwsh)
    source open-gaze/bin/activate  # (linux / mac)
    ```

2. Install the dependencies.

    ```shell
    pip install -r requirements.txt
    ```

3. Install the `template` package in development mode.

    ```shell
    pip install --editable .
    ```

4. Start training or testing, read more about [Tools for Open Gaze Estimation](tools/README.md).

## FAQs

In case of any problem, checkout the following sections first. In case of any other questions, please feel free to open an issue.

### Module Not Found Error

In general, the `template` package should be visible to the Python interpreter. We achieve this by installing the package in development mode. Please make sure the Python interpreter knows where to find the `template` package (consists of files in the `template` folder). If you are using a virtual environment, make sure you've activated the virtual environment before installation.

Alternatively, you can set the `PYTHONPATH` environment variable to include the parent directory of the `template` folder.

## License

This project is released under the [MIT License](LICENSE).
