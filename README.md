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

3. Install the `opengaze` package in development mode.

    ```shell
    pip install --editable .
    ```

4. Download external resources following these [instructions](resource/README.md).

## License

This project is released under the [MIT License](LICENSE).
