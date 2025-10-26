# Open Implementations for Gaze Estimation

An unofficial implementation for several gaze estimation algorithms, trained on open gaze datasets including MPIIGaze, MPIIFaceGaze, etc. This repository invokes the [mmengine](https://github.com/open-mmlab/mmengine) library under the hood. The code is organized in a modular fashion, mimicking the style of other open-source projects in the OpenMMLab family.

## Installation

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

## Quickstart

For preprocessing, please refer to the scripts in `prepare` folder.

For training, please refer to the scripts in `scripts` folder and configs in `configs` folder.

## Model List

This repository provides implementations of various state-of-the-art gaze estimation models, organized by their output space dimensionality. Each model accepts specific input modalities and generates either 2D screen coordinates or 3D gaze vectors. These implementations are based on the architectures and methodologies described in the respective original papers.

Please note that while efforts have been made to faithfully reproduce the models according to the paper descriptions, some implementation details may differ from the original authors' versions due to variations in interpretation or missing details in the publications. For definitive information about each model, we recommend referring directly to the original papers.

### 2D Gaze Estimation Models

#### ITracker
- **Paper**: Krafka, Kyle, et al. "Eye Tracking for Everyone." CVPR 2016.
- **ArXiv**: https://arxiv.org/abs/1606.05814
- **Input**:
  - face crop (B, 3, 224, 224)
  - reye crop (B, 3, 224, 224)
  - leye crop (B, 3, 224, 224)
  - face grid (B, 625)
- **Output**: point of gaze (gx, gy) with shape (B, 2)

#### AFFNet
- **Paper**: Bao, Yiwei, et al. "Adaptive Feature Fusion Network for Gaze Tracking in Mobile Tablets." arXiv 2021.
- **ArXiv**: https://arxiv.org/abs/2103.11119
- **Input**:
  - face crop (B, 3, 224, 224)
  - reye crop (B, 3, 112, 112), flip: true
  - leye crop (B, 3, 112, 112), flip: false
  - crop rect (B, 12)
- **Output**: point of gaze (gx, gy) with shape (B, 2)

### 3D Gaze Estimation Models

#### LeNet
- **Paper**: Zhang, Xucong, et al. "Appearance-Based Gaze Estimation in the Wild." CVPR 2015.
- **ArXiv**: https://arxiv.org/abs/1504.02863
- **Input**:
  - normalized eye patch (B, 1, 36, 60)
  - normalized head pose (B, 2)
- **Output**: gaze vector (pitch, yaw) with shape (B, 2)

#### GazeNet
- **Paper**: Zhang, Xucong, et al. "MPIIGaze: Real-World Dataset and Deep Appearance-Based Gaze Estimation." TPAMI 2017.
- **ArXiv**: https://arxiv.org/abs/1711.09017
- **Input**:
  - normalized eye patch (B, 3, 36, 60)
  - normalized head pose (B, 2)
- **Output**: gaze vector (pitch, yaw) with shape (B, 2)

#### DilatedNet
- **Paper**: Chen, Zhaokang, and Bertram E. Shi. "Appearance-Based Gaze Estimation Using Dilated-Convolutions." arXiv 2019.
- **ArXiv**: https://doi.org/10.48550/arXiv.1903.07296
- **Input**:
  - normalized face patch (B, 3, 96, 96)
  - normalized reye patch (B, 3, 64, 96)
  - normalized leye patch (B, 3, 64, 96)
- **Output**: gaze vector (pitch, yaw) with shape (B, 2)

#### FullFace
- **Paper**: Zhang, Xucong, et al. "It's Written All Over Your Face: Full-Face Appearance-Based Gaze Estimation." arXiv 2016.
- **ArXiv**: https://arxiv.org/abs/1611.08860
- **Input**: normalized face patch (B, 3, 224, 224)
- **Output**: gaze vector with shape (B, 2)

#### CANet
- **Paper**: Cheng, Yihua, et al. "A Coarse-to-Fine Adaptive Network for Appearance-Based Gaze Estimation." arXiv 2020.
- **ArXiv**: https://arxiv.org/abs/2001.00187
- **Input**:
  - normalized face patch (B, 3, 224, 224)
  - normalized reye patch (B, 3, 36, 60)
  - normalized leye patch (B, 3, 36, 60)
- **Output**: gaze vector (pitch, yaw) with shape (B, 2)

#### XGaze224
- **Paper**: Zhang, Xucong, et al. "ETH-XGaze: A Large Scale Dataset for Gaze Estimation under Extreme Head Pose and Gaze Variation." arXiv 2020.
- **ArXiv**: https://arxiv.org/abs/2007.15837
- **Input**: normalized face patch (B, 3, 224, 224)
- **Output**: gaze vector (pitch, yaw) with shape (B, 2)

#### GazeTR
- **Paper**: Cheng, Yihua, and Feng Lu. "Gaze Estimation Using Transformer." arXiv 2021.
- **ArXiv**: https://arxiv.org/abs/2105.14424
- **Input**: normalized face patch (B, 3, 224, 224)
- **Output**: gaze vector (pitch, yaw) with shape (B, 2)

## License

This project is released under the [MIT License](LICENSE).
