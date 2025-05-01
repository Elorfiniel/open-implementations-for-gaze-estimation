# External Resources

This folder contains external resources used in the project. Please manually download these resources and place them in this folder. See below for details:

```text
resource/
├── 3ddfa-v2/
│   ├── bfm-noneck-v3.pkl
│   ├── param-mean-std.pkl
│   └── tddfa-v2-mb1.onnx
├── face-models/
│   └── mpiigaze-generic.mat
└── mediapipe/
    ├── face_detection.tflite
    └── face_landmarker.task
```

## Download Links

You can download the external resources from the following links (please rename the files according to the folder structure mentioned above):

- [3ddfa-v2/bfm-noneck-v3.pkl](https://github.com/cleardusk/3DDFA_V2/blob/master/configs/bfm_noneck_v3.pkl): download the `bfm_noneck_v3.pkl` file from 3DDFA-v2.

- [3ddfa-v2/param-mean-std.pkl](https://github.com/cleardusk/3DDFA_V2/blob/master/configs/param_mean_std_62d_120x120.pkl): download the `param_mean_std_62d_120x120.pkl` file from 3DDFA-v2.

- [3ddfa-v2/tddfa-v2-mb1.onnx](https://github.com/cleardusk/3DDFA_V2/tree/master/weights): download the pre-converted `mb1_120x120.onnx` file from 3DDFA-v2.

- [face-models/mpiigaze-generic.mat](http://datasets.d2.mpi-inf.mpg.de/MPIIGaze/MPIIGaze.tar.gz): extract the `6 points-based face model.mat` file from MPIIGaze dataset.

- [mediapipe/face_detection.tflite](https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/latest/blaze_face_short_range.tflite): download the `blaze_face_short_range.tflite` file from MediaPipe.

- [mediapipe/face_landmarker.task](https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task): download the `face_landmarker.task` file from MediaPipe.
