# OpenCV 4 — Deep Learning for Computer Vision

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-blue.svg)](https://opencv.org)
[![Visual Studio](https://img.shields.io/badge/Visual%20Studio-2017-purple.svg)](https://visualstudio.microsoft.com)
[![Platform](https://img.shields.io/badge/Platform-Windows%2010%20x64-lightgrey.svg)](#)
[![YouTube](https://img.shields.io/badge/YouTube-Tutorials-red.svg)](https://www.youtube.com/tiziran)

OpenCV 4 with Deep Learning model inference (TensorFlow, Caffe) for Visual Studio 2017 (C++) on Windows 10 x64.

> **Project by Dr. Farshid Pirahansiah** — [www.tiziran.com](https://www.tiziran.com) | [YouTube](https://www.youtube.com/tiziran)

## What's Included

- OpenCV 4.x compiled from source (main + contrib modules)
- Visual Studio 2017 C++ project configuration
- Example code for loading and running Caffe and TensorFlow models
- PDF presentation with step-by-step setup guide

## Quick Start

### 1. Build OpenCV 4 from Source

```bash
# Clone OpenCV 4.x
git clone https://github.com/opencv/opencv.git
git clone https://github.com/opencv/opencv_contrib.git

# Build with CMake (enable DNN, contrib)
cmake -D CMAKE_BUILD_TYPE=Release \
      -D OPENCV_EXTRA_MODULES_PATH=../../opencv_contrib/modules \
      -D BUILD_opencv_dnn=ON \
      -G "Visual Studio 15 2017 Win64" ..
```

### 2. Configure Visual Studio 2017

- Set include path to OpenCV headers
- Set library path to built `.lib` files
- Set DLL path in system PATH

### 3. Run Deep Learning Models

See `opencvtest.cpp` for examples using Caffe and TensorFlow models with OpenCV's DNN module.

## Video Tutorials

| Topic | Link |
|-------|------|
| Compile OpenCV 4 from source | [Watch](https://www.youtube.com/watch?v=VK70YdaMD44) |
| VS2017 project setup | [Watch](https://www.youtube.com/watch?v=J0Exttz4_m4) |
| Using Caffe models | [Watch](https://www.youtube.com/watch?v=lj7AxKDoP9I) |
| Using TensorFlow models | [Watch](https://www.youtube.com/watch?v=5JqX0CbxtNk) |

## Presentation

[View Google Slides presentation](https://docs.google.com/presentation/d/12OqgsInMveeGbJYPnNOG8jwMQbp76DjqK4_NZtJE7bA/edit?usp=sharing)

## Deep Learning with OpenCV 4 — 2025-2026 State of the Art

### Model Format Support

| Format | Status | Notes |
|--------|--------|-------|
| ONNX | Preferred | Cross-platform, best supported in 2025+ |
| TensorFlow Lite | Supported | Edge/mobile deployment |
| Caffe | Legacy | Still functional, no longer actively developed |
| Darknet (YOLO) | Supported | Real-time object detection |
| Torch | Supported | PyTorch model export via ONNX |

### Recommended Inference Backends (2025-2026)

| Backend | Best For | Speed |
|---------|----------|-------|
| **ONNX Runtime** | Cross-platform CPU/GPU | 2-10x over OpenCV DNN |
| **NVIDIA TensorRT** | NVIDIA GPU production | 5-20x over CPU |
| **OpenVINO** | Intel CPU/iGPU/VPU | 2-4x on Intel hardware |
| **OpenCV DNN + CUDA** | Quick GPU offload | 3-8x over CPU |
| **Qualcomm QNN** | Snapdragon edge devices | Optimized for mobile |

### Modern Architecture Recommendations

1. **Export models to ONNX** — Universal format, best tooling support
2. **Use quantization (QDQ INT8)** — 4x compression, <1% accuracy loss
3. **Deploy with ONNX Runtime** — Cross-platform, hardware-optimized
4. **Edge AI** — NVIDIA Jetson (TensorRT), Raspberry Pi (NCNN), OpenVINO

## Related Projects

- [OpenCV 3.x (VS2015)](https://github.com/pirahansiah/opencv)
- [OpenCV 5 (VS2022)](https://github.com/pirahansiah/opencv5vs2022)
- [OpenCV Python Workshop](https://github.com/pirahansiah/opencv_python)

## Resources

- [OpenCV DNN Module Documentation](https://docs.opencv.org/4.x/d6/d0f/tutorial_dnn_bottom.html)
- [ONNX Runtime](https://onnxruntime.ai)
- [TensorRT Developer Guide](https://developer.nvidia.com/tensorrt)
- [OpenVINO Documentation](https://docs.openvino.ai)
- [YouTube Channel](https://www.youtube.com/tiziran)

## License

See repository for license details.
