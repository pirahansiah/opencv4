# OpenCV 4 — Development Roadmap

## 12-Month Vision

Transform the OpenCV 4 C++ inference project from a build-from-source reference into a production-ready, cross-platform deep learning deployment toolkit with hardware-accelerated backends and automated CI/CD.

### Q1: Foundation & Migration
- Migrate build system to CMake Presets for multi-generator support (VS2022, Ninja, MinGW)
- Add OpenCV 4.10+ submodule pinning for reproducible builds
- Replace hardcoded file paths with configurable runtime paths (environment variables / config files)
- Add unit tests for DNN inference pipeline (GoogleTest)
- Deprecate VS2017 toolchain, target VS2022 as primary

### Q2: Backend Expansion
- Integrate ONNX Runtime 1.18+ as primary inference backend
- Add TensorRT backend support for NVIDIA GPU targets
- Implement model quantization pipeline (QDQ INT8) for edge deployment
- Create Docker-based build environment for Linux (Ubuntu 24.04) cross-compilation
- Add benchmark suite comparing CPU vs GPU vs NPU inference times

### Q3: Platform & Edge
- Add ARM64 cross-compilation support (Raspberry Pi 5, NVIDIA Jetson Orin)
- Implement OpenVINO backend for Intel CPU/iGPU/VPU targets
- Create platform-specific performance tuning guides
- Add WebAssembly build target for browser-based inference (OpenCV.js)
- Implement model caching and warm-up routines for cold-start optimization

### Q4: Production & Polish
- Add CI/CD pipeline (GitHub Actions) with automated build, test, and release
- Implement streaming video inference pipeline with GStreamer integration
- Add multi-model ensemble inference support
- Create comprehensive API documentation with Doxygen
- Release v1.0 with semantic versioning and changelog

## Technical Debt

| Item | Priority | Impact | Effort |
|------|----------|--------|--------|
| Hardcoded file paths in opencvtest.cpp | High | Breaks portability | Low |
| No build automation (manual CMake steps) | High | Reduces adoption | Medium |
| Missing error handling in DNN pipeline | Medium | Silent failures | Low |
| No cross-platform support (Windows only) | High | Limits user base | High |
| Outdated VS2017 toolchain dependency | Medium | Security, compatibility | Medium |
| No test suite | High | Regression risk | Medium |
| No CI/CD pipeline | Medium | Manual releases | Medium |
| Missing .gitignore for build artifacts | Low | Repo bloat | Low |

## Future Features

| Feature | Description | Priority |
|---------|-------------|----------|
| ONNX Runtime Backend | Primary inference engine with CPU/GPU/NPU support | High |
| TensorRT Integration | NVIDIA GPU-accelerated inference with INT8 quantization | High |
| Video Stream Pipeline | Real-time inference on video files and RTSP streams | High |
| Multi-model Ensemble | Run multiple models in parallel for improved accuracy | Medium |
| WebAssembly Build | Browser-based inference via OpenCV.js | Medium |
| Python Bindings | PyPI package for Python 3.12+ integration | Medium |
| Model Zoo | Curated collection of pre-quantized ONNX models | Medium |
| REST API Server | HTTP endpoint for model inference (FastAPI + ONNX Runtime) | Low |
| Edge Deployment Kit | Pre-configured images for Jetson, Raspberry Pi, Intel NUC | Low |
| Benchmark Dashboard | Web-based performance monitoring across hardware targets | Low |
