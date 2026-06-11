# Quantization Notes

PyTorch dynamic quantization targets linear layers and stores weights as INT8 values. It is most
useful for CPU inference because it reduces model size and memory bandwidth during matrix
multiplication.

ONNX export provides a portable graph that can run with ONNX Runtime. The project exports encoder
backbones with dynamic batch and sequence axes, then applies ONNX Runtime dynamic quantization to
produce a smaller model artifact.

For edge deployments, the recommended workflow is to benchmark five configurations: baseline
PyTorch FP32, PyTorch INT8 dynamic quantization, ONNX FP32, ONNX INT8, and the hashing fallback for
emergency no-model operation. Track p50 latency, p95 latency, memory footprint, and NDCG@10.
