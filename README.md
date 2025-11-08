# Simple Model Serialization

## Background

The purpose of this model and project is to explore the different export mechanisms within PyTorch.
The goal is to eventually identify and document a robust model export process that can be used within a CI/CD
pipeline for applications that use a JVM to perform the inference.

## Current State

Exports for the TorchScript, `torch.export`, ONNX, and executorch for XNNPACK appear to work.
Inference on loaded model has been tested for every API except TorchScript.

Also, torch==2.9.0 adds the new `torch.export.draft_export` API which might be a useful option to include
behind a command line switch.  This project is hard-coded to use torch 2.8.0 and executorch 0.7.0.  Forwards and backwards compatibility requires additional testing.

Finally, for CPU-only execution when the operating environment is guaranteed to be Intel hardware,
an ONNX model can be compiled using [OpenVINO](https://docs.openvino.ai/2025/get-started/install-openvino/install-openvino-pip.html).

## Future Direction

The list below is an non-exclusive collection of additional export options that are being pondered.

* OpenVINO IR: OpenVINO has optimizations for newer CPUs (mostly Intel, but there is some support like AVX2 that would apply to AMD as well)
* AOTInductor:  The model is compiled into a shared object that can be natively loaded by PyTorch

## References

* [Executorch on GitHub](https://github.com/pytorch/executorch)
* [Executorch docs (v0.7)](https://docs.pytorch.org/executorch/0.7/getting-started.html)
* [torch.export API docs](https://docs.pytorch.org/docs/2.8/export.html)
* [torch.export.draft_export API docs](https://docs.pytorch.org/docs/2.9/export/draft_export.html)
* [torch export programming model](https://docs.pytorch.org/docs/stable/export/programming_model.html)
* [XNNPACK on GitHub](https://github.com/google/XNNPACK)
* [ONNX export docs](https://docs.pytorch.org/docs/stable/onnx_export.html)
* [ONNX control flow tutorial](https://docs.pytorch.org/tutorials/beginner/onnx/export_control_flow_model_to_onnx_tutorial.html)
* [ONNX project site](https://onnxruntime.ai/)
* [ONNX EP w/ xnnpack for Linux](https://onnxruntime.ai/docs/build/eps.html#build-for-linux)
* [Netron](https://netron.app/)
* [PyTorch wrapper for optimizing CPU execution](https://docs.pytorch.org/tutorials/recipes/xeon_run_cpu.html)
* [AOTI Export](https://docs.pytorch.org/tutorials/recipes/torch_export_aoti_python.html)
* [OpenVINO IR conversion](https://github.com/openvinotoolkit/openvino_notebooks/tree/latest/notebooks/vision-background-removal)
