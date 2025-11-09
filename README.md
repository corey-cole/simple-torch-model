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

## Specific Format Notes

### AOTI

Manual inspection of the compiled so shows the following dependencies:

```
ldd simple_model_aot/data/aotinductor/model/cucx52yljzzj6ezeg42pplszc7qonhznxsqxrgl4yq3ustgtbxww.wrapper.so
	linux-vdso.so.1 (0x00007ffb6fbe4000)
	libtorch_cpu.so => not found
	libstdc++.so.6 => /lib/x86_64-linux-gnu/libstdc++.so.6 (0x00007ffb6f800000)
	libgcc_s.so.1 => /lib/x86_64-linux-gnu/libgcc_s.so.1 (0x00007ffb6fb4c000)
	libc.so.6 => /lib/x86_64-linux-gnu/libc.so.6 (0x00007ffb6f400000)
	libm.so.6 => /lib/x86_64-linux-gnu/libm.so.6 (0x00007ffb6f717000)
	/lib64/ld-linux-x86-64.so.2 (0x00007ffb6fbe6000)
```
The `not found` label for `libtorch_cpu.so` is a false negative, as the library is present in the virtual environment's lib path when active

`./lib/python3.12/site-packages/torch/lib/libtorch_cpu.so`

The pt2 archive also includes both wrapper and kernel source along with JSON metadata showing the device used for compilation.

```json
{
  "AOTI_DEVICE_KEY": "cpu"
}
```

The wrapper source appears to be mostly boilerplate for setting up execution, while the kernel is the model itself.  The compile and link commands
are both embedded as comments in the kernel cpp file.


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
