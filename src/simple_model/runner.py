from __future__ import annotations

import onnxruntime as ort
import torch

from executorch.runtime import Runtime


from simple_model.cli.common import ExportFormat

# This class runs the exported SimpleModel or ConditionalModel.  It doesn't take input
# and instead uses a random tensor for demonstration purposes.
class ModelRunner:
    def __init__(self):
        pass

    def run_xnnpack_model(self, input_path: str) -> None:
        runtime = Runtime.get()
        input_tensor = torch.randn(1, 10)
        program = runtime.load_program(input_path)
        method = program.load_method("forward")
        if method is None:
            raise RuntimeError("Failed to load 'forward' method from program.")
        output = method.execute([input_tensor])
        print(f"Model output: {output}")
        return
    
    def run_onnx_model(self, input_path: str) -> None:
        input_tensor = torch.randn(1, 10).numpy()
        session = ort.InferenceSession(input_path)
        input_name = session.get_inputs()[0].name
        outputs = session.run(None, {input_name: input_tensor})
        print(f"Model output: {outputs[0]}")
        return
    
    def run_torch_export_model(self, input_path: str) -> None:
        loaded_model = torch.export.load(input_path)
        input_tensor = torch.randn(1, 10)
        module = loaded_model.module()
        if module is None:
            raise RuntimeError("Failed to get module from loaded model.")
        if not hasattr(module, "forward") or not callable(module.forward):
            raise RuntimeError("Loaded module does not have a 'forward' method.")
        output = module.forward(input_tensor)
        print(f"Model output: {output}")
        return

