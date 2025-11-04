# This class is the interface between the CLI and the different Torch export APIs
from __future__ import annotations
from pprint import pprint
import torch
import torch.nn as nn
import torch.onnx.verification

from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
from executorch.exir import to_edge_transform_and_lower
from simple_model.model import SimpleModel


class ModelExporter:
    def __init__(
        self,
        model: nn.Module | None = None,
        example_input: torch.Tensor | None = None,
    ):
        if model is None:
            self.model = SimpleModel()
        else:
            self.model = model

        if example_input is None:
            self.example_input = torch.randn(1, 10)
        else:
            self.example_input = example_input

    def export_torchscript(self, file_path: str) -> None:
        scripted_model = torch.jit.script(self.model)
        scripted_model.save(file_path)

    def export_onnx(self, file_path: str) -> None:
        # ONNX export uses torch.export internally
        # API docs recommend using f=None (the default, but setting explicitly here for clarity)
        # and then calling save(...) on the resulting torch.onnx.ONNXProgram
        onnx_program = torch.onnx.export(
            self.model,
            (self.example_input,),
            f=None,
            input_names=["input"],
            dynamo=True,
        )
        if onnx_program is None:
            raise RuntimeError("ONNX export failed")
        onnx_program.optimize()
        onnx_program.save(file_path)

        verification_results = torch.onnx.verification.verify_onnx_program(
            onnx_program,
            (self.example_input,),
        )
        print("ONNX verification results:")
        pprint(verification_results)


    def export_torch_export(self, file_path: str) -> None:
        exported_model = torch.export.export(self.model, (self.example_input,))
        # https://docs.pytorch.org/docs/stable/export/api_reference.html#torch.export.save
        # If the exported model is saved to an io.BytesIO entity, then extra data can be added
        # to the output via the extra_files named argument (Not demonstrated here)
        torch.export.save(exported_model, file_path)

    def export_xnnpack(self, file_path: str) -> None:
        # The documentation is unclear on the need for putting the model into eval mode
        self.model.eval()
        exported_program = torch.export.export(self.model, (self.example_input,))
        # Failure to run decompositions will result in the following error message:
        """
        RuntimeError: Node aten_linear_default with op <EdgeOpOverload: aten.linear.default>: schema = aten::linear(Tensor input, Tensor weight, Tensor? bias=None) -> Tensor was not decomposed or delegated.
        """
        with torch.no_grad():
            exported_program = exported_program.run_decompositions(decomp_table=None)
        # The default decomposition should be into the core Aten IR
        # Check for aten.linear as a single node
        print(exported_program.graph_module.print_readable(print_output=False))
        et_program = to_edge_transform_and_lower(
            exported_program,
            partitioner=[XnnpackPartitioner()],
        ).to_executorch()

        with open(file_path, "wb") as f:
            f.write(et_program.buffer)
