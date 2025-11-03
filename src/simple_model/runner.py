from __future__ import annotations

import click
import torch

from executorch.runtime import Runtime

from simple_model.cli.common import ExportFormat


def main(format: ExportFormat, input_path: str) -> None:
    print(f"Loading model from {input_path} in {format.value} format.")

    match format:
        # TODO: Move this out of the CLI module
        # TODO: Execute the same input_tensor against the original model for comparison
        case ExportFormat.XNNPACK:
            runtime = Runtime.get()
            input_tensor = torch.randn(1, 10)
            program = runtime.load_program(input_path)
            method = program.load_method("forward")
            if method is None:
                raise RuntimeError("Failed to load 'forward' method from program.")
            output = method.execute([input_tensor])
            print(f"Model output: {output}")
        case _:
            raise ValueError(f"Unsupported export format: {format}")
