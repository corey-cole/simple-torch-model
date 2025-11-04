from __future__ import annotations

import click
import torch

from executorch.runtime import Runtime

from simple_model.cli.common import ExportFormat
from simple_model.runner import ModelRunner


@click.command(context_settings={"show_default": True})
@click.option(
    "--format",
    type=click.Choice(ExportFormat),
    default=ExportFormat.TORCHSCRIPT,
    help="Export format.",
)
@click.option(
    "--input-path",
    type=click.Path(),
    default="output/simple_model.pte",
    help="Path to the saved model.",
)
def main(format: ExportFormat, input_path: str) -> None:
    print(f"Loading model from {input_path} in {format.value} format.")

    runner = ModelRunner()
    match format:
        case ExportFormat.XNNPACK:
            runner.run_xnnpack_model(input_path)
        case ExportFormat.ONNX:
            runner.run_onnx_model(input_path)
        case ExportFormat.TORCH_EXPORT:
            runner.run_torch_export_model(input_path)
        case _:
            raise ValueError(f"Unsupported export format: {format}")

