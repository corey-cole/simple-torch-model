import enum
import os

import click

from simple_model.cli.common import ExportFormat
from simple_model.exporter import ModelExporter
from simple_model.model import ConditionalModel, SimpleModel


@click.command(context_settings={"show_default": True})
@click.option(
    "--format",
    type=click.Choice(ExportFormat),
    default=ExportFormat.TORCHSCRIPT,
    help="Export format.",
)
@click.option(
    "--output-path",
    type=click.Path(),
    default="output/model.pt",
    help="Path to save the exported model.",
)
def main(format: ExportFormat, output_path: str) -> None:
    print(f"Exporting model to {output_path} in {format.value} format.")
    # Make the output path directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    if format == ExportFormat.TORCHSCRIPT:
        exporter = ModelExporter()  # Use default of SimpleModel
    else:
        exporter = ModelExporter(ConditionalModel())  # Use ConditionalModel
    match format:
        case ExportFormat.TORCHSCRIPT:
            exporter.export_torchscript(output_path)
        case ExportFormat.TORCH_EXPORT:
            exporter.export_torch_export(output_path)
        case ExportFormat.ONNX:
            exporter.export_onnx(output_path)
        case ExportFormat.XNNPACK:
            exporter.export_xnnpack(output_path)
        case _:
            raise ValueError(f"Unsupported export format: {format}")
