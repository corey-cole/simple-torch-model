import enum


class ExportFormat(enum.Enum):
    TORCHSCRIPT = "torchscript"
    TORCH_EXPORT = "torch_export"
    ONNX = "onnx"
    XNNPACK = "xnnpack"
