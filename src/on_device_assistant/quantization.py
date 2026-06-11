from __future__ import annotations

from pathlib import Path


def export_transformer_encoder_to_onnx(
    model_name: str,
    output_path: Path,
    opset: int = 17,
    max_length: int = 128,
) -> Path:
    """Export a Hugging Face encoder backbone to ONNX for local inference."""

    import torch
    from transformers import AutoModel, AutoTokenizer

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()

    inputs = tokenizer(
        "on-device retrieval augmented generation",
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )

    torch.onnx.export(
        model,
        (inputs["input_ids"], inputs["attention_mask"]),
        str(output_path),
        input_names=["input_ids", "attention_mask"],
        output_names=["last_hidden_state"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "sequence"},
            "attention_mask": {0: "batch", 1: "sequence"},
            "last_hidden_state": {0: "batch", 1: "sequence"},
        },
        opset_version=opset,
    )
    return output_path


def quantize_onnx_dynamic(input_path: Path, output_path: Path) -> Path:
    """Create an INT8 dynamic-quantized ONNX model."""

    from onnxruntime.quantization import QuantType, quantize_dynamic

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    quantize_dynamic(
        model_input=str(input_path),
        model_output=str(output_path),
        weight_type=QuantType.QInt8,
    )
    return output_path


def quantize_torch_dynamic(model_name: str, output_path: Path) -> Path:
    """Dynamically quantize Linear layers and save a Torch state dict."""

    import torch
    from transformers import AutoModel

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model = AutoModel.from_pretrained(model_name)
    model.eval()
    quantized = torch.quantization.quantize_dynamic(
        model,
        {torch.nn.Linear},
        dtype=torch.qint8,
    )
    torch.save(quantized.state_dict(), output_path)
    return output_path
