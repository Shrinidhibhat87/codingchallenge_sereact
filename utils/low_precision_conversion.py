"""
Utility file to convert the model into lower precision model format for deployment.
"""

import os

import tensorrt as trt
import torch
from omegaconf import DictConfig


def convert_model_to_low_precision(
    cfg: DictConfig, model: torch.nn.Module, DEVICE: torch.device
) -> None:
    """
    Convert the trained model to low-precision formats (ONNX and TensorRT).

    Args:
        cfg (DictConfig): Configuration parameters.
        model (torch.nn.Module): The trained model to convert.
        DEVICE (torch.device): The device the model is currently on.
    """
    try:
        # Ensure the model is in evaluation mode
        model.eval()

        # Create output directory if it doesn't exist
        output_dir = cfg.output_folder_path
        os.makedirs(output_dir, exist_ok=True)

        # Export model to ONNX
        onnx_path = os.path.join(output_dir, 'model.onnx')
        print(f'Exporting model to ONNX format at {onnx_path}')

        # Simply a placeholder input for the model
        dummy_input = torch.randn(1, 3, 224, 224).to(DEVICE)
        torch.onnx.export(
            model,
            dummy_input,
            onnx_path,
            verbose=True,
            input_names=['input'],
            output_names=['output'],
            opset_version=13,
        )

        # Convert ONNX to TensorRT
        print('Converting ONNX to TensorRT...')
        trt_path = os.path.join(output_dir, 'model_trt.pth')

        # Use TensorRT's ONNX parser
        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        with (
            trt.Builder(TRT_LOGGER) as builder,
            builder.create_network() as network,
            trt.OnnxParser(network, TRT_LOGGER) as parser,
        ):
            # Set builder configuration
            builder.max_batch_size = 1
            builder.max_workspace_size = 1 << 30
            builder.fp16_mode = True  # Enable FP16 precision

            # Parse the ONNX model
            if not os.path.exists(onnx_path):
                raise FileNotFoundError(f'ONNX file {onnx_path} not found.')

            with open(onnx_path, 'rb') as model_file:
                if not parser.parse(model_file.read()):
                    for error in parser.get_errors():
                        print(f'ONNX Parser Error: {error}')
                    raise RuntimeError('Failed to parse ONNX file.')

            # Build the TensorRT engine
            engine = builder.build_cuda_engine(network)

            # Serialize the engine
            with open(trt_path, 'wb') as trt_file:
                trt_file.write(engine.serialize())

            print(f'TensorRT model saved at {trt_path}')

    except Exception as e:
        print(f'Error during model conversion: {str(e)}')
        raise
