"""File that contains utility functions for model realted operations."""

import torch


def load_weights(
    model: torch.nn.Module,
    pre_trained_weights_path: str,
    device: torch.device,
    encoder_only: bool = True,
) -> None:
    """Load the pretrained weights from the path specified

    Args:
        model (torch.nn.Module): The model to load the weights into.
        pre_trained_weights_path (str): The path that contains the pretrained weights.
        device (torch.device): The device the model is on.
        encoder_only (bool, optional): Boolean to choose if the weight is encoder only.
            Defaults to True.
    """
    # Load the pretrained weights from the path
    pretrained_state_dict = torch.load(pre_trained_weights_path, map_location=device)

    # Get the current state of the model
    model_state_dict = model.state_dict()

    # By default we have to load only encoder weights, so we filter them out.
    filtered_state_dict = {}
    if encoder_only:
        for name, param in model_state_dict.items():
            # Check if the parameters are from the pre_encoder or the encoder part
            if name.startswith('pre_encoder.') or name.startswith('encoder.'):
                # Additional check for matching the size
                if (
                    name in pretrained_state_dict['model']
                    and pretrained_state_dict['model'][name].shape == param.shape
                ):
                    filtered_state_dict[name] = pretrained_state_dict['model'][name]
                else:
                    print(
                        f'Skipping parameter {name} as it is not found in the pretrained weights or the shapes mismatch.'
                    )
                    if name in pretrained_state_dict['model']:
                        print(
                            f'Expected shape: {param.shape}, Got shape: {pretrained_state_dict[name].shape}'
                        )

        # Load the filtered state dict
        missing_keys, unexpected_keys = model.load_state_dict(filtered_state_dict, strict=False)

        # Print the summary
        print(f'Successfully loaded {len(filtered_state_dict)} encoder weights.')
        print(f'Missing keys: {missing_keys}')
        print(f'Unexpected keys: {unexpected_keys}')

        # Extra prints that can be removed later on.
        for name, param in model.named_parameters():
            if name.startswith('pre_encoder') or name.startswith('encoder'):
                status = 'YES' if name in filtered_state_dict else 'NO'
                print(f'{status} {name}: {param.shape}')
    else:
        print('Loading the full model weights is not implemented yet!')
