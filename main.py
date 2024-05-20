"""
Main file.
Currently capable of only loading data.
Future work would inspire application of methods to understand the scene.
"""

import argparse
import os
import sys
from typing import List

from dataloader import SereactDataloader


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed command-line arguments containing the folder path.
    """
    parser = argparse.ArgumentParser(description='Dataloader parser.')
    parser.add_argument(
        'input_folder_path',
        type=str,
        help='Path to the folder containing subfolders with required files',
    )
    parser.add_argument('--debug', type=bool, default=False, help='Flag to degug and visualize')
    parser.add_argument('--ds_number', type=int, default=15, help='Data set index to visualize')
    return parser.parse_args()


def validate_folder_structure(folder_path: str) -> str:
    """
    Validate the folder structure to ensure it contains subdirectories, each with specific required files.

    Args:
        folder_path (str): Path to the main folder to be validated.

    Raises:
        NotADirectoryError: If the provided path is not a directory.
        ValueError: If the folder does not contain any subdirectories.
        FileNotFoundError: If any required file is missing in the subdirectories.

    Returns:
        str: The validated folder path.
    """
    # Check if the provided path is a directory
    if not os.path.isdir(folder_path):
        raise NotADirectoryError(f'The path {folder_path} does not exist or is not a directory.')

    # List all subdirectories in the given folder
    subdirectories = [
        os.path.join(folder_path, subdir)
        for subdir in os.listdir(folder_path)
        if os.path.isdir(os.path.join(folder_path, subdir))
    ]

    # Ensure there is at least one subdirectory
    if not subdirectories:
        raise ValueError(f'The folder {folder_path} does not contain any subdirectories.')

    # Files that must be present in each subdirectory
    required_files: List[str] = ['rgb.jpg', 'bbox3d.npy', 'mask.npy', 'pc.npy']

    for subdir in subdirectories:
        for required_file in required_files:
            file_path = os.path.join(subdir, required_file)
            if not os.path.isfile(file_path):
                raise FileNotFoundError(
                    f'The required file {required_file} is missing in the subdirectory {subdir}.'
                )

    return folder_path


def main() -> None:
    """
    Main function to parse arguments, validate the folder structure, and handle any exceptions.
    """
    args = parse_args()
    try:
        folder_path = validate_folder_structure(args.input_folder_path)
        print('Valid input folder path')
        datasettester = SereactDataloader(source_path=folder_path, debug=args.debug)
        datasettester.visualize_data(args.ds_number)

    except (NotADirectoryError, FileNotFoundError, ValueError) as e:
        print(e, file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
