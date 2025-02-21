# 3D Bounding Box Prediction Code

The intention of the project is to formulate and End-to-End deep learning approach that results in a model that is capable of predicting bounding box coordinates in 3D space when given a pointcloud input.

## Project Components

The project consists of several key components:

1. **Custom Dataloader with Data Augmentation Methods**
    - The dataloader is designed to handle the Sereact data, providing functionalities to display RGB images, colored pointclouds, and pointclouds with bounding boxes.
    - Data augmentation methods are included to enhance the training process.

2. **Trainer Class**
    - Responsible for training and evaluating the model.
    - Includes methods for setting up the training pipeline, managing epochs, and evaluating performance.

3. **[3DETR Implementation](https://github.com/Shrinidhibhat87/codingchallenge_sereact/tree/main/models#3detr-model)**
    - The project includes an implementation of the 3DETR model with modifications to suit the specific requirements of this project.
    - [Original 3DETR Implementation](https://github.com/facebookresearch/3detr/tree/main)

4. **[Loss Function](https://github.com/Shrinidhibhat87/codingchallenge_sereact/tree/main/losses#loss-function-design-and-components)**
    - Utilizes Hungarian Matching for set-to-set prediction.
    - Employs a simple L1 loss for bounding box coordinate predictions.

5. **Utility Folder**
    - Contains essential functions for importing and exporting models, including support for lower precision models.

These components work together to create an end-to-end deep learning solution for 3D bounding box prediction from pointcloud data.

## System Overview

![SystemOverview](https://github.com/user-attachments/assets/1a9b0ea0-0f18-4258-96ac-86e484c40e1c)

This image provides a rough and crude idea of the project's overall architecture and workflow.


## Getting Started

To get started with the project, follow these steps:

### Cloning the Repository

First, clone the repository to your local machine using the following command:

```bash
git clone https://github.com/Shrinidhibhat87/codingchallenge_sereact.git
cd codingchallenge_sereact
```

### Setting Up a Python Virtual Environment

Create a Python virtual environment to manage dependencies:

```bash
python3 -m venv venv
source venv/bin/activate
```

### Installing Required Libraries

Install the required libraries from the `requirements.txt` file:

```bash
pip install -r requirements.txt
```

### Handling the `open3d` Library

The `open3d` library is used for visualization purposes and can sometimes be troublesome to install. If you encounter issues, you can either comment out the `open3d` import statements in the code or install it separately:

```bash
pip install open3d
```

### Dataset Preparation

Ensure that you have the dataset stored in a dedicated path. The dataloader expects the datasets to be organized in a specific folder structure. Update the dataset path in the configuration file accordingly.

### Running the Project

After setting up the environment and updating the dataset path, you can run the main script:

```bash
python main.py
```

## Additional Setup Explanation
Because this project uses C++/CUDA files, especially for point sampling before the TransformerEncoder, ensure that there is no Dynamic linking issue.

To run the script, in the [detr3d repository](models/detr3d/setup.py) we need to go in the repository and run
```bash
python3 setup.py build_ext --inplace
```
This can be confirmed by opening a terminal and simply trying out

```python
import _ext_src
```

## Resolving Dynamic Linking Issues
If the above step does not work, then there could be an issue with the Dynamic linker to the .so file. Therefore we need to add this to the bashrc in order for it to find the library.

To ensure there are no dynamic linking issues with the C++/CUDA files, follow these steps:

1. **Find the Directory Containing `libc10.so`**
    - Locate the directory where the `libc10.so` file is stored. This is typically within the PyTorch library directory.

2. **Export the Path to `LD_LIBRARY_PATH`**
    - Use the following command to export the path:
    ```bash
    export LD_LIBRARY_PATH=/path/to/torch/lib:$LD_LIBRARY_PATH
    ```

3. **Add the Path to `bashrc` and Source It**
    - Add the export command to your `~/.bashrc` file to ensure it is set for future sessions:
    ```bash
    echo 'export LD_LIBRARY_PATH=/path/to/torch/lib:$LD_LIBRARY_PATH' >> ~/.bashrc
    ```
    - Source the `~/.bashrc` file to apply the changes immediately:
    ```bash
    source ~/.bashrc
    ```

By following these steps, you can resolve any dynamic linking issues related to the `libc10.so` file.

## Project Structure Overview

The repository is organized as follows:

```
codingchallenge_sereact/
├── config/                 # Contains the configuration files to run the project
├── dataloader/             # Dataloader specific to the dataset
├── losses/                 # Folder dedicated to handle losses specific to the project
├── models/                 # Contains model definitions and related scripts
│   ├── detr3d/             # Implementation of the 3DETR model
│   ├── _ext_src/            # C++/CUDA extensions for point sampling
├── trainer/                # Folder containting the trainer class
├── utility/                # utility folder containing utility functions
├── requirements.txt        # List of required Python libraries
├── README.md               # Project documentation
├── .pre-commit-config.yaml # Configuration for pre-commit hooks
├── pyproject.toml          # Project configuration file
└── main.py                 # Main script to run the project
```

This structure ensures a clear separation of different components, making the project easy to navigate and maintain.
## Example outputs

**Red Color --> Predicted Boxes**

**Green Color --> Ground truth Boxes**

![Red -> Predicted Boxes ; Green -> GT Boxes](https://github.com/user-attachments/assets/030865d2-045f-4a0d-b8e8-b284deec92ec)


https://github.com/user-attachments/assets/1f1b1ccf-5a8f-4654-b9d1-fad85c12e9f5


## Future TODO

1. Complete ReadMe
2. Code cleanup.
3. Add more type checkers in order to have uniform coding style.
4. Add Unit tests using Pytest.
