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

3. **[3DETR Implementation](https://github.com/facebookresearch/3detr/tree/main)**
    - The project includes an implementation of the 3DETR model with modifications to suit the specific requirements of this project.

4. **Loss Function**
    - Utilizes Hungarian Matching for set-to-set prediction.
    - Employs a simple L1 loss for bounding box coordinate predictions.

5. **Utility Folder**
    - Contains essential functions for importing and exporting models, including support for lower precision models.

These components work together to create an end-to-end deep learning solution for 3D bounding box prediction from pointcloud data.

## System Overview

![System Overview]![SystemOverview](https://github.com/user-attachments/assets/1a9b0ea0-0f18-4258-96ac-86e484c40e1c)

This image provides a rough and crude idea of the project's overall architecture and workflow.

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

## Example Images

Color Image

![Color Image](https://github.com/Shrinidhibhat87/codingchallenge_sereact/assets/36807594/13413e67-431b-41d8-93a3-4528204756b8)


PointCloud Image
![PointCloud Image](https://github.com/Shrinidhibhat87/codingchallenge_sereact/assets/36807594/2cfab025-5846-4a68-929a-f0fec378f56f)


Bounding box in PointCloud
![Bounding Box in PointCloud](https://github.com/Shrinidhibhat87/codingchallenge_sereact/assets/36807594/d1318e70-076b-4b6f-b4fa-d411a42b4b36)

## Future TODO

1. Complete ReadMe
    - Add ways to clone the repository and run the script.
    - Explain the structure overview.
2. Code cleanup.
3. Add more type checkers in order to have uniform coding style.
4. Add Unit tests using Pytest.
