# 3D Bounding Box Dataset

The folder is meant to load the dataset from the path specified in the config file. The script uses PyTorch Dataloader.
Each sub-folder in the dataset folder is organised as follows:
```
dataset/
├── object_1/               # Each sub-folder is a separate data
│   ├── bbox3d.npy          # Bounding box data in a .npy file
│   ├── mask.npy            # Bounding box mask in a .npy file.
│   ├── pc.npy              # Pointcloud data stored in a .npy file.
│   ├── rgb.png             # Color image file.
├── object_2/               # Each sub-folder is a separate data
├── object_3/               # Each sub-folder is a separate data
...
```

## Example Images from one particular data

![color_img](https://github.com/user-attachments/assets/baf7be5f-68a8-4495-aadb-7c12ac0d2096)


![pcd](https://github.com/user-attachments/assets/153caf81-7c96-436e-9c89-bf3a71048636)


![pcd_with_bbox](https://github.com/user-attachments/assets/50663e14-bc01-43ac-99ed-e13432c67dbc)
