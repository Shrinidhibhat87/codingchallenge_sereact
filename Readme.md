# Custom Dataloader for Sereact Data

## About the Custom Dataloader

The dataloader object for the Sereact data is written in this project. The dataloader is capable of showing the RGB image, colored pointcloud, and the colored pointcloud with a bounding box in it.

To visualize the data, one needs to set the debug flag to True. The syntax is:
```bash
python3 main.py <path to folder> --debug<> --ds_number<>
```
Example:
```bash
python3 main.py /home/shrinidhibhat/TransformerCoding/sereactcoding/data/dl_challenge --debug True --ds_number 15
```

## Example Images

Color Image

![Color Image](https://github.com/Shrinidhibhat87/codingchallenge_sereact/assets/36807594/13413e67-431b-41d8-93a3-4528204756b8)


PointCloud Image
![PointCloud Image](https://github.com/Shrinidhibhat87/codingchallenge_sereact/assets/36807594/2cfab025-5846-4a68-929a-f0fec378f56f)


Bounding box in PointCloud
![Bounding Box in PointCloud](https://github.com/Shrinidhibhat87/codingchallenge_sereact/assets/36807594/d1318e70-076b-4b6f-b4fa-d411a42b4b36)

## Future TODO

1. Expand the project to build methods that can do 3D bounding box predictions based on the data.
    - Research literature out there to get relevant works which can be done in the limited hardware capacity available.
    - Construct a model or use open-source models based on the literature survey.
    - Build a training pipeline for model training, including transforms, optimizers, etc.
2. Once the model is trained, store weights or best checkpoints in order to use later.
3. Enable infrastructure to deploy the model.
    - Create a Docker Compose script capable of loading the model using the pre-trained weights.
    - Ensure the server created has all the necessary dependencies.
