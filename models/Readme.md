# 3DETR Model

As mentioned previously, the model is heavily inspired by the original implementation of [3DETR](https://github.com/facebookresearch/3detr/tree/main)

## Model Component Breakdown

![mermaid-diagram-2025-02-21-112447](https://github.com/user-attachments/assets/bbb7a14e-a872-4e83-9602-ad00a68fea94)

There are primarily 5 components, although the diagram shows only 4 as it omits the prediction heads:

1. **Pre-Encoder**
    - Responsible for PointCloud preprocessing and reducing the points to fit the Transformer Encoder.
    - Downsample and use of Farthest-Point-Sampling to sample 2048 points randomly.
    - Each point is projected to a 256 dimensional feature space after set-aggregation.
    - Therefore the output from the pre_encoder is a 2048 x 256 matrix that acts as an input to the Transformer Encoder.

2. **Transformer Encoder**
    - Configurable number of layers, but chose the default 3 layer encoder layer.
    - Each layer consists of self-attention blocks that uses multihead attention, which in this case is also the default 4.
    - The attention block is followed by a MLP.
    - Output of the attention block is a 2048 x 2048 matrix that is used to attend to the 256 dimensional feature space output.
    - The Transformer Encoder layer also uses the default ReLU and LayerNorm.
    - NOTE: The masked encoder, although present in code, has not been tested.

3. **Query Positonal Embeddings**
    - Use the default non-parametric query embeddings and low-frequency Fourier positional embeddings.
    - The point features after the downsampling is used along with a set of query embeddings as input to the Transformer Decoder.
    - Use of the positional embeddings in the decoder benefits the decoder as it does not have direct access to the coordinates.

4. **Transformer Decoder**
    - The two inputs here are: Encoder feature outputs (2048x256) and location query embeddings (B x 256).
    - The decoder has configurable layers that use cross-attention between the location embeddings, encoder features and self-attention between the box_features.
    - As usual, there is a presence of LayerNorm, dropout and also ReLU non-linearity.
  
4. **Box Output Prediction using MLP**
    - The output prediction is based on the box features from the decoder.
    - There are changes from the original implementation where there is no need for prediciton of class.
    - There are separate MLPs to predict different features and the corners are calculated based on the prediction.


## Use of Pre-trained weights
Used the 4th entry of the Pretrained [Model table](https://github.com/facebookresearch/3detr/tree/main?tab=readme-ov-file#pretrained-models). The pre-trained model was trained on the Scannet dataset and was trained for 1080 epochs.

## Example Model Weights
| Epoch | Val/IoU@0.25 | Val/IoU@0.50 | Val/Mean IoU |
|-------|--------------|--------------|--------------|
|   10  |    0.34807   |    0.063536  |    0.21278   |
|   30  |    0.39227   |    0.060773  |    0.22693   |

NOTE: The model weights are available and can be shared upon request.

## Future TODO

1. Build a MLP head that directly predicts the corner coordinates based on the ground truth.
2. Perform a complete analysis using different model weights.
3. Try and test out masked encoder and see its effects on model performance.

