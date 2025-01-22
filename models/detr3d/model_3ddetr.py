"""
The model is inspired by: https://facebookresearch.github.io/3detr
Although this is not a straight copy paste of the code, most of the code is inspired by the above link
"""

import math

import numpy as np
import torch
import torch.nn as nn

from functools import partial
from typing import Optional
from torch import Tensor, nn

from models.detr3d.helpers import (ACTIVATION_DICT, NORM_DICT, WEIGHT_INIT_DICT)
from models.detr3d.transformer_detr import(
    MaskedTransformerEncoder, TransformerDecoder,
    TransformerDecoderLayer, TransformerEncoder,
    TransformerEncoderLayer
)
from utils.miscellaneous import shift_scale_points, scale_points, farthest_point_sample



class GenericMLP(nn.Module):
    """
    A generic Multi-Layer Perceptron (MLP) class that can be configured with various options.
    The class can be used to create both linear and convolutional MLPs for comparison of results.

    Args:
        input_dim (int): The dimension of the input features.
        hidden_dims (list of int): A list containing the dimensions of the hidden layers.
        output_dim (int): The dimension of the output layer.
        norm_fn_name (str, optional): The name of the normalization function to use. Default is None.
        activation (str, optional): The activation function to use. Default is "relu".
        use_conv (bool, optional): Whether to use convolutional layers instead of linear layers. Default is False.
        dropout (float or list of float, optional): Dropout probability for hidden layers. Default is None.
        hidden_use_bias (bool, optional): Whether to use bias in hidden layers. Default is False.
        output_use_bias (bool, optional): Whether to use bias in the output layer. Default is True.
        output_use_activation (bool, optional): Whether to apply activation function on the output. Default is False.
        output_use_norm (bool, optional): Whether to apply normalization on the output. Default is False.
        weight_init_name (str, optional): The name of the weight initialization method. Default is None.
    """
    def __init__(
        self,
        input_dim,
        hidden_dims,
        output_dim,
        norm_fn_name=None,
        activation="relu",
        use_conv=False,
        dropout=None,
        hidden_use_bias=False,
        output_use_bias=True,
        output_use_activation=False,
        output_use_norm=False,
        weight_init_name=None,
    ):
        super().__init__()
        
        # Select activation function
        activation = ACTIVATION_DICT[activation]
        norm = None
        
        # Select normalization function
        if norm_fn_name is not None:
            norm = NORM_DICT[norm_fn_name]
        if norm_fn_name == "ln" and use_conv:
            # Use GroupNorm as a substitute for LayerNorm in Conv1d
            norm = lambda x: nn.GroupNorm(1, x) # NEED TO READ MORE ABOUT THIS

        # Ensure dropout is a list if specified
        if dropout is not None:
            if not isinstance(dropout, list):
                dropout = [dropout for _ in range(len(hidden_dims))]

        # Initialize layers
        layers = []
        prev_dim = input_dim
        for idx, x in enumerate(hidden_dims):
            if use_conv:
                layer = nn.Conv1d(prev_dim, x, 1, bias=hidden_use_bias)
            else:
                layer = nn.Linear(prev_dim, x, bias=hidden_use_bias)
            layers.append(layer)
            
            # Add normalization if specified
            if norm:
                layers.append(norm(x))
            
            # Add activation
            layers.append(activation())
            
            # Add dropout if specified
            if dropout is not None:
                layers.append(nn.Dropout(p=dropout[idx]))
            prev_dim = x

        # Add final layer
        if use_conv:
            layer = nn.Conv1d(prev_dim, output_dim, 1, bias=output_use_bias)
        else:
            layer = nn.Linear(prev_dim, output_dim, bias=output_use_bias)
        layers.append(layer)

        # Add optional output normalization
        if output_use_norm:
            layers.append(norm(output_dim))

        # Add optional output activation
        if output_use_activation:
            layers.append(activation())

        self.layers = nn.Sequential(*layers)

        # Apply weight initialization if specified
        if weight_init_name is not None:
            self.do_weight_init(weight_init_name)

    def do_weight_init(self, weight_init_name):
        """
        Apply weight initialization to the layers.

        Args:
            weight_init_name (str): The name of the weight initialization method.
        """
        func = WEIGHT_INIT_DICT[weight_init_name]
        for (_, param) in self.named_parameters():
            if param.dim() > 1:  # Skip batch normalization/layer normalization parameters
                func(param)

    def forward(self, x):
        """
        Forward pass of the MLP.

        Args:
            x (Tensor): The input tensor.

        Returns:
            Tensor: The output of the MLP.
        """
        return self.layers(x)


class PositionEmbeddingCoordsSine(nn.Module):
    """
    Position Embedding using Sine or Fourier transformations.

    Args:
        temperature (int, optional): Temperature parameter for scaling. Default is 10000.
        normalize (bool, optional): Whether to normalize the input coordinates. Default is False.
        scale (float, optional): Scaling factor. Must be set if `normalize` is True. Default is None.
        pos_type (str, optional): Type of positional encoding, either "sine" or "fourier". Default is "fourier".
        d_pos (int, optional): Dimensionality of the position encoding. Required for "fourier".
        d_in (int, optional): Input dimensionality. Default is 3.
        gauss_scale (float, optional): Scale for Gaussian random matrix used in Fourier embeddings. Default is 1.0.
    """
    def __init__(
        self,
        temperature=10000,
        normalize=False,
        scale=None,
        pos_type="fourier",
        d_pos=None,
        d_in=3,
        gauss_scale=1.0,
    ):
        super().__init__()
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and not normalize:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        assert pos_type in ["sine", "fourier"], "pos_type must be 'sine' or 'fourier'"
        self.pos_type = pos_type
        self.scale = scale
        
        if pos_type == "fourier":
            assert d_pos is not None and d_pos % 2 == 0, "d_pos must be provided and even for 'fourier'"
            B = torch.empty((d_in, d_pos // 2)).normal_() * gauss_scale
            self.register_buffer("gauss_B", B)
            self.d_pos = d_pos

    def get_sine_embeddings(self, xyz, num_channels, input_range):
        """
        Compute sine positional embeddings.

        Args:
            xyz (Tensor): Input coordinates of shape (batch_size, num_points, d_in).
            num_channels (int): Number of output channels.
            input_range (tuple): Range for normalization if applicable.

        Returns:
            Tensor: Sine positional embeddings of shape (batch_size, num_channels, num_points).
        """
        xyz = xyz.clone()

        if self.normalize:
            xyz = shift_scale_points(xyz, src_range=input_range)

        ndim = num_channels // xyz.shape[2]
        if ndim % 2 != 0:
            ndim -= 1
        rems = num_channels - (ndim * xyz.shape[2])
        
        final_embeds = []
        for d in range(xyz.shape[2]):
            cdim = ndim + (2 if rems > 0 else 0)
            rems -= 2 if rems > 0 else 0
            
            dim_t = torch.arange(cdim, dtype=torch.float32, device=xyz.device)
            dim_t = self.temperature ** (2 * (dim_t // 2) / cdim)
            
            raw_pos = xyz[:, :, d] * self.scale
            pos = raw_pos[:, :, None] / dim_t
            pos = torch.stack((pos[:, :, 0::2].sin(), pos[:, :, 1::2].cos()), dim=3).flatten(2)
            final_embeds.append(pos)
        
        return torch.cat(final_embeds, dim=2).permute(0, 2, 1)

    def get_fourier_embeddings(self, xyz, num_channels=None, input_range=None):
        """
        Compute Fourier positional embeddings.

        Args:
            xyz (Tensor): Input coordinates of shape (batch_size, num_points, d_in).
            num_channels (int, optional): Number of output channels. Default is None.
            input_range (tuple, optional): Range for normalization if applicable. Default is None.

        Returns:
            Tensor: Fourier positional embeddings of shape (batch_size, num_channels, num_points).
        """
        if num_channels is None:
            num_channels = self.gauss_B.shape[1] * 2

        bsize, npoints = xyz.shape[0], xyz.shape[1]
        d_out = num_channels // 2
        
        assert d_out <= self.gauss_B.shape[1]
        assert xyz.shape[-1] == self.gauss_B.shape[0]
        
        xyz = xyz.clone()
        if self.normalize:
            xyz = shift_scale_points(xyz, src_range=input_range)
        
        xyz_proj = torch.mm(xyz.view(-1, xyz.shape[-1]), self.gauss_B[:, :d_out]).view(bsize, npoints, d_out)
        return torch.cat((xyz_proj.sin(), xyz_proj.cos()), dim=2).permute(0, 2, 1)

    def forward(self, xyz, num_channels=None, input_range=None):
        """
        Forward method to compute positional embeddings.

        Args:
            xyz (Tensor): Input coordinates of shape (batch_size, num_points, d_in).
            num_channels (int, optional): Number of output channels. Default is None.
            input_range (tuple, optional): Range for normalization if applicable. Default is None.

        Returns:
            Tensor: Positional embeddings of shape (batch_size, num_channels, num_points).
        """
        if self.pos_type == "sine":
            return self.get_sine_embeddings(xyz, num_channels, input_range)
        elif self.pos_type == "fourier":
            return self.get_fourier_embeddings(xyz, num_channels, input_range)
        else:
            raise ValueError(f"Unknown pos_type: {self.pos_type}")

    def extra_repr(self):
        """
        Extra representation string for printing module info.

        Returns:
            str: Extra information about the module.
        """
        return f"type={self.pos_type}, scale={self.scale}, normalize={self.normalize}, " + \
               (f"gaussB_shape={self.gauss_B.shape}, gaussB_sum={self.gauss_B.sum().item()}" if hasattr(self, "gauss_B") else "")


class BoxProcessor(object):
    """
    Convert the output of 3D DETR MLP heads into Bounding boxes.

    Args:
        config (dict): Configuration dictionary containing parameters for box processing.
    """
    def __init__(self, config):
        self.config = config
    
    def compute_predicted_center(
        self,
        center_offset,
        query_xyz,
        point_cloud_dims
    ):
        """
        Compute the predicted center of the bounding box.

        Args:
            center_offset (Tensor): Offset of the predicted center from the query points.
            query_xyz (Tensor): Query points coordinates.
            point_cloud_dims (list of Tensor): Dimensions of the point cloud as [min, max] coordinates.

        Returns:
            Tuple[Tensor, Tensor]: Normalized and unnormalized predicted center coordinates.
        """
        center_unnormalized = center_offset + query_xyz
        # Normalize the center by shift scaling
        center_normalized = shift_scale_points(center_unnormalized, point_cloud_dims)
        
        return center_normalized, center_unnormalized
    
    def compute_predicted_size(
        self,
        size_unnormalized,
        point_cloud_dims
    ):
        """
        Compute the predicted size of the bounding box.

        Args:
            size_unnormalized (Tensor): Unnormalized size of the bounding box.
            point_cloud_dims (list of Tensor): Dimensions of the point cloud as [min, max] coordinates.

        Returns:
            Tensor: Normalized size of the bounding box.
        """
        # Normalize the size by shift scaling
        scene_scale = point_cloud_dims[1] - point_cloud_dims[0]
        # Clamp the size to be within the scene scale
        scene_scale = torch.clamp(scene_scale, min=1e-1)
        size_unnormalized = scale_points(size_unnormalized, scene_scale)
        
        return size_unnormalized
    
    def compute_predicted_angle(
        self,
        angle_logits,
        angle_residuals
    ):
        """
        Compute the predicted angle of the bounding box.

        Args:
            angle_logits (Tensor): Logits for the angle classification.
            angle_residuals (Tensor): Residuals for the angle regression.

        Returns:
            Tensor: Predicted angle of the bounding box.
        """
        if angle_logits.shape[-1] == 1:
            # Special case for datasets with no rotation angle
            angle = angle_logits * 0 + angle_residuals * 0
            angle = angle.squeeze(-1).clamp(min=0)
        else:
            angle_per_cls = 2 * np.pi / 12  # Paper mentions that the angle is quantized into 12 bins
            pred_angle_class = angle_logits.argmax(dim=-1).detach()
            angle_center = angle_per_cls * pred_angle_class
            angle = angle_center + angle_residuals.gather(
                2, pred_angle_class.unsqueeze(-1)
            ).squeeze(-1)
            mask = angle > np.pi
            angle[mask] = angle[mask] - 2 * np.pi
        
        return angle
    
    def compute_objectness_and_class_probability(
        self,
        cls_logits
    ):
        """
        Compute objectness and class probability for a binary classification problem.

        Args:
            cls_logits (Tensor): Logits for the semantic classes of shape (batch_size, num_queries, 2).

        Returns:
            Tuple[Tensor, Tensor]: Class probabilities and objectness probabilities.
        """
        assert cls_logits.shape[-1] == 2, "Semantic class logits must have 2 classes"
        # Compute objectness probability directly using sigmoid
        objectness_prob = torch.sigmoid(cls_logits[..., 1])
        # Class probability for the background class
        class_prob = 1 - objectness_prob
        
        return torch.stack([class_prob, objectness_prob], dim=-1), objectness_prob
    
    def box_parameterization_to_corners(
        self,
        unnormalized_box_center,
        unnormalized_box_size,
        box_angle
    ):
        """
        Convert box parameterization to corner coordinates.

        Args:
            unnormalized_box_center (Tensor): Unnormalized center coordinates of the bounding box.
            unnormalized_box_size (Tensor): Unnormalized size of the bounding box.
            box_angle (Tensor): Angle of the bounding box.

        Returns:
            Tensor: Corner coordinates of the bounding box.
        """
        return self.config.box_parameterization_to_corners(
            unnormalized_box_center,
            unnormalized_box_size,
            box_angle
        )


class Model3DDETR(nn.Module):
    """
    The main 3D DETR model as described by the paper:
    http://arxiv.org/abs/2109.08141
    
    Important points of the model:
    1) Pre-Encoder module is responsible for taking in raw point cloud data and converting to N'X D feature map 
    2) Encoder architecture containing Multihead attention
    3) Decoder architecture is based on transformer architecture
    """
    
    def __init__(
        self,
        pre_encoder,
        encoder,
        decoder,
        config,
        encoder_dim=256,
        decoder_dim=256,
        position_embedding='Fourier',
        mlp_dropout=0.3,
        num_queries=256
    ):
        # Calling parent constructor and initializing the variables
        super().__init__()
        self.pre_encoder = pre_encoder
        self.encoder = encoder
        
        # If inductive bias is needed we can then add masking checks to introduce inductive bias
        # Create a member for encoder to decoder projection
        self.encoder_decoder_projection = GenericMLP(
            input_dim=encoder_dim,
            hidden_dims=[encoder_dim], # if inductive bias is needed, we need to add a hidden layer
            output_dim=decoder_dim,
            norm_fn_name='bn1d',
            activation='relu',
            use_conv=True,
            output_use_activation=True,
            output_use_norm=True,
            output_use_bias=False
        )
        
        # Member for positional embedding
        self.positional_embedding = PositionEmbeddingCoordsSine(
            d_pos=decoder_dim,
            pos_type=position_embedding,
            normalize=True
        )

        # Member for query projection
        self.query_projection = GenericMLP(
            input_dim=encoder_dim,
            hidden_dims=[decoder_dim],
            output_dim=decoder_dim,
            use_conv=True,
            output_use_activation=True,
            hidden_use_bias=True
        )
        
        # Member for decoder
        self.decoder = decoder
        
        # Need to build MLP heads
        self.build_mlp_heads(config, decoder_dim, mlp_dropout)
        
        # Member for number of queries
        self.num_queries = num_queries
        
        # Member for converting MLP output to bounding boxes
        self.box_processor = BoxProcessor(config)
        
        
    def build_mlp_heads(
        self,
        config, # No need for this parameter if not dependent on the config file
        decoder_dim,
        mlp_dropout
    ):
        """Builds the MLP heads for the 3D detection model.

        Args:
            config (dict): Configuration dictionary (not used in this function).
            decoder_dim (int): Dimension of the decoder output.
            mlp_dropout (float): Dropout rate for the MLP layers.
        Returns:
            None
        """
        # Define an MLP function
        mlp_func = partial(
            GenericMLP,
            norm_fn_name='bn1d',
            activation='relu',
            use_conv=True,
            hidden_dims=[decoder_dim, decoder_dim],
            dropout=mlp_dropout,
            input_dim=decoder_dim
        )
        
        # There are no semantic classes when detecting the bounding box here
        # But the two classes that could be detected is background and object
        semantic_class_head = mlp_func(output_dim=2)
        
        # The bounding box head is a 3D bounding box
        # MLP head for the various 3D bounding box parameters
        center_head = mlp_func(output_dim=3)
        size_head = mlp_func(output_dim=3)
        
        # The paper mentions quantizing the angles into 12 bins
        # Angle classification head: Which bin does the angle belong to
        angle_cls_head = mlp_func(output_dim=12)
        # Angle regression head: Finetune the residual within the classification to give continuous value
        angle_reg_head = mlp_func(output_dim=12)
        
        # Aggregate the individual heads
        self.mlp_heads = nn.ModuleDict([
            ("sem_cls_head", semantic_class_head),
            ("center_head", center_head),
            ("size_head", size_head),
            ("angle_cls_head", angle_cls_head),
            ("angle_residual_head", angle_reg_head),
        ])
    
    
    def get_query_embedding(
        self,
        encoder_xyz,
        point_cloud_dims
    ):
        """
        Generate query embeddings by sampling points from the encoder output and applying positional encoding.

        Args:
            encoder_xyz (Tensor): Encoder output coordinates of shape (batch_size, num_points, 3).
            point_cloud_dims (list of Tensor): Dimensions of the point cloud as [min, max] coordinates.

        Returns:
            Tuple[Tensor, Tensor]: Query coordinates and query embeddings.
        """
        # The farthest point sample function is inspired from the PointNet++ paper
        # The input is the encoder_xyz and we need to sample the num_queries sample points
        query_indices = farthest_point_sample(encoder_xyz, self.num_queries)
        # Convert to long
        query_indices = query_indices.long()
        
        # Convert to xyz
        """
        query_xyz = [torch.gather(encoder_xyz[..., x], 1, query_indices) for x in range(3)]
        query_xyz = torch.stack(query_xyz)
        query_xyz = query_xyz.permute(1, 2, 0)
        """
        query_xyz = torch.gather(encoder_xyz, 1, query_indices.unsqueeze(-1).expand(-1, -1, encoder_xyz.size(-1)))
        
        # Positional embedding for the query points using default Fourier transform
        positional_embedding = self.positional_embedding(query_xyz, input_range=point_cloud_dims)
        # Query embeddings
        query_embeddings = self.query_projection(positional_embedding)
        
        return query_xyz, query_embeddings
    
    def _break_up_pc(
        self,
        pc
    ):
        """Break up the point cloud into coordinates and features.
            pc (torch.Tensor): A point cloud tensor of shape (N, M, C) where N is the batch size,
                       M is the number of points, and C is the number of channels. The first
                       three channels are expected to be the XYZ coordinates. Additional
                       channels may include colors, normals, etc.
        Returns:
            tuple: A tuple containing:
            - xyz (torch.Tensor): A tensor of shape (N, M, 3) containing the XYZ coordinates.
            - features (torch.Tensor or None): A tensor of shape (N, C-3, M) containing the additional
                               features, or None if there are no additional features.
        """
        
        # Point cloud may contain colours and/or normals, so we need to break them up
        xyz = pc[..., :3].contiguous()
        features = pc[..., 3:].transpose(1, 2).contiguous() if pc.size(-1) > 3 else None
        
        return xyz, features
    
    def run_encoder(
        self,
        point_clouds
    ):
        """Runs the encoder on the given point clouds.
            point_clouds (torch.Tensor): The input point clouds with shape (batch_size, num_points, num_features).
        
        Returns:
            tuple: A tuple containing:
            - encoder_xyz (torch.Tensor): The encoded xyz coordinates with shape (batch_size, num_points, 3).
            - encoder_features (torch.Tensor): The encoded features with shape (num_points, batch_size, num_features).
            - encoder_indices (torch.Tensor): The indices of the points used in the encoder with shape (batch_size, num_points).
        """
        # break-up point cloud into xyz and features
        xyz, features = self._break_up_pc(point_clouds)
        
        # Pass the point cloud through the pre-encoder
        pre_encoder_xyz, pre_encoder_features, pre_encoder_indices = self.pre_encoder(xyz, features)
        
        # Dimensions are:
        # xyz: (batch_size, num_points, 3)
        # features: (batch_size, num_features, num_points)
        # indices: (batch_size, num_points)
        
        # Multihead attention in encoder expects num_points x batch x num_features
        pre_encoder_features = pre_encoder_features.permute(2, 0, 1)
        
        # XYZ points are in (batch, num_points, 3) order
        encoder_xyz, encoder_features, encoder_indices = self.encoder(
            pre_encoder_features,
            xyz=pre_encoder_xyz
        )
        
        # Checks
        if encoder_indices is None:
            # Encoder does not perform dowmsampling
            encoder_indices = pre_encoder_indices
        else:
            # Use gather to ensure that it works for both FPS and random sampling
            encoder_indices = torch.gather(pre_encoder_indices, 1, encoder_indices.type(torch.int64))
        
        return encoder_xyz, encoder_features, encoder_indices
    
    def get_box_prediction(
        self,
        query_xyz,
        point_cloud_dims,
        box_features
    ):
        """
        Generate box predictions from the decoder output features.

        Args:
            query_xyz (Tensor): Query coordinates of shape (batch_size, num_queries, 3).
            point_cloud_dims (list of Tensor): Dimensions of the point cloud as [min, max] coordinates.
            box_features (Tensor): Decoder output features of shape (num_layers, num_queries, batch, channel).

        Returns:
            dict: A dictionary containing the final and auxiliary outputs with keys:
                - "outputs": Output from the last layer of the decoder.
                - "auxiliary_outputs": Outputs from the intermediate layers of the decoder.
        """
        
        # Dimensions are:
        # query_xyz: (batch_size, num_queries, 3)
        # point_cloud_dims: [min, max] coordinates
        # box_features: (num_layers, num_queries, batch, channel)
        
        # Change box_features to ((num_layers x batch), channel, num_queries)
        box_features = box_features.permute(0, 2, 3, 1)
        num_layers, batch_size, channel, num_queries = (
            box_features.shape[0],
            box_features.shape[1],
            box_features.shape[2],
            box_features.shape[3]
        )
        box_features = box_features.reshape(num_layers * batch_size, channel, num_queries)
        
        # The MLP head outputs are ((num_layers x batch), num_output, num_queries) so transpose the last two dimensions
        class_logits = self.mlp_heads["sem_cls_head"](box_features).transpose(1, 2)
        center_offset = (
            self.mlp_heads["center_head"](box_features).sigmoid().transpose(1, 2) - 0.5
        )
        size_normalized = (
            self.mlp_heads["size_head"](box_features).sigmoid().transpose(1, 2)
        )
        angle_logits = self.mlp_heads["angle_cls_head"](box_features).transpose(1, 2)
        angle_residual_normalized = self.mlp_heads["angle_residual_head"](box_features).transpose(1, 2)
        
        # Reshape the outputs to (num_layers, batch, num_queries, num_output)
        class_logits = class_logits.reshape(num_layers, batch_size, num_queries, -1)
        center_offset = center_offset.reshape(num_layers, batch_size, num_queries, -1)
        size_normalized = size_normalized.reshape(num_layers, batch_size, num_queries, -1)
        angle_logits = angle_logits.reshape(num_layers, batch_size, num_queries, -1)
        angle_residual_normalized = angle_residual_normalized.reshape(num_layers, batch_size, num_queries, -1)
        
        # Get the angle residual values
        angle_residual = angle_residual_normalized * (np.pi / angle_residual_normalized.shape[-1])
        
        # Placeholder for outputs
        outputs = []
        
        for i in range(num_layers):
            # The box processor class converts the MLP outputs to bounding boxes
            (center_normalized, center_unnormalized) = self.box_processor.compute_predicted_center(
                center_offset[i],
                query_xyz,
                point_cloud_dims
            )
            # Get the angle
            angle_contiguous = self.box_processor.compute_predicted_angle(
                angle_logits[i],
                angle_residual[i]
            )
            # Get size unnormlized
            size_unnormalized = self.box_processor.compute_predicted_size(
                size_normalized[i],
                point_cloud_dims
            )
            # Get the corners of the bounding box
            box_corners = self.box_processor.box_parameterization_to_corners(
                center_unnormalized,
                size_unnormalized,
                angle_contiguous
            )
            
            # We use the below part only for evaluation (matching/mAP) and not loss
            with torch.no_grad():
                # Compute objectness and class probability
                (semantic_class_prob, objectness_prob) = self.box_processor.compute_objectness_and_class_probability(
                    class_logits[i]
                )

            box_prediction = {
                "semantic_class_logits": class_logits[i],
                "center_normalized": center_normalized.contiguous(),
                "center_unnormalized": center_unnormalized,
                "size_normalized": size_normalized[i],
                "angle_logits": angle_logits[i],
                "angle_residual": angle_residual[i],
                "angle_residual_normalized": angle_residual_normalized[i],
                "angle_contiguous": angle_contiguous,
                "objectness_prob": objectness_prob,
                "semantic_class_prob": semantic_class_prob,
                "box_corners": box_corners
            }
            
            outputs.append(box_prediction)
        
        # Intermediate decoder layer outputs are only used during training
        auxiliary_outputs = outputs[:-1]
        outputs = outputs[-1]
        
        return {
            "outputs": outputs, # output from the last layer of the decoder
            "auxiliary_outputs": auxiliary_outputs # Output from the intermediate layers of the decoder
        }
        
    def forward(
        self,
        inputs,
        encoder_only=False
    ):
        """_summary_

        Args:
            inputs (_type_): _description_
            encoder_only (bool, optional): _description_. Defaults to False.
        """
        # Get the point clouds from the input
        point_clouds = inputs["point_clouds"]
        
        # Run it through the encoder
        encoder_xyz, encoder_features, encoder_indices = self.run_encoder(point_clouds)
        # Modify the shape encoder features
        encoder_features = self.encoder_decoder_projection(
            encoder_features.permute(1, 2, 0)
        ).permute(2, 0, 1)
        
        # Note down the shape of the intermediate features
        if encoder_only:
            return encoder_xyz, encoder_features.transpose(0, 1)
        
        # Append the Point cloud dimensions
        point_cloud_dims = [inputs["point_cloud_dims_min"], inputs["point_cloud_dims_max"]]
        
        # Get the query embeddings
        query_xyz, query_embeddings = self.get_query_embedding(encoder_xyz, point_cloud_dims)
        # Query embedding shape: (batch_size, channels, num_points)
        encoder_pos = self.positional_embedding(encoder_xyz, input_range=point_cloud_dims)
        
        # The decoder expects the query embeddings to be in the shape (num_points, batch_size, channels)
        encoder_pos = encoder_pos.permute(2, 0, 1)
        # Similarly, the query embeddings are in the shape (batch_size, channels, num_points)
        query_embeddings = query_embeddings.permute(2, 0, 1)
        target = torch.zeros_like(query_embeddings)
        
        # Get the box features
        box_features = self.decoder(
            target,
            encoder_features,
            query_embeddings, # Query position is embeddings
            encoder_pos # Position is encoder positional embedding
        )[0]
        
        # Predict the bounding boxes
        box_predictions = self.get_box_prediction(query_xyz, point_cloud_dims, box_features)
        
        return box_predictions

"""
All the code for the set-aggregation downsampling operation as mentioned in the PointNet++ paper
"""

# Function to build pre_encoder
def build_preencoder(args):
    """
    Builds the preencoder configuration for the model.

    Args:
        args: An argument parser object that contains the following attributes:
            - use_color (int): A flag indicating whether to use color information (1 for True, 0 for False).
            - encoder_dim (int): The dimension of the encoder output.

    Returns:
        list: A list of integers representing the dimensions of the MLP (Multi-Layer Perceptron) used in the preencoder.
    """
    mlp_dimensions = [3 * int(args.use_color), 64, 128, args.encoder_dim]
    # something to do with PointnetSAMModuleVotes
    return mlp_dimensions

# Function to build encoder
def build_encoder(args):
    """
    Builds and returns an encoder based on the specified arguments.
    Args:
        args: An object containing the following attributes:
            - encoder_type (str): The type of encoder to build. Can be "Vanilla" or "masked".
            - encoder_dim (int): The dimension of the encoder model.
            - encoder_nhead (int): The number of heads in the multi-head attention mechanism.
            - encoder_ffn_dim (int): The dimension of the feedforward network.
            - encoder_dropout (float): The dropout rate.
            - encoder_activation (str): The activation function to use.
            - encoder_num_layers (int): The number of layers in the encoder.
    Returns:
        encoder: An instance of TransformerEncoder or MaskedTransformerEncoder based on the specified encoder type.
    Raises:
        ValueError: If the encoder_type is not recognized.
    """
    if args.encoder_type == "Vanilla":
        encoder_layer = TransformerEncoderLayer(
            d_model = args.encoder_dim,
            nhead=args.encoder_nhead,
            dim_feedforward=args.encoder_ffn_dim,
            dropout=args.encoder_dropout,
            activation=args.encoder_activation
        )
        encoder = TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=args.encoder_num_layers
        )
    
    elif args.encoder_type in ["masked"]:
        encoder_layer = TransformerEncoderLayer(
            d_model = args.encoder_dim,
            nhead=args.encoder_nhead,
            dim_feedforward=args.encoder_ffn_dim,
            dropout=args.encoder_dropout,
            activation=args.encoder_activation
        )
        # There is a need for interim downsampling that uses the PointNet++ set abstraction
        # But we are currently ignoring that part
        masking_radius = [math.pow(x, 2) for x in [0.4, 0.8, 1.2]]
        
        encoder = MaskedTransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=3,
            interim_downsampling=None,
            masking_radius=masking_radius,
        )
    else:
        raise ValueError(f"Unknown encoder type: {args.encoder_type}")
    
    return encoder

# Function to build the decoder
def build_decoder(args):
    """
    Builds a Transformer decoder using the provided arguments.
    Args:
        args: An object containing the following attributes:
            - decoder_dim (int): The dimension of the decoder model.
            - decoder_nhead (int): The number of heads in the multihead attention mechanism.
            - decoder_ffn_dim (int): The dimension of the feedforward network.
            - decoder_dropout (float): The dropout rate for the decoder.
            - decoder_num_layers (int): The number of layers in the decoder.
    Returns:
        TransformerDecoder: A Transformer decoder instance configured with the specified parameters.
    """
    decoder_layer = TransformerDecoderLayer(
        d_model=args.decoder_dim,
        nhead=args.decoder_nhead,
        dim_feedforward=args.decoder_ffn_dim,
        dropout=args.decoder_dropout,
    )
    decoder = TransformerDecoder(
        decoder_layer=decoder_layer,
        num_layers=args.decoder_num_layers,
        return_intermediate=True
    )
    
    return decoder


# Function to build the 3D DETR model
def build_3ddetr_model(args, config):
    """
    Build the 3DDETR model and its output processor.
    Args:
        args (Namespace): The arguments containing model hyperparameters and configurations.
            - encoder_dim (int): Dimension of the encoder.
            - decoder_dim (int): Dimension of the decoder.
            - position_embedding (str): Type of position embedding to use.
            - mlp_dropout (float): Dropout rate for the MLP layers.
            - num_queries (int): Number of queries for the decoder.
        config (dict): Configuration dictionary for the model and processor.
    Returns:
        tuple: A tuple containing:
            - model (Model3DDETR): The constructed 3DDETR model.
            - output_processor (BoxProcessor): The processor for the model's output.
    """
    pre_encoder = build_preencoder(args)
    encoder = build_encoder(args)
    decoder = build_decoder(args)
    model = Model3DDETR(
        pre_encoder=pre_encoder,
        encoder=encoder,
        decoder=decoder,
        config=config,
        encoder_dim=args.encoder_dim,
        decoder_dim=args.decoder_dim,
        position_embedding=args.position_embedding,
        mlp_dropout=args.mlp_dropout,
        num_queries=args.num_queries
    )
    output_processor = BoxProcessor(config=config)
    
    return model, output_processor
