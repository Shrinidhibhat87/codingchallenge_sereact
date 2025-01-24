// File taken from https://github.com/facebookresearch/3detr/tree/main/third_party/pointnet2/_ext_src/include


#pragma once
#include <torch/extension.h>

at::Tensor group_points(at::Tensor points, at::Tensor idx);
at::Tensor group_points_grad(at::Tensor grad_out, at::Tensor idx, const int n);
