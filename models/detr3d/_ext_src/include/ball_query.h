// File taken from https://github.com/facebookresearch/3detr/tree/main/third_party/pointnet2/_ext_src/include

#pragma once
#include <torch/extension.h>

at::Tensor ball_query(at::Tensor new_xyz, at::Tensor xyz, const float radius,
                      const int nsample);
