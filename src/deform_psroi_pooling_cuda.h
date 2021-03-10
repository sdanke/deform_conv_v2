#pragma once
#include <torch/extension.h>

#ifdef __cplusplus
extern "C"
{
#endif

    std::tuple<at::Tensor, at::Tensor>
    deform_psroi_pooling_cuda_forward(const at::Tensor &input,
                                    const at::Tensor &bbox,
                                    const at::Tensor &trans,
                                    const int64_t no_trans,
                                    const double spatial_scale,
                                    const int64_t output_dim,
                                    const int64_t group_size,
                                    const int64_t pooled_size,
                                    const int64_t part_size,
                                    const int64_t sample_per_part,
                                    const double trans_std);

    std::tuple<at::Tensor, at::Tensor>
    deform_psroi_pooling_cuda_backward(const at::Tensor &out_grad,
                                    const at::Tensor &input,
                                    const at::Tensor &bbox,
                                    const at::Tensor &trans,
                                    const at::Tensor &top_count,
                                    const int64_t no_trans,
                                    const double spatial_scale,
                                    const int64_t output_dim,
                                    const int64_t group_size,
                                    const int64_t pooled_size,
                                    const int64_t part_size,
                                    const int64_t sample_per_part,
                                    const double trans_std);

#ifdef __cplusplus
}
#endif