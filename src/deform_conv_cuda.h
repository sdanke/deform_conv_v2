#pragma once
#include <torch/extension.h>

#ifdef __cplusplus
extern "C"
{
#endif

    at::Tensor
    deform_conv_cuda_forward(const at::Tensor &input,
                        const at::Tensor &weight,
                        const at::Tensor &bias,
                        const at::Tensor &offset,
                        const int64_t kernel_h,
                        const int64_t kernel_w,
                        const int64_t stride_h,
                        const int64_t stride_w,
                        const int64_t pad_h,
                        const int64_t pad_w,
                        const int64_t dilation_h,
                        const int64_t dilation_w,
                        const int64_t group,
                        const int64_t deformable_group, 
                        const int64_t im2col_step);

    std::vector<at::Tensor>
    deform_conv_cuda_backward(const at::Tensor &input,
                        const at::Tensor &weight,
                        const at::Tensor &bias,
                        const at::Tensor &offset,
                        const at::Tensor &grad_output,
                        const int64_t kernel_h, 
                        const int64_t kernel_w,
                        const int64_t stride_h, 
                        const int64_t stride_w,
                        const int64_t pad_h, 
                        const int64_t pad_w,
                        const int64_t dilation_h, 
                        const int64_t dilation_w,
                        const int64_t group,
                        const int64_t deformable_group, 
                        const int64_t im2col_step);

#ifdef __cplusplus
}
#endif