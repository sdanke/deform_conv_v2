#include <torch/script.h>
// #include "deform_conv_cpu.h"
// #include "deform_psroi_pooling_cpu.h"
// #include "modulated_deform_conv_cpu.h"

// #ifdef WITH_CUDA
#include "deform_conv_cuda.h"
#include "deform_psroi_pooling_cuda.h"
#include "modulated_deform_conv_cuda.h"
// #endif

void init() { 

}

at::Tensor
deform_conv_forward(const at::Tensor &input,
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
               const int64_t im2col_step)
{
    if (input.type().is_cuda()) {
    #ifdef WITH_CUDA
        return deform_conv_cuda_forward(input, weight, bias, offset,
                                   kernel_h, kernel_w,
                                   stride_h, stride_w,
                                   pad_h, pad_w,
                                   dilation_h, dilation_w,
                                   group,
                                   deformable_group, 
                                   im2col_step);
    #else
        AT_ERROR("Not compiled with GPU support");
    #endif
    }
    AT_ERROR("Not implemented on the CPU");
}

std::vector<at::Tensor>
deform_conv_backward(const at::Tensor &input,
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
                const int64_t im2col_step)
{
    if (input.type().is_cuda()) {
    #ifdef WITH_CUDA
        return deform_conv_cuda_backward(input,
                                    weight,
                                    bias,
                                    offset,
                                    grad_output,
                                    kernel_h, kernel_w,
                                    stride_h, stride_w,
                                    pad_h, pad_w,
                                    dilation_h, dilation_w,
                                    group,
                                    deformable_group,
                                    im2col_step);
    #else
        AT_ERROR("Not compiled with GPU support");
    #endif
    }
    AT_ERROR("Not implemented on the CPU");
}


std::tuple<at::Tensor, at::Tensor>
deform_psroi_pooling_forward(const at::Tensor &input,
                             const at::Tensor &bbox,
                             const at::Tensor &trans,
                             const int64_t no_trans,
                             const double spatial_scale,
                             const int64_t output_dim,
                             const int64_t group_size,
                             const int64_t pooled_size,
                             const int64_t part_size,
                             const int64_t sample_per_part,
                             const double trans_std)
{
    if (input.type().is_cuda())
    {
#ifdef WITH_CUDA
        return deform_psroi_pooling_cuda_forward(input,
                                                 bbox,
                                                 trans,
                                                 no_trans,
                                                 spatial_scale,
                                                 output_dim,
                                                 group_size,
                                                 pooled_size,
                                                 part_size,
                                                 sample_per_part,
                                                 trans_std);
#else
        AT_ERROR("Not compiled with GPU support");
#endif
    }
    AT_ERROR("Not implemented on the CPU");
}

std::tuple<at::Tensor, at::Tensor>
deform_psroi_pooling_backward(const at::Tensor &out_grad,
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
                              const double trans_std)
{
    if (input.type().is_cuda())
    {
#ifdef WITH_CUDA
        return deform_psroi_pooling_cuda_backward(out_grad,
                                                  input,
                                                  bbox,
                                                  trans,
                                                  top_count,
                                                  no_trans,
                                                  spatial_scale,
                                                  output_dim,
                                                  group_size,
                                                  pooled_size,
                                                  part_size,
                                                  sample_per_part,
                                                  trans_std);
#else
        AT_ERROR("Not compiled with GPU support");
#endif
    }
    AT_ERROR("Not implemented on the CPU");
}


at::Tensor
modulated_deform_conv_forward(const at::Tensor &input,
               const at::Tensor &weight,
               const at::Tensor &bias,
               const at::Tensor &offset,
               const at::Tensor &mask,
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
               const int64_t im2col_step)
{
    if (input.type().is_cuda())
    {
#ifdef WITH_CUDA
        return modulated_deform_conv_cuda_forward(input, weight, bias, offset, mask,
                                   kernel_h, kernel_w,
                                   stride_h, stride_w,
                                   pad_h, pad_w,
                                   dilation_h, dilation_w,
                                   group,
                                   deformable_group,
                                   im2col_step);
#else
        AT_ERROR("Not compiled with GPU support");
#endif
    }
    AT_ERROR("Not implemented on the CPU");
}

std::vector<at::Tensor>
modulated_deform_conv_backward(const at::Tensor &input,
                const at::Tensor &weight,
                const at::Tensor &bias,
                const at::Tensor &offset,
                const at::Tensor &mask,
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
                const int64_t im2col_step)
{
    if (input.type().is_cuda())
    {
#ifdef WITH_CUDA
        return modulated_deform_conv_cuda_backward(input,
                                    weight,
                                    bias,
                                    offset,
                                    mask,
                                    grad_output,
                                    kernel_h, kernel_w,
                                    stride_h, stride_w,
                                    pad_h, pad_w,
                                    dilation_h, dilation_w,
                                    group,
                                    deformable_group,
                                    im2col_step);
#else
        AT_ERROR("Not compiled with GPU support");
#endif
    }
    AT_ERROR("Not implemented on the CPU");
}

static auto registry =
    torch::RegisterOperators("dcn_v2_ops::deform_conv_forward", &deform_conv_forward)
    .op("dcn_v2_ops::deform_conv_backward", &deform_conv_backward)
    .op("dcn_v2_ops::deform_psroi_pooling_forward", &deform_psroi_pooling_forward)
    .op("dcn_v2_ops::deform_psroi_pooling_backward", &deform_psroi_pooling_backward)
    .op("dcn_v2_ops::modulated_deform_conv_forward", &modulated_deform_conv_forward)
    .op("dcn_v2_ops::modulated_deform_conv_backward", &modulated_deform_conv_backward);
